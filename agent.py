"""
agent.py
--------
"The analyst," now running on Google's Gemini API instead of Anthropic's --
Gemini's Flash models have a genuine no-credit-card free tier. Same overall
architecture as before: a loop that calls the model, runs whichever tool it
asks for, and feeds the result back, until the model returns plain text.

Three tools:
  1. web_search           -> WE implement this using Tavily's free search
                              API (1,000 free searches/month, built for LLM
                              agents). Only offered to the model at all if
                              TAVILY_API_KEY is configured.
  2. download_dataset      -> fetches a URL, previews it if it looks tabular.
  3. run_python_analysis   -> runs model-written Python (pandas/numpy) to
                              actually compute an answer, not guess one.
"""

import os
import subprocess
import sys
import tempfile
import time

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import config
import data_tools

# By default the SDK secretly retries a failed call up to 5 times (with
# growing delays) before it ever raises an error to us -- including on 429
# "quota exceeded" errors, which quietly burns extra requests we can't see.
# attempts=1 turns that off, so every generate_content() call is exactly one
# HTTP request, and WE fully control what happens after a failure (see
# _call_model below) instead of the SDK doing its own invisible thing.
_HTTP_OPTIONS = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=_HTTP_OPTIONS)


SYSTEM_PROMPT = """You are a helpful assistant running inside a Telegram bot. You can
answer general questions directly, and you can also do real data analysis
when asked -- possibly with inline data, possibly pointing at a public
dataset (MOSPI or similar).

Rules:
1. If the question is general and doesn't need a lookup or computation,
   just answer it directly -- do NOT call any tools "just in case."
2. For data questions: never guess a number or fact you could look up or
   compute instead. Use web_search to find the right source, download_dataset
   to fetch it, run_python_analysis to actually compute from it. Print
   intermediate values so you can sanity-check your own work.
3. download_dataset saves files into a shared sandbox directory and tells
   you the local path -- pass that same path into your run_python_analysis
   code to open it (e.g. pd.read_csv("<local_path>")).
4. Pay close attention to exact wording: units, rounding, "top state" vs
   "top 3 states", percentage vs raw count, etc.
5. Be economical with tool calls -- each one costs real API quota. Don't
   search for the same thing twice, and do all your computation in ONE
   run_python_analysis call rather than several small ones whenever you can.
6. If the question specifies an exact JSON output shape (as data-analysis
   questions here typically do, e.g. {"answer": {"state": "<state name>"},
   "log_url": "..."}), your final message must be ONLY the raw JSON value
   for the "answer" key -- no markdown, no code fences, no explanation, no
   surrounding object, no "log_url" (the system adds that separately).
   Example: if the question wants {"answer": {"state": "Assam"}, ...}, your
   final message should be exactly: {"state": "Assam"}
   Otherwise (a plain conversational question), just answer in plain text.
"""

_WEB_SEARCH_DECL = types.FunctionDeclaration(
    name="web_search",
    description=(
        "Search the web for a query. Returns a short list of "
        "{title, url, snippet} results to help find the right source or dataset."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"query": types.Schema(type="STRING", description="Search query")},
        required=["query"],
    ),
)

_DOWNLOAD_DECL = types.FunctionDeclaration(
    name="download_dataset",
    description=(
        "Download a file from a URL into a shared sandbox directory. If it "
        "looks tabular (csv/tsv/xlsx/json) you get back its columns, shape, "
        "and first few rows. Returns the local file path to use in "
        "run_python_analysis."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"url": types.Schema(type="STRING", description="URL to download")},
        required=["url"],
    ),
)

_RUN_PYTHON_DECL = types.FunctionDeclaration(
    name="run_python_analysis",
    description=(
        "Run a Python 3 script to compute an answer. pandas, numpy, and "
        "requests are installed. Use pd.read_csv/read_excel on paths "
        "returned by download_dataset. print() whatever you need to see -- "
        "stdout/stderr are returned to you."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"code": types.Schema(type="STRING", description="Full python script to run")},
        required=["code"],
    ),
)


def _build_tools():
    decls = [_DOWNLOAD_DECL, _RUN_PYTHON_DECL]
    if config.TAVILY_API_KEY:
        decls.insert(0, _WEB_SEARCH_DECL)
    return [types.Tool(function_declarations=decls)]


TOOLS = _build_tools()


def _tool_web_search(query: str) -> dict:
    if not config.TAVILY_API_KEY:
        return {"error": "web_search is not configured (no TAVILY_API_KEY set)"}
    try:
        return {"results": data_tools.web_search(query, config.TAVILY_API_KEY)}
    except Exception as e:
        return {"error": str(e)}


def _tool_download_dataset(url: str) -> dict:
    try:
        path = data_tools.download_file(url)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        df = data_tools.load_tabular(path)
        return {"ok": True, "local_path": str(path), **data_tools.preview(df)}
    except Exception:
        return {
            "ok": True,
            "local_path": str(path),
            "note": "downloaded but not auto-parsed as tabular; open it manually in run_python_analysis",
        }


def _tool_run_python(code: str, timeout: int = 60) -> dict:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir=str(data_tools.SANDBOX_DIR)
    ) as f:
        f.write(code)
        script_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(data_tools.SANDBOX_DIR),
        )
        return {
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-2000:],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"TIMEOUT after {timeout}s", "returncode": -1}
    finally:
        os.unlink(script_path)


def _history_to_contents(history):
    """history: list of {"role": "user"|"model", "text": str}"""
    return [
        types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        for turn in history
    ]


_last_call_at = 0.0


def _pace():
    """Sleeps just long enough to keep calls at least GEMINI_MIN_INTERVAL_SECONDS
    apart, so a fast multi-step question can't out-run the free-tier rate limit."""
    global _last_call_at
    remaining = config.GEMINI_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call_at)
    if remaining > 0:
        time.sleep(remaining)
    _last_call_at = time.monotonic()


def _call_model(contents, gen_config):
    """One Gemini call, paced, with a single polite retry on 429 (quota
    exceeded). If the retry also fails, the error is raised to the caller --
    we deliberately don't keep hammering the API beyond that one extra try."""
    _pace()
    try:
        return client.models.generate_content(model=config.GEMINI_MODEL, contents=contents, config=gen_config)
    except genai_errors.ClientError as e:
        if e.code != 429:
            raise
        time.sleep(config.GEMINI_RATE_LIMIT_RETRY_SECONDS)
        _pace()
        return client.models.generate_content(model=config.GEMINI_MODEL, contents=contents, config=gen_config)


def run_agent(history, log_fn, max_steps: int = 8):
    """
    history: running conversation for this chat, oldest first, as
             {"role": "user"|"model", "text": str} dicts.
    log_fn:  callable(dict) -- appends one readable event to the run log.
    returns: the model's final raw text (expected to be a JSON value).
    """
    contents = _history_to_contents(history)

    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for step in range(max_steps):
        response = _call_model(contents, gen_config)

        candidate_content = response.candidates[0].content
        contents.append(candidate_content)

        text_parts = [p.text for p in candidate_content.parts if getattr(p, "text", None)]
        if text_parts:
            log_fn({"event": "agent_note", "step": step, "text": " ".join(text_parts)[:1000]})

        function_calls = [
            p.function_call for p in candidate_content.parts if getattr(p, "function_call", None)
        ]

        if not function_calls:
            return "".join(text_parts).strip()

        response_parts = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}

            if name == "web_search":
                result = _tool_web_search(args.get("query", ""))
            elif name == "download_dataset":
                result = _tool_download_dataset(args.get("url", ""))
            elif name == "run_python_analysis":
                result = _tool_run_python(args.get("code", ""))
            else:
                result = {"error": f"unknown tool: {name}"}

            log_fn({"event": "tool_call", "step": step, "tool": name, "input": args, "output": result})
            response_parts.append(types.Part.from_function_response(name=name, response=result))

        contents.append(types.Content(role="user", parts=response_parts))

    # Ran out of steps. Rather than silently returning nothing, spend ONE
    # more call asking the model to give its best answer right now, with no
    # tools offered, based on whatever it already found above. This turns a
    # dead end into a usable (if not perfect) answer most of the time.
    log_fn({"event": "max_steps_reached", "note": "asking for a best-effort final answer, no tools"})
    contents.append(types.Content(role="user", parts=[types.Part(text=(
        "You're out of tool-call budget for this turn. Based only on what "
        "you've already found above, give your single best-effort answer "
        "now, in the exact format requested -- no tools, no more searching."
    ))]))
    final_config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    response = _call_model(contents, final_config)
    text_parts = [p.text for p in response.candidates[0].content.parts if getattr(p, "text", None)]
    final_text = "".join(text_parts).strip()
    log_fn({"event": "final_answer_after_max_steps", "text": final_text[:1000]})
    return final_text or None