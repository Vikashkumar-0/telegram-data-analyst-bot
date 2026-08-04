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

from google import genai
from google.genai import types

import config
import data_tools

client = genai.Client(api_key=config.GEMINI_API_KEY)


SYSTEM_PROMPT = """You are a careful data-analyst agent operating inside a Telegram bot.

You'll be given a data-analysis question, possibly with inline data, possibly
pointing at a public dataset (MOSPI or similar). The question describes the
exact JSON shape the final answer should have, e.g.
{"answer": {"state": "<state name>"}, "log_url": "..."}.

Rules:
1. Never guess a number or fact you could look up or compute instead. If
   web_search is available, use it to find the right source; use
   download_dataset to fetch it; use run_python_analysis to actually
   compute from it. Print intermediate values so you can sanity-check your
   own work before finalizing.
2. download_dataset saves files into a shared sandbox directory and tells
   you the local path -- pass that same path into your run_python_analysis
   code to open it (e.g. pd.read_csv("<local_path>")).
3. Pay close attention to exact wording: units, rounding, "top state" vs
   "top 3 states", percentage vs raw count, etc.
4. When -- and only when -- you're fully done and confident, reply with a
   single line containing ONLY the JSON value for the "answer" key. No
   markdown, no code fences, no explanation, no surrounding object, no
   "log_url" (the system adds that separately). Just the raw JSON value.
   Example: if the question wants {"answer": {"state": "Assam"}, ...},
   your final message should be exactly: {"state": "Assam"}
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


def run_agent(history, log_fn, max_steps: int = 12):
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
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=gen_config,
        )


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

    log_fn({"event": "error", "error": "max_steps exceeded without a final answer"})
    return None

