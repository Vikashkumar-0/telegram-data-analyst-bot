"""
agent.py
--------
"The analyst" -- a loop that calls an LLM, runs whichever tool it asks for,
and feeds the result back, until the model replies with plain text instead
of a tool call. That's the whole architecture; everything below is just the
three tools and some bookkeeping around that one loop.

Runs on Groq (fast, cheap hosting for open-weight models) via the official
`openai` Python package, since Groq's API speaks the exact same "chat
completions" shape as OpenAI -- same message format, same tool-calling
format. We just point the client at Groq's URL and use a Groq key.

Three tools:
  1. web_search           -> Tavily's free search API (only offered to the
                              model if TAVILY_API_KEY is configured).
  2. download_dataset      -> fetches a URL, previews it if it looks tabular.
  3. run_python_analysis   -> runs model-written Python (pandas/numpy) to
                              actually compute an answer, not guess one.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

from openai import OpenAI, RateLimitError

import config
import data_tools

# max_retries=0: the SDK would otherwise silently retry a failed call a
# couple of times on its own -- including on 429 "rate limited" errors --
# before telling us. We turn that off so WE decide what happens after a
# failure (see _call_model below), instead of the SDK doing it invisibly.
client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL, max_retries=0)


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

# Tool definitions, in plain OpenAI "function" format -- just a dict, no
# SDK-specific type to construct.
_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for a query. Returns a short list of "
            "{title, url, snippet} results to help find the right source or dataset."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}

_DOWNLOAD_TOOL = {
    "type": "function",
    "function": {
        "name": "download_dataset",
        "description": (
            "Download a file from a URL into a shared sandbox directory. If it "
            "looks tabular (csv/tsv/xlsx/json) you get back its columns, shape, "
            "and first few rows. Returns the local file path to use in "
            "run_python_analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to download"}},
            "required": ["url"],
        },
    },
}

_RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python_analysis",
        "description": (
            "Run a Python 3 script to compute an answer. pandas, numpy, and "
            "requests are installed. Use pd.read_csv/read_excel on paths "
            "returned by download_dataset. print() whatever you need to see -- "
            "stdout/stderr are returned to you."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Full python script to run"}},
            "required": ["code"],
        },
    },
}


def _build_tools():
    tools = [_DOWNLOAD_TOOL, _RUN_PYTHON_TOOL]
    if config.TAVILY_API_KEY:
        tools.insert(0, _WEB_SEARCH_TOOL)
    return tools


TOOLS = _build_tools()


def _tool_web_search(query: str = "") -> dict:
    if not config.TAVILY_API_KEY:
        return {"error": "web_search is not configured (no TAVILY_API_KEY set)"}
    try:
        return {"results": data_tools.web_search(query, config.TAVILY_API_KEY)}
    except Exception as e:
        return {"error": str(e)}


def _tool_download_dataset(url: str = "") -> dict:
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


def _tool_run_python(code: str = "", timeout: int = 60) -> dict:
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


# name -> function, so the loop below can dispatch with a single dict lookup
# instead of an if/elif chain.
TOOL_FUNCTIONS = {
    "web_search": _tool_web_search,
    "download_dataset": _tool_download_dataset,
    "run_python_analysis": _tool_run_python,
}


def _history_to_messages(history):
    """history: list of {"role": "user"|"assistant", "text": str}"""
    return [{"role": turn["role"], "content": turn["text"]} for turn in history]


_last_call_at = 0.0


def _pace():
    """Sleeps just long enough to keep calls at least GROQ_MIN_INTERVAL_SECONDS
    apart, so a fast multi-step question can't out-run the free-tier rate limit."""
    global _last_call_at
    remaining = config.GROQ_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call_at)
    if remaining > 0:
        time.sleep(remaining)
    _last_call_at = time.monotonic()


def _call_model(messages, **kwargs):
    """One Groq call, paced, with a single polite retry on 429 (rate
    limited). If the retry also fails, the error is raised to the caller --
    we deliberately don't keep hammering the API beyond that one extra try."""
    _pace()
    try:
        return client.chat.completions.create(model=config.GROQ_MODEL, messages=messages, **kwargs)
    except RateLimitError:
        time.sleep(config.GROQ_RATE_LIMIT_RETRY_SECONDS)
        _pace()
        return client.chat.completions.create(model=config.GROQ_MODEL, messages=messages, **kwargs)


def run_agent(history, log_fn, max_steps: int = 8):
    """
    history: running conversation for this chat, oldest first, as
             {"role": "user"|"assistant", "text": str} dicts.
    log_fn:  callable(dict) -- appends one readable event to the run log.
    returns: the model's final raw text (expected to be a JSON value).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _history_to_messages(history)

    for step in range(max_steps):
        response = _call_model(messages, tools=TOOLS)
        message = response.choices[0].message

        if message.content:
            log_fn({"event": "agent_note", "step": step, "text": message.content[:1000]})

        if not message.tool_calls:
            return (message.content or "").strip()

        # Keep the assistant's tool-call turn in the conversation so the
        # model (and Groq, which requires it) can see what it already asked
        # for when we send the tool results back next.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            tool_fn = TOOL_FUNCTIONS.get(name)
            result = tool_fn(**args) if tool_fn else {"error": f"unknown tool: {name}"}

            log_fn({"event": "tool_call", "step": step, "tool": name, "input": args, "output": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    # Ran out of steps. Rather than silently returning nothing, spend ONE
    # more call asking the model to give its best answer right now, with no
    # tools offered, based on whatever it already found above. This turns a
    # dead end into a usable (if not perfect) answer most of the time.
    log_fn({"event": "max_steps_reached", "note": "asking for a best-effort final answer, no tools"})
    messages.append({
        "role": "user",
        "content": (
            "You're out of tool-call budget for this turn. Based only on what "
            "you've already found above, give your single best-effort answer "
            "now, in the exact format requested -- no tools, no more searching."
        ),
    })
    response = _call_model(messages)
    final_text = (response.choices[0].message.content or "").strip()
    log_fn({"event": "final_answer_after_max_steps", "text": final_text[:1000]})
    return final_text or None