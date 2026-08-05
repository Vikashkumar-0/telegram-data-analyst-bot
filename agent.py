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
import re
import subprocess
import sys
import tempfile
import time


from openai import OpenAI, RateLimitError, BadRequestError

import config
import data_tools

# max_retries=0: the SDK would otherwise silently retry a failed call a
# couple of times on its own -- including on 429 "rate limited" errors --
# before telling us. We turn that off so WE decide what happens after a
# failure (see _call_model below), instead of the SDK doing it invisibly.
client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL, max_retries=0)


def _build_system_prompt():
    if config.TAVILY_API_KEY:
        data_rule = (
            "3. For questions needing outside data: never guess a fact you could look\n"
            "   up instead. Use web_search to find the right source, download_dataset\n"
            "   to fetch it, run_python_analysis to actually compute from it. Print\n"
            "   intermediate values so you can sanity-check your own work."
        )
    else:
        # No TAVILY_API_KEY configured -> web_search is NOT in the tools list
        # this session. Telling the model to use it anyway (when it isn't
        # actually offered) is exactly what caused it to attempt an
        # undeclared tool call and crash the run with a 400 -- so this
        # branch must never mention web_search.
        data_rule = (
            "3. You have no internet search capability this session. For data\n"
            "   questions, work only from a dataset URL already given in the\n"
            "   question, or use download_dataset + run_python_analysis on it. If\n"
            "   you can't find a source without searching, say so honestly instead\n"
            "   of guessing."
        )
    return f"""You are a helpful assistant running inside a Telegram bot. You can
answer general questions directly, and you can also do real data analysis
when asked -- possibly with inline data, possibly pointing at a public
dataset (MOSPI or similar).

Rules:
1. If the question is general knowledge or conversation and involves no
   arithmetic or data, just answer it directly -- do NOT call any tools
   "just in case."
2. NEVER do arithmetic in your head, no matter how simple it looks --
   averages, sums, percentages, differences, counts, all of it. Even
   "what's the average of 5 numbers" MUST go through run_python_analysis.
   You are a language model; your mental math is not reliable enough to
   trust, and there is no reason to guess when a calculator is one tool
   call away.
{data_rule}
4. download_dataset saves files into a shared sandbox directory and tells
   you the local path -- pass that same path into your run_python_analysis
   code to open it (e.g. pd.read_csv("<local_path>")).
5. Pay close attention to exact wording: units, rounding, "top state" vs
   "top 3 states", percentage vs raw count, etc.
6. Be economical with tool calls -- each one costs real API quota. Don't
   search for the same thing twice, and do all your computation in ONE
   run_python_analysis call rather than several small ones whenever you can.
7. Only ever call a tool from the list you were given. If you're not sure
   something is available, don't call it -- answer with what you have.
8. If the question specifies an output schema (e.g. {{"answer": {{"state": "<state name>"}}, "log_url": "..."}}),
   your final message should be ONLY the target value for the "answer" field (e.g. {{"state": "Assam"}}, 30.0, or tabular text).
   Do NOT wrap your final message in an outer {{"answer": ...}} dict or duplicate JSON lines, as the system adds the outer
   wrapper and log_url automatically.

"""


# Built once at import time from whatever TAVILY_API_KEY actually is --
# kept in sync with _build_tools() below so the prompt never promises a
# tool that isn't really offered (that mismatch is what caused Groq to
# reject an undeclared "web_search" call with a 400 before this fix).
SYSTEM_PROMPT = _build_system_prompt()


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
    """One Groq call, paced, with two narrow safety nets:
      - a single polite retry on 429 (rate limited) -- not a retry storm.
      - a single retry with tools turned OFF if the model tries to call a
        tool that wasn't actually declared (a "tool_use_failed" 400). This
        can happen if the system prompt and the real tool list ever drift
        out of sync again -- instead of crashing the whole run, we force a
        plain-text answer from whatever the model already knows.
    If either safety net doesn't apply, the error is raised as-is."""
    _pace()
    try:
        return client.chat.completions.create(model=config.GROQ_MODEL, messages=messages, **kwargs)
    except RateLimitError:
        time.sleep(config.GROQ_RATE_LIMIT_RETRY_SECONDS)
        _pace()
        return client.chat.completions.create(model=config.GROQ_MODEL, messages=messages, **kwargs)
    except BadRequestError as e:
        if "tool_use_failed" in str(e) and "tools" in kwargs:
            retry_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
            _pace()
            return client.chat.completions.create(model=config.GROQ_MODEL, messages=messages, **retry_kwargs)
        raise


def _extract_pseudo_tool_call(content: str):
    """
    Detects pseudo tool calls in text content when open-weight models output syntax
    like <function.run_python_analysis{...}</function> or {"code": "..."} instead of
    populating native tool_calls.
    """
    if not content or not isinstance(content, str):
        return None, None

    text = content.strip()

    # Pattern 1: <function.tool_name{"arg": "val"}</function> or <function tool_name>...
    m1 = re.search(r"<function[.=:\s]+(\w+)\s*(\{.*?\})\s*(?:</function>|>)?", text, flags=re.DOTALL)
    if m1:
        name = m1.group(1)
        raw_args = m1.group(2)
        try:
            args = json.loads(raw_args)
            return name, args
        except Exception:
            pass

    # Pattern 2: {"name": "tool_name", "arguments": {...}}
    m2 = re.search(r"\{\s*\"(?:name|function)\"\s*:\s*\"(\w+)\"\s*,\s*\"(?:arguments|parameters|args)\"\s*:\s*(\{.*\}|\".*\")\s*\}", text, flags=re.DOTALL)
    if m2:
        name = m2.group(1)
        raw_args = m2.group(2)
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.startswith("{") else json.loads(raw_args)
            return name, args
        except Exception:
            pass

    # Pattern 3: Raw JSON object containing tool argument keys like "code", "query", or "url"
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                if "code" in parsed and isinstance(parsed["code"], str):
                    return "run_python_analysis", {"code": parsed["code"]}
                if "query" in parsed and isinstance(parsed["query"], str):
                    return "web_search", {"query": parsed["query"]}
                if "url" in parsed and isinstance(parsed["url"], str) and parsed["url"].startswith("http"):
                    return "download_dataset", {"url": parsed["url"]}
        except Exception:
            pass

    return None, None


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
            pseudo_name, pseudo_args = _extract_pseudo_tool_call(message.content or "")
            if pseudo_name and pseudo_name in TOOL_FUNCTIONS:
                tool_fn = TOOL_FUNCTIONS[pseudo_name]
                result = tool_fn(**pseudo_args)
                log_fn({"event": "pseudo_tool_call_intercepted", "step": step, "tool": pseudo_name, "input": pseudo_args, "output": result})

                stdout = ""
                if isinstance(result, dict) and "stdout" in result:
                    stdout = (result.get("stdout") or "").strip()
                elif isinstance(result, dict) and "results" in result:
                    stdout = json.dumps(result.get("results"), ensure_ascii=False)

                messages.append({
                    "role": "assistant",
                    "content": f"Executed tool {pseudo_name}",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool '{pseudo_name}' result:\n{json.dumps(result, ensure_ascii=False)}\n"
                        f"State your final answer clearly in plain text or table based on this result. Do NOT output function tags."
                    ),
                })
                try:
                    followup = _call_model(messages)
                    final_text = (followup.choices[0].message.content or "").strip()
                    if final_text:
                        return final_text
                except Exception as e:
                    log_fn({"event": "followup_error", "error": str(e)})

                if stdout:
                    return stdout

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