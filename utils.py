"""
utils.py
--------
Small, pure, dependency-free helpers. Kept separate from agent.py so they can
be unit tested without needing an API key or network access (see tests/).
"""

import json
import re


def _unwrap_answer(val):
    """
    Recursively unwraps dicts like {"answer": ...} or stringified JSON inside "answer"
    to return the underlying native Python value (float, int, dict, list, str).
    """
    max_depth = 5
    depth = 0
    while depth < max_depth:
        depth += 1
        if isinstance(val, dict):
            if len(val) == 1 and "answer" in val:
                val = val["answer"]
                continue
            elif "answer" in val and len(val) <= 2 and ("log_url" in val or "status" in val):
                val = val["answer"]
                continue
        if isinstance(val, str):
            val_str = val.strip()
            if re.match(r"^-?\d+\.\d+$", val_str):
                return float(val_str)
            if re.match(r"^-?\d+$", val_str):
                return int(val_str)
            if (val_str.startswith("{") and val_str.endswith("}")) or (val_str.startswith("[") and val_str.endswith("]")):
                try:
                    parsed = json.loads(val_str)
                    if isinstance(parsed, (dict, list, int, float, bool)):
                        val = parsed
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            blocks = re.findall(r"(\{.*?\}|\[.*?\])", val_str, flags=re.DOTALL)
            if blocks:
                for b in blocks:
                    try:
                        parsed = json.loads(b)
                        val = parsed
                        break
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    break
                continue
        break

    if isinstance(val, str):
        val_str = val.strip()
        if re.match(r"^-?\d+\.\d+$", val_str):
            return float(val_str)
        if re.match(r"^-?\d+$", val_str):
            return int(val_str)

    return val



def extract_answer_value(raw_text):
    """
    Takes whatever raw text the LLM produced as its final message and turns it
    into a clean Python value (dict/list/number/string) that belongs under the "answer" key.

    Handles edge cases:
      - ```json ... ``` code blocks
      - Double-nested dicts: {"answer": {"answer": 30.0}} -> 30.0
      - Concatenated JSON strings: {"answer": 25.0}{"answer": 25.0} -> 25.0
      - Pseudo function tags: <function...{...}</function>
      - Numeric strings & raw text markdown
    """
    if raw_text is None:
        return None

    text = str(raw_text).strip()
    if not text:
        return ""

    # Strip leading/trailing markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    # Case 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        return _unwrap_answer(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Case 2: Concatenated or multiple JSON objects (e.g. {"answer": 25.0}{"answer": 25.0})
    json_blocks = re.findall(r"(\{.*?\}|\[.*?\])", text, flags=re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            return _unwrap_answer(parsed)
        except (json.JSONDecodeError, ValueError):
            continue

    # Case 3: Inner JSON inside <function...{...}</function>
    m_func = re.search(r"<function[.=:\s]*\w*\s*(\{.*?\})\s*(?:</function>|>)?", text, flags=re.DOTALL)
    if m_func:
        try:
            parsed = json.loads(m_func.group(1))
            unwrapped = _unwrap_answer(parsed)
            if unwrapped:
                return unwrapped
        except Exception:
            pass

    # Case 4: Pure numeric string
    try:
        if re.match(r"^-?\d+\.\d+$", text):
            return float(text)
        if re.match(r"^-?\d+$", text):
            return int(text)
    except ValueError:
        pass

    # Case 5: Safe tag cleanup (removes <function...> and </function> tags without deleting content)
    cleaned = re.sub(r"<function[.=:\s]*\w*", "", text)
    cleaned = re.sub(r"</function>", "", cleaned).strip()
    if cleaned:
        return cleaned

    return text


