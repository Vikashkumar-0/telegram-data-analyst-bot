"""
utils.py
--------
Small, pure, dependency-free helpers. Kept separate from agent.py so they can
be unit tested without needing an API key or network access (see tests/).
"""

import json
import re


def extract_answer_value(raw_text):
    """
    Takes whatever raw text the LLM produced as its final message and tries
    hard to turn it into a real Python value (dict/list/number/string) that
    belongs under the "answer" key.

    Handles the common ways a model slightly misbehaves:
      - wraps the JSON in ```json ... ``` fences anyway
      - adds a stray sentence before/after the JSON
      - just returns a bare string/number (no JSON at all)
    """
    if raw_text is None:
        return None

    text = raw_text.strip()

    # strip a leading/trailing markdown code fence if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # last resort: grab the first {...} or [...] block anywhere in the text
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # give up gracefully -- return the raw string rather than crashing
    return text
