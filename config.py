"""
config.py
---------
Single place that reads and validates environment variables. Every other
module imports from here instead of touching os.environ directly, so if
something is missing you get one clear error message pointing at exactly
what to set, instead of a random KeyError three files deep.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    pass


def _require(name: str, fallback_env: str = None) -> str:
    value = os.environ.get(name) or (os.environ.get(fallback_env) if fallback_env else None)
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in (or set it in your "
            f"host's dashboard)."
        )
    return value


TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")

# Gemini's free tier needs no credit card -- console: aistudio.google.com/apikey
GEMINI_API_KEY = _require("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# The free tier only allows a handful of requests per minute (check your
# live number at aistudio.google.com/rate-limit), and one question can
# trigger several Gemini calls in a row (search -> download -> analyze ->
# answer). So instead of firing calls as fast as possible and tripping that
# limit, agent.py waits at least this many seconds between calls.
GEMINI_MIN_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", 5))

# If we still hit a 429 (quota exceeded) despite pacing, wait this long once
# and try that one call again. Just one polite retry -- not a retry storm.
GEMINI_RATE_LIMIT_RETRY_SECONDS = float(os.environ.get("GEMINI_RATE_LIMIT_RETRY_SECONDS", 10))


# Optional but recommended: free web search built for LLM agents, 1,000
# searches/month free, no credit card -- tavily.com. If unset, the agent
# simply won't be offered a web_search tool and will rely on URLs already
# present in the question plus its own knowledge.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# The public HTTPS URL your host gives this service, e.g.
# https://your-app-name.onrender.com -- used to build each log_url.
# For local testing this can be http://localhost:8000 (just won't be
# publicly wget-able until you deploy).
PUBLIC_BASE_URL = _require("PUBLIC_BASE_URL", fallback_env="BASE_URL").rstrip("/")

LOG_DIR = os.environ.get("LOG_DIR", "logs")
PORT = int(os.environ.get("PORT", 8000))