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


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "dummy-telegram-token")

# Groq hosts open-weight models on fast custom chips -- console.groq.com/keys.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or "dummy-groq-key"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Free tier pacing
GROQ_MIN_INTERVAL_SECONDS = float(os.environ.get("GROQ_MIN_INTERVAL_SECONDS", 2))
GROQ_RATE_LIMIT_RETRY_SECONDS = float(os.environ.get("GROQ_RATE_LIMIT_RETRY_SECONDS", 10))

# Optional Tavily key for web search
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Public HTTPS URL (e.g. https://telegram-data-analyst-bot-ogij.onrender.com)
PUBLIC_BASE_URL = (os.environ.get("PUBLIC_BASE_URL") or os.environ.get("BASE_URL") or "http://localhost:8000").rstrip("/")


WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

# Polling vs Webhook mode determination:
# Default to polling only if explicitly requested OR if PUBLIC_BASE_URL is localhost / 127.0.0.1.
# On Render (where PUBLIC_BASE_URL is https://telegram-data-analyst-bot-ogij.onrender.com),
# this automatically resolves to False (Webhook mode).
_use_polling_env = os.environ.get("USE_POLLING", "").strip().lower()
if _use_polling_env in ("true", "1", "yes"):
    USE_POLLING = True
elif _use_polling_env in ("false", "0", "no"):
    USE_POLLING = False
else:
    USE_POLLING = "localhost" in PUBLIC_BASE_URL or "127.0.0.1" in PUBLIC_BASE_URL

LOG_DIR = os.environ.get("LOG_DIR", "logs")
PORT = int(os.environ.get("PORT", 8000))