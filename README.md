# Data Analyst Telegram Bot

An LLM agent that answers data-analysis questions over Telegram and replies
with a single JSON object. It searches the web, downloads real datasets, and
runs actual Python (pandas) to compute answers instead of letting the model
guess numbers from memory.

## Architecture

```
Telegram user
      │  data question
      ▼
┌─────────────────────────────┐
│  bot.py (FastAPI + polling) │
│  - receives the message     │
│  - keeps per-chat history   │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  agent.py                   │
│  reasoning loop: call the   │
│  model, run whatever tool   │
│  it asks for, repeat        │
└───────┬─────────────┬───────┘
        ▼             ▼
┌───────────────┐ ┌───────────────────┐
│ web_search     │ │ download_dataset   │
│ (built-in,     │ │ run_python_analysis│
│  server-side)  │ │ (data_tools.py)    │
└───────┬────────┘ └─────────┬──────────┘
        └───────────┬────────┘
                     ▼
        final JSON value ("answer")
                     │
                     ▼
┌─────────────────────────────┐        ┌─────────────────────────────┐
│  Telegram reply:             │        │  logger.py writes each step  │
│  {"answer": ..., "log_url"}  │        │  to logs/<chat_id>.jsonl,     │
│                               │        │  served by bot.py itself at  │
│                               │        │  GET /logs/<chat_id>.jsonl   │
└─────────────────────────────┘        └─────────────────────────────┘
```

The key design choice: **the same FastAPI app that runs the bot also serves
its own logs over HTTP.** No GitHub commits at runtime, no third-party
storage account to set up — your host's public URL for this app *is* your
`log_url` base.

## Files

| File | What it does |
|---|---|
| `bot.py` | FastAPI app: starts Telegram polling, handles messages, serves `/logs/<chat_id>.jsonl` |
| `agent.py` | The reasoning loop: calls Gemini, runs whichever tool it requests, repeats until it has a final answer |
| `data_tools.py` | `download_file`, `load_tabular`, `preview` — the actual "fetch and understand a dataset" logic |
| `logger.py` | Appends structured JSONL events to disk as they happen |
| `config.py` | Reads and validates all environment variables in one place |
| `utils.py` | `extract_answer_value` — pulls a clean JSON value out of the model's final text, even if it added stray formatting |
| `tests/test_agent.py` | Unit tests for `utils.py` and `data_tools.py` (no API key or network needed) |

## 1. Get your credentials

| What | How | Cost |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Message `@BotFather` on Telegram → `/newbot` → pick a name ending in `bot` | free |
| `GEMINI_API_KEY` | aistudio.google.com/apikey — no credit card needed | free tier (Flash models) |
| `TAVILY_API_KEY` | tavily.com — sign up, copy the key. Optional but recommended: without it, the model has no `web_search` tool and can only work from URLs already in the question | free, 1,000 searches/month |
| `PUBLIC_BASE_URL` | filled in *after* you deploy (step 4) — the URL your host gives you | — |

**On free-tier limits:** Gemini's Flash models are ~15 requests/minute and roughly 1,000–1,500 requests/day on the free tier as of mid-2026 — plenty for this project, but these numbers do change, so check ai.google.dev/gemini-api/docs/pricing if you hit 429 errors. Tavily's free tier is 1,000 searches/month, no card required. Neither needs billing enabled to work.

Copy `.env.example` to `.env` and fill in real values for local testing.
`.env` is already in `.gitignore` — never commit it.

## 2. Run the tests

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

These don't need any credentials — they check the JSON-extraction logic and
dataset loading in isolation.

## 3. Run it locally

```bash
export $(cat .env | xargs)       # loads your .env into the shell (mac/linux)
uvicorn bot:app --reload
```

Message your bot on Telegram from your own account. You should get back one
line of JSON. Check `logs/<chat_id>.jsonl` locally, and confirm
`http://localhost:8000/logs/<chat_id>.jsonl` shows the same content in your
browser.

## 4. Deploy so it's reachable 24/7

This now needs to be a **web service** (not a background worker), because it
serves the `/logs/...` endpoint itself.

**Render.com**
1. New → Web Service → connect this GitHub repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
4. Add `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`, `TAVILY_API_KEY` (optional), `LOG_DIR=logs` as env vars
5. Deploy once to get your assigned URL (e.g. `https://your-app.onrender.com`)
6. Go back and add `PUBLIC_BASE_URL=https://your-app.onrender.com`, redeploy

**Railway.app** — same idea: deploy from repo (it picks up the `Procfile`),
add the same env vars, then set `PUBLIC_BASE_URL` to whatever URL Railway
assigns once it's live.

## 5. Test with the official grading harness

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
# follow its README to point it at your bot username, add a few of your
# own questions to evals/questions.json, and run it
```

Try a few different answer shapes (a single number, a list, a nested
object) to make sure the model is actually following the "reply with ONLY
the value under answer" instruction — `extract_answer_value` has a fallback
for stray formatting, but it's worth confirming in practice.

## Known limitations (worth knowing, and worth being able to explain)

- **Ephemeral disk.** Free-tier web services on most hosts don't guarantee
  local files survive a redeploy. Logs persist fine while the service stays
  up, but a fresh deploy can wipe `logs/`. Fine for a live grading window;
  if you want logs to survive indefinitely, the next step is a persistent
  disk add-on or swapping `logger.py`'s writes for an S3-compatible bucket
  — the `RunLogger` interface wouldn't need to change, just its internals.
- **No sandboxing on `run_python_analysis`.** It trusts model-written code
  and runs it directly on your server. Acceptable for a project you control
  end-to-end; not something to expose beyond this.
- **In-memory chat history** resets on restart. Doesn't affect grading since
  each message still gets a complete, independent answer.
- Double-check `GEMINI_MODEL` in `config.py`/`.env` against Google's current
  docs before deploying — model names get retired (e.g. Gemini 2.0 Flash
  was shut down mid-2026) and free-tier eligibility shifts between model
  generations.
- If you skip `TAVILY_API_KEY`, the bot still works — it just won't have a
  `web_search` tool, so it depends on the question already containing a
  dataset URL, or on the model's own knowledge of common sources like MOSPI.
