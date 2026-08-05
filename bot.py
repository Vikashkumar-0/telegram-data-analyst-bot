"""
bot.py
------
Runs two things in one process:
  1. The Telegram long-polling loop ("the mailroom clerk") -- no public URL
     needed for this part.
  2. A tiny FastAPI web server whose ONLY real job is to serve
     GET /logs/<chat_id>.jsonl straight off local disk. That endpoint's
     public URL (given to you by your host, e.g. Render) is exactly what
     goes in log_url -- no GitHub, no third-party storage.

Run locally with:  uvicorn bot:app --reload
Run in production with the Procfile's command.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

import config
from agent import run_agent
from logger import LOG_DIR, RunLogger
from utils import extract_answer_value

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("databot")

# Per-chat history kept in memory. Resets if the process restarts -- fine
# here since every incoming message still gets a fresh, complete answer.
HISTORY: dict[int, list] = {}

telegram_app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    log.info("Received from %s: %s", chat_id, text[:200])

    history = HISTORY.setdefault(chat_id, [])
    history.append({"role": "user", "text": text})

    run_logger = RunLogger(chat_id)
    run_logger.log({"event": "message_received", "text": text})

    try:
        raw_answer = run_agent(history, run_logger.log)
        history.append({"role": "assistant", "text": raw_answer or ""})
        answer_value = extract_answer_value(raw_answer)
        run_logger.log({"event": "final_answer", "answer": answer_value})
    except Exception as e:
        log.exception("agent failed")
        run_logger.log({"event": "error", "error": str(e)})
        answer_value = None

    reply_obj = {"answer": answer_value, "log_url": run_logger.url}
    reply_str = json.dumps(reply_obj)
    run_logger.log({"event": "reply_sent", "reply": reply_obj})

    await update.message.reply_text(reply_str)


telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()
    log.info("Telegram polling started")
    yield
    await telegram_app.updater.stop()
    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/logs/{chat_id}.jsonl")
async def get_log(chat_id: str):
    path = Path(LOG_DIR) / f"{chat_id}.jsonl"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    # text/plain (not application/x-ndjson) so it opens as readable text in
    # a browser and is fetchable by tools that treat unfamiliar MIME types
    # as opaque binary -- the content itself is unchanged either way.
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.PORT)