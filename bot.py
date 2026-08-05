"""
bot.py
------
FastAPI web app serving both:
  1. Telegram bot interface:
     - Webhook mode (production / Render): Telegram POSTs updates to /webhook.
       Render receives the HTTP request and automatically wakes up the service!
       FastAPI processes updates in the background and returns HTTP 200 OK instantly.
     - Long-polling mode (local development): runs getUpdates loop when USE_POLLING=True.
  2. Public GET /logs/<chat_id>.jsonl endpoint serving execution logs.

Run locally with: uvicorn bot:app --reload
Run in production with the Procfile command.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

import config
from agent import run_agent
from logger import LOG_DIR, RunLogger
from utils import extract_answer_value

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("databot")

# Per-chat history kept in memory.
HISTORY: dict[int, list] = {}

telegram_app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat or not update.message:
        return

    chat_id = update.effective_chat.id
    text = update.message.text or ""
    log.info("Received from %s: %s", chat_id, text[:200])

    history = HISTORY.setdefault(chat_id, [])
    history.append({"role": "user", "text": text})

    run_logger = RunLogger(chat_id)
    run_logger.log({"event": "message_received", "text": text})

    try:
        raw_answer = run_agent(history, run_logger.log)
        if not raw_answer:
            log.warning("Empty response from run_agent, retrying once...")
            raw_answer = run_agent(history, run_logger.log)

        if not raw_answer:
            raw_answer = "Error: Unable to compute an answer for this request."

        history.append({"role": "assistant", "text": raw_answer or ""})
        answer_value = extract_answer_value(raw_answer)
        run_logger.log({"event": "final_answer", "answer": answer_value})
    except Exception as e:
        log.exception("agent failed")
        run_logger.log({"event": "error", "error": str(e)})
        answer_value = f"Error processing request: {str(e)}"

    reply_obj = {"answer": answer_value, "log_url": run_logger.url}
    # ensure_ascii=False ensures unicode characters like ₹, →, -, % are rendered as clean UTF-8
    reply_str = json.dumps(reply_obj, ensure_ascii=False)
    run_logger.log({"event": "reply_sent", "reply": reply_obj})

    await update.message.reply_text(reply_str)


telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await telegram_app.initialize()
    await telegram_app.start()

    if config.USE_POLLING:
        log.info("Starting Telegram bot in POLLING mode...")
        await telegram_app.updater.start_polling()
        log.info("Telegram polling started successfully")
    else:
        webhook_url = f"{config.PUBLIC_BASE_URL}/webhook"
        log.info("Starting Telegram bot in WEBHOOK mode at %s...", webhook_url)
        await telegram_app.bot.set_webhook(
            url=webhook_url,
            secret_token=config.WEBHOOK_SECRET or None,
            drop_pending_updates=False,
        )
        log.info("Telegram webhook set successfully to %s", webhook_url)

    yield

    if config.USE_POLLING and telegram_app.updater and telegram_app.updater.running:
        await telegram_app.updater.stop()

    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health():
    return {
        "status": "ok",
        "mode": "polling" if config.USE_POLLING else "webhook",
        "public_base_url": config.PUBLIC_BASE_URL,
    }


async def _process_update_task(update: Update):
    try:
        await telegram_app.process_update(update)
    except Exception as e:
        log.exception("Unhandled error processing update %s", getattr(update, "update_id", None))


@app.post("/webhook")
async def telegram_webhook(request: Request):
    if config.WEBHOOK_SECRET:
        header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if header_secret != config.WEBHOOK_SECRET:
            return JSONResponse({"error": "Unauthorized secret token"}, status_code=401)

    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        if update:
            asyncio.create_task(_process_update_task(update))
    except Exception as e:
        log.exception("Failed to parse webhook update")
        return JSONResponse({"error": str(e)}, status_code=400)

    return {"status": "ok"}



@app.get("/logs/{chat_id}.jsonl")
async def get_log(chat_id: str):
    path = Path(LOG_DIR) / f"{chat_id}.jsonl"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.PORT)