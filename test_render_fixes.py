"""
test_render_fixes.py
---------------------
Explicit test suite covering Tests A through H required for Render production deployment.
"""

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bot
import config
from utils import extract_answer_value


class TestRenderFixesAthroughH(unittest.TestCase):
    def setUp(self):
        bot.HISTORY.clear()

    def test_A_import(self):
        """Test A — import completes quickly without hanging."""
        self.assertIsNotNone(bot.app)

    def test_B_and_C_health_endpoint(self):
        """Test B & C — production style app listening & GET / returns 200."""
        @asynccontextmanager
        async def dummy_lifespan(app):
            yield

        with patch.object(bot.app.router, "lifespan_context", dummy_lifespan):
            with TestClient(bot.app) as client:
                resp = client.get("/")
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["status"], "ok")
                self.assertIn("mode", data)
                self.assertIn("public_base_url", data)

    def test_D_webhook_authentication(self):
        """Test D — webhook authentication secret verification."""
        @asynccontextmanager
        async def dummy_lifespan(app):
            yield

        with patch.object(bot.app.router, "lifespan_context", dummy_lifespan):
            with TestClient(bot.app) as client:
                with patch.object(bot.config, "WEBHOOK_SECRET", "secret-key-123"):
                    resp1 = client.post("/webhook", json={"update_id": 1})
                    self.assertEqual(resp1.status_code, 401)

                    resp2 = client.post(
                        "/webhook",
                        json={"update_id": 1},
                        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"}
                    )
                    self.assertEqual(resp2.status_code, 401)

                    with patch("bot._process_update_task"):
                        with patch("bot.Update.de_json"):
                            resp3 = client.post(
                                "/webhook",
                                json={"update_id": 1},
                                headers={"X-Telegram-Bot-Api-Secret-Token": "secret-key-123"}
                            )
                            self.assertEqual(resp3.status_code, 200)

    def test_E_mocked_telegram_update(self):
        """Test E — mocked Telegram update processing in background."""
        mock_update = MagicMock()
        mock_update.effective_chat.id = 12345
        mock_update.message.text = "What is 2 + 2?"
        mock_update.message.reply_text = AsyncMock()

        async def run_handler():
            await bot.handle_message(mock_update, None)

        with patch("bot.run_agent", return_value="4"):
            asyncio.run(run_handler())

        mock_update.message.reply_text.assert_called_once()
        sent_reply = mock_update.message.reply_text.call_args[0][0]
        parsed = json.loads(sent_reply)
        self.assertEqual(parsed["answer"], 4)
        self.assertIn("12345.jsonl", parsed["log_url"])

    def test_F_agent_failure(self):
        """Test F — agent failure leaves server alive & sends error JSON response."""
        mock_update = MagicMock()
        mock_update.effective_chat.id = 67890
        mock_update.message.text = "Crash test"
        mock_update.message.reply_text = AsyncMock()

        async def run_handler():
            await bot.handle_message(mock_update, None)

        with patch("bot.run_agent", side_effect=RuntimeError("Groq Outage")):
            asyncio.run(run_handler())

        mock_update.message.reply_text.assert_called_once()
        sent_reply = mock_update.message.reply_text.call_args[0][0]
        parsed = json.loads(sent_reply)
        self.assertIn("Error processing request", parsed["answer"])
        self.assertIn("Groq Outage", parsed["answer"])

        @asynccontextmanager
        async def dummy_lifespan(app):
            yield

        with patch.object(bot.app.router, "lifespan_context", dummy_lifespan):
            with TestClient(bot.app) as client:
                health_resp = client.get("/")
                self.assertEqual(health_resp.status_code, 200)

    def test_G_telegram_api_unavailable(self):
        """Test G — Telegram API setup failure does NOT block FastAPI server."""
        @asynccontextmanager
        async def error_lifespan(app):
            bot.log.error("Simulated Telegram setup failure")
            yield

        with patch.object(bot.app.router, "lifespan_context", error_lifespan):
            with TestClient(bot.app) as client:
                resp = client.get("/")
                self.assertEqual(resp.status_code, 200)

    def test_H_logs_endpoint(self):
        """Test H — GET /logs/<chat_id>.jsonl returns valid JSONL."""
        log_dir = Path(bot.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        test_chat_file = log_dir / "7777.jsonl"
        test_chat_file.write_text('{"event": "test"}\n', encoding="utf-8")

        @asynccontextmanager
        async def dummy_lifespan(app):
            yield

        with patch.object(bot.app.router, "lifespan_context", dummy_lifespan):
            with TestClient(bot.app) as client:
                resp = client.get("/logs/7777.jsonl")
                self.assertEqual(resp.status_code, 200)
                self.assertIn('{"event": "test"}', resp.text)


if __name__ == "__main__":
    unittest.main()

