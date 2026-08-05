"""
tests/test_agent_e2e.py
------------------------
Comprehensive deterministic end-to-end test suite for Data Analyst Telegram Bot.

Runs 18 explicit scenario tests exercising the complete application pipeline:
  - Simple arithmetic & numerical extraction
  - Averages & statistics with Python sandbox
  - Percentage changes & UTF-8 symbol preservation
  - General non-data explanations (no tool calls)
  - General factual questions (Tavily present & absent)
  - Inline sales data analysis & table formatting
  - Multi-step public dataset workflow (search -> download -> analyze)
  - MOSPI-style question envelope shaping
  - Multi-turn conversation context retention
  - Duplicate JSON string regression fixes
  - Model fake envelope stripping & real log_url assignment
  - Empty model response fallback handling
  - Multi-step tool-call loop ID linkage
  - Rate-limit retry handling (Groq 429)
  - Undeclared tool retry handling (Groq 400)
  - FastAPI /webhook endpoint processing & secret validation
  - FastAPI health check GET /
  - Public JSONL log file retrieval GET /logs/<chat_id>.jsonl
"""

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
from openai import RateLimitError, BadRequestError
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent


def _completion(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return ChatCompletion.model_validate({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": agent.config.GROQ_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    })


def _tool_call(call_id, name, arguments: dict):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _rate_limit_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request, json={"error": {"message": "rate limited"}})
    return RateLimitError("rate limited", response=response, body=None)


def _tool_use_failed_error(tool_name="web_search"):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        status_code=400,
        request=request,
        json={"error": {
            "message": f"Tool call validation failed: attempted to call tool '{tool_name}' which was not in request.tools",
            "code": "tool_use_failed",
        }},
    )
    return BadRequestError("tool_use_failed", response=response, body=None)


class Test18RequiredScenarios(unittest.TestCase):
    def setUp(self):
        agent._last_call_at = 0.0

    def _run_full_pipeline(self, question, responses, stub_tools=None, chat_id=1021167690, history=None):
        from utils import extract_answer_value
        from logger import RunLogger
        import config

        if history is None:
            history = []
        history.append({"role": "user", "text": question})

        run_logger = RunLogger(chat_id)
        run_logger.log({"event": "message_received", "text": question})

        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses):
                with patch.dict(agent.TOOL_FUNCTIONS, stub_tools or {}):
                    raw_answer = agent.run_agent(history, run_logger.log)

        answer_value = extract_answer_value(raw_answer)
        run_logger.log({"event": "final_answer", "answer": answer_value})

        reply_obj = {"answer": answer_value, "log_url": run_logger.url}
        reply_str = json.dumps(reply_obj, ensure_ascii=False)
        run_logger.log({"event": "reply_sent", "reply": reply_obj})
        return raw_answer, answer_value, reply_str, run_logger

    def test_1_simple_arithmetic(self):
        responses = [
            _completion(tool_calls=[_tool_call("c1", "run_python_analysis", {"code": "print(2+2)"})]),
            _completion(content="4"),
        ]
        raw, val, reply, logger = self._run_full_pipeline("What is 2 + 2?", responses)
        parsed = json.loads(reply)
        self.assertEqual(set(parsed.keys()), {"answer", "log_url"})
        self.assertEqual(parsed["answer"], 4)
        self.assertIn("1021167690.jsonl", parsed["log_url"])

    def test_2_average(self):
        responses = [
            _completion(tool_calls=[_tool_call("c1", "run_python_analysis", {"code": "print(sum([10,20,30,40,50])/5)"})]),
            _completion(content="30"),
        ]
        stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "30.0\n", "stderr": "", "returncode": 0}}
        raw, val, reply, logger = self._run_full_pipeline("What is the average of 10, 20, 30, 40, and 50?", responses, stub_tools)
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], 30)

    def test_3_percentage_increase_unicode(self):
        responses = [
            _completion(tool_calls=[_tool_call("c1", "run_python_analysis", {"code": "print(((1000-800)/800)*100)"})]),
            _completion(content="25.0"),
        ]
        stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "25.0\n", "stderr": "", "returncode": 0}}
        try:
            raw, val, reply, logger = self._run_full_pipeline("A product price increased from ₹800 to ₹1,000. What is the percentage increase?", responses, stub_tools)
            parsed = json.loads(reply)
            self.assertEqual(parsed["answer"], 25.0)
            self.assertIsInstance(parsed["answer"], float)
        except BaseException as err:
            import traceback
            print("TEST 3 ERROR:", repr(err))
            traceback.print_exc()
            raise



    def test_4_general_non_data_question(self):
        explanation = "Overfitting occurs when a machine learning model learns the detail and noise in the training data to the extent that it negatively impacts performance on new data."
        responses = [_completion(content=explanation)]
        raw, val, reply, logger = self._run_full_pipeline("Explain what overfitting means in machine learning.", responses)
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], explanation)

    def test_5_general_factual_question_with_and_without_tavily(self):
        responses_a = [_completion(content="The current Prime Minister of India is Narendra Modi.")]
        with patch.object(agent.config, "TAVILY_API_KEY", "real-key"):
            raw, val, reply, logger = self._run_full_pipeline("Who is the current Prime Minister of India?", responses_a)
            parsed = json.loads(reply)
            self.assertIn("Narendra Modi", parsed["answer"])

        agent._last_call_at = 0.0
        responses_b = [_completion(content="The current Prime Minister of India is Narendra Modi.")]
        with patch.object(agent.config, "TAVILY_API_KEY", None):
            raw, val, reply, logger = self._run_full_pipeline("Who is the current Prime Minister of India?", responses_b)
            parsed = json.loads(reply)
            self.assertIn("Narendra Modi", parsed["answer"])

    def test_6_inline_sales_data(self):
        table_output = "| Month | Sales | Change |\n|---|---|---|\n| Jan | 100 | - |\n| Feb | 120 | 20% |\n| Mar | 150 | 25% |\n| Apr | 130 | -13.33% |\n| May | 200 | 53.85% |\nMean: 140, Median: 130, Highest: May"
        responses = [
            _completion(tool_calls=[_tool_call("c1", "run_python_analysis", {"code": "import pandas as pd..."})]),
            _completion(content=table_output),
        ]
        stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "Mean: 140\nMedian: 130\n", "stderr": "", "returncode": 0}}
        raw, val, reply, logger = self._run_full_pipeline(
            "Sales were: January 100, February 120, March 150, April 130, May 200. Calculate the mean, median, month-to-month percentage changes, and highest month.",
            responses, stub_tools
        )
        parsed = json.loads(reply)
        self.assertIn("140", parsed["answer"])
        self.assertIn("130", parsed["answer"])
        self.assertIn("May", parsed["answer"])

    def test_7_public_dataset_workflow(self):
        responses = [
            _completion(tool_calls=[_tool_call("c1", "web_search", {"query": "MOSPI MMR dataset"})]),
            _completion(tool_calls=[_tool_call("c2", "download_dataset", {"url": "https://mospi.gov.in/mmr.csv"})]),
            _completion(tool_calls=[_tool_call("c3", "run_python_analysis", {"code": "pd.read_csv('/tmp/sandbox/mmr.csv')"})]),
            _completion(content="Assam has the highest MMR."),
        ]
        calls_made = []
        stub_tools = {
            "web_search": lambda query="": calls_made.append("search") or {"results": [{"url": "https://mospi.gov.in/mmr.csv"}]},
            "download_dataset": lambda url="": calls_made.append("download") or {"ok": True, "local_path": "/tmp/sandbox/mmr.csv"},
            "run_python_analysis": lambda code="": calls_made.append("python") or {"stdout": "Assam: 215", "stderr": "", "returncode": 0},
        }
        with patch.object(agent.config, "TAVILY_API_KEY", "real-key"):
            raw, val, reply, logger = self._run_full_pipeline("Find highest MMR per MOSPI", responses, stub_tools)
        parsed = json.loads(reply)
        self.assertEqual(calls_made, ["search", "download", "python"])
        self.assertIn("Assam", parsed["answer"])

    def test_8_mospi_style_question(self):
        responses = [
            _completion(tool_calls=[_tool_call("c1", "web_search", {"query": "MOSPI MMR state"})]),
            _completion(content='{"state": "Assam"}'),
        ]
        stub_tools = {"web_search": lambda **kw: {"results": [{"snippet": "Assam MMR is 215"}]}}
        with patch.object(agent.config, "TAVILY_API_KEY", "real-key"):
            raw, val, reply, logger = self._run_full_pipeline(
                'Which state has the highest maternal mortality rate based on MOSPI data? Reply with ONLY this JSON object and nothing else: {"answer": {"state": "<state name>"}, "log_url": "<url>"}',
                responses, stub_tools
            )
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], {"state": "Assam"})

    def test_9_multi_turn_conversation(self):
        history = []
        responses1 = [_completion(content="Data received: Jan 100, Feb 150, Mar 200.")]
        raw1, val1, reply1, log1 = self._run_full_pipeline("Here is sales data: Jan 100, Feb 150, Mar 200.", responses1, history=history)
        history.append({"role": "assistant", "text": raw1})

        responses2 = [
            _completion(tool_calls=[_tool_call("c2", "run_python_analysis", {"code": "print(sum([100,150,200])/3)"})]),
            _completion(content="150"),
        ]
        stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "150.0\n", "stderr": "", "returncode": 0}}
        raw2, val2, reply2, log2 = self._run_full_pipeline("What is the average?", responses2, stub_tools=stub_tools, history=history)
        parsed2 = json.loads(reply2)
        self.assertEqual(parsed2["answer"], 150)
        history.append({"role": "assistant", "text": raw2})

        responses3 = [
            _completion(tool_calls=[_tool_call("c3", "run_python_analysis", {"code": "print(((200-100)/100)*100)"})]),
            _completion(content="100%"),
        ]
        raw3, val3, reply3, log3 = self._run_full_pipeline("What about the percentage increase from January to March?", responses3, stub_tools=stub_tools, history=history)
        parsed3 = json.loads(reply3)
        self.assertEqual(parsed3["answer"], "100%")

    def test_10_duplicate_json_regression(self):
        responses = [_completion(content='{"answer": 25.0}{"answer": 25.0}')]
        raw, val, reply, logger = self._run_full_pipeline("Percentage increase?", responses)
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], 25.0)

    def test_11_model_returns_full_fake_envelope(self):
        responses = [_completion(content='{"answer": 25.0, "log_url": "https://fake/log.jsonl", "status": "ok"}')]
        raw, val, reply, logger = self._run_full_pipeline("Percentage increase?", responses)
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], 25.0)
        self.assertNotIn("fake", reply)

    def test_12_empty_model_response(self):
        responses = [_completion(content=""), _completion(content="Retried answer: 42")]
        raw, val, reply, logger = self._run_full_pipeline("Question?", responses)
        parsed = json.loads(reply)
        self.assertNotEqual(parsed["answer"], "")
        self.assertIsNotNone(parsed["answer"])

    def test_13_tool_call_loop_ids_linked(self):
        responses = [
            _completion(tool_calls=[_tool_call("id_search", "web_search", {"query": "MOSPI MMR"})]),
            _completion(tool_calls=[_tool_call("id_dl", "download_dataset", {"url": "https://mospi.gov.in/data.csv"})]),
            _completion(tool_calls=[_tool_call("id_py", "run_python_analysis", {"code": "print('Assam')"}),]),

            _completion(content="Assam"),
        ]
        stub_tools = {
            "web_search": lambda **kw: {"results": [{"url": "https://mospi.gov.in/data.csv"}]},
            "download_dataset": lambda **kw: {"ok": True, "local_path": "/tmp/data.csv"},
            "run_python_analysis": lambda **kw: {"stdout": "Assam\n", "stderr": "", "returncode": 0},
        }
        with patch.object(agent.config, "TAVILY_API_KEY", "real-key"):
            raw, val, reply, logger = self._run_full_pipeline("MOSPI question", responses, stub_tools)
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], "Assam")

    def test_14_rate_limit(self):
        responses = [_rate_limit_error(), _completion(content="Recovered after rate limit")]
        raw, val, reply, logger = self._run_full_pipeline("Hi", responses)
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], "Recovered after rate limit")

    def test_15_undeclared_tool_regression(self):
        responses = [_tool_use_failed_error(), _completion(content="Answer without tools")]
        raw, val, reply, logger = self._run_full_pipeline("Undeclared tool prompt", responses)
        parsed = json.loads(reply)
        self.assertEqual(parsed["answer"], "Answer without tools")

    def test_16_webhook_endpoint(self):
        from fastapi.testclient import TestClient
        from contextlib import asynccontextmanager
        import bot

        @asynccontextmanager
        async def dummy_lifespan(app):
            yield

        with patch.object(bot.app.router, "lifespan_context", dummy_lifespan):
            with TestClient(bot.app) as client:
                with patch("bot.asyncio.create_task"):
                    with patch("bot.Update.de_json"):
                        resp = client.post("/webhook", json={"update_id": 1})
                        self.assertEqual(resp.status_code, 200)

                with patch.object(bot.config, "WEBHOOK_SECRET", "my-secret"):
                    resp = client.post("/webhook", json={"update_id": 1}, headers={"X-Telegram-Bot-Api-Secret-Token": "bad"})
                    self.assertEqual(resp.status_code, 401)

    def test_17_health_endpoint(self):
        from fastapi.testclient import TestClient
        from contextlib import asynccontextmanager
        import bot

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

    def test_18_log_endpoint(self):
        from fastapi.testclient import TestClient
        from contextlib import asynccontextmanager
        import bot

        @asynccontextmanager
        async def dummy_lifespan(app):
            yield

        raw, val, reply, logger = self._run_full_pipeline("Log test question", [_completion(content="Ok")], chat_id=888888)
        with patch.object(bot.app.router, "lifespan_context", dummy_lifespan):
            with TestClient(bot.app) as client:
                resp = client.get("/logs/888888.jsonl")
                self.assertEqual(resp.status_code, 200)
                lines = resp.text.strip().split("\n")
                self.assertTrue(len(lines) >= 3)
                for line in lines:
                    parsed_line = json.loads(line)
                    self.assertIn("event", parsed_line)
                    self.assertNotIn("GROQ_API_KEY", line)
                    self.assertNotIn("TELEGRAM_BOT_TOKEN", line)






if __name__ == "__main__":
    unittest.main()