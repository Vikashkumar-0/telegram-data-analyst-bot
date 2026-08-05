"""
tests/test_agent_e2e.py
------------------------
End-to-end tests for run_agent()'s whole loop -- multi-step tool calling,
rate-limit retry, and the max-steps fallback -- WITHOUT calling the real
Groq API. We patch client.chat.completions.create() to return realistic
ChatCompletion objects (built from the actual openai SDK's response model,
so the shape is exactly what Groq/OpenAI would send back).

This is the standard way to test code that calls an external API: fast,
free, deterministic, and it doesn't burn real rate-limit quota every time
you run your test suite.

Run with:  pytest test_agent_e2e.py -v
"""

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import RateLimitError, BadRequestError
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent


def _completion(content=None, tool_calls=None):
    """Builds a real ChatCompletion object (same class the SDK returns) from
    plain Python values, so tests exercise the exact same code paths
    (tc.model_dump(), message.tool_calls, etc.) as a live response would."""
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
    """Mirrors the exact failure we saw in production: the model tries to
    call a tool that wasn't declared in this request's `tools` list."""
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


class TestRunAgentEndToEnd(unittest.TestCase):
    def setUp(self):
        # Reset pacing state and make sleeps instant so tests run fast.
        agent._last_call_at = 0.0
        self.sleep_patch = patch("agent.time.sleep")
        self.mock_sleep = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def _patch_create(self, side_effect):
        p = patch.object(agent.client.chat.completions, "create", side_effect=side_effect)
        mock = p.start()
        self.addCleanup(p.stop)
        return mock

    def _patch_create_recording(self, responses):
        """Like _patch_create, but also snapshots (deep-copies) the
        `messages` argument at the moment of each call. Needed because
        run_agent mutates the same list object across the loop -- looking at
        mock.call_args_list *after* the run finishes would show every call
        with the FINAL message list, not what was actually sent each time."""
        seen = []

        def fake_create(model, messages, **kwargs):
            seen.append((json.loads(json.dumps(messages)), kwargs))
            return responses[len(seen) - 1]

        p = patch.object(agent.client.chat.completions, "create", side_effect=fake_create)
        p.start()
        self.addCleanup(p.stop)
        return seen

    def test_simple_question_answers_in_one_call_no_tools(self):
        mock = self._patch_create([_completion(content="Paris is the capital of France.")])

        result = agent.run_agent([{"role": "user", "text": "What's the capital of France?"}], log_fn=lambda e: None)

        self.assertEqual(result, "Paris is the capital of France.")
        self.assertEqual(mock.call_count, 1)

    def test_multi_step_tool_calling_then_final_answer(self):
        events = []
        responses = [
            _completion(tool_calls=[_tool_call("call_1", "download_dataset", {"url": "https://example.com/data.csv"})]),
            _completion(tool_calls=[_tool_call("call_2", "run_python_analysis", {"code": "print('Assam')"})]),
            _completion(content='{"state": "Assam"}'),
        ]
        seen = self._patch_create_recording(responses)

        # Stub the tool IMPLEMENTATIONS via the dispatch dict, so this test
        # exercises the AGENT LOOP's wiring (does it call the right tool,
        # feed the result back correctly, stop at the right time) without
        # touching the real network or filesystem. The tools themselves
        # have their own unit tests in test_agent.py.
        stub_tools = {
            "download_dataset": lambda **kw: {"ok": True, "local_path": "/tmp/data.csv"},
            "run_python_analysis": lambda **kw: {"stdout": "Assam\n", "stderr": "", "returncode": 0},
        }
        with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
            result = agent.run_agent(
                [{"role": "user", "text": "Which state has the highest maternal mortality rate?"}],
                log_fn=events.append,
            )

        self.assertEqual(result, '{"state": "Assam"}')
        self.assertEqual(len(seen), 3)

        # Second call's message list must contain the first tool's result,
        # correctly linked back by tool_call_id -- this is what Groq
        # requires to know which call each result answers.
        second_call_messages, _ = seen[1]
        tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(tool_messages[0]["content"]), {"ok": True, "local_path": "/tmp/data.csv"})

        tool_call_events = [e for e in events if e["event"] == "tool_call"]
        self.assertEqual([e["tool"] for e in tool_call_events], ["download_dataset", "run_python_analysis"])

    def test_rate_limit_retries_once_then_succeeds(self):
        mock = self._patch_create([_rate_limit_error(), _completion(content="ok")])

        result = agent.run_agent([{"role": "user", "text": "hi"}], log_fn=lambda e: None)

        self.assertEqual(result, "ok")
        self.assertEqual(mock.call_count, 2)
        # One polite wait using the configured retry delay -- not a retry storm.
        self.mock_sleep.assert_any_call(agent.config.GROQ_RATE_LIMIT_RETRY_SECONDS)

    def test_rate_limit_twice_raises_instead_of_retrying_forever(self):
        self._patch_create([_rate_limit_error(), _rate_limit_error()])

        with self.assertRaises(RateLimitError):
            agent.run_agent([{"role": "user", "text": "hi"}], log_fn=lambda e: None)

    def test_max_steps_exhausted_forces_a_final_no_tools_answer(self):
        # Model keeps asking for the same tool forever -- simulates a
        # confused/looping model. After max_steps, we should force ONE more
        # call with no tools and return whatever it says, not None.
        looping_response = _completion(
            tool_calls=[_tool_call("call_x", "run_python_analysis", {"code": "print(1)"})]
        )
        forced_final = _completion(content='{"state": "unknown"}')
        mock = self._patch_create([looping_response] * 8 + [forced_final])

        stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "1\n", "stderr": "", "returncode": 0}}
        with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
            result = agent.run_agent(
                [{"role": "user", "text": "some ambiguous question"}], log_fn=lambda e: None, max_steps=8
            )

        self.assertEqual(result, '{"state": "unknown"}')
        self.assertEqual(mock.call_count, 9)  # 8 looping steps + 1 forced final
        # The forced final call must NOT offer tools -- otherwise it could
        # just loop again instead of answering.
        final_call_kwargs = mock.call_args_list[-1].kwargs
        self.assertNotIn("tools", final_call_kwargs)

    def test_pacing_waits_between_calls(self):
        agent._last_call_at = time.monotonic()  # pretend a call JUST happened
        self._patch_create([_completion(content="ok")])

        agent.run_agent([{"role": "user", "text": "hi"}], log_fn=lambda e: None)

        # Should have slept roughly GROQ_MIN_INTERVAL_SECONDS before calling,
        # since the "previous call" was just now.
        waited = [c.args[0] for c in self.mock_sleep.call_args_list if c.args]
        self.assertTrue(any(w > 0 for w in waited))

    def test_undeclared_tool_call_retries_once_without_tools(self):
        # Regression test for the production bug: the model tried to call
        # 'web_search' when it wasn't in the declared tools list (because
        # TAVILY_API_KEY wasn't set but the prompt still mentioned it). That
        # crashed the whole run. Now _call_model should catch the 400,
        # retry ONCE with tools stripped, and get a normal text answer.
        mock = self._patch_create([_tool_use_failed_error(), _completion(content='{"state": "unknown"}')])

        result = agent.run_agent(
            [{"role": "user", "text": "Which state has the highest maternal mortality rate?"}],
            log_fn=lambda e: None,
        )

        self.assertEqual(result, '{"state": "unknown"}')
        self.assertEqual(mock.call_count, 2)
        # The retry must have dropped "tools" -- otherwise the model could
        # just try to call the same undeclared tool again.
        self.assertNotIn("tools", mock.call_args_list[1].kwargs)

    def test_other_400_errors_are_not_swallowed(self):
        # Only the specific "tool_use_failed" case gets a retry. Any other
        # 400 (bad request body, invalid model name, etc.) should still
        # surface immediately -- silently retrying every 400 would hide
        # real bugs instead of fixing this one specific known failure mode.
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(status_code=400, request=request, json={"error": {"message": "invalid model"}})
        other_error = BadRequestError("invalid model", response=response, body=None)
        self._patch_create([other_error])

        with self.assertRaises(BadRequestError):
            agent.run_agent([{"role": "user", "text": "hi"}], log_fn=lambda e: None)

    def test_system_prompt_never_mentions_web_search_when_tavily_key_missing(self):
        # Regression test: this exact mismatch (prompt says "use
        # web_search", tools list doesn't include it because no
        # TAVILY_API_KEY) is what caused the production 400 in the first
        # place. Guards against it coming back.
        with patch.object(agent.config, "TAVILY_API_KEY", None):
            prompt = agent._build_system_prompt()
        self.assertNotIn("web_search", prompt)

    def test_system_prompt_mentions_web_search_when_tavily_key_present(self):
        with patch.object(agent.config, "TAVILY_API_KEY", "fake-key-for-test"):
            prompt = agent._build_system_prompt()
        self.assertIn("web_search", prompt)


class TestExtractAnswerValue(unittest.TestCase):
    def test_extract_nested_answer(self):
        from utils import extract_answer_value
        raw = '{"answer": {"answer": 30.0}}'
        self.assertEqual(extract_answer_value(raw), 30.0)

    def test_extract_nested_dict(self):
        from utils import extract_answer_value
        raw = '{"answer": {"state": "Assam"}}'
        self.assertEqual(extract_answer_value(raw), {"state": "Assam"})

    def test_extract_concatenated_json(self):
        from utils import extract_answer_value
        raw = '{"answer": 25.0}{"answer": 25.0}'
        self.assertEqual(extract_answer_value(raw), 25.0)

    def test_extract_code_fenced_json(self):
        from utils import extract_answer_value
        raw = "```json\n{\"answer\": 42}\n```"
        self.assertEqual(extract_answer_value(raw), 42)

    def test_extract_plain_text_and_tables(self):
        from utils import extract_answer_value
        raw = "| Metric | Value |\n|---|---|\n| Mean | 140 |"
        self.assertEqual(extract_answer_value(raw), "| Metric | Value |\n|---|---|\n| Mean | 140 |")


class TestBotWebhook(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import bot
        self.bot = bot
        self.client = TestClient(bot.app)

    def test_health_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("mode", data)

    @patch("bot.asyncio.create_task")
    @patch("bot.Update.de_json")
    def test_webhook_post_success(self, mock_de_json, mock_create_task):
        mock_de_json.return_value = unittest.mock.MagicMock()
        payload = {
            "update_id": 10001,
            "message": {
                "message_id": 1,
                "date": 1441645532,
                "chat": {"id": 1021167690, "type": "private"},
                "text": "2+2?",
            },
        }
        resp = self.client.post("/webhook", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})
        self.assertTrue(mock_create_task.called)

    def test_webhook_secret_unauthorized(self):
        with patch.object(self.bot.config, "WEBHOOK_SECRET", "super-secret-123"):
            resp = self.client.post("/webhook", json={}, headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"})
            self.assertEqual(resp.status_code, 401)
            self.assertIn("Unauthorized", resp.json()["error"])


class TestRealWorldQuestions(unittest.TestCase):
    def test_easy_arithmetic_2_plus_2(self):
        from utils import extract_answer_value
        raw_llm_outputs = ['4', '{"answer": 4}', '{"answer": {"answer": 4}}']
        for raw in raw_llm_outputs:
            extracted = extract_answer_value(raw)
            reply = {"answer": extracted, "log_url": "https://example.com/logs/1.jsonl"}
            reply_json = json.dumps(reply, ensure_ascii=False)
            self.assertIn('"answer": 4', reply_json)

    def test_unicode_table_formatting(self):
        from utils import extract_answer_value
        table_text = "| Month | Change |\n|---|---|\n| Jan → Feb | 20% |\n| Price | ₹800 → ₹1,000 |"
        extracted = extract_answer_value(table_text)
        reply = {"answer": extracted, "log_url": "https://example.com/logs/1.jsonl"}
        reply_json = json.dumps(reply, ensure_ascii=False)
        self.assertIn("₹800", reply_json)
        self.assertIn("Jan → Feb", reply_json)
        self.assertNotIn(r"\u2192", reply_json)


class TestPseudoToolCallInterception(unittest.TestCase):
    def test_extract_pseudo_tool_call_patterns(self):
        from agent import _extract_pseudo_tool_call

        # Pattern 1: <function.run_python_analysis{...}</function>
        text1 = '<function.run_python_analysis{"code": "print(25.0)"}</function>'
        name1, args1 = _extract_pseudo_tool_call(text1)
        self.assertEqual(name1, "run_python_analysis")
        self.assertEqual(args1, {"code": "print(25.0)"})

        # Pattern 2: {"code": "..."}
        text2 = '{"code": "import numpy as np\\nprint(25.0)"}'
        name2, args2 = _extract_pseudo_tool_call(text2)
        self.assertEqual(name2, "run_python_analysis")
        self.assertEqual(args2, {"code": "import numpy as np\nprint(25.0)"})

        # Pattern 3: {"query": "..."}
        text3 = '{"query": "MOSPI maternal mortality rate"}'
        name3, args3 = _extract_pseudo_tool_call(text3)
        self.assertEqual(name3, "web_search")
        self.assertEqual(args3, {"query": "MOSPI maternal mortality rate"})

    def test_run_agent_intercepts_pseudo_tool_call(self):
        responses = [
            _completion(content='<function.run_python_analysis{"code": "print(25.0)"}</function>'),
            _completion(content='25.0'),
        ]
        agent._last_call_at = 0.0
        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses):
                stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "25.0\n", "stderr": "", "returncode": 0}}
                with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
                    result = agent.run_agent(
                        [{"role": "user", "text": "Percentage increase?"}],
                        log_fn=lambda e: None,
                    )
        self.assertEqual(result, '25.0')

    def test_pseudo_tool_call_fallback_to_stdout_when_model_silent(self):
        # If model outputs empty string after tool execution, agent falls back to stdout instead of returning empty answer!
        responses = [
            _completion(content='<function.run_python_analysis{"code": "print(30.0)"}</function>'),
            _completion(content=''),
        ]
        agent._last_call_at = 0.0
        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses):
                stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "30.0\n", "stderr": "", "returncode": 0}}
                with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
                    result = agent.run_agent(
                        [{"role": "user", "text": "Average?"}],
                        log_fn=lambda e: None,
                    )
        self.assertEqual(result, '30.0')


class TestUserSixScenarios(unittest.TestCase):
    def setUp(self):
        agent._last_call_at = 0.0

    def test_case_1_average_calculation(self):
        from utils import extract_answer_value
        responses = [
            _completion(tool_calls=[_tool_call("call_1", "run_python_analysis", {"code": "print(sum([10,20,30,40,50])/5)"})]),
            _completion(content="30.0"),
        ]
        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses):
                stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "30.0\n", "stderr": "", "returncode": 0}}
                with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
                    raw = agent.run_agent([{"role": "user", "text": "What is the average of 10, 20, 30, 40, and 50?"}], log_fn=lambda e: None)
                    val = extract_answer_value(raw)
                    self.assertIn(val, (30, 30.0, "30", "30.0"))
                    reply = json.dumps({"answer": val, "log_url": "https://example.com/logs/1.jsonl"}, ensure_ascii=False)
                    self.assertEqual(reply, '{"answer": 30.0, "log_url": "https://example.com/logs/1.jsonl"}')

    def test_case_2_percentage_increase(self):
        from utils import extract_answer_value
        responses = [
            _completion(tool_calls=[_tool_call("call_1", "run_python_analysis", {"code": "print(((1000-800)/800)*100)"})]),
            _completion(content="25.0"),
        ]
        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses):
                stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "25.0\n", "stderr": "", "returncode": 0}}
                with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
                    raw = agent.run_agent([{"role": "user", "text": "A product price increased from ₹800 to ₹1,000. What is the percentage increase?"}], log_fn=lambda e: None)
                    val = extract_answer_value(raw)
                    self.assertIn(val, (25, 25.0, "25", "25.0"))
                    reply = json.dumps({"answer": val, "log_url": "https://example.com/logs/1.jsonl"}, ensure_ascii=False)
                    self.assertEqual(reply, '{"answer": 25.0, "log_url": "https://example.com/logs/1.jsonl"}')

    def test_case_3_sales_data_analysis_table(self):
        from utils import extract_answer_value
        table_output = (
            "| Metric | Value |\n"
            "|---|---|\n"
            "| Mean | 140.0 |\n"
            "| Median | 130.0 |\n"
            "| Jan→Feb | 20.0% |\n"
            "| Feb→Mar | 25.0% |\n"
            "| Mar→Apr | -13.33% |\n"
            "| Apr→May | 53.85% |\n"
            "| Highest Month | May |"
        )
        responses = [
            _completion(tool_calls=[_tool_call("call_1", "run_python_analysis", {"code": "import pandas as pd..."})]),
            _completion(content=table_output),
        ]
        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses):
                stub_tools = {"run_python_analysis": lambda **kw: {"stdout": "Mean: 140\nMedian: 130\n", "stderr": "", "returncode": 0}}
                with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
                    raw = agent.run_agent([{"role": "user", "text": "I have sales data: January 100, February 120, March 150, April 130, May 200..."}], log_fn=lambda e: None)
                    val = extract_answer_value(raw)
                    self.assertIn("140", val)
                    self.assertIn("130", val)
                    self.assertIn("20", val)
                    self.assertIn("25", val)
                    self.assertIn("-13.33", val)
                    self.assertIn("53.85", val)
                    self.assertIn("May", val)
                    reply = json.dumps({"answer": val, "log_url": "https://example.com/logs/1.jsonl"}, ensure_ascii=False)
                    self.assertNotIn(r"\u2192", reply)

    def test_case_4_rmsle_explanation_no_tools(self):
        from utils import extract_answer_value
        explanation = "RMSLE measures relative error and penalizes underestimates more than overestimates..."
        responses = [_completion(content=explanation)]
        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses) as mock_create:
                raw = agent.run_agent([{"role": "user", "text": "Explain why RMSLE is useful for predicting heavy equipment selling prices..."}], log_fn=lambda e: None)
                val = extract_answer_value(raw)
                self.assertEqual(val, explanation)
                self.assertEqual(mock_create.call_count, 1)

    def test_case_5_prime_minister_general_question(self):
        from utils import extract_answer_value
        responses = [_completion(content="The current Prime Minister of India is Narendra Modi.")]
        with patch("agent.time.sleep"):
            with patch.object(agent.client.chat.completions, "create", side_effect=responses) as mock_create:
                raw = agent.run_agent([{"role": "user", "text": "Who is the current Prime Minister of India?"}], log_fn=lambda e: None)
                val = extract_answer_value(raw)
                self.assertIn("Narendra Modi", val)
                self.assertEqual(mock_create.call_count, 1)

    def test_case_6_genuine_external_data_search_execution(self):
        from utils import extract_answer_value
        responses = [
            _completion(tool_calls=[_tool_call("call_search", "web_search", {"query": "MOSPI maternal mortality rate state"})]),
            _completion(content="According to the latest MOSPI report retrieved via web search, Assam reported the highest maternal mortality rate."),
        ]
        with patch("agent.time.sleep"):
            with patch.object(agent.config, "TAVILY_API_KEY", "real-tavily-key"):
                with patch.object(agent.client.chat.completions, "create", side_effect=responses):
                    search_called = []
                    def stub_search(query):
                        search_called.append(query)
                        return [{"title": "MOSPI Data", "url": "https://mospi.gov.in/mmr", "snippet": "Assam MMR is 215 per 100k"}]

                    stub_tools = {
                        "web_search": lambda query="": {"results": stub_search(query)},
                    }
                    with patch.dict(agent.TOOL_FUNCTIONS, stub_tools):
                        raw = agent.run_agent([{"role": "user", "text": "Which state has highest MMR according to MOSPI data?"}], log_fn=lambda e: None)
                        val = extract_answer_value(raw)
                        self.assertEqual(len(search_called), 1)
                        self.assertIn("Assam", val)
                        self.assertNotIn("example.com", val)


if __name__ == "__main__":
    unittest.main()