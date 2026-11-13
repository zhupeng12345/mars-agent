import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.platform.contracts import DiagnosisDraftPayload
from agentic.platform.llm_gateway import LLMGateway


class GatewayTests(unittest.TestCase):
    def test_fake_provider_returns_structured_json(self) -> None:
        response = LLMGateway().chat(
            "fake_json",
            [{"role": "user", "content": "test"}],
        )
        self.assertEqual(response.status, "completed")
        self.assertIsInstance(response.content, dict)
        self.assertTrue(response.request_id.startswith("req-"))

    def test_fake_provider_obeys_diagnosis_payload_schema(self) -> None:
        response = LLMGateway().chat(
            "fake_json",
            [{"role": "user", "content": "test"}],
            response_schema=DiagnosisDraftPayload,
        )
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.content["risk_level"], "unknown")
        self.assertIn("requires_human_review", response.content)

    def test_ollama_receives_json_schema_and_response_is_validated(self) -> None:
        class Response:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"message": {"content": json.dumps(DiagnosisDraftPayload.fake_payload())}}

        class Session:
            trust_env = True

            def __init__(self):
                self.payload = None

            def post(self, _url, *, json, timeout):
                self.payload = json
                return Response()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({"local": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "qwen3-vl:latest", "retries": 0, "capabilities": ["text"]}}), encoding="utf-8")
            session = Session()
            response = LLMGateway(path, session=session).chat("local", [{"role": "user", "content": "json"}], response_schema=DiagnosisDraftPayload)
        self.assertEqual(response.status, "completed")
        self.assertEqual(session.payload["format"], DiagnosisDraftPayload.json_schema())

    def test_openai_compatible_json_schema_request_and_content_part_response(self) -> None:
        payload = DiagnosisDraftPayload.fake_payload()

        class Response:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"choices": [{"finish_reason": "stop", "message": {"content": [{"type": "text", "text": json.dumps(payload)}]}}], "usage": {"total_tokens": 12}}

        class Session:
            def __init__(self):
                self.payload = None

            def post(self, _url, *, headers, json, timeout):
                self.payload = json
                return Response()

        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"__MARS_TEST_API_KEY__": "test-key"}):
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({"qwen": {"provider": "openai_compatible", "base_url": "https://example.invalid/v1", "model": "qwen", "api_key_env": "__MARS_TEST_API_KEY__", "structured_output_mode": "json_schema", "capabilities": ["text"], "retries": 0}}), encoding="utf-8")
            session = Session()
            response = LLMGateway(path, session=session).chat("qwen", [{"role": "user", "content": "json"}], response_schema=DiagnosisDraftPayload)
        self.assertEqual(response.status, "completed")
        sent_schema = session.payload["response_format"]["json_schema"]
        self.assertEqual(sent_schema["name"], "diagnosisdraftpayload")
        self.assertTrue(sent_schema["strict"])
        self.assertEqual(sent_schema["schema"], DiagnosisDraftPayload.json_schema())
        self.assertEqual(response.content["risk_level"], "unknown")
        self.assertEqual(response.usage["total_tokens"], 12)

    def test_schema_validation_failure_is_not_retried_or_reported_as_provider_success(self) -> None:
        class Response:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"message": {"content": '{"risk_level":"not-a-risk"}'}}

        class Session:
            trust_env = True

            def __init__(self):
                self.calls = 0

            def post(self, _url, *, json, timeout):
                self.calls += 1
                return Response()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({"local": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "qwen3-vl:latest", "retries": 2, "capabilities": ["text"]}}), encoding="utf-8")
            session = Session()
            response = LLMGateway(path, session=session).chat("local", [{"role": "user", "content": "json"}], response_schema=DiagnosisDraftPayload)
        self.assertEqual(response.status, "schema_error")
        self.assertEqual(response.error_type, "schema_error")
        self.assertEqual(response.retries, 0)
        self.assertEqual(session.calls, 1)

    def test_provider_refusal_and_truncation_are_explicit(self) -> None:
        class Session:
            def __init__(self, body):
                self.body = body

            def post(self, _url, *, headers, json, timeout):
                body = self.body
                return type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: body})()

        profile = {"provider": "openai_compatible", "base_url": "https://example.invalid/v1", "model": "qwen", "api_key_env": "__MARS_TEST_API_KEY__", "capabilities": ["text"], "retries": 0}
        for body, expected in (
            ({"choices": [{"finish_reason": "stop", "message": {"refusal": "not allowed", "content": None}}]}, "provider_refusal"),
            ({"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}, "provider_truncated_response"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"__MARS_TEST_API_KEY__": "test-key"}):
                path = Path(directory) / "profiles.json"
                path.write_text(json.dumps({"qwen": profile}), encoding="utf-8")
                response = LLMGateway(path, session=Session(body)).chat("qwen", [{"role": "user", "content": "json"}])
            self.assertEqual(response.status, "provider_error")
            self.assertIn(expected, response.error or "")

    def test_missing_api_key_is_visible_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({"qwen": {"provider": "openai_compatible", "base_url": "https://example.invalid/v1", "model": "qwen", "api_key_env": "__MARS_TEST_MISSING_KEY__", "capabilities": ["text"]}}), encoding="utf-8")
            response = LLMGateway(path).chat("qwen", [{"role": "user", "content": "test"}])
        self.assertEqual(response.status, "configuration_error")
        self.assertIn("missing_api_key", response.error or "")

    def test_ollama_profile_can_disable_thinking_and_returns_content(self) -> None:
        class Response:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"message": {"content": '{"ok":true}', "thinking": "private reasoning"}, "eval_count": 4}

        class Session:
            trust_env = True

            def __init__(self):
                self.payload = None

            def post(self, _url, *, json, timeout):
                self.payload = json
                return Response()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps({"local": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "qwen3-vl:latest", "think": False, "retries": 0, "capabilities": ["text"]}}), encoding="utf-8")
            session = Session()
            response = LLMGateway(path, session=session).chat("local", [{"role": "user", "content": "json"}])
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.content, {"ok": True})
        self.assertFalse(session.payload["think"])
        self.assertNotIn("thinking", response.to_dict())


if __name__ == "__main__":
    unittest.main()
