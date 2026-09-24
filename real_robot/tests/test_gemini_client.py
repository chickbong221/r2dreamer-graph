"""How the Gemini client handles failures: nothing is asked again identically unless it can help."""

from __future__ import annotations

import glob
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ..preprocessing.gemini_client import (
    GeminiClient,
    GeminiError,
    GeminiMalformed,
    GeminiRequestFailed,
    GeminiTruncated,
    TextPart,
    VideoPart,
    normalize_usage,
)

USAGE = {"prompt_token_count": 1200, "candidates_token_count": 80, "cached_content_token_count": 1000}


class Transient(Exception):
    code = 503


class Scripted(GeminiClient):
    def __init__(self, cfg, script):
        super().__init__(cfg)
        self.script = list(script)
        self.calls = 0

    def _generate_content(self, parts, schema):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def settings(cache_dir, **changes):
    return {"backend": "generate_content", "model": "test-model", "cache_dir": cache_dir, "temperature": 0.0,
            "max_output_tokens": 100, "media_resolution": "default", "max_retries": 3, "retry_base_seconds": 0.0,
            "malformed_json_retries": 1, **changes}


class Failures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.parts = [TextPart("annotate this")]

    def tearDown(self):
        self.tmp.cleanup()

    def failures(self):
        return glob.glob(os.path.join(self.tmp.name, "failures", "*.json"))

    def test_forcing_a_new_answer_bypasses_success_cache(self):
        client = Scripted(settings(self.tmp.name), [('{"a": 1}', USAGE, "STOP"),
                                                   ('{"a": 2}', USAGE, "STOP")])
        client.generate_json(self.parts, {}, "test")
        client.use_response_cache = False
        answer, record = client.generate_json(self.parts, {}, "test")
        self.assertEqual(answer, {"a": 2})
        self.assertFalse(record["cached"])
        self.assertEqual(client.calls, 2)

    def test_token_preflight_checks_input_and_output_limits(self):
        client = GeminiClient(settings(self.tmp.name, input_token_headroom=10))
        models = Mock()
        client._client = SimpleNamespace(models=models)
        models.get.return_value = SimpleNamespace(input_token_limit=1000, output_token_limit=200)
        models.count_tokens.return_value = SimpleNamespace(total_tokens=800)
        client._check_token_budget(self.parts)
        models.count_tokens.return_value.total_tokens = 950
        with self.assertRaisesRegex(GeminiError, "whole episode"):
            client._check_token_budget(self.parts)
        client.cfg["max_output_tokens"] = 201
        with self.assertRaisesRegex(GeminiError, "max_output_tokens"):
            client._check_token_budget(self.parts)
        models.generate_content.assert_not_called()

    def test_actual_sdk_accepts_video_metadata_and_nullable_answer_schema(self):
        try:
            from google.genai import types
        except ImportError:
            self.skipTest("requires google-genai")
        from ..preprocessing.annotate_episode import answer_schema
        from ..graphs.validate import full_scope
        from . import synthetic

        spec = synthetic.spec("blue_on_red")
        schema = answer_schema(spec, full_scope(spec))
        client = GeminiClient(settings(self.tmp.name, preflight_tokens=True))
        models = Mock()
        client._client = SimpleNamespace(models=models)
        models.get.return_value = SimpleNamespace(input_token_limit=1000000, output_token_limit=65536)
        models.count_tokens.return_value = SimpleNamespace(total_tokens=1000)
        models.generate_content.return_value = SimpleNamespace(text='{}', candidates=[], usage_metadata=None)
        client.upload = Mock(return_value={"uri": "https://example.org/test.mp4", "mime_type": "video/mp4"})
        client._generate_content([TextPart("test"), VideoPart("unused.mp4", 10, 10)], schema)
        kwargs = models.generate_content.call_args.kwargs
        self.assertIsInstance(kwargs["config"], types.GenerateContentConfig)
        self.assertEqual(kwargs["config"].response_json_schema, schema)
        self.assertEqual(kwargs["contents"][0].parts[1].video_metadata.fps, 10)

    def test_a_truncated_answer_is_not_asked_again_identically(self):
        client = Scripted(settings(self.tmp.name), [('{"facts": [', USAGE, "FinishReason.MAX_TOKENS")])
        with self.assertRaises(GeminiTruncated) as caught:
            client.generate_json(self.parts, {"type": "object"}, "episode/events")
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(self.failures()), 1)
        # The cut-off answer was billed; the exception says for how much.
        self.assertEqual(caught.exception.usage["input_tokens"], 1200)

    def test_malformed_json_is_asked_once_more_and_saved(self):
        client = Scripted(settings(self.tmp.name), [("{not json", USAGE, "STOP"), ('{"a": 1}', USAGE, "STOP")])
        parsed, record = client.generate_json(self.parts, {"type": "object"}, "episode/events")
        self.assertEqual(parsed, {"a": 1})
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(self.failures()), 1)
        self.assertEqual(record["usage_normalized"]["cached_tokens"], 1000)
        # Both attempts were billed: the call reports both, not just the answer that parsed.
        self.assertEqual(record["usage_total"]["input_tokens"], 2400)
        self.assertEqual([a["outcome"] for a in record["attempts"]], ["malformed", "ok"])
        again, cached = client.generate_json(self.parts, {"type": "object"}, "episode/events")
        self.assertTrue(cached["cached"])
        self.assertEqual(client.calls, 2)

    def test_malformed_json_twice_raises(self):
        client = Scripted(settings(self.tmp.name), [("{no", USAGE, "STOP"), ("{still no", USAGE, "STOP")])
        with self.assertRaises(GeminiMalformed) as caught:
            client.generate_json(self.parts, {"type": "object"}, "episode/events")
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(self.failures()), 2)
        self.assertEqual(caught.exception.usage["input_tokens"], 2400)

    def test_failures_in_quick_succession_are_all_kept(self):
        client = Scripted(settings(self.tmp.name, malformed_json_retries=7), [("{no", USAGE, "STOP")] * 8)
        with self.assertRaises(GeminiMalformed):
            client.generate_json(self.parts, {"type": "object"}, "episode/events")
        self.assertEqual(len(self.failures()), 8)

    def test_transient_service_errors_are_retried(self):
        client = Scripted(settings(self.tmp.name), [Transient("busy"), ('{"a": 2}', USAGE, "STOP")])
        parsed, _ = client.generate_json(self.parts, {"type": "object"}, "episode/events")
        self.assertEqual(parsed, {"a": 2})
        self.assertEqual(client.calls, 2)

    def test_request_errors_are_not_retried(self):
        client = Scripted(settings(self.tmp.name), [ValueError("bad schema"), ('{"a": 3}', USAGE, "STOP")])
        with self.assertRaises(GeminiRequestFailed):
            client.generate_json(self.parts, {"type": "object"}, "episode/events")
        self.assertEqual(client.calls, 1)

    def test_the_interactions_backend_sends_the_configured_temperature(self):
        sent = []

        class Interactions:
            def create(self, **kwargs):
                sent.append(kwargs)
                return type("Interaction", (), {"usage": None, "status": "completed", "output_text": '{"a": 1}'})()

        for temperature, expected in ((0.0, {"max_output_tokens": 100, "temperature": 0.0}),
                                      (None, {"max_output_tokens": 100})):
            client = GeminiClient(settings(self.tmp.name, backend="interactions", temperature=temperature))
            client._client = type("Client", (), {"interactions": Interactions()})()
            text, _, status = client._interactions(self.parts, {"type": "object"})
            self.assertEqual(sent[-1]["generation_config"], expected)
            self.assertEqual(client.settings()["temperature"], temperature)
            self.assertEqual((text, status), ('{"a": 1}', "completed"))

    def test_usage_from_either_backend(self):
        self.assertEqual(normalize_usage(USAGE)["input_tokens"], 1200)
        other = normalize_usage({"total_input_tokens": 5, "total_output_tokens": 6, "total_cached_tokens": 4})
        self.assertEqual((other["input_tokens"], other["output_tokens"], other["cached_tokens"]), (5, 6, 4))


if __name__ == "__main__":
    unittest.main()
