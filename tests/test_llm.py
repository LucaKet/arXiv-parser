"""Tests for the fallback provider chain."""

from __future__ import annotations

import io
import json
import unittest
import urllib.request
from unittest import mock

from arxiv_parser.llm import (
    FallbackProvider,
    LLMError,
    LLMProvider,
    OpenAICompatProvider,
)


class Stub(LLMProvider):
    def __init__(self, name: str, reply: str | Exception) -> None:
        self._name = name
        self._reply = reply
        self.calls = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    def complete(self, system: str, user: str, temperature: float) -> str:
        self.calls += 1
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


class TestFallbackProvider(unittest.TestCase):
    def test_first_provider_success_no_fallback(self):
        a, b = Stub("a", "ok-a"), Stub("b", "ok-b")
        p = FallbackProvider([a, b])
        self.assertEqual(p.complete("s", "u", 0.2), "ok-a")
        self.assertEqual(a.calls, 1)
        self.assertEqual(b.calls, 0)

    def test_primary_fails_uses_fallback(self):
        a, b = Stub("a", LLMError("down")), Stub("b", "ok-b")
        p = FallbackProvider([a, b])
        self.assertEqual(p.complete("s", "u", 0.2), "ok-b")
        self.assertEqual(a.calls, 1)
        self.assertEqual(b.calls, 1)

    def test_chain_of_three(self):
        a = Stub("a", LLMError("down"))
        b = Stub("b", LLMError("down too"))
        c = Stub("c", "ok-c")
        p = FallbackProvider([a, b, c])
        self.assertEqual(p.complete("s", "u", 0.2), "ok-c")

    def test_all_fail_raises_combined(self):
        a, b = Stub("a", LLMError("down")), Stub("b", LLMError("down too"))
        p = FallbackProvider([a, b])
        with self.assertRaises(LLMError) as cm:
            p.complete("s", "u", 0.2)
        self.assertIn("a: down", str(cm.exception))
        self.assertIn("b: down too", str(cm.exception))

    def test_empty_chain_rejected(self):
        with self.assertRaises(ValueError):
            FallbackProvider([])

    def test_name_shows_chain(self):
        p = FallbackProvider([Stub("a", "x"), Stub("b", "y")])
        self.assertEqual(p.name, "a -> b")


class TestOpenAICompatProvider(unittest.TestCase):
    def _raise_http_error(self, code: int, body: bytes):
        import urllib.error

        def urlopen(req, timeout):
            raise urllib.error.HTTPError(req.full_url, code, "msg", None, io.BytesIO(body))

        return mock.patch.object(urllib.request, "urlopen", side_effect=urlopen)

    def test_error_message_extracted_from_json_object(self):
        body = json.dumps({"error": {"message": "model overloaded"}}).encode()
        with self._raise_http_error(503, body):
            with self.assertRaises(LLMError) as cm:
                OpenAICompatProvider("http://x/v1", "m", "k").complete("s", "u", 0.2)
        self.assertIn("HTTP 503: model overloaded", str(cm.exception))
        self.assertNotIn("{", str(cm.exception))

    def test_error_message_extracted_from_json_array(self):
        body = json.dumps([{"error": {"message": "high demand"}}]).encode()
        with self._raise_http_error(503, body):
            with self.assertRaises(LLMError) as cm:
                OpenAICompatProvider("http://x/v1", "m", "k").complete("s", "u", 0.2)
        self.assertIn("HTTP 503: high demand", str(cm.exception))

    def test_non_json_body_kept_verbatim(self):
        with self._raise_http_error(502, b"<html>bad gateway</html>"):
            with self.assertRaises(LLMError) as cm:
                OpenAICompatProvider("http://x/v1", "m", "k").complete("s", "u", 0.2)
        self.assertIn("bad gateway", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
