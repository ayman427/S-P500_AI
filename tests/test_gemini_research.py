import asyncio
import json
import os
import unittest
from unittest.mock import Mock, patch

from sp500_ml import gemini_research as gemini
from sp500_ml.auth import AuthenticatedUser
from tests.test_auth import invoke_asgi


QUOTE = "Apple announced a share repurchase program after reporting quarterly results."

def grounded():
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps({"findings": [{"source": 1, "quote": QUOTE, "analysis": "The announcement may explain interest; timing alone does not establish causation."}]})}]}}]}


class GeminiTests(unittest.TestCase):
    def setUp(self):
        for p in [patch.object(gemini, "load_environment"), patch.object(gemini, "load_evidence", return_value={"headlines": [{"title": "Apple announcement", "url": "https://www.apple.com/newsroom/report", "published_at_utc": "2024-05-02T12:00:00Z", "excerpt": QUOTE}], "documents": []}), patch.object(gemini, "_company_aliases", return_value=["Apple", "AAPL"]),
                  patch.object(gemini, "read_spikes", return_value={"ticker": "AAPL", "events": [{"date": "2024-05-03", "previousDate": "2024-05-02"}]}),
                  patch.dict(os.environ, {"GEMINI_API_KEY": "test-private-gemini-key"})]:
            p.start(); self.addCleanup(p.stop)

    def response(self, status=200, payload=None):
        return Mock(status_code=status, content=b"response", is_redirect=False, json=Mock(return_value=grounded() if payload is None else payload))

    def test_grounded_research_uses_server_key_search_and_never_saves_or_labels_training_data(self):
        with patch.object(gemini.requests, "post", return_value=self.response()) as post, patch("sp500_ml.spike_evidence.save_evidence") as save:
            result = gemini.analyze_saved_evidence("AAPL", "2024-05-03")
        self.assertFalse(result["grounded"])
        self.assertEqual(result["mode"], "evidence")
        self.assertFalse(result["stored"])
        self.assertFalse(result["usableForTraining"])
        self.assertFalse(result["independentlyVerified"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "test-private-gemini-key")
        self.assertNotIn("tools", post.call_args.kwargs["json"])
        self.assertEqual(post.call_args.kwargs["json"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("test-private-gemini-key", repr(result))
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        save.assert_not_called()

    def test_unicode_source_text_is_sent_without_double_escaped_quotes(self):
        quote = "Apple announced its \u201cquarterly results\u201d and a repurchase program."
        archive = {"documents": [{"title": "Apple results", "url": "https://www.apple.com/report", "publishedAt": "2024-05-02", "text": quote}]}
        payload = grounded()
        payload["candidates"][0]["content"]["parts"][0]["text"] = json.dumps({"findings": [{"source": 1, "quote": quote, "analysis": "A business announcement, with uncertain price impact."}]})
        with patch.object(gemini, "load_evidence", return_value=archive), patch.object(gemini.requests, "post", return_value=self.response(payload=payload)) as post:
            result = gemini.analyze_saved_evidence("AAPL", "2024-05-03")
        self.assertIn(quote, post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"])
        self.assertIn(quote, result["text"])

    def test_fabricated_quotes_sources_and_incomplete_answers_are_rejected(self):
        for change in ["quote", "source", "boolean", "finish", "secret", "json"]:
            with self.subTest(change=change):
                payload = grounded()
                candidate = payload["candidates"][0]
                findings = json.loads(candidate["content"]["parts"][0]["text"])
                if change == "quote": findings["findings"][0]["quote"] = "Invented announcement with no original source"
                if change == "source": findings["findings"][0]["source"] = 9
                if change == "boolean": findings["findings"][0]["source"] = True
                if change == "secret": findings["findings"][0]["analysis"] = "test-private-gemini-key"
                candidate["content"]["parts"][0]["text"] = json.dumps(findings) if change != "json" else "invalid"
                if change == "finish": candidate["finishReason"] = "MAX_TOKENS"
                with patch.object(gemini.requests, "post", return_value=self.response(payload=payload)), self.assertRaises(gemini.GeminiResearchError):
                    gemini.analyze_saved_evidence("AAPL", "2024-05-03")

    def test_rate_limits_access_errors_and_invalid_dates_do_not_leak_provider_body(self):
        for status, expected in [(429, 429), (403, 503), (404, 503), (400, 503)]:
            response = self.response(status, {"error": "test-private-gemini-key"})
            with patch.object(gemini.requests, "post", return_value=response), self.assertRaises(gemini.GeminiResearchError) as error:
                gemini.analyze_saved_evidence("AAPL", "2024-05-03")
            self.assertEqual(error.exception.status, expected)
            self.assertNotIn("test-private-gemini-key", str(error.exception))
            response.json.assert_not_called()
        with patch.object(gemini.requests, "post") as post, self.assertRaises(ValueError):
            gemini.analyze_saved_evidence("AAPL", "not-a-date")
        post.assert_not_called()

    def test_missing_unrelated_undated_and_unsafe_evidence_never_calls_gemini(self):
        for changes in [{}, {"excerpt": "Unrelated company report", "title": "Other company"}, {"published_at_utc": "2020-01-01"}, {"url": "https://127.0.0.1/private"}]:
            row = {"title": "Apple", "excerpt": QUOTE, "url": "https://www.apple.com/report", "published_at_utc": "2024-05-02", **changes}
            archive = {"headlines": [row]} if changes else {}
            with patch.object(gemini, "load_evidence", return_value=archive), patch.object(gemini.requests, "post") as post, self.assertRaises(gemini.GeminiResearchError) as error:
                gemini.analyze_saved_evidence("AAPL", "2024-05-03")
            self.assertEqual(error.exception.status, 422)
            post.assert_not_called()


class GeminiApiTests(unittest.TestCase):
    def test_route_requires_signin_and_returns_sanitized_provider_errors(self):
        with patch.object(gemini, "analyze_saved_evidence", return_value={"grounded": True}) as research:
            self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/gemini", {"ticker": "AAPL", "date": "2024-05-03"})), 401)
            research.assert_not_called()
            with patch("vercel_app.authenticate", return_value=AuthenticatedUser("alice", "token")):
                self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/gemini", {"ticker": "AAPL", "date": "2024-05-03"})), 200)
                research.side_effect = gemini.GeminiResearchError("Quota reached", 429)
                self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/gemini", {"ticker": "AAPL", "date": "2024-05-03"})), 429)
