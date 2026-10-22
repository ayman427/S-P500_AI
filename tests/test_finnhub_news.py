import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from sp500_ml.finnhub_news import fetch_company_news, FinnhubNewsError
from sp500_ml.risk_news import read_event_news, normalized_headlines


def article(timestamp="2020-01-03T15:00:00Z", related="AAPL"):
    return {"datetime": int(pd.Timestamp(timestamp).timestamp()), "related": related,
            "headline": "Company earnings report", "source": "Original Publisher",
            "url": "https://example.com/article"}


class FinnhubRequestTests(unittest.TestCase):
    def response(self, payload, status=200):
        response = Mock(status_code=status)
        response.json.return_value = payload
        return response

    def test_api_key_stays_in_header_and_dates_cover_new_york_boundary(self):
        response = self.response([article(), article("2020-01-04T02:00:00Z"),
                                  article("2020-01-04T15:00:00Z"), article(related="MSFT")])
        with patch("sp500_ml.finnhub_news.requests.get", return_value=response) as get:
            rows = fetch_company_news("AAPL", "2020-01-03", 0, "private-test-key")
        args, kwargs = get.call_args
        self.assertNotIn("private-test-key", args[0])
        self.assertNotIn("token", kwargs["params"])
        self.assertEqual(kwargs["headers"]["X-Finnhub-Token"], "private-test-key")
        self.assertEqual(kwargs["params"], {"symbol": "AAPL", "from": "2020-01-03", "to": "2020-01-04"})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["publicationDate"] == "2020-01-03" for row in rows))
        self.assertTrue(all(row["source"] == "Finnhub" for row in rows))
        self.assertTrue(all(row["publisher"] == "Original Publisher" for row in rows))

    def test_provider_failures_are_safe_and_distinct(self):
        for status, expected in [(401, "unauthorized"), (403, "forbidden"), (429, "rate_limited")]:
            with self.subTest(status=status), patch("sp500_ml.finnhub_news.requests.get", return_value=self.response({}, status)):
                with self.assertRaises(FinnhubNewsError) as caught:
                    fetch_company_news("AAPL", "2020-01-03", 0, "private-test-key")
                self.assertEqual(caught.exception.status, expected)
                self.assertNotIn("private-test-key", str(caught.exception))
        for payload in ({"error": "private-test-key"}, "invalid response"):
            with self.subTest(payload=payload), patch("sp500_ml.finnhub_news.requests.get", return_value=self.response(payload)):
                with self.assertRaises(FinnhubNewsError) as caught:
                    fetch_company_news("AAPL", "2020-01-03", 0, "private-test-key")
                self.assertNotIn("private-test-key", str(caught.exception))

    def test_network_exception_does_not_expose_request_credentials(self):
        with patch("sp500_ml.finnhub_news.requests.get", side_effect=requests.Timeout("private-test-key")):
            with self.assertRaises(FinnhubNewsError) as caught:
                fetch_company_news("AAPL", "2020-01-03", 0, "private-test-key")
            self.assertNotIn("private-test-key", str(caught.exception))

    def test_archive_normalization_preserves_provider_provenance(self):
        rows = normalized_headlines([{"published_at_utc": "2020-01-03T15:00:00Z", "title": "News",
                                     "url": "https://example.com/article", "source": "Finnhub"}])
        self.assertEqual(rows[0]["source"], "Finnhub")


class FinnhubCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in [patch("sp500_ml.risk_news.NEWS_DIR", Path(self.directory.name)),
                        patch("sp500_ml.prediction_store.load_environment"),
                        patch.dict(os.environ, {"FINNHUB_API_KEY": "private-test-key"})]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_event_cache_prevents_repeat_calls_and_archives_matching_articles(self):
        fresh = [{"published_at_utc": "2020-01-03T15:00:00Z", "title": "Historical news",
                  "publisher": "Original Publisher", "source": "Finnhub", "url": "https://example.com/article"}]
        with patch("sp500_ml.finnhub_news.fetch_company_news", return_value=fresh) as fetch:
            first = read_event_news("AAPL", "2020-01-03")
            second = read_event_news("AAPL", "2020-01-03")
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(first["providerStatus"], "refreshed")
            self.assertEqual(second["providerStatus"], "cached")
            self.assertEqual(second["provider"], "Finnhub")
            self.assertEqual(second["matchCount"], 1)
            self.assertEqual(second["headlines"][0]["source"], "Finnhub")
            # A different date window must not reuse the same cached response.
            read_event_news("AAPL", "2020-01-03", 3)
            self.assertEqual(fetch.call_count, 2)
        for path in Path(self.directory.name).rglob("*.json"):
            self.assertNotIn("private-test-key", path.read_text())

    def test_empty_success_is_cached_but_not_presented_as_proof_of_no_news(self):
        with patch("sp500_ml.finnhub_news.fetch_company_news", return_value=[]) as fetch:
            read_event_news("AAPL", "2020-01-03")
            result = read_event_news("AAPL", "2020-01-03")
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(result["matchCount"], 0)
            self.assertIn("coverage", result["message"])

    def test_rate_limit_keeps_archived_articles_and_explains_failure(self):
        root = Path(self.directory.name)
        (root / "archive").mkdir()
        (root / "archive/AAPL.json").write_text(json.dumps({"headlines": [{"published_at_utc": "2020-01-03T15:00:00Z", "title": "Saved Yahoo article", "url": "https://example.com/yahoo"}]}))
        with patch("sp500_ml.finnhub_news.fetch_company_news", side_effect=FinnhubNewsError("rate_limited", "Try again shortly.")):
            result = read_event_news("AAPL", "2020-01-03")
        self.assertEqual(result["providerStatus"], "rate_limited")
        self.assertEqual(result["providerMessage"], "Try again shortly.")
        self.assertEqual(result["headlines"][0]["source"], "Yahoo Finance feed")
        self.assertFalse((root / "events").exists())
