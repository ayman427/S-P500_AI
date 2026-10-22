import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from sp500_ml.api_ninjas_trial import ApiNinjasError, fetch_news, select_events


class ApiNinjasTrialTests(unittest.TestCase):
    def article(self, stamp="2020-01-04T02:00:00Z", ticker="AAPL"):
        return {"published": int(pd.Timestamp(stamp).timestamp()), "tickers": [ticker],
                "title": "Apple shares surge after earnings beat", "url": "https://example.com/story", "source": "Publisher"}

    def test_dates_tags_provenance_and_header(self):
        response = Mock(status_code=200)
        response.json.return_value = [self.article(), self.article(ticker="MSFT"), self.article("2020-01-04T16:00:00Z")]
        with patch("sp500_ml.api_ninjas_trial.requests.get", return_value=response) as get:
            result = fetch_news("AAPL", "2020-01-03", "2020-01-03", "secret-test")
        self.assertEqual(len(result["headlines"]), 1)
        self.assertEqual(result["headlines"][0]["source"], "API Ninjas")
        self.assertNotIn("excerpt", result["headlines"][0])
        self.assertFalse(result["truncated"])
        self.assertEqual(get.call_args.kwargs["headers"]["X-Api-Key"], "secret-test")
        self.assertNotIn("secret-test", str(get.call_args.args) + str(get.call_args.kwargs["params"]))
        self.assertEqual(get.call_args.kwargs["params"]["end_date"], "2020-01-04")

    def test_failures_do_not_leak_provider_or_credentials(self):
        for status in (400, 401, 403, 429, 200):
            response = Mock(status_code=status)
            response.json.return_value = {"error": "secret-test"}
            with self.subTest(status=status), patch("sp500_ml.api_ninjas_trial.requests.get", return_value=response):
                with self.assertRaises(ApiNinjasError) as caught:
                    fetch_news("AAPL", "2020-01-03", "2020-01-03", "secret-test")
                self.assertNotIn("secret-test", str(caught.exception))
        with patch("sp500_ml.api_ninjas_trial.requests.get", side_effect=requests.Timeout("secret-test")):
            with self.assertRaises(ApiNinjasError) as caught:
                fetch_news("AAPL", "2020-01-03", "2020-01-03", "secret-test")
            self.assertNotIn("secret-test", str(caught.exception))

    def test_pagination_is_bounded_and_honest(self):
        response = Mock(status_code=200)
        response.json.return_value = [self.article()] * 50
        with patch("sp500_ml.api_ninjas_trial.requests.get", return_value=response) as get:
            result = fetch_news("AAPL", "2020-01-03", "2020-01-03", "secret-test")
        self.assertEqual(get.call_count, 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["headlines"]), 1)

    def test_selection_spans_available_history_and_invalid_dates_make_no_requests(self):
        events = [{"date": f"20{year:02d}-01-03"} for year in range(10)]
        selected = select_events(list(reversed(events)))
        self.assertEqual(selected, [events[i] for i in (0, 3, 6, 9)])
        with patch("sp500_ml.api_ninjas_trial.requests.get") as get:
            for start, end in (("bad", "2020-01-03"), ("2020-02-03", "2020-01-03")):
                with self.assertRaises(ValueError):
                    fetch_news("AAPL", start, end, "secret-test")
            get.assert_not_called()
