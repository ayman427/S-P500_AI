import asyncio
import json
import os
import threading
import unittest
from http.client import HTTPConnection
from unittest.mock import patch, Mock

import pandas as pd

from sp500_ml.auth import AuthenticatedUser
from sp500_ml.finnhub_news import FinnhubNewsError
from sp500_ml.spike_research import classify_sources, research_spike, read_investigation, _rpc
from tests.test_auth import invoke_asgi
from sp500_ml.sessions import _calendar


EVENT = {"date": "2024-05-03", "previousDate": "2024-05-02", "previousClose": 100., "close": 110., "return": .1, "threshold": .05}


def article(title="Apple shares surge after earnings beat estimates", excerpt="Apple shares surged after earnings beat expectations.", published="2024-05-02T21:00:00Z"):
    return {"title": title, "excerpt": excerpt, "published_at_utc": published,
            "publisher": "Example publisher", "source": "Finnhub", "url": "https://example.com/" + title.replace(" ", "-")}


class SourceTests(unittest.TestCase):
    def classify(self, rows, previous="2024-05-02", event="2024-05-03"):
        return classify_sources(rows, ["AAPL", "Apple"], previous, event, "2026-10-19T12:00:00Z")

    def test_evidence_is_focused_deduplicated_and_not_an_invented_cause(self):
        rows = self.classify([article(), article(), article("Microsoft jumps after earnings beat estimates", "Microsoft shares surged after earnings beat expectations."),
                              article("Best stocks to buy: Apple earnings beat estimates"), article("Apple opens a store", "Apple opened a store.")])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["categories"], ["Earnings / guidance"])
        self.assertTrue(rows[0]["reportedLink"])
        self.assertEqual(rows[0]["basis"], "provider_excerpt")
        self.assertEqual(rows[0]["summary"], article()["excerpt"])

    def test_current_other_date_and_invalid_links_are_not_substituted(self):
        invalid = article(); invalid["url"] = "javascript:alert(1)"
        self.assertEqual(self.classify([article(published="2026-10-01T12:00:00Z"), article(published="2024-04-20T12:00:00Z"), invalid]), [])

    def test_publication_times_use_exchange_close_including_early_close(self):
        rows = self.classify([article(published="2024-05-02T19:00:00Z"),
                              article("Apple raises guidance", published="2024-05-03T18:00:00Z"),
                              article("Apple announces share repurchase", published="2024-05-03T21:00:00Z")])
        self.assertEqual({row["timing"] for row in rows}, {"before_move", "during_move", "after_close"})
        early = self.classify([article(published="2024-11-29T18:30:00Z")], previous="2024-11-27", event="2024-11-29")
        self.assertEqual(early[0]["timing"], "after_close")

    def test_headline_only_is_labeled_and_separate_sentences_do_not_prove_a_link(self):
        row = self.classify([article("Apple earnings beat estimates", "Apple shares surged. Apple earnings beat estimates after the report.")])[0]
        self.assertFalse(row["reportedLink"])
        row = self.classify([article(excerpt="")])[0]
        self.assertEqual(row["basis"], "headline_only")

    def test_provider_tag_does_not_allow_other_company_excerpt_to_add_topics(self):
        row = self.classify([article("Apple announces share repurchase", "Microsoft shares surged after earnings beat estimates.")])[0]
        self.assertEqual(row["categories"], ["Buybacks / dividends"])
        self.assertFalse(row["reportedLink"])

    def test_surging_product_demand_is_not_reported_as_a_surging_stock(self):
        row = self.classify([article("Apple earnings beat estimates", "Apple earnings beat expectations as demand for its phones surged.")])[0]
        self.assertFalse(row["reportedLink"])

    def test_wider_window_labels_context_and_does_not_link_an_old_rally_to_this_spike(self):
        rows = self.classify([article(published="2024-04-28T15:00:00Z"), article("Apple shares surge after earnings release", published="2024-05-05T15:00:00Z")])
        self.assertEqual({row["contextPhase"] for row in rows}, {"earlier_context", "later_reporting"})
        self.assertFalse(any(row["reportedLink"] for row in rows))
        self.assertEqual(self.classify([article(published="2024-04-25T15:00:00Z")]), [])

    def test_focused_company_pronouns_can_supply_the_missing_topic(self):
        row = self.classify([article("Apple shares surge", "The company raised its guidance.")])[0]
        self.assertEqual(row["categories"], ["Earnings / guidance"])

    def test_presidential_statements_are_dated_company_context_not_automatic_manipulation(self):
        row = self.classify([article("Apple shares surge after president's remarks", "Apple shares surged after the US president said it would benefit from a policy.")])[0]
        self.assertIn("Political / public statements", row["categories"])
        self.assertTrue(row["reportedLink"])
        self.assertEqual(row["contextPhase"], "move_window")
        self.assertEqual(self.classify([article("Apple president says shares will rise", "Apple company president said the stock will rise.")]), [])


class ResearchTests(unittest.TestCase):
    def setUp(self):
        for patcher in [patch("sp500_ml.risk_model.risk_score", return_value={"status": "not_trained", "probability": None, "target": "20-session downside risk", "message": "Not trained"}),
                        patch("sp500_ml.spike_research.read_spikes", return_value={"ticker": "AAPL", "events": [EVENT]}),
                        patch("sp500_ml.spike_research._company_aliases", return_value=["Apple", "AAPL"]),
                        patch("sp500_ml.spike_research.load_environment"),
                        patch("sp500_ml.spike_research.load_evidence", return_value={"headlines": [], "documents": []}),
                        patch("sp500_ml.spike_research.save_evidence"),
                        patch("sp500_ml.spike_research.fetch_sec_evidence", return_value={"status": "empty", "message": "No filings", "sources": [], "documentsChecked": 0, "truncated": False}),
                        patch("sp500_ml.spike_research.fetch_company_announcements", return_value={"status": "not_supported", "message": "No adapter", "sources": [], "documentsChecked": 0, "truncated": False}),
                        patch.dict(os.environ, {"FINNHUB_API_KEY": "test-private-key"})]:
            patcher.start(); self.addCleanup(patcher.stop)

    def test_save_requires_receipt_and_records_research_not_predictive_labels(self):
        with patch("sp500_ml.spike_research._rpc", side_effect=[None, {"id": "confirmed-id"}]) as rpc, patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[article()]):
            row = research_spike("AAPL", EVENT["date"])
        self.assertTrue(row["saved"])
        self.assertEqual(row["recordId"], "confirmed-id")
        self.assertFalse(row["usableForTraining"])
        self.assertEqual(row["confidence"], "moderate")
        self.assertEqual(row["assessment"]["status"], "reported_business")
        self.assertIsNone(row["assessment"]["return20"])
        self.assertIn("not verified", row["overview"])
        self.assertNotIn("test-private-key", json.dumps(rpc.call_args.args[2]))
        with patch("sp500_ml.spike_research._rpc", side_effect=[None, {}]), patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "confirm"):
                research_spike("AAPL", EVENT["date"])

    def test_empty_coverage_stays_unknown_and_provider_failures_are_distinct(self):
        for failure in (None, FinnhubNewsError("rate_limited", "Try again later")):
            with patch("sp500_ml.spike_research._rpc", side_effect=[None, {"id": "id"}]), patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[], side_effect=failure):
                row = research_spike("AAPL", EVENT["date"])
                self.assertEqual(row["confidence"], "unknown")
                self.assertEqual(row["categories"], [])
                self.assertEqual(row["providerStatus"], "rate_limited" if failure else "empty")

    def test_invalid_or_undetected_dates_do_no_provider_work(self):
        with patch("sp500_ml.spike_research._rpc") as rpc, patch("sp500_ml.spike_research.fetch_company_news_range") as fetch:
            for date in ("2024-01-01", "tomorrow", None):
                with self.assertRaises(ValueError): research_spike("AAPL", date)
            rpc.assert_not_called(); fetch.assert_not_called()

    def test_primary_fallback_is_saved_when_finnhub_has_no_context(self):
        source = {"title": "Issuer release", "url": "https://example.com/release", "publishedAt": "2024-05-02", "categories": ["Buybacks / dividends"], "reportedLink": False, "basis": "company_announcement", "timing": "unknown_time"}
        primary = {"status": "ok", "message": "Checked", "sources": [source], "documentsChecked": 1, "truncated": False}
        with patch("sp500_ml.spike_research._rpc", side_effect=[None, {"id": "id"}]), patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[]), patch("sp500_ml.spike_research.fetch_company_announcements", return_value=primary):
            row = research_spike("AAPL", EVENT["date"])
        self.assertEqual(row["evidenceStatus"], "possible_explanation")
        self.assertEqual(row["categories"], ["Buybacks / dividends"])
        self.assertEqual(row["coverage"][-1]["status"], "ok")
        self.assertFalse(row["usableForTraining"])

    def test_unavailable_and_inconclusive_are_different_from_no_documents(self):
        for count, status, expected in [(0, "empty", "no_documents_found"), (1, "ok", "inconclusive"), (0, "unavailable", "sources_unavailable")]:
            fallback = {"status": status, "message": "Checked", "sources": [], "documentsChecked": count, "truncated": False}
            with patch("sp500_ml.spike_research._rpc", side_effect=[None, {"id": "id"}]), patch("sp500_ml.spike_research.fetch_company_news_range", side_effect=FinnhubNewsError("unavailable", "Unavailable")), patch("sp500_ml.spike_research.fetch_sec_evidence", return_value=fallback):
                self.assertEqual(research_spike("AAPL", EVENT["date"])["evidenceStatus"], expected)

    def test_focused_news_still_checks_primary_business_evidence(self):
        with patch("sp500_ml.spike_research._rpc", side_effect=[None, {"id": "id"}]), patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[article()]), patch("sp500_ml.spike_research.fetch_sec_evidence", return_value={"status": "empty", "message": "", "sources": [], "documentsChecked": 0, "truncated": False}) as sec:
            research_spike("AAPL", EVENT["date"])
            sec.assert_called_once()

    def test_retained_news_is_reused_when_provider_has_lost_coverage(self):
        archived = {"headlines": [{**article(), "retrievedAt": "2026-10-18T12:00:00Z"}], "documents": []}
        with patch("sp500_ml.spike_research._rpc", side_effect=[None, {"id": "id"}]), patch("sp500_ml.spike_research.load_evidence", return_value=archived), patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[]), patch("sp500_ml.spike_research.fetch_sec_evidence", return_value={"status": "empty", "message": "", "sources": [], "documentsChecked": 0, "truncated": False}) as sec:
            row = research_spike("AAPL", EVENT["date"])
        self.assertEqual(row["sources"][0]["retrievedAt"], "2026-10-18T12:00:00Z")
        self.assertEqual(row["documentsRetrieved"], 1)
        self.assertEqual(row["documentsMatched"], 1)
        self.assertTrue(any(item["provider"] == "Saved evidence" for item in row["coverage"]))
        sec.assert_called_once()

    def test_cache_avoids_provider_requests_and_failed_refresh_preserves_evidence(self):
        cached = {"_id": "id", "fetchedAt": pd.Timestamp.now(tz="UTC").isoformat(), "providerStatus": "ok", "sources": [article()]}
        with patch("sp500_ml.spike_research._rpc", return_value=cached), patch("sp500_ml.spike_research.fetch_company_news_range") as fetch:
            self.assertTrue(research_spike("AAPL", EVENT["date"])["cached"])
            fetch.assert_not_called()
        cached["fetchedAt"] = "2024-05-04T12:00:00Z"
        with patch("sp500_ml.spike_research._rpc", return_value=cached) as rpc, patch("sp500_ml.spike_research.fetch_company_news_range", side_effect=FinnhubNewsError("rate_limited", "Wait")):
            row = research_spike("AAPL", EVENT["date"])
            self.assertEqual(row["sources"], cached["sources"])
            self.assertEqual(row["providerStatus"], "rate_limited")
            self.assertEqual(rpc.call_count, 1)
        with patch("sp500_ml.spike_research._rpc", return_value=cached) as rpc, patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[]):
            self.assertEqual(research_spike("AAPL", EVENT["date"])["sources"], cached["sources"])
            self.assertEqual(rpc.call_count, 1)

    def test_cached_evidence_does_not_freeze_later_price_outcomes(self):
        sessions = _calendar(2024).sessions_window(EVENT["date"], 21)
        history = [{"date": day.date().isoformat(), "close": EVENT["close"]} for day in sessions]
        cached = {"_id": "saved", "fetchedAt": pd.Timestamp.now(tz="UTC").isoformat(), "sources": [],
                  "coverage": [{"status": "empty", "truncated": False}]}
        with patch("sp500_ml.spike_research.read_spikes", return_value={"ticker": "AAPL", "events": [EVENT], "history": history}), \
                patch("sp500_ml.spike_research._rpc", return_value=cached) as rpc, \
                patch("sp500_ml.spike_research.fetch_company_news_range") as fetch:
            self.assertAlmostEqual(research_spike("AAPL", EVENT["date"])["assessment"]["return20"], 0.)
            history[-1]["close"] = 99.
            self.assertAlmostEqual(research_spike("AAPL", EVENT["date"])["assessment"]["return20"], -.1)
            fetch.assert_not_called()
            self.assertTrue(all(call.args[0] == "query" for call in rpc.call_args_list))

    def test_http_200_with_convex_error_is_not_a_save(self):
        response = Mock(); response.json.return_value = {"status": "error", "errorMessage": "test-private-key"}
        with patch("sp500_ml.spike_research.configured", return_value=True), patch.dict(os.environ, {"CONVEX_URL": "https://test.convex.cloud", "PREDICTION_STORE_SECRET": "test-private-key"}), patch("sp500_ml.spike_research.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "no save was confirmed") as error:
                _rpc("mutation", "save", {})
            self.assertNotIn("test-private-key", str(error.exception))

    def test_view_reads_old_saved_evidence_without_external_collection_or_models(self):
        cached = {"_id": "saved", "fetchedAt": "2024-05-04T12:00:00Z", "sources": [], "coverage": []}
        with patch("sp500_ml.spike_research._rpc", return_value=cached) as rpc, patch("sp500_ml.spike_research.fetch_company_news_range") as news, patch("sp500_ml.spike_research.fetch_sec_evidence") as sec, patch("sp500_ml.risk_model.risk_score") as score:
            row = read_investigation("AAPL", EVENT["date"])
        self.assertTrue(row["saved"])
        self.assertEqual(row["assessment"]["status"], "insufficient_evidence")
        self.assertNotIn("riskScore", row)
        self.assertTrue(all(call.args[0] == "query" for call in rpc.call_args_list))
        news.assert_not_called(); sec.assert_not_called(); score.assert_not_called()

    def test_view_without_cache_uses_archive_and_records_pending_collection(self):
        with patch("sp500_ml.spike_research._rpc", side_effect=[None, {"id": "saved"}]), patch("sp500_ml.spike_research.fetch_company_news_range") as news, patch("sp500_ml.spike_research.fetch_sec_evidence") as sec, patch("sp500_ml.spike_research.fetch_company_announcements") as issuer:
            row = read_investigation("AAPL", EVENT["date"])
        self.assertTrue(row["saved"])
        self.assertEqual(row["providerStatus"], "not_collected")
        self.assertEqual(row["assessment"]["status"], "insufficient_evidence")
        news.assert_not_called(); sec.assert_not_called(); issuer.assert_not_called()

    def test_collector_does_not_treat_pending_view_as_completed_collection(self):
        cached = {"_id": "saved", "fetchedAt": pd.Timestamp.now(tz="UTC").isoformat(), "sources": [], "coverage": [{"status": "not_collected"}]}
        with patch("sp500_ml.spike_research._rpc", side_effect=[cached, {"id": "saved"}]), patch("sp500_ml.spike_research.fetch_company_news_range", return_value=[article()]) as news:
            row = research_spike("AAPL", EVENT["date"])
        news.assert_called_once()
        self.assertEqual(row["providerStatus"], "ok")

    def test_collection_cli_filters_recent_events_and_bounds_each_company(self):
        from sp500_ml.spike_research import main
        recent = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=1)).date().isoformat()
        row = {"ticker": "AAPL", "eventDate": recent, "evidenceStatus": "possible_explanation", "confidence": "limited",
               "providerStatus": "ok", "recordId": "saved", "saved": True, "coverage": []}
        events = [{**EVENT, "date": "2000-01-01"}, {**EVENT, "date": recent}, {**EVENT, "date": recent}]
        with patch("sys.argv", ["collect", "AAPL", "MSFT", "--recent-days", "30", "--limit", "1"]), \
                patch("sp500_ml.spike_research.read_spikes", return_value={"events": events}), \
                patch("sp500_ml.spike_research.research_spike", return_value=row) as collect, patch("builtins.print"):
            main()
        self.assertEqual([call.args for call in collect.call_args_list], [("AAPL", recent), ("MSFT", recent)])

    def test_collection_cli_isolates_company_failures_and_reports_nonzero_exit(self):
        from sp500_ml.spike_research import main
        row = {"ticker": "MSFT", "eventDate": EVENT["date"], "evidenceStatus": "no_documents_found", "confidence": "unknown",
               "providerStatus": "empty", "recordId": "saved", "saved": True, "coverage": []}
        with patch("sys.argv", ["collect", "AAPL", "MSFT", "--date", EVENT["date"]]), \
                patch("sp500_ml.spike_research.research_spike", side_effect=[RuntimeError("private provider message"), row]) as collect, \
                patch("builtins.print") as output, self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(collect.call_count, 2)
        self.assertNotIn("private provider message", repr(output.call_args_list))


class ResearchApiTests(unittest.TestCase):
    def test_asgi_auth_and_validated_inputs(self):
        self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/research", body={"ticker": "AAPL", "date": EVENT["date"]})), 401)
        with patch("vercel_app.authenticate", return_value=AuthenticatedUser("alice", "token")), patch("sp500_ml.spike_research.read_investigation", return_value={"saved": True}) as research:
            self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/research", body={"ticker": "AAPL", "date": EVENT["date"]})), 200)
            research.assert_called_once_with("AAPL", EVENT["date"])
            research.side_effect = ValueError("Not a spike")
            self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/spikes/research", body={"ticker": "AAPL", "date": "2024-01-01"})), 400)

    def test_local_server_checks_auth_and_returns_confirmed_research(self):
        import dashboard_server
        server = dashboard_server.DashboardServer(("127.0.0.1", 0), dashboard_server.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            client.request("POST", "/api/spikes/research", body="{}")
            response = client.getresponse(); self.assertEqual(response.status, 401); response.read()
            with patch.object(dashboard_server, "authenticate", return_value=AuthenticatedUser("alice", "token")), patch("sp500_ml.spike_research.read_investigation", return_value={"saved": True, "recordId": "id"}):
                client.request("POST", "/api/spikes/research", body=json.dumps({"ticker": "AAPL", "date": EVENT["date"]}))
                response = client.getresponse(); self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["recordId"], "id")
            client.close()
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
