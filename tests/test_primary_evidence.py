import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sp500_ml.primary_evidence import _allowed_url, _download, PrimaryEvidenceError, fetch_sec_evidence, fetch_company_announcements


ROOT = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/"
RELEASE = b"<html><h1>Apple reports financial results</h1><article><p>Apple reports financial results for its second quarter.</p><p>The Board authorized a program to repurchase up to $110 billion of common stock.</p><h2>Forward-looking statements</h2><p>Potential future acquisitions may occur.</p></article></html>"


class SecEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        (path / "company_tickers.json").write_text(json.dumps({"0": {"ticker": "AAPL", "title": "Apple Inc.", "cik_str": 320193}}))
        for patcher in [patch("sp500_ml.primary_evidence.FUNDAMENTAL_DIR", path), patch("sp500_ml.primary_evidence._get_sec_user_agent", return_value="Research contact@example.com")]:
            patcher.start(); self.addCleanup(patcher.stop)

    def test_older_history_and_linked_announcement_keep_sec_availability_time(self):
        columns = {"filingDate": ["2024-05-02"], "form": ["8-K"], "accessionNumber": ["0000320193-24-000001"], "primaryDocument": ["aapl-20240502.htm"], "acceptanceDateTime": ["2024-05-02T20:30:00Z"]}
        submissions = {"filings": {"recent": {}, "files": [{"name": "CIK0000320193-submissions-001.json", "filingFrom": "2023-01-01", "filingTo": "2024-12-31"}]}}
        primary = b'<html><p>Apple furnished a company announcement.</p><a href="ex991.htm">Exhibit 99.1</a><a href="https://127.0.0.1/private">99.1</a><a href="https://www.sec.gov/Archives/edgar/data/999/000032019324000001/ex991.htm">99.1</a></html>'
        def download(url, headers):
            if url.endswith("CIK0000320193.json"): return json.dumps(submissions).encode()
            if url.endswith("submissions-001.json"): return json.dumps(columns).encode()
            if url == ROOT + "aapl-20240502.htm": return primary
            if url == ROOT + "ex991.htm": return RELEASE
            raise AssertionError("Unexpected URL: " + url)
        with patch("sp500_ml.primary_evidence._download", side_effect=download):
            result = fetch_sec_evidence("AAPL", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["documentsChecked"], 2)
        self.assertEqual(result["sources"][0]["categories"], ["Earnings / guidance", "Buybacks / dividends"])
        self.assertEqual(result["sources"][0]["timing"], "during_move")
        self.assertEqual(result["sources"][0]["timestampBasis"], "sec_acceptance")
        self.assertIn("repurchase", result["sources"][0]["summary"])
        self.assertNotIn("Acquisition / deal", result["sources"][0]["categories"])

    def test_missing_acceptance_time_is_not_fabricated(self):
        metadata = {"filings": {"recent": {"filingDate": ["2024-05-02"], "form": ["8-K"], "accessionNumber": ["0000320193-24-000001"], "primaryDocument": ["aapl.htm"]}}}
        with patch("sp500_ml.primary_evidence._download", side_effect=[json.dumps(metadata).encode(), RELEASE]):
            result = fetch_sec_evidence("AAPL", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z")
        self.assertEqual(result["sources"][0]["publishedAt"], "2024-05-02")
        self.assertEqual(result["sources"][0]["timing"], "unknown_time")

    def test_config_and_network_failures_are_not_empty_success(self):
        with patch("sp500_ml.primary_evidence._get_sec_user_agent", return_value=""), patch("sp500_ml.primary_evidence._download") as get:
            self.assertEqual(fetch_sec_evidence("AAPL", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z")["status"], "not_configured")
            get.assert_not_called()
        with patch("sp500_ml.primary_evidence._download", side_effect=PrimaryEvidenceError("unavailable", "Source unreachable")):
            self.assertEqual(fetch_sec_evidence("AAPL", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z")["status"], "unavailable")


class IssuerArchiveTests(unittest.TestCase):
    def test_nvidia_archive_uses_dates_and_retains_later_reporting_as_context(self):
        archive = b'<div class="index-item-text"><h3><a href="/news/nvidia-results">NVIDIA financial results</a></h3><span class="index-item-text-info-date">February 21, 2024</span></div>'
        release = b'<div class="article-body"><p>NVIDIA announced financial results.</p></div><aside>Completed an acquisition.</aside>'
        with patch("sp500_ml.primary_evidence._download", side_effect=[archive, release, b""]):
            result = fetch_company_announcements("NVDA", "2024-02-21", "2024-02-22", "2026-10-19T12:00:00Z")
        self.assertEqual(result["sources"][0]["categories"], ["Earnings / guidance"])
        self.assertEqual(result["documentsChecked"], 1)
        self.assertEqual(result["status"], "ok")

    def test_microsoft_uses_release_dateline_instead_of_fiscal_period_end(self):
        release = b'<h1>Earnings</h1><p>REDMOND, Wash. - April 25, 2023 - Microsoft Corp. today announced the following results for the quarter ended March 31, 2023.</p>'
        with patch("sp500_ml.primary_evidence._download", return_value=release):
            result = fetch_company_announcements("MSFT", "2023-04-25", "2023-04-26", "2026-10-19T12:00:00Z")
        self.assertEqual(result["sources"][0]["publishedAt"], "2023-04-25")
        self.assertEqual(result["sources"][0]["timing"], "unknown_time")

    def test_nvidia_stops_requesting_pages_once_document_cap_is_reached(self):
        archive = ('<div class="index-item-text"><h3><a href="/news/release-{0}">NVIDIA results</a></h3><span class="index-item-text-info-date">February 21, 2024</span></div>')
        listing = ''.join(archive.format(i) for i in range(5)).encode()
        release = b'<p>NVIDIA announced financial results.</p>'
        with patch("sp500_ml.primary_evidence._download", side_effect=[listing, release, release, release, release]) as download:
            result = fetch_company_announcements("NVDA", "2024-02-21", "2024-02-22", "2026-10-19T12:00:00Z")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["truncated"])
        self.assertEqual(download.call_count, 5)

    def test_month_pagination_finds_a_release_without_current_news_substitution(self):
        archive = b'<span class="pagination-total">2</span><a href="/newsroom/2024/05/unrelated/">Apple opens a store May 20, 2024</a>'
        second = b'<span class="pagination-total">2</span><a href="/newsroom/2024/05/apple-reports-second-quarter-results/">PRESS RELEASE Apple reports second quarter results May 2, 2024</a>'
        with patch("sp500_ml.primary_evidence._download", side_effect=[b"<html>No April releases</html>", archive, second, RELEASE]) as download:
            result = fetch_company_announcements("AAPL", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["documentsChecked"], 1)
        self.assertIn("?page=2", download.call_args_list[2].args[0])
        self.assertEqual(result["sources"][0]["timing"], "unknown_time")
        self.assertEqual(result["sources"][0]["basis"], "company_announcement")

    def test_other_issuers_are_not_given_apple_results(self):
        with patch("sp500_ml.primary_evidence._download") as download:
            self.assertEqual(fetch_company_announcements("ZZZZ", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z")["status"], "not_supported")
            download.assert_not_called()

    def test_pagination_cap_is_recorded(self):
        archive = b'<span class="pagination-total">9</span>'
        with patch("sp500_ml.primary_evidence._download", return_value=archive) as download:
            result = fetch_company_announcements("AAPL", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z")
        self.assertTrue(result["truncated"])
        self.assertEqual(download.call_count, 8)


class DownloadBoundaryTests(unittest.TestCase):
    def test_private_external_encoded_traversal_and_credential_urls_are_rejected(self):
        for url in ["https://127.0.0.1/private", "https://www.sec.gov.attacker.com/file", "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/%2e%2e/ex.htm", "https://user:pass@www.apple.com/newsroom/2024/05/test/"]:
            self.assertFalse(_allowed_url(url))
            with patch("sp500_ml.primary_evidence.requests.get") as get:
                with self.assertRaises(PrimaryEvidenceError): _download(url, {})
                get.assert_not_called()

    def test_redirects_are_not_followed_and_body_size_is_bounded(self):
        response = Mock(status_code=302, is_redirect=True)
        for redirect in (True, False):
            response.is_redirect = redirect
            response.iter_content.return_value = [b"x" * 8_000_001]
            with patch("sp500_ml.primary_evidence.requests.get") as get:
                get.return_value.__enter__.return_value = response
                with self.assertRaises(PrimaryEvidenceError): _download(ROOT + "ex991.htm", {})
                self.assertFalse(get.call_args.kwargs["allow_redirects"])


if __name__ == "__main__":
    unittest.main()
