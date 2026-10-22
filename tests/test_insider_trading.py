import os
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from sp500_ml.insider_trading import fetch_candidates, parse_filing, verify_filing, collect_insider_evidence, InsiderSourceError, _store

ACCESSION = "0000899243-15-001800"
BASE = "https://www.sec.gov/Archives/edgar/data/320193/000089924315001800/"
NOW = "2026-10-19T22:00:00+00:00"
INDEX = f'''<html><body>SEC Accession No. {ACCESSION}<div class="infoHead">Accepted</div><div class="info">2015-07-28 18:31:06</div>
<div>Filing Date</div><div>2015-07-28</div><table class="tableFile"><tr><td><a href="/xslF345X03/doc4.xml">Form</a></td><td>4</td></tr></table></body></html>'''.encode()
XML = b'''<ownershipDocument><documentType>4</documentType><issuer><issuerCik>0000320193</issuerCik></issuer>
<reportingOwner><reportingOwnerId><rptOwnerCik>12345</rptOwnerCik><rptOwnerName>Example Buyer</rptOwnerName></reportingOwnerId><reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>
<nonDerivativeTable><nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle><transactionDate><value>2015-07-27</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding><transactionAmounts><transactionShares><value>100</value></transactionShares><transactionPricePerShare><value>25</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>'''
CANDIDATE = {"ticker": "AAPL", "cik": "320193", "accession_number": ACCESSION, "transaction_code": "P"}


def records():
    return parse_filing(INDEX, XML, "AAPL", "320193", ACCESSION, BASE + ACCESSION + "-index.html", BASE + "doc4.xml", NOW)


class InsiderParsingTests(unittest.TestCase):
    def test_sec_acceptance_time_issuer_and_purchase_values(self):
        row = records()[0]
        self.assertEqual(row["acceptedAt"], "2015-07-28T22:31:06+00:00")
        self.assertEqual(row["transactionDate"], "2015-07-27")
        self.assertEqual(row["value"], 2500.)
        self.assertEqual(row["insiderIds"], ["12345"])
        self.assertEqual(row["position"], "Director")
        self.assertFalse(row["usableForTraining"])
        self.assertEqual(row["currency"], "unverified")
        self.assertEqual(len(row["filingFingerprint"]), 64)

    def test_awards_derivatives_and_invalid_identity_are_not_purchases(self):
        for xml in (XML.replace(b"<transactionCode>P", b"<transactionCode>A"),
                    XML.replace(b"nonDerivative", b"derivative"),
                    XML.replace(b"<transactionShares><value>100", b"<transactionShares><value>-1")):
            self.assertEqual(parse_filing(INDEX, xml, "AAPL", "320193", ACCESSION, "index", "xml", NOW), [])
        for index, xml in ((INDEX.replace(b"Accepted", b"Unknown"), XML), (INDEX, XML.replace(b"0000320193", b"0000000001")), (INDEX, XML.replace(b"<documentType>4", b"<documentType>4/A"))):
            with self.assertRaises(ValueError):
                parse_filing(index, xml, "AAPL", "320193", ACCESSION, "index", "xml", NOW)

    def test_provider_url_is_ignored_and_requests_stay_on_sec(self):
        candidate = {**CANDIDATE, "sec_filing_url": "http://127.0.0.1/private"}
        with patch("sp500_ml.insider_trading._download", side_effect=[INDEX, XML]) as download:
            row = verify_filing(candidate, "AAPL", NOW, "private-agent")[0]
        self.assertEqual(row["sourceUrl"], BASE + "doc4.xml")
        self.assertEqual(download.call_args_list[0].args[0], BASE + ACCESSION + "-index.html")
        self.assertEqual(download.call_args_list[1].args[0], BASE + "doc4.xml")

    def test_malformed_purchase_row_does_not_discard_valid_rows(self):
        first = XML.split(b"<nonDerivativeTransaction>")[1].split(b"</nonDerivativeTransaction>")[0]
        malformed = b"<nonDerivativeTransaction>" + first.replace(b"<transactionShares><value>100", b"<transactionShares><value>invalid") + b"</nonDerivativeTransaction>"
        mixed = XML.replace(b"</nonDerivativeTable>", malformed + b"</nonDerivativeTable>")
        rows = parse_filing(INDEX, mixed, "AAPL", "320193", ACCESSION, "index", "xml", NOW)
        self.assertEqual(len(rows), 1)


class InsiderProviderTests(unittest.TestCase):
    def test_header_filters_and_no_premium_pagination(self):
        response = Mock(status_code=200, is_redirect=False)
        response.json.return_value = [CANDIDATE]
        with patch("sp500_ml.insider_trading.requests.get", return_value=response) as get:
            self.assertEqual(fetch_candidates("AAPL", "2015-07-01", "2015-07-29", "private-key"), [CANDIDATE])
        self.assertEqual(get.call_args.kwargs["headers"]["X-Api-Key"], "private-key")
        self.assertNotIn("limit", get.call_args.kwargs["params"])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_provider_error_is_not_empty_coverage_or_a_key_leak(self):
        for status, payload in ((403, {}), (429, {}), (200, {"error": "private-key"})):
            response = Mock(status_code=status, is_redirect=False)
            response.json.return_value = payload
            with patch("sp500_ml.insider_trading.requests.get", return_value=response):
                with self.assertRaises(InsiderSourceError) as caught:
                    fetch_candidates("AAPL", "2015-07-01", "2015-07-29", "private-key")
                self.assertNotIn("private-key", str(caught.exception))

    def test_database_http_200_error_is_not_confirmed(self):
        response = Mock()
        response.json.return_value = {"status": "error", "errorMessage": "private-key"}
        with patch("sp500_ml.insider_trading.load_environment"), patch("sp500_ml.insider_trading.configured", return_value=True), patch.dict(os.environ, {"CONVEX_URL": "https://test.convex.cloud", "PREDICTION_STORE_SECRET": "private-key"}), patch("sp500_ml.insider_trading.requests.post", return_value=response):
            with self.assertRaises(RuntimeError) as caught:
                _store("mutation", "save", {})
            self.assertNotIn("private-key", str(caught.exception))


class InsiderCollectionTests(unittest.TestCase):
    def setUp(self):
        for patcher in (patch("sp500_ml.insider_trading.load_environment"), patch("sp500_ml.insider_trading._get_sec_user_agent", return_value="private-agent"), patch.dict(os.environ, {"API_NINJAS_API_KEY": "private-key"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_after_close_disclosure_is_excluded_until_next_session(self):
        for as_of, count in (("2015-07-28", 0), ("2015-07-29", 1)):
            with patch("sp500_ml.insider_trading._store", side_effect=[{"rows": [], "truncated": False}, {"ids": ["id"]}]) as store, patch("sp500_ml.insider_trading.fetch_candidates", return_value=[CANDIDATE]), patch("sp500_ml.insider_trading.verify_filing", return_value=records()):
                result = collect_insider_evidence("AAPL", as_of, NOW)
                self.assertEqual(result["transactions"], count)
                self.assertEqual(result["coverage"], "partial")
                self.assertEqual(store.call_count, 2)  # Retain after-close disclosures, but hide them before cutoff.

    def test_capped_discovery_and_empty_results_never_claim_no_buying(self):
        with patch("sp500_ml.insider_trading._store", return_value={"rows": [], "truncated": False}), patch("sp500_ml.insider_trading.fetch_candidates", return_value=[]):
            result = collect_insider_evidence("AAPL", "2015-07-29", NOW)
            self.assertEqual(result["status"], "coverage_incomplete")
        with patch("sp500_ml.insider_trading._store", side_effect=[{"rows": [], "truncated": False}, {"ids": ["id"]}]), patch("sp500_ml.insider_trading.fetch_candidates", return_value=[CANDIDATE]*10), patch("sp500_ml.insider_trading.verify_filing", return_value=records()) as verify:
            result = collect_insider_evidence("AAPL", "2015-07-29", NOW)
            self.assertTrue(result["truncated"])
            self.assertEqual(result["buyers"], 1)
            self.assertEqual(result["transactions"], 1)
            verify.assert_called_once()

    def test_provider_failure_keeps_retained_evidence_but_save_failure_raises(self):
        with patch("sp500_ml.insider_trading._store", return_value={"rows": records(), "truncated": False}), patch("sp500_ml.insider_trading.fetch_candidates", side_effect=InsiderSourceError("Provider unavailable")):
            result = collect_insider_evidence("AAPL", "2015-07-29", NOW)
            self.assertEqual(result["status"], "buying_observed")
            self.assertEqual(result["providerStatus"], "unavailable")
        with patch("sp500_ml.insider_trading._store", side_effect=[{"rows": [], "truncated": False}, {"ids": []}]), patch("sp500_ml.insider_trading.fetch_candidates", return_value=[CANDIDATE]), patch("sp500_ml.insider_trading.verify_filing", return_value=records()):
            with self.assertRaises(RuntimeError):
                collect_insider_evidence("AAPL", "2015-07-29", NOW)
