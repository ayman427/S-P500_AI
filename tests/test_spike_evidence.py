from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sp500_ml.spike_evidence import collect_documents, load_evidence, save_evidence
from sp500_ml.primary_evidence import _source, sources_from_documents


class EvidenceArchiveTests(unittest.TestCase):
    def test_unmatched_documents_are_retained_versions_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch("sp500_ml.spike_evidence.NEWS_DIR", Path(directory)):
            with collect_documents() as documents:
                self.assertIsNone(_source(b"<p>Apple opened a new office.</p>", "Apple office", "https://www.apple.com/newsroom/2024/05/office/", "2024-05-02", "date", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z", "company_announcement"))
            self.assertEqual(len(documents), 1)
            headline = {"url": "https://example.com/report", "published_at_utc": "2024-05-02T21:00:00Z", "title": "Apple results", "excerpt": "First version"}
            save_evidence("AAPL", "2024-05-03", [headline], documents, "2026-10-19T12:00:00Z")
            save_evidence("AAPL", "2024-05-03", [headline, {**headline, "excerpt": "Revised version"}], documents, "2026-10-20T12:00:00Z")
            retained = load_evidence("AAPL", "2024-05-03")
            self.assertEqual(len(retained["headlines"]), 2)
            self.assertEqual(retained["headlines"][0]["retrievedAt"], "2026-10-19T12:00:00Z")
            self.assertEqual(len(retained["documents"]), 1)

    def test_saved_document_reclassification_preserves_its_observation_time(self):
        with collect_documents() as documents:
            _source(b"<p>Apple announced financial results.</p>", "Apple results", "https://www.apple.com/newsroom/2024/05/results/", "2024-05-02", "date", "2024-05-02", "2024-05-03", "2026-10-19T12:00:00Z", "company_announcement")
        source = sources_from_documents(documents, "2024-05-02", "2024-05-03")[0]
        self.assertEqual(source["retrievedAt"], "2026-10-19T12:00:00Z")
        self.assertEqual(source["publicationPrecision"], "date")


if __name__ == "__main__": unittest.main()
