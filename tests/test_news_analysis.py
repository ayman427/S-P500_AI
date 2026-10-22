import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sp500_ml.news_analysis import rank_news, summarize_article, read_article, analyze_event_news


def report(title="Shares plunge after earnings miss", excerpt="The company cut its outlook after weak demand. Shares fell sharply after the earnings release."):
    return {"title": title, "url": "https://finance.yahoo.com/news/report.html", "excerpt": excerpt,
            "publisher": "Publisher", "published_at_utc": "2020-01-03T15:00:00Z",
            "publicationDate": "2020-01-03", "sameDay": True, "daysFromEvent": 0}


class NewsAnalysisTests(unittest.TestCase):
    def setUp(self):
        aliases = patch("sp500_ml.news_analysis._company_aliases", return_value=["AAPL", "company"])
        aliases.start()
        self.addCleanup(aliases.stop)

    def test_other_companies_are_excluded_even_when_provider_tags_the_selected_stock(self):
        rows = rank_news([report("GE shares plunge after earnings miss", "GE shares fell after weak demand."),
                          report("Apple shares fall after weak demand", "Apple cut its outlook after weak demand.")], ["AAPL", "Apple"])
        selected = [row for row in rows if row["relevant"]]
        self.assertEqual(len(selected), 1)
        self.assertIn("Apple", selected[0]["title"])
    def test_catalyst_reporting_beats_investment_opinion_and_duplicates(self):
        rows = rank_news([report("Best stocks to buy after the earnings miss"), report(), report(), report("Company opens a new store", "A new location opened today.")])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["title"], "Shares plunge after earnings miss")
        self.assertTrue(rows[0]["relevant"])
        self.assertIn("Earnings / guidance", rows[0]["topics"])
        self.assertFalse(rows[-1]["relevant"])

    def test_public_article_sentences_are_read_and_summary_does_not_invent_claims(self):
        body = "The company cut its outlook after weak demand. Shares fell sharply after the earnings release. Investors watched trading throughout the session."
        row = rank_news([report()])[0]
        with patch("sp500_ml.news_analysis.read_article", return_value=body):
            result = summarize_article(row)
        self.assertEqual(result["basis"], "article")
        self.assertIn("cut its outlook", result["summary"])
        self.assertIn(result["summary"], body)

    def test_inaccessible_article_uses_labeled_excerpt_or_no_summary(self):
        with patch("sp500_ml.news_analysis.read_article", return_value=""):
            result = summarize_article(rank_news([report()])[0])
            self.assertEqual(result["basis"], "provider_excerpt")
            self.assertTrue(result["summary"])
            result = summarize_article(rank_news([report(excerpt="")])[0])
            self.assertEqual(result["basis"], "headline_only")
            self.assertEqual(result["summary"], "")

    def test_reader_rejects_private_hosts_and_redirects_without_fetching_them(self):
        with patch("sp500_ml.news_analysis.requests.get") as get:
            self.assertEqual(read_article("https://127.0.0.1/private"), "")
            self.assertEqual(read_article("https://finance.yahoo.com.attacker.example/news"), "")
            get.assert_not_called()
            response = Mock(is_redirect=True, headers={"Location": "https://127.0.0.1/private"})
            get.return_value.__enter__ = Mock(return_value=response)
            get.return_value.__exit__ = Mock(return_value=False)
            self.assertEqual(read_article("https://finance.yahoo.com/news/report"), "")
            self.assertEqual(get.call_count, 1)

    def test_reader_extracts_structured_article_body_not_navigation(self):
        body = "The company cut its earnings outlook after slower sales. " * 5
        html = '<html><nav>Subscribe now</nav><script type="application/ld+json">' + json.dumps({"@type": "NewsArticle", "articleBody": body}) + '</script></html>'
        response = Mock(is_redirect=False, status_code=200, headers={"Content-Type": "text/html"})
        response.iter_content.return_value = [html.encode()]
        with patch("sp500_ml.news_analysis.requests.get") as get:
            get.return_value.__enter__ = Mock(return_value=response)
            get.return_value.__exit__ = Mock(return_value=False)
            self.assertIn("earnings outlook", read_article("https://finance.yahoo.com/news/report"))

    def test_analysis_cache_avoids_reading_articles_twice_and_preserves_selection(self):
        news = {"headlines": [report()], "providerMessage": ""}
        with tempfile.TemporaryDirectory() as directory, patch("sp500_ml.news_analysis.NEWS_DIR", Path(directory)), patch("sp500_ml.risk_news.read_event_news", return_value=news), patch("sp500_ml.news_analysis.read_article", return_value="") as read:
            first = analyze_event_news("AAPL", "2020-01-03")
            second = analyze_event_news("AAPL", "2020-01-03")
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(read.call_count, 1)
            self.assertEqual(second["articles"][0]["basis"], "provider_excerpt")
            self.assertEqual(second["eventDate"], "2020-01-03")

    def test_no_relevant_news_does_not_create_a_crash_explanation(self):
        news = {"headlines": [report("Company opens a store", "A new store opened today.")], "providerMessage": ""}
        with tempfile.TemporaryDirectory() as directory, patch("sp500_ml.news_analysis.NEWS_DIR", Path(directory)), patch("sp500_ml.risk_news.read_event_news", return_value=news), patch("sp500_ml.news_analysis.read_article") as read:
            result = analyze_event_news("AAPL", "2020-01-03")
            self.assertEqual(result["articles"], [])
            self.assertIn("No strong matches", result["overview"])
            read.assert_not_called()
