import unittest

from sp500_ml.rise_risk import assess_rise
from sp500_ml.sessions import _calendar


EVENT = {"date": "2024-05-03", "close": 100., "previousClose": 80., "return": .25}


def source(categories, **extra):
    return {"categories": categories, "basis": "provider_excerpt", "contextPhase": "move_window",
            "timing": "during_move", "reportedLink": True, "title": "", "summary": "", **extra}


class RiseRiskTests(unittest.TestCase):
    def assess(self, sources=(), history=(), coverage=()):
        return assess_rise(EVENT, history, sources, coverage)

    def test_public_statement_is_reported_context_not_proof_or_missing_business_evidence(self):
        row = self.assess([source(["Political / public statements"])])
        self.assertEqual(row["status"], "narrative_linked")
        self.assertEqual(row["businessSources"], [])
        self.assertIn("unverified", row["label"])
        self.assertFalse(row["usableForTraining"])
        self.assertEqual(self.assess()["status"], "insufficient_evidence")

    def test_primary_positive_developments_and_mixed_evidence_are_distinct(self):
        primary = source(["Earnings / guidance"], basis="sec_filing", summary="The company raised its guidance.")
        self.assertEqual(self.assess([primary])["status"], "documented_development")
        row = self.assess([source(["Speculative activity"]), primary])
        self.assertEqual(row["status"], "mixed_evidence")
        self.assertEqual(row["businessSources"], [1])
        self.assertEqual(self.assess([source(["Earnings / guidance"])])["status"], "reported_business")

    def test_generic_negated_rumored_and_late_documents_do_not_verify_positive_support(self):
        for summary, extra in [("Quarterly financial results.", {}), ("The company did not raise guidance.", {}),
                               ("The company could announce a buyback.", {}), ("The company authorized no repurchase.", {}),
                               ("The company raised guidance.", {"timing": "after_close"}),
                               ("The company raised guidance.", {"contextPhase": "earlier_context"}),
                               ("The company raised guidance.", {"timing": "unknown_time"})]:
            row = self.assess([source(["Earnings / guidance"], basis="sec_filing", reportedLink=False, summary=summary, **extra)])
            self.assertEqual(row["businessSources"], [])

    def test_late_and_headline_only_narratives_do_not_establish_a_link(self):
        for extra in ({"timing": "after_close"}, {"contextPhase": "later_reporting"}, {"basis": "headline_only"}, {"reportedLink": False}):
            self.assertEqual(self.assess([source(["Political / public statements"], **extra)])["narrativeSources"], [])

    def test_previous_day_date_only_primary_release_is_valid_context_but_same_day_is_unresolved(self):
        primary = source(["Buybacks / dividends"], basis="company_announcement", reportedLink=False,
                         summary="Apple announced a share repurchase program.", timing="unknown_time",
                         publicationPrecision="date", publishedAt="2024-05-02")
        self.assertEqual(self.assess([primary])["businessSources"], [0])
        self.assertEqual(self.assess([{**primary, "publishedAt": "2024-05-03"}])["businessSources"], [])
        self.assertEqual(self.assess([{**primary, "contextPhase": "earlier_context"}])["businessSources"], [])

    def test_outcomes_follow_exchange_sessions_and_stop_at_twenty(self):
        sessions = _calendar(2024).sessions_window(EVENT["date"], 22)
        history = [{"date": day.date().isoformat(), "close": 120. if i == 1 else 90. if i else 100.}
                   for i, day in enumerate(sessions)]
        history[-1]["close"] = 1.
        row = self.assess(history=history)
        self.assertEqual(row["observedSessions"], 20)
        self.assertAlmostEqual(row["return5"], -.1)
        self.assertAlmostEqual(row["return20"], -.1)
        self.assertAlmostEqual(row["maxDrawdown"], -.25)
        self.assertAlmostEqual(row["gainReversed"], .5)
        self.assertEqual(row["peakDate"], history[1]["date"])
        self.assertEqual(row["outcomeThrough"], history[20]["date"])

    def test_missing_sessions_do_not_shift_outcomes_and_unavailable_sources_stay_unknown(self):
        sessions = _calendar(2024).sessions_window(EVENT["date"], 21)
        history = [{"date": day.date().isoformat(), "close": 110.} for i, day in enumerate(sessions) if i != 4]
        row = self.assess(history=history, coverage=[{"status": "unavailable"}])
        self.assertEqual(row["observedSessions"], 3)
        self.assertIsNone(row["return5"])
        self.assertIsNone(row["return20"])
        self.assertEqual(row["status"], "insufficient_evidence")
        self.assertIn("incomplete or unavailable", " ".join(row["gaps"]))

    def test_assessment_explains_all_five_evidence_states_without_fake_certainty(self):
        primary = source(["Earnings / guidance"], basis="sec_filing", summary="The company raised guidance.")
        business = source(["Earnings / guidance"])
        narrative = source(["Political / public statements"])
        for sources, expected in [([], "insufficient_evidence"), ([narrative], "narrative_linked"),
                                  ([primary], "documented_development"), ([business], "reported_business"),
                                  ([narrative, primary], "mixed_evidence")]:
            with self.subTest(status=expected):
                result = self.assess(sources)
                self.assertEqual(result["status"], expected)
                self.assertGreater(len(result["explanation"]), 20)
                self.assertLess(len(result["explanation"]), 180)
                self.assertNotIn("probability", result)
                self.assertNotEqual(result["evidenceStrength"], "supported")
        self.assertEqual(self.assess([primary, business])["evidenceStrength"], "supported")
        self.assertEqual(self.assess([primary, business], coverage=[{"status": "unavailable"}])["evidenceStrength"], "limited")

