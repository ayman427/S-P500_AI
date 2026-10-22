"""Descriptive rise investigation; source gaps are never evidence of manipulation."""
import re
import math

import pandas as pd

from .sessions import _calendar

BUSINESS = {"Earnings / guidance", "Buybacks / dividends", "Acquisition / deal", "Products / approvals / contracts"}
POSITIVE_BUSINESS = re.compile(
    r"\b(?:beat|exceed)\w*.{0,35}(?:estimates|expectations)|"
    r"\b(?:rais|increas)\w*.{0,25}(?:guidance|outlook|dividend)|"
    r"\b(?:revenue|sales|profit|earnings)\b.{0,30}\b(?:grew|rose|increased|higher)\b|"
    r"\b(?:record|higher)\s+(?:revenue|sales|profit)|"
    r"\b(?:authoriz|announc)\w*.{0,60}(?:repurchase|buyback)|"
    r"\b(?:wins?|won|awarded)\b.{0,30}\bcontract|\b(?:FDA|regulatory) approv\w*|"
    r"\b(?:agree|complet)\w*.{0,45}(?:merger|acquisition)", re.I)


def positive_development(text):
    return bool(POSITIVE_BUSINESS.search(text) and not re.search(
        r"\b(?:no|not|denied|unconfirmed|rumou?red|might|could|considering|proposed)\b", text, re.I))


def price_outcome(event, closes):
    sessions = _calendar(pd.Timestamp(event["date"]).year).sessions_window(event["date"], 21)
    observed = []
    for session in sessions[1:]:
        day = session.date().isoformat()
        if day not in closes:
            break
        observed.append((day, closes[day]))
    peak, worst, lowest, peak_date = event["close"], 0., event["close"], event["date"]
    for day, close in observed:
        if close > peak:
            peak, peak_date = close, day
        worst = min(worst, close / peak - 1)
        lowest = min(lowest, close)
    return observed, worst, lowest / event["close"] - 1, peak_date


def assess_rise(event, history, sources, coverage):
    timely = [(i, s) for i, s in enumerate(sources)
              if s.get("contextPhase") == "move_window" and s.get("timing") in {"before_move", "during_move"}]
    narratives = [i for i, s in timely if s.get("reportedLink") and s.get("basis") == "provider_excerpt"
                  and {"Political / public statements", "Speculative activity"}.intersection(s.get("categories", []))]
    # A date-only release on the previous day predates this close, even without an exact clock time.
    primary_timely = [(i, s) for i, s in enumerate(sources) if s.get("contextPhase") == "move_window"
                      and (s.get("timing") in {"before_move", "during_move"}
                           or (s.get("timing") == "unknown_time" and s.get("publicationPrecision") == "date"
                               and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.get("publishedAt", ""))
                               and s["publishedAt"] < event["date"]))]
    documented = [i for i, s in primary_timely if s.get("basis") in {"sec_filing", "company_announcement"}
                  and BUSINESS.intersection(s.get("categories", []))
                  and positive_development(s.get("title", "") + " " + s.get("summary", ""))]
    reported = [i for i, s in timely if s.get("reportedLink") and BUSINESS.intersection(s.get("categories", []))]
    status = "mixed_evidence" if narratives and documented else "narrative_linked" if narratives else "documented_development" if documented else "reported_business" if reported else "insufficient_evidence"
    labels = {"mixed_evidence": "Narrative and business developments documented",
              "narrative_linked": "Narrative-linked rise; business support unverified",
              "documented_development": "Business development documented",
              "reported_business": "Business catalyst reported; verification incomplete",
              "insufficient_evidence": "Insufficient evidence"}
    closes = {row["date"]: float(row["close"]) for row in history if pd.notna(row.get("close")) and math.isfinite(float(row["close"])) and float(row["close"]) > 0}
    observed, worst, _, peak_date = price_outcome(event, closes)
    last_day, last_close = observed[-1] if observed else (event["date"], event["close"])
    gain = event["close"] - event["previousClose"]
    gaps = []
    if not documented:
        gaps.append("No timely positive business development verified in the checked primary sources.")
    if not narratives:
        gaps.append("No dated report linking this rise to a public statement or speculative activity was verified.")
    if any(c.get("status") not in {"ok", "empty", "not_needed", "not_listed"} or c.get("truncated") for c in coverage):
        gaps.append("Some evidence sources are incomplete or unavailable.")
    if len(observed) < 20:
        gaps.append("The 20-session price window is incomplete; missing sessions are not skipped.")
    incomplete = any(c.get("status") not in {"ok", "empty", "not_needed", "not_listed"} or c.get("truncated") for c in coverage)
    explanations = {
        "mixed_evidence": "Dated reporting links the rise to a statement or speculation. A positive business development is also documented.",
        "narrative_linked": "Dated reporting links the rise to a public statement or speculation. Positive business support remains unverified.",
        "documented_development": "A timely primary source documents a positive business development near this rise.",
        "reported_business": "A business catalyst is reported, but timely primary verification is incomplete.",
        "insufficient_evidence": "No timely explanation was verified in the available evidence."}
    corroborated = any(sources[i].get("basis") == "provider_excerpt" for i in reported)
    strength = "supported" if documented and corroborated and not incomplete else "limited" if narratives or documented or reported else "insufficient"
    return {"method": "rise_investigation_v2", "status": status, "label": labels[status],
            "explanation": explanations[status], "evidenceStrength": strength,
            "confidenceReason": "Timely primary business evidence and reporting are available." if strength == "supported" else
                "Relevant evidence exists, but verification or source coverage is incomplete." if strength == "limited" else
                "No timely source link was verified; missing coverage is not negative business evidence.",
            "narrativeSources": narratives, "businessSources": documented, "reportedBusinessSources": reported,
            "gaps": gaps, "eventReturn": event["return"], "outcomeThrough": last_day,
            "observedSessions": len(observed), "peakDate": peak_date,
            "return5": observed[4][1] / event["close"] - 1 if len(observed) >= 5 else None,
            "return20": observed[19][1] / event["close"] - 1 if len(observed) >= 20 else None,
            "maxDrawdown": worst if observed else None,
            "gainReversed": max(0., (event["close"] - last_close) / gain) if observed and gain > 0 else None,
            "usableForTraining": False}
