"""Offline fixed-rule audit; writes reports to Convex, never changes production models."""
import argparse
import hashlib
import inspect
import json
import math
from uuid import uuid4

import numpy as np
import pandas as pd

from .paths import BACKTEST_DIR
from .spike_anomalies import METHOD, analyze_event, read_anomalies
from .spike_rules import spike_signals

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "PG", "WMT", "CAT"]
VARIANTS = ["price_only", "volume_only", "market_only", "acceleration_only", "multi_signal"]


def wilson(hits, total):
    if not total:
        return None
    p, z = hits / total, 1.959963984540054
    centre = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., centre-half), min(1., centre+half)]


def metrics(rows, variant):
    tp = sum(r[variant] and r["reversal"] for r in rows)
    fp = sum(r[variant] and not r["reversal"] for r in rows)
    fn = sum(not r[variant] and r["reversal"] for r in rows)
    tn = len(rows) - tp - fp - fn
    return {"events": len(rows), "flags": tp+fp, "truePositives": tp, "falseAlerts": fp, "missedReversals": fn,
            "trueNegatives": tn, "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None, "falsePositiveRate": fp/(fp+tn) if fp+tn else None,
            "precisionWilson95": wilson(tp, tp+fp)}


def block_lift(rows, repeats=2000):
    quarters = sorted({str(pd.Period(r["date"], freq="Q")) for r in rows})
    if len(quarters) < 2:
        return {"interval95": None, "quarters": len(quarters), "validResamples": 0}
    blocks = []
    for quarter in quarters:
        group = [r for r in rows if str(pd.Period(r["date"], freq="Q")) == quarter]
        blocks.append([sum(r["multi_signal"] and r["reversal"] for r in group), sum(r["multi_signal"] for r in group), sum(r["reversal"] for r in group), len(group)])
    sampled = np.random.default_rng(42).integers(0, len(blocks), size=(repeats, len(blocks)))
    totals = np.asarray(blocks)[sampled].sum(axis=1)
    totals = totals[totals[:, 1] > 0]
    lift = totals[:, 0] / totals[:, 1] - totals[:, 2] / totals[:, 3]
    return {"interval95": np.quantile(lift, [.025, .975]).tolist() if len(lift) else None,
            "quarters": len(quarters), "validResamples": len(lift)}


def select_rows(ticker, events, holdout):
    rows, excluded = [], {"incompleteOutcome": 0, "missingSignals": 0, "notConfirmedPriceAnomaly": 0, "crossesSplit": 0, "overlap": 0}
    next_allowed = "0000-00-00"
    for e in sorted(events, key=lambda e: e["date"]):
        a, outcome = e["analysis"], e["analysis"]["historicalOutcome"]
        signals = {s["id"]: s for s in a["signals"]}
        if outcome["sessions"] != 20:
            excluded["incompleteOutcome"] += 1
            continue
        if any(s["status"] == "unavailable" for s in signals.values()):
            excluded["missingSignals"] += 1
            continue
        if signals["price"]["status"] != "elevated":
            excluded["notConfirmedPriceAnomaly"] += 1
            continue
        if e["date"] < holdout <= outcome["through"]:
            excluded["crossesSplit"] += 1
            continue
        if e["date"] <= next_allowed:
            excluded["overlap"] += 1
            continue
        next_allowed = outcome["through"]
        row = {"ticker": ticker, "date": e["date"], "through": outcome["through"],
               "partition": "development" if e["date"] < holdout else "holdout",
               "reversal": outcome["largestDecline"] <= -.1 + 1e-12,
               "price_only": True, "volume_only": signals["volume"]["status"] == "elevated",
               "market_only": signals["market"]["status"] == "elevated",
               "acceleration_only": signals["acceleration"]["status"] == "elevated",
               "rise": e["return"], "previousClose": e["previousClose"], "close": e["close"],
               "signals": a["signals"], "return20": outcome["return20"], "largestDecline": outcome["largestDecline"]}
        row["multi_signal"] = sum(row[k] for k in ("volume_only", "market_only", "acceleration_only")) >= 2
        rows.append(row)
    return rows, excluded


def audit_examples(result, rows):
    """Independently recalculate displayed measurements for one event per split/company."""
    checks = []
    history = {r["date"]: r for r in result["history"]}
    for partition in ("development", "holdout"):
        selected = next((r for r in rows if r["partition"] == partition), None)
        if not selected:
            continue
        r = selected
        from .sessions import _calendar
        day = pd.Timestamp(r["date"])
        days = _calendar(day.year).sessions_in_range(day-pd.Timedelta(days=180), day)
        prior = days[days < day][-20:]
        volume = history[r["date"]]["volume"] / np.mean([history[d.date().isoformat()]["volume"] for d in prior])
        measured = {s["id"]: s["value"] for s in r["signals"]}
        close_ok = math.isclose(r["close"] / r["previousClose"] - 1, r["rise"], abs_tol=1e-10)
        volume_ok = math.isclose(volume, measured["volume"], abs_tol=1e-10)
        checks.append({"ticker": r["ticker"], "date": r["date"], "partition": partition,
                       "priceArithmeticPassed": close_ok, "volumeArithmeticPassed": volume_ok,
                       "manualReview": "not_performed"})
        if not close_ok or not volume_ok:
            raise RuntimeError("Independent sample arithmetic failed; report not accepted.")
    return checks


def build_report(rows, protocol, coverage, checks):
    partitions = {}
    for name in ("development", "holdout"):
        sample = [r for r in rows if r["partition"] == name]
        partitions[name] = {"from": min((r["date"] for r in sample), default=None),
            "through": max((r["through"] for r in sample), default=None),
            "variants": {v: metrics(sample, v) for v in VARIANTS}, "primaryLiftBlockBootstrap": block_lift(sample)}
    h = partitions["holdout"]
    primary, baseline = h["variants"]["multi_signal"], h["variants"]["price_only"]
    lift = primary["precision"] - baseline["precision"] if primary["precision"] is not None and baseline["precision"] is not None else None
    interval = h["primaryLiftBlockBootstrap"]["interval95"]
    enough = primary["flags"] >= 30 and baseline["events"] >= 100 and h["primaryLiftBlockBootstrap"]["quarters"] >= 8
    conclusion = "insufficient_sample" if not enough else "descriptive_lift_observed" if interval and interval[0] > 0 else "no_demonstrated_lift"
    return {"protocol": protocol, "coverage": coverage, "partitions": partitions, "holdoutPrecisionLift": lift,
            "conclusion": conclusion, "sampleChecks": checks, "validatedForPrediction": False,
            "limitations": ["Historical data has been explored before; this is not a guaranteed virgin holdout.",
                "Fixed nonrandom current-constituent sample introduces selection and survivorship bias.",
                "Candidates are price spikes, not all trading days; missed non-spike reversals and all-market false positives are not measured.",
                "Wilson intervals are approximate; quarterly block bootstrap retains contemporaneous cross-stock events but cannot remove all temporal dependence.",
                "Signal ablations are exploratory; do not select or retune a winner using holdout outcomes.",
                "Missing-input and incomplete-outcome exclusions can bias coverage. Daily closes miss intraday losses.",
                "Sample arithmetic checks are automated, not independent manual market-data verification.",
                "No model training, trading-profit evaluation, sector adjustment or manipulation detection is performed."]}



def render_report(report):
    lines = ["# Fixed-signal chronological audit", "", f"Conclusion: **{report['conclusion']}**", "",
        "Rules were frozen before this run. This is not a guarantee of never-seen historical data, nor validation for prediction.", ""]
    for name, partition in report["partitions"].items():
        lines += [f"## {name.title()}", "", f"Event/outcome coverage: {partition['from']} through {partition['through']}.", "",
            "| Rule | Flags | Reversals caught | False alerts | Missed reversals | Precision |", "|---|---:|---:|---:|---:|---:|"]
        for name, m in partition["variants"].items():
            rate = f"{100*m['precision']:.2f}%" if m['precision'] is not None else "Unavailable"
            lines.append(f"| {name} | {m['flags']} | {m['truePositives']} | {m['falseAlerts']} | {m['missedReversals']} | {rate} |")
        interval = partition["primaryLiftBlockBootstrap"]["interval95"]
        lines += ["", "Primary precision lift 95% quarterly-block bootstrap interval: " + (f"{interval[0]*100:.2f} to {interval[1]*100:.2f} percentage points." if interval else "Unavailable."), ""]
    lines += ["## Coverage and checks", "", f"{sum(c['status']=='ok' for c in report['coverage'])}/{len(report['coverage'])} companies evaluated. {len(report['sampleChecks'])} automated sample arithmetic checks; no manual secondary-vendor review.", "",
        "## Limits", "", *["- " + limit for limit in report["limitations"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--holdout", default="2022-01-01")
    args = parser.parse_args()
    if pd.Timestamp(args.holdout).date().isoformat() != args.holdout:
        parser.error("Use a YYYY-MM-DD holdout date")
    from .data import _normalize_ticker
    tickers = list(dict.fromkeys(_normalize_ticker(t) for t in args.tickers))
    run = uuid4().hex
    folder = BACKTEST_DIR / "anomaly_evaluation" / run
    protocol = {"version": "fixed_signal_audit_v1", "runId": run, "createdAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": METHOD, "tickers": tickers, "holdoutFrom": args.holdout, "target": "Close at least 10% below rise-day close within 20 sessions",
        "primary": "At least two of volume>=2x, stock-minus-SPY>=3pp, five-session acceleration>=5pp",
        "codeSha256": hashlib.sha256((inspect.getsource(analyze_event)+inspect.getsource(spike_signals)).encode()).hexdigest(),
        "rulesFrozenBeforeCollection": True, "bootstrapSeed": 42, "bootstrapResamples": 2000}
    (folder / "protocol.json").write_text(json.dumps(protocol, allow_nan=False))
    rows, coverage, checks = [], [], []
    for ticker in tickers:
        print(f"Evaluating {ticker}...", flush=True)
        try:
            result = read_anomalies(ticker, 0)
            selected, excluded = select_rows(ticker, result["events"], args.holdout)
            sampled = audit_examples(result, selected)
            rows.extend(selected); checks.extend(sampled)
            coverage.append({"ticker": ticker, "status": "ok", "asOf": result["asOf"], "candidates": len(result["events"]),
                "eligible": len(selected), "excluded": excluded,
                "priceHistorySha256": hashlib.sha256(json.dumps(result["history"], sort_keys=True, allow_nan=False).encode()).hexdigest()})
        except Exception as error:
            coverage.append({"ticker": ticker, "status": "failed", "errorType": type(error).__name__})
            print(f"{ticker}: excluded after {type(error).__name__}", flush=True)
    report = build_report(rows, protocol, coverage, checks)
    report["coverageComplete"] = all(c["status"] == "ok" for c in coverage)
    if not report["coverageComplete"]:
        report["conclusion"] = "incomplete_coverage"
    if not rows:
        raise RuntimeError("No eligible events. Protocol retained; no completed report published.")
    for name, text in (("events.json", json.dumps(rows, sort_keys=True, allow_nan=False)), ("report.json", json.dumps(report, sort_keys=True, allow_nan=False)), ("report.md", render_report(report))):
        path = folder / name
        path.write_text(text)
        if hashlib.sha256(path.read_text().encode()).hexdigest() != hashlib.sha256(text.encode()).hexdigest():
            raise RuntimeError("Cloud report verification failed.")
    print(json.dumps({"report": str(folder / "report.json"), "conclusion": report["conclusion"],
        "holdout": report["partitions"]["holdout"], "holdoutPrecisionLift": report["holdoutPrecisionLift"], "coverageComplete": report["coverageComplete"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
