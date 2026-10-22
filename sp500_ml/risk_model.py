"""Calibrated downside classifier, separate from return forecasts and retrospective explanations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import re
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import publish_generation, resolve_generation
from .cloud_data import read_csv, list_datasets
from .data import _normalize_ticker, _ticker_file, get_sp500_tickers
from .news_analysis import _mentions
from .paths import MODEL_DIR, BACKTEST_DIR, NEWS_DIR, DATA_DIR
from .rise_risk import positive_development
from .search_prices import latest_completed_session
from .sessions import execution_window, utc_now
from .spike_rules import prepare_prices, spike_signals, DETECTOR_VERSION

PREFIX = "rise_downside_v1"
TARGET = "Close at least 10% below rise-day close within the next 20 trading sessions"
PRICE_FEATURES = ["rise_return", "return_5", "return_20", "volatility_20", "volatility_60", "volume_ratio", "drawdown_60",
                  "market_return_1", "market_return_5", "market_return_20", "market_volatility_20"]
EVIDENCE_FEATURES = ["public_statements", "speculative_reports", "business_reports", "primary_developments", "dated_sources"]


def company_aliases(ticker, name):
    cleaned = re.sub(r"\b(?:incorporated|inc|corporation|corp|plc|ltd|limited|holdings)\b\.?", "", name, flags=re.I).strip(" ,.")
    return [alias for alias in dict.fromkeys([ticker, cleaned]) if alias]


def price_frame(prices, market=None):
    prices = prepare_prices(prices)
    calendar = xcals.get_calendar("XNYS", start=prices.index[0], end=prices.index[-1])
    prices = prices.reindex(calendar.sessions_in_range(prices.index[0], prices.index[-1]))
    close, volume = prices.Close, prices.Volume
    returns, _, rises = spike_signals(close)
    frame = pd.DataFrame({"rise_return": returns, "return_5": close.pct_change(5, fill_method=None),
        "return_20": close.pct_change(20, fill_method=None), "volatility_20": returns.rolling(20).std(),
        "volatility_60": returns.rolling(60).std(), "volume_ratio": volume / volume.shift(1).rolling(20).mean().replace(0, np.nan),
        "drawdown_60": close / close.rolling(60).max() - 1})
    market_close = market.Close.reindex(frame.index) if market is not None else pd.Series(np.nan, index=frame.index)
    for days in (1, 5, 20):
        frame[f"market_return_{days}"] = market_close.pct_change(days, fill_method=None)
    frame["market_volatility_20"] = market_close.pct_change(fill_method=None).rolling(20).std()
    future = pd.concat([close.shift(-day) for day in range(1, 21)], axis=1)
    frame["outcome"] = (future.min(axis=1) / close - 1 <= -.10 + 1e-12).astype(float).where(future.notna().all(axis=1))
    frame["label_end"] = pd.Series(frame.index, index=frame.index).shift(-20)
    frame["is_rise"] = rises & (close.rolling(61).count() == 61)
    return frame.replace([np.inf, -np.inf], np.nan)


def evidence_features(archive, cutoff, aliases):
    """Use the actual observed version, not merely an old publication date."""
    from .spike_research import TOPICS
    result = dict.fromkeys(EVIDENCE_FEATURES, 0.)
    seen = set()
    for row in archive.get("headlines", []) + archive.get("documents", []):
        published = pd.to_datetime(row.get("published_at_utc", row.get("publishedAt")), utc=True, errors="coerce")
        retrieved = pd.to_datetime(row.get("retrievedAt"), utc=True, errors="coerce")
        if pd.isna(published) or pd.isna(retrieved) or published > cutoff or retrieved > cutoff or published < cutoff - pd.Timedelta(days=7):
            continue
        if row.get("publicationPrecision") == "date" or str(row.get("source", "")).startswith("Gemini"):
            continue
        text = row.get("title", "") + " " + row.get("excerpt", row.get("text", ""))
        if not _mentions(text, aliases):
            continue
        identity = (row.get("url"), row.get("contentHash", text))
        if identity in seen:
            continue
        seen.add(identity)
        categories = {label for label, pattern in TOPICS if re.search(pattern, text, re.I)}
        result["dated_sources"] += 1
        result["public_statements"] += "Political / public statements" in categories
        result["speculative_reports"] += "Speculative activity" in categories
        result["business_reports"] += bool(categories.intersection({"Earnings / guidance", "Buybacks / dividends", "Acquisition / deal", "Products / approvals / contracts"}))
        result["primary_developments"] += row.get("basis") in {"sec_filing", "company_announcement"} and positive_development(text)
    return result


def event_rows(ticker, prices, market=None, archives=None, aliases=None, training=True):
    frame = price_frame(prices, market)
    rows, next_allowed = [], pd.Timestamp.min
    for day, values in frame.loc[frame.is_rise].iterrows():
        if training and (pd.isna(values.outcome) or day < next_allowed):
            continue
        archive = (archives or {}).get(day.date().isoformat())
        evidence = evidence_features(archive, execution_window(day)[1] - pd.Timedelta(nanoseconds=1), aliases or [ticker]) if archive else dict.fromkeys(EVIDENCE_FEATURES, 0.)
        rows.append({"ticker": ticker, "date": day, "label_end": values.label_end,
                     "outcome": values.outcome, **values[PRICE_FEATURES].to_dict(), **evidence})
        if training:
            next_allowed = values.label_end + pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["ticker", "date", "label_end", "outcome", *PRICE_FEATURES, *EVIDENCE_FEATURES])


def chronological_splits(rows):
    dates = sorted(rows.date.unique())
    if len(dates) < 100:
        raise ValueError("At least 100 distinct rise dates are required for chronological validation.")
    calibration_start, test_start = dates[int(len(dates) * .6)], dates[int(len(dates) * .8)]
    train = rows.loc[(rows.date < calibration_start) & (rows.label_end < calibration_start)]
    calibration = rows.loc[(rows.date >= calibration_start) & (rows.date < test_start) & (rows.label_end < test_start)]
    test = rows.loc[rows.date >= test_start]
    for name, part in zip(("training", "calibration", "holdout"), (train, calibration, test)):
        counts = part.outcome.value_counts()
        if len(part) < 100 or counts.get(0, 0) < 20 or counts.get(1, 0) < 20:
            raise ValueError(f"Insufficient {name} support: need 100 resolved rises including 20 outcomes of each class.")
    return train, calibration, test


def _logits(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_classifier(train, calibration, features):
    estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(C=1., max_iter=1000))
    estimator.fit(train[features], train.outcome)
    calibrator = LogisticRegression(C=1., max_iter=1000)
    calibrator.fit(_logits(estimator.predict_proba(calibration[features])[:, 1]), calibration.outcome)
    return {"estimator": estimator, "calibrator": calibrator, "features": features}


def probabilities(model, rows):
    return model["calibrator"].predict_proba(_logits(model["estimator"].predict_proba(rows[model["features"]])[:, 1]))[:, 1]


def metrics(y, probability, reference):
    y = np.asarray(y)
    bins = np.minimum((probability * 5).astype(int), 4)
    groups = [{"count": int(sum(bins == i)), "predicted": float(probability[bins == i].mean()), "observed": float(y[bins == i].mean())}
              for i in range(5) if sum(bins == i)]
    return {"samples": len(y), "positives": int(y.sum()), "brier": float(brier_score_loss(y, probability)),
            "constantBrier": float(brier_score_loss(y, np.full(len(y), reference))), "rocAuc": float(roc_auc_score(y, probability)),
            "calibrationError": sum(g["count"] * abs(g["predicted"] - g["observed"]) for g in groups) / len(y), "calibrationBins": groups}


def train_dataset(rows):
    rows = rows.sort_values(["date", "ticker"])
    if rows.duplicated(["ticker", "date"]).any() or rows.outcome.isna().any():
        raise ValueError("Training rows must be unique resolved stock/date outcomes.")
    train, calibration, test = chronological_splits(rows)
    price = fit_classifier(train, calibration, PRICE_FEATURES)
    probability = probabilities(price, test)
    validation = metrics(test.outcome, probability, float(train.outcome.mean()))
    approved = validation["brier"] < .95 * validation["constantBrier"] and validation["rocAuc"] >= .60 and validation["calibrationError"] <= .10
    report = {"target": TARGET, "version": PREFIX, "detector": DETECTOR_VERSION, "status": "experimental_validated" if approved else "validation_failed",
        "featureMode": "price_only", "rows": len(rows), "tickers": int(rows.ticker.nunique()), "trainRows": len(train), "calibrationRows": len(calibration),
        "trainedThrough": calibration.label_end.max().date().isoformat(), "holdoutFrom": test.date.min().date().isoformat(),
        "holdoutThrough": test.date.max().date().isoformat(), "holdoutLabelsThrough": test.label_end.max().date().isoformat(),
        "priceValidation": validation, "evidenceRows": int((rows.dated_sources > 0).sum()), "combinedStatus": "insufficient_point_in_time_evidence",
        "evidenceCutoffPolicy": "Publication and exact observed content version must predate the next session open; unknown evidence uses the price-only model.",
        "limitations": ["Current S&P 500 constituents introduce survivorship bias.", "Closing-price declines only; intraday lows are not labels.",
                        "One chronological holdout is not independent live validation.", "A downside score does not establish manipulation."]}
    combined = None
    evidence_parts = [part.loc[part.dated_sources > 0] for part in (train, calibration, test)]
    if all(len(part) >= 100 and min(part.outcome.value_counts().reindex([0, 1], fill_value=0)) >= 20 for part in evidence_parts):
        candidate = fit_classifier(evidence_parts[0], evidence_parts[1], PRICE_FEATURES + EVIDENCE_FEATURES)
        subset = evidence_parts[2]
        candidate_probability = probabilities(candidate, subset)
        candidate_metrics = metrics(subset.outcome, candidate_probability, float(evidence_parts[0].outcome.mean()))
        price_brier = brier_score_loss(subset.outcome, probabilities(price, subset))
        report["combinedValidation"] = candidate_metrics
        report["combinedPriceBaselineBrier"] = float(price_brier)
        valid = approved and candidate_metrics["brier"] < .95 * price_brier and candidate_metrics["rocAuc"] >= .60 and candidate_metrics["calibrationError"] <= .10
        report["combinedStatus"] = "experimental_validated" if valid else "validation_failed"
        if valid:
            combined = candidate
    model = {"version": PREFIX, "target": TARGET, "price": price, "combined": combined, "report": report}
    held_out = test[["ticker", "date", "label_end", "outcome"]].copy()
    held_out["probability"] = probability
    return model, report, held_out


def train_risk(tickers=None):
    tickers = sorted(set(_normalize_ticker(t) for t in (tickers or get_sp500_tickers())))
    expected = latest_completed_session(utc_now())
    market = prepare_prices(read_csv(_ticker_file("SPY"), parse_dates=["Date"], index_col="Date"))
    names = json.loads((DATA_DIR.parent / "companies.json").read_text())
    archive_keys = {row["key"] for row in list_datasets("news/spike_evidence/")}
    def load(ticker):
        prices = read_csv(_ticker_file(ticker), parse_dates=["Date"], index_col="Date")
        archive = {key.split("/")[-2]: json.loads((NEWS_DIR / key.removeprefix("news/")).read_text())
                   for key in archive_keys if key.endswith("/" + ticker + ".json")}
        return event_rows(ticker, prices.loc[prices.index <= expected], market, archive, company_aliases(ticker, names.get(ticker, ticker)))
    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for checked, (ticker, future) in enumerate(zip(tickers, [pool.submit(load, ticker) for ticker in tickers]), 1):
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                failures.append(ticker)
            print(f"Risk dataset: {checked}/{len(tickers)} stocks checked", flush=True)
    if not frames:
        raise ValueError("No resolved rise events were available; no risk model was published.")
    rows = pd.concat(frames, ignore_index=True)
    model, report, held_out = train_dataset(rows)
    report.update(requestedTickers=len(tickers), unavailableTickers=failures, builtAt=utc_now().isoformat(), priceAsOf=expected.date().isoformat())
    model["report"] = report
    # Immutable training examples are uploaded before their model generation is published.
    from .cloud_data import write_csv
    dataset = BACKTEST_DIR / "risk_training" / f"{uuid4().hex}.csv"
    write_csv(rows, dataset, index=False, date_format="%Y-%m-%d")
    report["datasetKey"] = str(dataset)
    model["companyNames"] = names
    publish_generation(MODEL_DIR, PREFIX, model, report, {"target": TARGET, "validation": report["priceValidation"]}, held_out)
    return report


def risk_score(ticker, event_date, history):
    unavailable = {"status": "not_trained", "target": TARGET, "probability": None,
                   "message": "Risk model has not been trained and validated yet."}
    try:
        paths = resolve_generation(MODEL_DIR, PREFIX)
        if paths is None:
            return unavailable
        model = joblib.load(BytesIO(paths[0].read_bytes()))
        if model.get("version") != PREFIX or model.get("target") != TARGET:
            raise ValueError("Incompatible risk model")
        report = model["report"]
        if report["status"] != "experimental_validated":
            return {**unavailable, "status": "validation_failed", "message": "The experimental model did not pass its validation gates; no score is shown."}
        if event_date <= report["trainedThrough"]:
            return {**unavailable, "status": "historical_in_sample", "message": "This event predates the model's training cutoff; no retrospective score is shown."}
        prices = pd.DataFrame(history).set_index("date")
        prices.index = pd.to_datetime(prices.index)
        prices = prices.rename(columns={"close": "Close", "volume": "Volume"})
        prices = prices.loc[prices.index <= pd.Timestamp(event_date)]
        market = prepare_prices(read_csv(_ticker_file("SPY"), parse_dates=["Date"], index_col="Date"))
        path = NEWS_DIR / "spike_evidence" / event_date / f"{ticker}.json"
        archives = {event_date: json.loads(path.read_text())} if path.exists() else {}
        rows = event_rows(ticker, prices, market, archives, company_aliases(ticker, model["companyNames"].get(ticker, ticker)), training=False)
        row = rows.loc[rows.date == pd.Timestamp(event_date)]
        if row.empty:
            return {**unavailable, "status": "insufficient_prices", "message": "This rise lacks the complete prior price window required by the model."}
        evidence_window_closed = utc_now() >= execution_window(event_date)[1]
        combined = model["combined"] if row.dated_sources.iloc[0] > 0 and evidence_window_closed else None
        selected = combined or model["price"]
        transformed = selected["estimator"].named_steps["simpleimputer"].transform(row[selected["features"]])
        standardized = selected["estimator"].named_steps["standardscaler"].transform(transformed)[0]
        contributions = standardized * selected["estimator"].named_steps["logisticregression"].coef_[0] * selected["calibrator"].coef_[0, 0]
        drivers = [{"feature": feature, "direction": "raises" if value > 0 else "lowers"}
                   for feature, value in sorted(zip(selected["features"], contributions), key=lambda pair: -abs(pair[1]))
                   if abs(value) > 1e-12 and pd.notna(row[feature].iloc[0])][:3]
        return {"status": "experimental", "target": TARGET, "probability": float(probabilities(selected, row)[0]),
                "featureMode": "price_and_evidence" if combined else "price_only", "modelVersion": model["run_id"],
                "trainedThrough": report["trainedThrough"], "evidenceSources": int(row.dated_sources.iloc[0]), "drivers": drivers,
                "missingPriceInputs": int(row[PRICE_FEATURES].isna().sum().sum()),
                "validation": report.get("combinedValidation") if combined else report["priceValidation"],
                "message": "Experimental downside estimate, not a manipulation score. Evidence model is awaiting sufficient validation." if model["combined"] is None else "Experimental downside estimate, not a manipulation score."}
    except Exception:
        return {**unavailable, "status": "unavailable", "message": "The risk model could not be verified or loaded; no score is shown."}


def main():
    parser = argparse.ArgumentParser(description="Train the separate rise downside-risk model using Convex datasets.")
    parser.add_argument("--tickers", nargs="+")
    args = parser.parse_args()
    print(json.dumps(train_risk(args.tickers), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
