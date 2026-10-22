# Rise investigation: price and volume

This view is independent of Forecast, prediction history/training and Declines. It reads existing completed-session stock and SPY prices from Convex. No news collection, Gemini call or experimental risk model is used by the view. Existing research records and maintenance tools are retained but are not prerequisites.

## Signals

Candidate dates retain the historical adjusted-price spike rule: daily rise of at least 5% and three standard deviations above the prior 60-row mean, minimum 20 prior returns. The new observation layer independently checks exchange-session alignment; candidates without the actual previous exchange session are excluded. Prior-volatility inputs require 20-60 complete prior exchange-session returns. Missing interior sessions are not filled or silently skipped.

- Price vs normal: daily return compared with max(5%, prior mean + 3 prior standard deviations).
- Above SPY: stock daily return minus SPY daily return on exactly the same two exchange dates. Elevated at 3 percentage points. This is a return difference, not a fitted beta adjustment.
- Trading volume: rise-day volume divided by the prior 20 exchange sessions' average. Elevated at 2x. Missing/invalid volume or a zero baseline is unavailable.
- Five-session acceleration: latest compounded five-session return minus the preceding disjoint five-session return. Elevated at 5 percentage points. It measures momentum improvement, including recovery from negative returns.

Rules are descriptive thresholds, not learned or validated reversal probabilities. Missing inputs remain unavailable. Sector comparison is not implemented because historical sector membership has not been established; no sector is guessed.

## Timing

All event signals use information through that event's close only. Prior mean, volatility and volume baselines exclude the event session. Prices after the event affect only the separate historical outcome: 5/20-session returns, largest close-based loss from rise-day close, and original gain reversed at the last available close. The outcome window stops at the first missing exchange session and at 20 subsequent sessions. Reversals never change the anomaly signals or summary.

## Historical check

Within the selected history period, use completed, non-overlapping outcome windows with all four signals available. Select newest qualifying windows first. A multi-signal event has at least two elevated signals beyond price. Report how many multi-signal events closed at least 10% below their rise-day close within 20 sessions, alongside counts for all eligible price spikes as a simple baseline. Missing-input, incomplete and overlapping windows are excluded and counted.

This is same-stock retrospective description. It is not an untouched holdout, a causal comparison, a prospective probability or evidence of predictive improvement. Small and current-constituent samples can be biased. Counts include only the selected price-spike universe and cannot measure all-market false positives.

## UI and storage

History selector, spike selector, interactive price/volume chart, short summary and four signal values. One Details drawer holds methods, data coverage, historical check and later outcomes. There are no news verdicts, manipulation labels or risk percentages. Data is read from Convex into memory; no new local dataset or cloud schema is needed. Existing source archives are not deleted.

## Limits

Unusual trading is not proof of manipulation, nor a recommendation to trade. Daily closes miss intraday movements. Volume is provider-reported and can be distorted across corporate actions. Market comparison does not account for sector-specific or beta effects. Baseline/future-price invariance, missing-data behaviour, market-wide moves, counts and UI are tested; future reversal prediction has not been validated.


## Offline chronological audit

Run `python -m sp500_ml.anomaly_evaluation` to evaluate the fixed primary multi-signal rule on a preselected, nonrandom 12-company sample (AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, JNJ, PG, WMT, CAT). Rules, source-code hash, sample and split are written to a new Convex run before reading outcomes. Default split: development before 2022-01-01 and chronological holdout from that date. Outcomes crossing the split are purged. Within each company, retain the earliest eligible non-overlapping event windows. Every baseline uses the same complete-input cohort.

The primary rule is unchanged: at least two additional elevated signals beyond the price anomaly. Volume-only, market-only and acceleration-only ablations are exploratory. Report true positives, false alerts, missed reversals, precision, recall and false-positive rates within the spike universe. Wilson intervals are approximate; primary precision lift against the all-price-spike baseline additionally uses 2,000 paired calendar-quarter block bootstrap resamples (seed 42), retaining contemporaneous cross-company groups. Positive descriptive lift requires an interval above zero, at least 30 flags, 100 eligible holdout events and eight populated quarters; it never enables a production risk score.

Artifacts are `backtests/anomaly_evaluation/<run-id>/protocol.json`, `events.json` and `report.json` in Convex, not local folders. Event records contain measurements and outcomes; report coverage records identify price-history hashes, data dates and every exclusion. Report/events are read back and verified after publication. Failures remain explicit and force an incomplete-coverage conclusion. Forecast models and UI are never changed.

The script independently recalculates stock-return and volume arithmetic for one development and one holdout example per supported company (up to 24 examples). These are automated checks, not manual verification against a second vendor. Historical data was explored previously: the chronological holdout is an audit partition, not a guarantee of never-seen data. A fresh prospective period is needed for stronger confirmation. No thresholds should be retuned against this report and then described as independently validated.
