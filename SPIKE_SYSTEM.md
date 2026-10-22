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
