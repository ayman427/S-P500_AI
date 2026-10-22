# Development benchmark ? October 14, 2026

This benchmark compares Ridge and HistGradientBoosting against zero-return/no-trade and always-long baselines. It evaluates development performance, not fresh confirmation of profitability.

## Fixed protocol

- Target: next-session open-to-close return, with the signal decided at the prior close.
- Sample chosen before the run: AAPL, MSFT, NVDA, AMD, AMZN, GOOGL, META, JPM, BAC, V, MA, UNH, JNJ, XOM, CVX, CAT, PG, KO, WMT, NEE, AMT, LIN.
- Latest local stock bar: October 5, 2026 for every selected stock. No prices were refreshed for this experiment.
- Inputs: stock OHLCV and SPY context. All eleven sector ETF histories and VIX are absent locally; SEC fundamentals are excluded for consistency across tickers.
- Evaluation: five expanding development folds, a five-session gap, and two inner time-series folds for parameter selection. The latest 252 labeled sessions per stock are excluded, with labels crossing into that period removed.
- Signal: predicted return must exceed the 20 bps round-trip cost. Entry threshold is zero; transaction cost is 10 bps per side. Long positions enter at the next open and exit at that session close.
- No thresholds or model grids were adjusted after inspecting this run. Production models and paper ledgers are preserved.

The sample spans sectors but is deliberately selected and includes several large technology companies. It is not a random sample. Current-constituent survivorship bias and dependence across stocks/dates limit interpretation. Counts of profitable tickers and pooled win rates are descriptive, not significance tests. Cumulative returns cover each ticker's concatenated validation windows; they are not annualized returns or a portfolio simulation.

## Artifacts

- `models/universe_intraday_comparison.json`: aggregate and per-ticker development results, protocol version, costs, and development dates.
- `backtests/development_benchmark_inputs.json`: hashes, coverage dates, and row counts for each input series; Python/scikit-learn versions; model and ledger hashes for preservation checks.
- `backtests/development_benchmark.log`: command output.
- Reproduction command: see the evaluation protocol in [README.md](README.md#evaluation-protocol).

## Results

All 22 selected stocks completed; none were skipped. Final holdout data was not evaluated.

| Method | Mean ticker return MAE | Beats zero-return MAE | Profitable tickers | Median ticker net return |
| --- | ---: | ---: | ---: | ---: |
| HistGradientBoosting | 1.0833% | 1/22 | 4/22 | -23.49% |
| Ridge | 1.0772% | 5/22 | 11/22 | -0.37% |
| Zero return / no trade | 1.0716% | 0/22 | 0/22 | +0.00% |

Always-long median ticker net return was -97.64%; always-long lost money on all 22 stocks under the assumed 20 bps round-trip daily cost. Both models beat always-long on every ticker, but no trading did too. That comparison does not demonstrate a useful trading edge.

The zero-return forecast has lower mean error than either model. Gradient boosting beats its error on only one stock; Ridge does so on five. Ridge performs better than gradient boosting on these aggregate measures, but its median net return is negative and only half the stocks are profitable. This run does not justify a claim of a consistent predictive or trading advantage. No model was promoted or retrained as a result.

The pooled net trade win rates are 46.35% for gradient boosting (6606 trades) and 48.00% for Ridge (5104 trades). These observations share dates and market conditions; they are not independent trials.

### Per-ticker development returns

These are cumulative net returns over each ticker?s validation windows. No trading returns 0% for every ticker. They are not returns from an equal-weight portfolio.

| Stock | Ridge | HistGradientBoosting | Always long |
| --- | ---: | ---: | ---: |
| AAPL | +22.47% | +25.25% | -90.56% |
| MSFT | +60.60% | +29.01% | -97.60% |
| NVDA | +196.12% | -1.29% | -96.12% |
| AMD | +4.82% | +22.68% | -98.98% |
| AMZN | -62.96% | -58.48% | -99.14% |
| GOOGL | +29.40% | -12.31% | -89.99% |
| META | +1.12% | -48.49% | -78.29% |
| JPM | -3.98% | -15.87% | -97.69% |
| BAC | -35.15% | -16.35% | -98.29% |
| V | +12.96% | -26.03% | -90.82% |
| MA | +3.12% | -38.45% | -94.94% |
| UNH | +10.27% | -48.03% | -98.35% |
| JNJ | -22.59% | +8.92% | -98.15% |
| XOM | -50.43% | -35.76% | -99.00% |
| CVX | -24.87% | -59.71% | -99.27% |
| CAT | -1.87% | -29.34% | -97.13% |
| PG | -17.16% | -20.96% | -96.64% |
| KO | -24.95% | -45.51% | -98.90% |
| WMT | +23.52% | -15.19% | -95.77% |
| NEE | -39.81% | -37.38% | -97.92% |
| AMT | -35.05% | -54.28% | -98.21% |
| LIN | +65.00% | -7.66% | -95.74% |

### Validation and preservation

- All 66 Python tests pass, including default holdout visibility and baseline alignment checks.
- Frontend lint and production build pass. Vite reports a bundle-size warning.
- Saved production model and paper-ledger hashes match the input snapshot.
- The completed report is archived in `backtests/development_benchmark_results.json`; the earlier comparison is preserved in `backtests/universe_intraday_comparison_previous.json`.

## Prospective validation

After choosing a model and rule on development data, record its fingerprint and keep it fixed while logging signals after each close. Start with current completed-session data; this saved dataset is stale. Do not count backfilled signals as prospective observations. Keep every ledger, including unsuccessful runs, and review paper returns separately from historical development results. The existing 30-day freeze warning supports this process but does not enforce it or establish sufficient evidence.

Final holdout results are hidden from default dashboard/API responses and require an explicit reveal. This cannot undo earlier inspection; previously viewed periods should not be called fresh holdouts. A later period of frozen-model paper observations remains necessary.
