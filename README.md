# S&P 500 price model

Downloads adjusted daily prices for the current S&P 500 constituents from Yahoo Finance, starting on 2000-01-01. Each ticker is stored in `data/prices/` as a CSV, so updates merge by date and can be safely rerun. The current constituent list is refreshed from Wikipedia whenever data is updated.

## Code layout

`sp500_ml/` is a package, split by responsibility: `data.py` (price download/storage), `fundamentals.py` (SEC point-in-time fundamentals), `feature_names.py` + `features.py` (feature engineering), `models.py` (estimator factory, artifact paths), `backtest.py` (walk-forward validation, holdout, comparisons), `training.py` (train/predict orchestration), `news.py` (headline fetch and informational, non-model entity extraction), `paper.py` (paper-trading ledgers), and `cli.py` (the `python -m sp500_ml` entry point).

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Dashboard

Install the frontend once:

```powershell
npm --prefix frontend install
```

Start the local API in one PowerShell terminal:

```powershell
& ".\.venv\Scripts\python.exe" dashboard_server.py
```

Start the React dev server from the project root in a second terminal:

```powershell
node frontend/node_modules/vite/bin/vite.js frontend --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. The dashboard reads local project data and exposes explicit price-sync, train, and paper-signal actions. It is bound to localhost and does not place orders.

## Run

```powershell
python -m sp500_ml update
python -m sp500_ml list
python -m sp500_ml news --ticker CCI
python -m sp500_ml fundamentals --ticker AAPL
python -m sp500_ml train --ticker AAPL
python -m sp500_ml train --ticker AAPL --model ridge --ridge-alpha 1.0
python -m sp500_ml compare --ticker AAPL
python -m sp500_ml predict --ticker AAPL
python -m sp500_ml predict --ticker AAPL --model ridge
python -m sp500_ml update
python -m sp500_ml paper-trade --ticker AAPL
python -m sp500_ml paper-status --ticker AAPL
```

`news --ticker CCI` fetches recent headlines and counterparty context without training a model or creating a paper signal. Results are cached in `data/news/CCI.json` and shown in the dashboard for that ticker.

`compare --ticker AAPL` compares Ridge with HistGradientBoosting on development folds only. It saves `models/AAPL_intraday_comparison.json` and does not replace the selected next-close production model or use the final holdout. The dashboard's **Compare models** button runs this experiment for the selected ticker.

`compare-all` runs the same Ridge-versus-HistGradientBoosting development-only comparison across every locally downloaded ticker and saves aggregate and per-ticker results to `models/universe_intraday_comparison.json`. It uses price and SPY features consistently, excludes each ticker's latest 252 sessions, and does not request SEC facts or change production models. Use `--limit 5` for a quick sample; the full local-universe run can take a while.

After changing the signal-decision rule, run `python -m sp500_ml reassess` to recompute development and holdout metrics for saved model reports and compare them against their old values in `models/methodology_reassessment.json`. The old trading figures used next-open information for entry and should be treated as optimistic upper bounds.

Training runs three development walk-forward folds and evaluates a separate, untouched final 252-session holdout. The production model is selectable per ticker: HistGradientBoosting or Ridge. HistGradientBoosting remains the default. Ridge uses a median imputer and feature standardization; tune its positive `--ridge-alpha` value when training. Ridge artifacts and paper ledgers are kept separate from the existing HistGradientBoosting files. `compare` and `compare-all` compare these estimators on next-session open-to-close returns using development folds only; they do not change the selected next-close model or use the final holdout for selection. The final training report is in `models/AAPL_metrics.json`, and its session-by-session trade log is in `backtests/AAPL_final_holdout_trades.csv`. Entry is decided at the signal close using the model's predicted close-to-close return less round-trip costs and the configured threshold; the next open cannot change that decision. The holdout also compares no-trade and always-long baselines.

The final holdout is computed on every `train`/`run` but written to its own sealed file (`models/AAPL_holdout.json`, plus the trades CSV), never to `models/AAPL_metrics.json`, and it is not printed unless you pass `--reveal-holdout`. Headline error is return-space MAE (compared with a previous-close baseline); dollar MAE is kept only for reference. HistGradientBoosting now trains with absolute-error loss, so retrained models differ from earlier ones.

Paper ledgers are named `{ticker}_{model_type}_{fingerprint}_v2.csv`. A model is treated as frozen for 30 days of paper trading: retraining earlier warns, because it starts a new ledger. `paper-status` settles pending signals against local prices. `predict` warns when the latest bar is 3 or more sessions old. SEC fundamentals are re-downloaded at most every 24 hours.

```powershell
python -m sp500_ml train --ticker AAPL --reveal-holdout
```

```powershell
python -m sp500_ml train --ticker AAPL --model ridge --ridge-alpha 0.1
python -m sp500_ml predict --ticker AAPL --model ridge
python -m sp500_ml paper-trade --ticker AAPL --model ridge
```

To require a further 0.5% projected edge and assume 10 bps cost per side:

```powershell
python -m sp500_ml train --ticker AAPL --entry-threshold 0.005 --transaction-cost-bps 10
```

Choose any ticker printed by `list`. Each ticker gets its own model and validation metrics. To update data, train the selected ticker, and predict its next-session close in one command:

```powershell
python -m sp500_ml run --ticker AAPL
```

The per-ticker gradient-boosting model estimates the next trading session's closing price using stock return, volatility, intraday range/gap, SPY market context, and point-in-time SEC fundamentals. Fundamentals are joined from the day after the SEC filing date to avoid using data before it was public. Set an identifying SEC User-Agent/contact in PowerShell before the first fundamentals download:

```powershell
$env:SEC_USER_AGENT = "S&P500_AI your.name@example.com"
```

On Windows, the application also reads `SEC_USER_AGENT` from the saved current-user environment registry value, so save it once with `[Environment]::SetEnvironmentVariable(..., "User")` and reuse it for every ticker and dashboard training action.

`fundamentals --ticker AAPL` refreshes the selected company's local SEC facts cache. Each `train` action now attempts to refresh SEC facts first. If SEC is unavailable, training uses that ticker's existing cache when possible; otherwise it continues with price and market features and records a warning in the metrics. The downloader stores adjusted OHLCV prices. Existing CSVs with only close and volume are automatically recognized as an old schema and re-downloaded from 2000 on the next `update`. Training uses three expanding walk-forward validation folds. Its trading report evaluates the forecast made at a prior close against the next session's observed open, enters long only when the projected close-to-open edge clears the threshold and estimated round-trip costs, and exits at that session's close. It reports wins/losses, net win rate and a 95% Wilson interval, cumulative return, drawdown, Sharpe ratio, profit factor, and an always-long comparison. The interval is a rough binomial interval and does not account for dependence between adjacent trading days; it is not proof of future profitability. Do not tune the threshold on these same folds and treat the result as unbiased; reserve a later untouched period for final confirmation. The selected ticker's prediction is written to `predictions.csv`; its fitted model and metrics are written under `models/`.

Historical news is not yet a model feature: the free historical-news probe returned HTTP 429, and current headlines cannot be used to train past periods without leakage. Add news only after configuring a reliable timestamped historical provider.

For prospective paper trading, first train a ticker model. After each market close, run `update` and then `paper-trade --ticker AAPL`. The command settles the previous pending forecast when the next session's open and close are present, then records the latest forecast as pending. Recent Yahoo Finance headlines are saved with publication timestamps in the ledger as context only. Simple explicit relationship phrases (such as acquires, invests in, or partners with) trigger a Wikipedia search for the named counterparty; the headline, extraction rule, search snippets, and source links are saved together. These are candidate relationships requiring review, not verified facts, and they are not model inputs or used in historical backtests. Paper histories are kept separately by model version; retraining starts a new ledger automatically and preserves the prior one. `paper-status --ticker AAPL` shows the ledger for the currently trained model. Re-running on the same signal date does not add duplicates.

Run `update` regularly, for example after the market close, to fetch new rows. It is incremental and also downloads full history for newly added constituents. Yahoo Finance/Wikipedia availability and rate limits can affect downloads; rerun `update` to retry. This uses today's constituent list for the entire history, so results have survivorship bias and do not represent historical S&P 500 membership. The model is an educational baseline, not investment advice.

## Tests

```powershell
python -m unittest discover -s tests
```
