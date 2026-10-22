# S&P 500 price model

Downloads adjusted daily prices for the current S&P 500 constituents from Yahoo Finance, starting on 2000-01-01. Convex uses separate indexed tables: `prices`, `fundamentals`, `news`, `companyMetadata`, `priceSync`, and `predictions`. Model binaries/manifests use `modelArtifacts`; backtest reports use `backtestReports`; paper ledgers use `paperLedgers`; CLI prediction exports use `predictionExports`. Large dataset contents live in Convex File Storage, referenced by their category's table. The app reads them into memory for calculations and writes updates back to Convex; it creates no local data, models, backtests, or paper-ledger folders. The current constituent list is refreshed from Wikipedia whenever data is updated. Configure Convex first using [CONVEX_SETUP.md](CONVEX_SETUP.md).

## Code layout

`sp500_ml/` is a package, split by responsibility: `data.py` (price download/storage), `fundamentals.py` (SEC point-in-time fundamentals), `feature_names.py` + `features.py` (feature engineering), `models.py` (estimator factory, artifact paths), `backtest.py` (walk-forward validation, holdout, comparisons), `training.py` (train/predict orchestration), `news.py` (headline fetch and informational, non-model entity extraction), `paper.py` (paper-trading ledgers), and `cli.py` (the `python -m sp500_ml` entry point).

## Setup

Clerk sign-in is required. See [CLERK_SETUP.md](CLERK_SETUP.md) for the connected
Clerk application, Convex integration, environment settings, and ownership rules.
Market data and reusable models are shared; each signed-in user has private
prediction history. Legacy ownerless records are kept service-only.

```powershell
python -m pip install -r requirements.txt
```

## Dashboard

### Rise investigation

Rise investigation uses saved prices and volume only. It measures the rise against prior volatility, same-session SPY performance, prior 20-session volume and five-session acceleration. News, Gemini, source collection and experimental risk models are not required. Forecast and Declines are unchanged.

The chart and four measured signals are visible by default. One Details drawer explains thresholds, missing inputs, historical reversal counts against a price-spike baseline, and the selected event's later outcome. Future prices never affect the event signals. Historical counts are descriptive, not calibrated probabilities or manipulation verdicts. See [SPIKE_SYSTEM.md](SPIKE_SYSTEM.md).

The sidebar has **Predictions for you** (private saved forecasts) and **Rise candidates**
(shared S&P 500 rankings). The checklist scans all current constituents independently
of searches, refreshes prices, and ranks stocks by volume buildup, momentum and breakout
conditions met. Progress, session dates and missing coverage are visible; expanding a
stock shows its checks. Restart the local API after updating to start the background
scanner. See [SPIKE_SYSTEM.md](SPIKE_SYSTEM.md) for worker setup on hosted deployments.

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

Open `http://127.0.0.1:5173`. The dashboard presents one stock-search panel. Enter a company name (such as Apple or Nvidia) or ticker to see its saved price history, return forecast for the selected period, and historical development results. If no model or validation report exists, search prepares one automatically and shows progress. Choose Gradient Boosting or Ridge with the compact model selector. The forecast and historical test results are labelled with the selected model, and that choice is remembered. Incompatible older targets are retrained automatically on search; missing required inputs produce an explanation instead of a forecast. Choose a one-session or five-session forecast period beside the model selector. Models and reports are kept separately for each period, and both selections are remembered. Ambiguous names, including companies with multiple share classes, require an explicit choice. The company directory uses company names and price inventory stored in Convex, so searches do not depend on a remote name lookup. Advanced comparisons, price syncing, and paper-trading actions remain available through the CLI. The local API is bound to localhost and does not place orders. Use the moon/sun button to switch themes; the browser remembers the choice across refreshes and future visits.

## Vercel

`vercel.json` routes `/api/*` to the FastAPI ASGI entrypoint `vercel_app:app` and all other paths to the Vite frontend service. Prices, fundamentals, news, and stock metadata persist in Convex. Set the same server-only Convex connection settings in the deployed Python service. Model binaries, evaluation reports, and paper ledgers also persist in Convex. Trained generations publish a checksum-verified manifest last; renewable cloud locks replace local lock files. Long training jobs still need adequate worker time and memory. Logs go to the terminal or hosting log collector. Convex predictions and actual-result comparisons are already cloud-backed.

## Run

```powershell
python -m sp500_ml update
python -m sp500_ml list
python -m sp500_ml news --ticker CCI
python -m sp500_ml fundamentals --ticker AAPL
python -m sp500_ml train --ticker AAPL
python -m sp500_ml train --ticker AAPL --model ridge --ridge-alpha 1.0
python -m sp500_ml train --ticker AAPL --target open_to_close_5d
python -m sp500_ml compare --ticker AAPL
python -m sp500_ml predict --ticker AAPL
python -m sp500_ml predict --ticker AAPL --model ridge
python -m sp500_ml update
python -m sp500_ml paper-trade --ticker AAPL
python -m sp500_ml paper-status --ticker AAPL
```

`news --ticker CCI` fetches recent headlines and counterparty context without training a model or creating a paper signal. Results are stored as the `news/CCI.json` dataset in Convex and shown in the dashboard for that ticker.

`compare --ticker AAPL` compares Ridge with HistGradientBoosting on development folds only. The default target is next-session open-to-close; use `--target open_to_close_5d` for a next-open through fifth-session-close return. It does not replace the selected production model or use the final holdout. Run this experiment through the CLI; the simplified dashboard only shows stock results.

`compare-all` runs the Ridge-versus-HistGradientBoosting development-only comparison across every stock ticker stored in Convex and saves aggregate and per-ticker results. It excludes each ticker's latest 252 sessions, and does not request SEC facts or change production models. Use `--limit 5` for a quick sample; the full universe run can take a while.

After changing the target or signal rule, run `python -m sp500_ml reassess` to recompute development and holdout metrics for saved model reports and compare them against their old values in `models/methodology_reassessment.json`. The old models predict close-to-close return and are not directly comparable to the new open-to-close target; their signal decisions also used next-open information.

Training uses five expanding-window walk-forward folds with a five-session purge gap and evaluates a separate, untouched final 252-session holdout. Nested time-series CV selects Ridge alpha or HistGradientBoosting iteration/leaf settings within each training fold. The default one-session target remains next-session open-to-close; `--target open_to_close_5d` adds a next-open through fifth-session-close target with separate model/report files. Auxiliary market inputs include all eleven sector ETFs and VIX, downloaded by `update`, plus momentum/volatility and stock/market interaction features. Those series are optional until downloaded; daily Yahoo bars do not provide order-flow, bid/ask, options, or timestamped historical news features. SEC features remain point-in-time signed-log changes plus margins/leverage ratios. Paper trading currently supports only the one-session target. The one-session holdout compares no-trade and always-long baselines; the five-session holdout uses non-overlapping entries.

The final holdout is computed on every `train`/`run` but written to its own sealed file (`models/AAPL_open_to_close_holdout.json`, plus the trades CSV), never to `models/AAPL_open_to_close_metrics.json`, and it is not printed unless you pass `--reveal-holdout`. Headline error is return-space MAE, compared with a zero-return baseline for the open-to-close target; dollar MAE is retained as a secondary reference. Open-to-close artifacts use separate filenames, preserving older close-to-close models and reports.

New verified paper ledgers are named `{ticker}_{model_type}_{fingerprint}_v3.csv`. Earlier ledgers are preserved as historical files and are not substituted into current paper results. A model is treated as frozen for 30 days of paper trading: retraining earlier warns, because it starts a new ledger. `paper-status` settles pending signals against Convex prices. `predict` warns when the latest bar is 3 or more sessions old. SEC fundamentals are re-downloaded at most every 24 hours.

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

Choose any ticker printed by `list`. Each ticker gets its own model and validation metrics. To update data, train the selected ticker, and predict its next-session open-to-close return in one command:

```powershell
python -m sp500_ml run --ticker AAPL
```

The per-ticker model predicts next-session open-to-close return using stock return, volatility, intraday range/gap, SPY market context, and point-in-time SEC fundamentals. Fundamentals are joined from the day after the SEC filing date to avoid using data before it was public. Set an identifying SEC User-Agent/contact in PowerShell before the first fundamentals download:

```powershell
$env:SEC_USER_AGENT = "S&P500_AI your.name@example.com"
```

On Windows, the application also reads `SEC_USER_AGENT` from the saved current-user environment registry value, so save it once with `[Environment]::SetEnvironmentVariable(..., "User")` and reuse it for every ticker and dashboard training action.

`fundamentals --ticker AAPL` refreshes the selected company's SEC fundamentals in Convex. Each `train` action now attempts to refresh SEC facts first. If SEC is unavailable, training uses that ticker's existing cache when possible; otherwise it continues with price and market features and records a warning in the metrics. The downloader stores adjusted OHLCV prices. Existing CSVs with only close and volume are automatically recognized as an old schema and re-downloaded from 2000 on the next `update`. Training uses five expanding walk-forward validation folds with a five-session purge gap and nested parameter tuning. Its trading report makes the entry decision at the prior close using the predicted return minus round-trip costs; it executes at the next open and exits at the target session close. The next open is used only to calculate the realized return. It reports wins/losses, net win rate and a 95% Wilson interval, cumulative return, drawdown, Sharpe ratio, profit factor, and an always-long comparison. The interval is a rough binomial interval and does not account for dependence between adjacent trading days; it is not proof of future profitability. Do not tune the threshold on these same folds and treat the result as unbiased; reserve a later untouched period for final confirmation. The selected ticker's prediction export is stored in Convex; its fitted model and metrics are uploaded to `modelArtifacts`.

Historical news is not yet a model feature: the free historical-news probe returned HTTP 429, and current headlines cannot be used to train past periods without leakage. Add news only after configuring a reliable timestamped historical provider.

For prospective paper trading, first train a ticker model. After each market close, run `update` and then `paper-trade --ticker AAPL`. The command settles the previous pending forecast when the next session's open and close are present, then records the latest forecast as pending. Recent Yahoo Finance headlines are saved with publication timestamps in the ledger as context only. Simple explicit relationship phrases (such as acquires, invests in, or partners with) trigger a Wikipedia search for the named counterparty; the headline, extraction rule, search snippets, and source links are saved together. These are candidate relationships requiring review, not verified facts, and they are not model inputs or used in historical backtests. Paper histories are kept separately by model version; retraining starts a new ledger automatically and preserves the prior one. `paper-status --ticker AAPL` shows the ledger for the currently trained model. Re-running on the same signal date does not add duplicates.

Run `update` regularly, for example after the market close, to fetch new rows. It is incremental and also downloads full history for newly added constituents. Yahoo Finance/Wikipedia availability and rate limits can affect downloads; rerun `update` to retry. This uses today's constituent list for the entire history, so results have survivorship bias and do not represent historical S&P 500 membership. The model is an educational baseline, not investment advice.

## Tests

```powershell
python -m unittest discover -s tests
```

## Evaluation protocol

The simplified dashboard shows development results and does not request final holdout results. Use the CLI `--reveal-holdout` flag only after choosing a model and trading rule on development data. This is a visibility safeguard, not a reset of research history: periods already inspected are no longer fresh confirmation data.

`compare` and `compare-all` now include a zero-return forecast/no-trade baseline on the same validation sessions. Always-long net returns use the same costs and dates. Zero-return directional accuracy is omitted because a constant zero forecast has no up/down decision. Universe summaries report how many tickers beat baseline MAE and always-long returns; pooled trades are descriptive and do not establish independent statistical evidence.

A reproducible broader sample can be run with:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m sp500_ml compare-all --tickers AAPL MSFT NVDA AMD AMZN GOOGL META JPM BAC V MA UNH JNJ XOM CVX CAT PG KO WMT NEE AMT LIN
```

This fixed 22-stock sample spans sectors but is not a random sample or an unbiased historical universe. It excludes each ticker's latest 252 labeled sessions, uses five purged development folds with nested tuning, a zero entry threshold, and 10 bps cost per side. It uses saved price/market data and excludes SEC features for uniformity. The report records its protocol version, creation time, ticker list, dates, and baseline comparisons in `models/universe_intraday_comparison.json`. Running it replaces that comparison report; it does not replace production models or paper ledgers. Current-constituent survivorship bias remains.

For prospective validation, train once, record the chosen model/target/costs, then update prices and log paper signals after each close without retraining for at least 30 days. Keep the fingerprinted ledger intact. Thirty days is an operational minimum, not sufficient evidence of an edge; final assessment needs more observations and uncertainty estimates. Historical results and prospective paper results should be reviewed separately.

## Result integrity

New training runs write an immutable generation under `models/.runs/`, containing the model, development report, sealed holdout report, and holdout trades. A checksum manifest is replaced atomically only when the full set is written. All model readers resolve this manifest and reject missing or changed files. A failed publication keeps the previous complete generation selected. Legacy models/reports remain readable with a visible unverified-history label; retraining publishes a verified generation. Historical files are not deleted or silently converted into verified results.

Paper signals use the XNYS calendar from [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars). A new signal must be created after its signal session closes and strictly before the next session opens. Its UTC creation time and exact expected execution date are recorded. Late or stale signals are rejected; update data and wait for the next valid window. Settlement requires that exact session's completed prices and refuses to substitute a later date if data is missing. Weekends, exchange holidays, early closes, and daylight-saving changes are handled by the calendar.

Current paper results require a matching ticker, model type, fingerprint, timing, and signal rule. Returns are reconciled against entry decisions, execution prices, and costs; duplicates, invalid values, and contradictory totals fail instead of producing a performance figure. Ledger writes are atomic and concurrent paper operations for the same ticker/model are rejected. A process crash can leave a lock file: inspect its recorded process ID and confirm that the process has ended before manually removing an abandoned lock.

Dashboard state requests reject responses from previous selections or superseded refreshes, and failed requests clear their previous results. Development results are visible in stock-search results; final holdout disclosure remains available through the CLI. These checks guard against accidental inconsistency; they cannot certify upstream price accuracy, a machine's clock, or deliberate edits to both data and its checksums. Validation statistics assess the training procedure on held-out folds, not an out-of-sample rerun of the final model fitted to all available labeled history.

Regression checks:

```powershell
python -m unittest discover -s tests
node --test frontend/src/requestGate.test.mjs frontend/src/stockSearch.test.mjs
```
# Live prediction history with Convex

Stock searches automatically refresh the selected stock, SPY, sector ETFs, and VIX
before inference. Forecasts use the latest completed XNYS session, with a 30-minute
publication buffer after closing; intraday quotes are not substituted for daily
model inputs. Refreshes are reused for up to 60 seconds. If an input is delayed,
forecasts use the latest shared completed session and show its actual date.
An input with no valid data still fails explicitly.

The stock search can save prospective forecasts to Convex and show predicted versus
actual returns for each model and forecast period. See [CONVEX_SETUP.md](CONVEX_SETUP.md)
for deployment, server environment variables, and the hourly comparison worker.
This requires a configured Convex deployment; it does not fabricate or backfill
live results from historical data.

## Historical risk

Stock results include a separate historical risk panel with 1-, 3-, 5-, 10-year
or all-available history. It compares adjusted closing prices, maximum/current
drawdown, annualized daily-return volatility, and the worst daily return. Choose
a 10%, 20%, or 30% decline threshold; select a chart pin or episode to inspect
its peak, first declining session, threshold confirmation, trough, and recovery.
The five deepest qualifying episodes are shown. Episodes and onset dates are
identified retrospectively, not as signals available at the historical onset.

The market comparison shows the stock and SPY's largest decline, volatility, and
period return on exactly the same stock sessions. Missing or invalid SPY data
marks the comparison unavailable without hiding the stock's risk figures. A
drawdown chart overlays the percentage below each series' running peak; the stock
uses a solid line and SPY a dashed line. Sampled chart histories preserve their
true worst drawdown and price extremes. All calculations use existing stored
adjusted prices, with no new downloads or model changes.

Results use two tabs beneath a shared company/price header. **Forecast** is the
default, containing model/period choices, the forecast, and
collapsed **Model performance** details. **Historical risk** contains the decline
chart, risk controls, and decline/recovery details, plus a compact Declines / Price spikes switch. Only one view is visible at a time; view settings are preserved. The SPY comparison and extra chart are collapsed by default. Risk data loads on first opening its tab;
switching tabs preserves its controls and does not retrain or save another
prediction. Searching a new stock resets to Forecast. The tab browser regression
is `python -B -m tests.risk_browser_smoke --tabs`.

**Predictions for you** is a separate sidebar on the right (below the main card
on smaller screens). It lists the logged-in user's saved predictions across all
stocks, models, and forecast periods, newest first, with saved time and pending
or actual results. It loads without a stock search, refreshes after confirmed
saves, and offers manual refresh and pagination for older records. The read-only
`GET /api/predictions` forwards the verified Clerk token to Convex `predictions:mine`,
which derives ownership from `ctx.auth.getUserIdentity()` and reads the
`by_owner` index. No browser-provided owner is accepted; legacy ownerless records
remain excluded. Deploy the new query/index with `npm run convex:dev -- --once`.

Historical risk does not display news, summaries, source links, or publication-date
controls, and makes no news or analysis requests. Existing backend news services,
CLI tools, Finnhub configuration, and archived Convex news remain available
independently; removing news from this tab does not delete stored data.

`/api/risk` requires Clerk authentication. Regression checks:
`python -B -m unittest discover -s tests -p test_risk.py` and
`python -B -m tests.risk_browser_smoke --tabs` (isolated Edge browser check).

**Price spikes** is an on-demand view within **Historical risk** backed by authenticated `GET /api/spikes`. It reads adjusted daily prices from Convex and flags positive close-to-close returns of at least 5% and at least the prior 60-session mean plus three standard deviations (minimum 20 prior returns). The current session is excluded from its own baseline, and incomplete sessions are excluded entirely. All qualifying events remain accessible via Show more; selecting an event displays an interactive daily-price chart with up to 10 sessions on either side. This descriptive heuristic does not establish manipulation or analyze news, volume, social media, or order books. It scans the searched stock, rather than running a background universe-wide scan. No local data or prediction records are written. Tests: `python -B -m unittest tests.test_spikes` and the tab browser regression above.


**Spike research** appears below the selected spike chart in Historical risk > Price spikes. Authenticated
`POST /api/spikes/research` accepts `{"ticker":"AAPL","date":"2024-05-03"}`
and verifies that the date is a detected spike before fetching news. Results are
shared research, stored separately in Convex `spikeResearch`, never in personal
predictions or local files. Deploy the schema/functions with
`npm run convex:dev -- --once` before using the endpoint.

Run a bounded research batch from the trusted project terminal:

```powershell
.\.venv\Scripts\python.exe -B -m sp500_ml.spike_research NVDA --limit 5
.\.venv\Scripts\python.exe -B -m sp500_ml.spike_research AAPL --date 2024-05-03
```

The batch researches the largest historical spikes (maximum 25 per invocation),
stops on provider errors, and prints confirmed record IDs. Existing successful
results are reused for six hours; a failed refresh retains prior evidence.
Each record stores the price fingerprint, possible topic categories, publication
and retrieval timestamps, source links, brief provider-excerpt/headline evidence,
coverage status and qualitative evidence strength. Moderate means a focused
report explicitly links a stock rise with a matching topic; limited means a
related topic without that explicit link; unknown means insufficient evidence.
These are unverified associations, not established causes or calibrated
probabilities. Only company-focused text or clearly identified market-policy
context is considered. No arbitrary article URLs are fetched.

Publication timing is classified against actual exchange closes, including early
closes: before the previous close, during the close-to-close move, or afterward.
Coverage is limited to the previous session's calendar day (at most three days
back) through the calendar day after the spike. All rows have
`usableForTraining: false`: provider timestamps alone do not establish that an
article or its current wording was available historically. This stage does not
train models, estimate cause frequencies, or assign red flags.

A bounded live Finnhub probe on 2026-10-19 returned recent AAPL news, but no
articles for the tested May 2024 and October 2025 windows. Empty results are
coverage gaps, not proof that no news occurred. Confirm coverage per event.
Regression checks: `python -B -m unittest tests.test_spike_research` and
`node --test tests/convex_spike_research.test.mjs`.


**Primary evidence fallbacks (research method v2).** If Finnhub returns no
company-focused explanation, research now checks SEC current reports (8-K/6-K)
and attached company announcements. It follows date-range pointers into older
submissions files, rather than assuming the recent-filings list covers the event.
At most three older history files, three filings, and two exhibits per filing
are inspected; coverage records show whether a limit truncated the search.
Use your existing `SEC_USER_AGENT` identity (process environment, `.env.local`,
or Windows user setting). Missing configuration, network failures, and access
limits are recorded separately from empty search results. SEC URLs and linked
exhibits are constrained to SEC hosts and the selected issuer/accession folder;
redirects and arbitrary external links are not followed. Downloads are bounded
and requests are paced. Documents remain in memory; selected evidence is saved
only in Convex.

The initial direct company-archive adapter supports Apple: it searches each
relevant month and up to four archive pages, then reads at most four date-matched
announcements. It discovers release links from the archive rather than keeping
a list of known event URLs. Other issuers currently use the generic SEC/exhibit
path; their direct website archives require issuer-specific parsers.

New records include per-source `coverage` and one of `possible_explanation`,
`inconclusive` (documents found but no relevant topic), `no_documents_found`, or
`sources_unavailable`. A document dated near a spike remains context, not proof
of causation. SEC sources use the filing's acceptance timestamp, not an assumed
earlier company publication time. Date-only announcements retain their original
calendar date with `publicationPrecision: date` and `timing: unknown_time`;
no midnight timestamp is invented. All research remains excluded from training.
V1 records are preserved; v2 keys keep the new retrieval method separate.

Validation on 2026-10-19: Apple's archive supplied its May 2, 2024 earnings and
buyback release for the May 3 spike. SEC hosts failed DNS resolution from the
local machine; this is recorded as unavailable, not as no filings. Generic SEC
history/exhibit handling is covered by isolated fixtures. Tests:
`python -B -m unittest tests.test_primary_evidence tests.test_spike_research`.


The **Possible explanation** card loads research automatically for the selected
spike via the authenticated POST endpoint. It shows topic context, one short
source excerpt and dated link, and qualitative evidence strength. Additional
sources and provider coverage stay collapsed. Date-only publications are labeled
with unavailable exact time. Unknown evidence, network failures, and retries
remain separate from price/chart loading. Changing the selected event cancels
old requests and clears its explanation; successful responses must match the
stock/date and include a confirmed Convex record ID. Unsafe source link schemes
are omitted. Restart the Python backend after backend changes and refresh the
frontend to load the card. Browser regression:
`python -B -m tests.risk_browser_smoke --tabs`.


**Research v3: coverage and retention.** This supersedes the v2 retrieval limits
above. An event's context window now spans seven calendar days before the spike
(or its previous price session if earlier) through three days afterward. Sources
are labeled as earlier context, the move window, or later reporting. A report
about an older rally cannot receive an explicit-link score for the selected
spike; all rows remain excluded from predictive training. Topic rules now also
recognize earnings releases, company-focused pronoun references, upgrades and
strategic partnerships, while excluding acquisition-cost accounting footnotes
from deal explanations and using issuer article bodies instead of related news.

When the system resolver cannot resolve SEC hosts, an application-only fallback
uses Google's HTTPS DNS API. It accepts public IPv4 addresses only for the two
SEC hosts and preserves the original HTTP Host, TLS SNI, hostname verification,
and CA validation. Proxy settings and explicit access denials are respected;
Windows network settings are unchanged. Requests 2.32.2+ is required. Live SEC
submissions and exhibit retrieval succeeded on 2026-10-19 with this fallback.

Direct issuer support now includes Apple month archives, NVIDIA's date-filtered
financial-release archive, and Microsoft's quarterly earnings releases. Other
issuers continue through SEC. NVIDIA searches up to four financial archive pages
per relevant year and reads up to four matching releases. Microsoft verifies the
actual release dateline, not the fiscal period end. Caps and partial failures
remain visible in coverage; these adapters do not guarantee all news categories.

Retrieved headlines and issuer/SEC document text are retained in Convex's news
storage under `news/spike_evidence/<event-date>/<ticker>.json`. This includes
unmatched documents. Content versions and first retrieval timestamps are kept;
primary-document text is capped at 100,000 characters with an explicit truncation
flag. The archive is reclassified when research is refreshed and can supply
context even if a provider later loses historical coverage. Writes use the
existing cloud operation lock and never create local data files. The normal
six-hour research cache still prevents repeat downloads.

The UI's collapsed Sources & coverage section shows separate retrieved-document
and topic-match counts, the date window, and per-provider document counts. A
bounded live backfill retained 19 primary documents across AAPL 2024-05-03,
NVDA 2024-02-22, and MSFT 2023-04-26, with source-backed context in all three cases.
Run additional bounded backfills with the documented `spike_research --limit`
command, or select a spike in the UI. This is not a universe-wide completed scan.

Additional checks: `python -B -m unittest tests.test_sec_connection tests.test_spike_evidence`
and the existing primary-evidence, research, and tab browser regressions.

# Spike watch checklist

Price spikes opens historical charts, volume, pre-event activity, and explanations
directly. Current watch has been removed from the frontend, including its automatic
requests. The backend research implementation and existing Convex records remain
available; the frontend no longer generates watch snapshots.

The backend price/volume rule remains in research mode. SEC-verified insider purchases
are retained as context with partial coverage; catalyst data is Unknown. Neither changes
the alert score, and production caution alerts remain disabled. Historical
validation uses non-overlapping five-session trials, two chronological periods,
and resolved outcomes only. See [SPIKE_SYSTEM.md](SPIKE_SYSTEM.md) for the pipeline,
data contract, limitations, and requirements before enabling predictive warnings.

# Experimental hybrid forecast model

Select **Hybrid (experimental)** alongside Gradient Boosting and Ridge. It uses
the existing point-in-time features and both forecast periods. The third model
combines two separately fitted component forecasts; it does not duplicate features
or assume that the existing models have different feature sets.

At initial training, fit both components on the first part of the training window,
leave a five-session gap, and choose a convex blend on the later 20% (at least 20
rows) by return MAE. Refit the components on all completed training outcomes with
that chosen weight. A component's weight may be zero if calibration favors the
other model. Walk-forward and sealed-holdout evaluation retain the existing outer
time boundaries. `hybrid_calibration` records the split, gap, errors and weights.
Historical metrics evaluate this initial blend; they do not establish the performance
of the later personalized feedback mechanism.

**Learning from actual prediction errors:** when an authenticated hybrid forecast
is saved, its original Ridge and boosting predictions, blend weight, learning rule,
feedback count and feedback cutoff are stored in the user's Convex prediction row.
The existing comparison worker later settles the actual market outcome. On a new
search, the hybrid reads only that user's settled forecasts for the same ticker,
period and exact model fingerprint. Pending outcomes, future exits, old model
versions and overlapping outcome windows are excluded. At most 60 independent
outcomes from the latest 200 selection records are used.

Each component receives a score `exp(-MAE / mean_component_MAE)`. Normalize those
scores into a new Ridge weight and shrink it toward the original calibration weight
by `n / (n + 10)`. Zero feedback keeps initial weights. This is an experimental
heuristic, not a guarantee of improvement. It adjusts the ensemble weights; it does
not incrementally refit Ridge coefficients or boosting trees. Saved forecasts stay
immutable, and the frontend displays the exact confirmed saved blend if it changed
between loading and saving. Retraining creates a new model version with a fresh
feedback history. Earlier predictions remain available for audit.

The local dashboard already runs the outcome-comparison worker. Other deployments
need the existing worker (`python -m sp500_ml.prediction_store`) running separately.
Prediction recording is allowed only after the signal close and before the next
session opens. No synthetic, backfilled or historical test prediction is inserted
as a real user feedback observation. Without a signed-in user, CLI forecasts use
the initial calibrated blend, never another user's feedback.

```powershell
.\.venv\Scripts\python.exe -B -m sp500_ml train --ticker AAPL --model hybrid
.\.venv\Scripts\python.exe -B -m sp500_ml compare --ticker AAPL
```

Hybrid models, metrics and sealed reports use their own `*_hybrid_*` artifact
prefixes and immutable Convex training generations. Existing model files are not
replaced. To remove the experiment later, remove the hybrid factory/selector entry
and custom estimator together, while retaining historical prediction records.
No continuous background retraining or production performance claim is included.

The first real AAPL training smoke run selected a 50/50 initial blend. Development
return MAE was 1.0682%, versus 1.0519% for the zero-return baseline. This run did not
establish an advantage. It was saved separately in Convex; no historical predictions
were inserted into a user's live feedback history.

Checks: `python -B -m unittest tests.test_hybrid tests.test_prediction_store` and
`node --test tests/convex_predictions.test.mjs`, plus the existing full suite and
`python -B -m tests.risk_browser_smoke --tabs`.

# API Ninjas coverage trial

Set `API_NINJAS_API_KEY` in the server's `.env.local` (never a `VITE_` variable).
The Stock News endpoint requires paid access. Run:

```powershell
.\.venv\Scripts\python.exe -B -m sp500_ml.api_ninjas_trial
```

This evaluates 20 detected spikes: four dates spaced across each stock's available
history for AAPL, NVDA, MSFT, AMZN, and TSLA. It compares topic-matched headlines
with existing research, retains retrieved headlines through the Convex news
archive, and writes an incremental trial report under the cloud news dataset's
`spike_trials` key. No result files are stored locally. Each event makes at most
two news requests (100 headlines); capped results are marked truncated. The trial
stops on provider errors and retains completed report entries. Re-running performs
a new trial and can use additional quota. Headlines are not full article content,
topic matches do not establish causes, and these records are excluded from training.
The trial is separate from automatic frontend research until coverage is evaluated.


### Fixed-signal audit

`python -m sp500_ml.anomaly_evaluation` compares the unchanged Rise investigation signals against price-spike and single-signal baselines, with a purged chronological development/holdout split. It stores an isolated protocol, event dataset and report in Convex under `backtests/anomaly_evaluation/<run-id>/`. No forecasting model or UI is changed. This is a historical audit, not proof of future predictive value. See [SPIKE_SYSTEM.md](SPIKE_SYSTEM.md).

### Convex usage

See [CONVEX_USAGE.md](CONVEX_USAGE.md) for RAM-only compressed caching, scan batching and polling limits. Restart the backend after updating. These controls reduce future egress and database I/O; they do not reset existing usage or delete saved records.
