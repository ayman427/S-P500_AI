Create and run the prediction database
======================================

Platform APIs now require Clerk authentication. Configure the Clerk issuer on
both Python and Convex using [CLERK_SETUP.md](CLERK_SETUP.md). Prediction records
are indexed by their verified owner; the service secret alone cannot access user
history or create user predictions without that user's Clerk token.

Convex is also the required source of prices, fundamentals, news, and company
metadata. Separate tables hold indexed metadata and SHA256 checksums:
`prices` (stock and market CSV histories), `fundamentals` (SEC event histories),
`news` (headlines and context), `companyMetadata` (company directory), and
`priceSync` (refresh timestamps). Records expose `ticker` and `format` for filtering.
Compressed dataset contents live in Convex File Storage. Forecast records remain
in the `predictions` table. The Python service downloads verified content into a bounded
32 MiB memory cache, never a disk cache. A missing connection or dataset raises an
error; runtime code does not fall back to the old `data` directory.

The existing datasets and model artifacts have been migrated and verified. The
one-time migration tools have been retired. Verification receipts remain in the
`backtestReports` table as `backtests/convex_data_migration.json`,
`backtests/convex_table_migration.json`, and `backtests/convex_artifact_migration.json`.
New installations connect to Convex directly; no local data folders are needed.

Runtime artifact storage uses `modelArtifacts`, `backtestReports`, `paperLedgers`,
and `predictionExports`. Model serialization happens in memory. Immutable model
generations retain per-file hashes and publish their manifest last. Renewable
`operationLocks` provide mutual exclusion and reject writes after a lease expires.
The CLI prediction export is a cloud dataset, not a local CSV. Logs go to the
terminal or hosting log collector, without redirecting them into the project.

Dataset uploads commit an atomic metadata pointer and require the expected
version, so concurrent commits fail rather than silently replacing one another.
Older storage objects are retired after an hour to allow active readers to finish.
Deployments share data only when they use the same CONVEX_URL and server secret.

From the project root, run `npm install` then `npm run convex:dev`.
Sign in to Convex and choose **Create a new project**, named `sp500-predictions`.
Keep the CLI running until it confirms functions are deployed. This creates the
deployment and generates local configuration; no account credentials belong in Git.

For the local database, keep `npm run convex:dev` running and run
`.venv\Scripts\python.exe setup_convex.py` once. It generates a secret, configures
Convex, and stores the server settings in ignored `.env.local`. The Python API and
worker read those settings automatically. Local storage persists on this computer;
the Convex dev process must run whenever the database is used.

For a hosted deployment, run `npx convex login`, then
`npx convex dev --configure new` and select a cloud deployment. Run the setup script
again for that deployment. Account sign-in requires the account owner's browser.

Alternatively, generate a random secret locally (at least 32 random bytes). Set the SAME value as
`PREDICTION_STORE_SECRET` in Convex Dashboard > Settings > Environment Variables
and in the Python server's environment. Set `CONVEX_URL` in the Python environment
to the deployment's **.convex.cloud** URL. Never use a `VITE_` variable for the secret.
Use a password manager or environment editor to avoid putting the secret in shell
history. A Convex deploy key is NOT the prediction-store secret.

Restart `dashboard_server.py` with both environment variables set. Searching a stock
records the server-generated forecast and loads the most recent 30 records for the
selected model and period. Browser requests cannot provide forecasts or actual
prices. Convex validates the recording window against its own clock and assigns
the immutable creation timestamp. Duplicate model/date/target records return the
original entry. Retrained models retain separate SHA256 identities.

The local dashboard runs a comparison worker immediately and every hour while the
server is running. It downloads adjusted Yahoo Finance daily prices, waits at least
one hour after the expected closing time, and uses only the exact entry and exit
sessions. Missing bars remain pending. Stored actual results are never overwritten.
For a serverless/Vercel deployment, run this worker on a persistent host separately:

```
.venv\Scripts\python.exe -m sp500_ml.prediction_store
```

Alternatively schedule `python -m sp500_ml.prediction_store --once` hourly with the
same environment variables. Vercel does not keep the local background worker alive.
Without a running worker, records remain pending; reopening the frontend alone
does not collect actual prices. No worker needs access to the trained model files.

Only predictions recorded after the signal session closes and before the next
session opens count as live forecasts. Stale saved data is deliberately rejected.
Refresh market data first; existing backtests are not imported as live results.
One-session predictions compare next session open to close; five-session predictions
compare next session open to fifth session close, using the XNYS holiday calendar.
Actual returns are before transaction costs, and absolute error is in percentage
points. Yahoo data can be delayed or revised; stored comparisons represent the
first valid adjusted prices collected, not a brokerage execution record.

For production deployment use `npm run convex:deploy` and configure both variables
for the production Python service and its worker as well as the production Convex
deployment. The trusted Python service/secret holder can write records; protect
that host and rotate the shared secret together on both sides when necessary.
