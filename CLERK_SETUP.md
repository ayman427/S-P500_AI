Clerk authentication
====================

The application requires sign-in before accessing any platform API. Clerk manages
accounts and sessions. Python validates the JWT signature against the configured
issuer's JWKS, expiry, not-before time, audience `convex`, authorized frontend
origin (`azp`), and completed-session status. Convex independently validates the
forwarded token and derives prediction ownership from `ctx.auth.getUserIdentity()`.
The browser cannot choose an owner or supply arbitrary forecast values.

The development Clerk issuer configured for this project is
`https://amazing-elk-6490.clerk.accounts.dev`.

1. In that Clerk application's Dashboard, activate the **Convex integration**.
   This makes the normal session token include `aud: "convex"`; no custom JWT
   template is needed. Enable the desired sign-in/sign-up methods in Clerk.
   If integration settings change during an existing session, the frontend refreshes
   its token before retrying. Older Clerk setups using a JWT template named `convex`
   are also supported. Both token types still require server-side signature,
   issuer, audience, expiry, and authorized-origin verification.
2. Set these values in ignored root `.env.local` (already configured for the
   current development application):

   ```dotenv
   VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_public_key
   CLERK_JWT_ISSUER_DOMAIN=https://your-app.clerk.accounts.dev
   CLERK_AUTHORIZED_PARTIES=http://127.0.0.1:5173,http://localhost:5173
   ```

3. Configure the same issuer on the matching Convex deployment and deploy:

   ```powershell
   node node_modules/convex/bin/main.js env set CLERK_JWT_ISSUER_DOMAIN https://your-app.clerk.accounts.dev
   npm run convex:dev -- --once
   ```

4. Install updated dependencies and restart both servers:

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   npm --prefix frontend install
   .venv\Scripts\python.exe -B dashboard_server.py
   # In another terminal:
   npm --prefix frontend run dev
   ```

Vite reads root `.env.local`; only `VITE_` variables are exposed to the browser.
Never prefix `PREDICTION_STORE_SECRET` or a Clerk secret key with `VITE_`.
A Clerk secret key is not needed for this integration.

Data ownership
--------------

- `predictions`: new records contain a verified `ownerId` (issuer + Clerk user
  identity). Both duplicate checks and history queries use an owner-prefixed index.
  Two users can save the same stock/model/session independently. Only that user's
  history is returned; identity cannot be set through request bodies.
- Existing ownerless predictions are preserved as legacy service-only records.
  They are not assigned to the first person who signs in and are not visible in
  user histories. Any future import must explicitly establish their real owner.
- Prices, fundamentals, news, company metadata, trained model artifacts, and
  historical model evaluations are shared platform resources. The web API requires
  authentication even for these shared resources; Convex dataset functions require
  the server credential and are never exposed directly to the browser.
- Old CLI paper ledgers and prediction exports are service-only archives. Web
  responses exclude legacy paper history, and bulk update/paper-trading/comparison
  actions are restricted to the trusted server CLI. The user interface does not
  expose those archives or let one user modify another user's ledger.
- The settlement worker uses the service credential to reconcile pending records
  for all users. It cannot change a record's owner or original forecast.

Signing out unmounts the stock-search UI and aborts its pending requests. A different
user or session mounts a fresh instance, clearing in-memory result/history state.
Saved model/period preferences are scoped to the Clerk user ID; the color theme
remains a browser preference.
Tokens are fetched from Clerk for each request; they are not saved in project files
or custom localStorage keys. API responses are marked `no-store`.

Sharing the development app through a tunnel
------------------------------------------

Add the exact HTTPS frontend origin (no trailing slash) to
`CLERK_AUTHORIZED_PARTIES` in root `.env.local`, keeping the localhost entries.
Vite uses the same setting to allow that hostname. Restart the frontend after
changing it, and restart the backend if it has an explicit environment override.
Open the tunnel URL and sign in there; a localhost session is separate.
Forward port 5173: its `/api` proxy reaches the local backend on port 8765, so a
second public backend tunnel is unnecessary. Your friend must also be allowed
through the tunnel provider's access controls. When a tunnel URL changes, replace
the old origin with the new exact origin. Do not allow every `devtunnels.ms` host.

Production deployment
---------------------

Use the production Clerk application, public key, issuer, and production Convex
deployment together. Set `CLERK_AUTHORIZED_PARTIES` to the exact deployed frontend
origin(s), including HTTPS, with no trailing slash. The local defaults deliberately
do not authorize arbitrary Vercel preview origins. Configure the Convex integration
in the production Clerk app too. Add the frontend domain in Clerk as required by
your Clerk application settings. There is no unauthenticated development bypass.

Run `python -B -m unittest discover -s tests` and
`node --test tests/convex_predictions.test.mjs tests/convex_datasets.test.mjs` for
token rejection and ownership isolation checks. `tests/browser_smoke.py` checks the
signed-out UI using a fresh temporary browser profile without creating an account.
