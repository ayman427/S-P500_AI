import test from "node:test";
import assert from "node:assert/strict";
import { commit, get, tableForKey, datasetLabels, acquireLock, renewLock, releaseLock } from "../convex/datasets.js";

process.env.PREDICTION_STORE_SECRET = "isolated-test-secret";
const args = {
  secret: "isolated-test-secret", key: "prices/AAA.csv", storageId: "upload",
  sha256: "ab".repeat(32), contentSha256: "cd".repeat(32), size: 100,
  modifiedAt: 1000, expectedVersion: null,
};
function context(current = null, hash = Buffer.from(args.sha256, "hex").toString("base64")) {
  const writes = [];
  return { writes, db: {
    system: { get: async () => ({sha256: hash}) },
    query: () => ({withIndex: () => ({unique: async () => current})}),
    insert: async (...value) => writes.push(value),
    replace: async (...value) => writes.push(value),
  }, storage: {delete: async () => {}, getUrl: async () => "https://file"},
  scheduler: {runAfter: async () => {}}, };
}
test("dataset commit validates the Convex base64 checksum and publishes a version", async () => {
  const ctx = context();
  const result = await commit._handler(ctx, args);
  assert.equal(result.version, 1);
  assert.equal(ctx.writes.length, 1);
  assert.equal(ctx.writes[0][0], "prices");
  assert.equal(result.ticker, "AAA");
});
test("data categories route to separate readable tables", () => {
  assert.equal(tableForKey("prices/AAPL.csv"), "prices");
  assert.equal(tableForKey("prices/AAPL.full_refresh"), "priceSync");
  assert.equal(tableForKey("fundamentals/AAPL.csv"), "fundamentals");
  assert.equal(tableForKey("fundamentals/company_tickers.json"), "companyMetadata");
  assert.equal(tableForKey("news/AAPL.json"), "news");
  assert.equal(tableForKey("models/.runs/AAPL/abc/model.joblib"), "modelArtifacts");
  assert.equal(tableForKey("backtests/results.json"), "backtestReports");
  assert.equal(tableForKey("paper_trades/AAPL.csv"), "paperLedgers");
  assert.equal(tableForKey("prediction_exports/predictions.csv"), "predictionExports");
  assert.deepEqual(datasetLabels("news/AAPL.json"), {ticker: "AAPL", format: "json"});
});
test("stale expected version cannot replace a concurrent dataset update", async () => {
  const ctx = context({_id: "existing", version: 3, contentSha256: "previous"});
  await assert.rejects(commit._handler(ctx, {...args, expectedVersion: 2}), /concurrently/);
  assert.equal(ctx.writes.length, 0);
});
test("corrupt uploaded data cannot be published", async () => {
  const ctx = context(null, "wrong");
  await assert.rejects(commit._handler(ctx, args), /checksum/);
  assert.equal(ctx.writes.length, 0);
});
test("unauthenticated dataset requests are denied", async () => {
  await assert.rejects(get._handler(context(), {secret: "incorrect", key: "prices/AAA.csv"}), /Unauthorized/);
});
test("an expired or missing cloud lock prevents publishing", async () => {
  const ctx = context();
  await assert.rejects(commit._handler(ctx, {...args, lease: {key: "model", owner: "owner"}}), /lock expired/);
  assert.equal(ctx.writes.length, 0);
});
test("an active cloud lock prevents a second writer", async () => {
  const ctx = context({owner: "first", expiresAt: Date.now() + 60000});
  await assert.rejects(acquireLock._handler(ctx, {secret: args.secret, key: "model", owner: "second"}), /Another operation/);
  assert.equal(ctx.writes.length, 0);
});

test("abandoned legacy scan leases recover after three minutes without changing other locks", async () => {
  const now = Date.now();
  const abandoned = {_id: "old-lock", owner: "old-worker", expiresAt: now + 7 * 60000};
  const ctx = context(abandoned);
  await acquireLock._handler(ctx, {secret: args.secret, key: "market-checklist", owner: "new-worker"});
  assert.equal(ctx.writes.length, 1);
  const value = ctx.writes[0][1];
  assert.equal(value.owner, "new-worker");
  assert.ok(value.heartbeatAt >= now);
  assert.equal(value.expiresAt - value.heartbeatAt, 180000);
  await assert.rejects(acquireLock._handler(context({...abandoned, expiresAt: now + 14 * 60000}),
    {secret: args.secret, key: "market-checklist", owner: "new-worker"}), /Another operation/);
  await assert.rejects(acquireLock._handler(context(abandoned), {secret: args.secret, key: "model", owner: "new-worker"}), /Another operation/);
});

test("fresh scan heartbeats block takeover and replaced owners cannot renew, release or write", async () => {
  const current = {_id: "live-lock", owner: "live-worker", heartbeatAt: Date.now(), expiresAt: Date.now() + 180000};
  const ctx = context(current);
  await assert.rejects(acquireLock._handler(ctx, {secret: args.secret, key: "market-checklist", owner: "other"}), /Another operation/);
  await assert.rejects(renewLock._handler(ctx, {secret: args.secret, key: "market-checklist", owner: "other"}), /lock lost/);
  await releaseLock._handler(ctx, {secret: args.secret, key: "market-checklist", owner: "other"});
  await assert.rejects(commit._handler(ctx, {...args, lease: {key: "market-checklist", owner: "other"}}), /lock expired/);
  assert.equal(ctx.writes.length, 0);
});

test("heartbeats renew the short scan lease and retain the lifetime of other operations", async () => {
  for (const [key, lifetime] of [["market-checklist", 180000], ["model", 900000]]) {
    const ctx = context({_id: "lock", owner: "worker", expiresAt: Date.now() + 60000});
    ctx.db.patch = async (id, value) => ctx.writes.push([id, value]);
    await renewLock._handler(ctx, {secret: args.secret, key, owner: "worker"});
    assert.equal(ctx.writes[0][0], "lock");
    assert.equal(ctx.writes[0][1].expiresAt - ctx.writes[0][1].heartbeatAt, lifetime);
  }
});
