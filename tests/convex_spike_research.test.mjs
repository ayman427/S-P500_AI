import test from "node:test";
import assert from "node:assert/strict";
import {get, save} from "../convex/spikeResearch.js";

process.env.PREDICTION_STORE_SECRET = "research-test-secret";
const secret = process.env.PREDICTION_STORE_SECRET;
const record = {key: "AAPL:2024-05-03:test", ticker: "AAPL", eventDate: "2024-05-03", method: "dated_topic_rules_v1",
  priceFingerprint: "fingerprint", previousDate: "2024-05-02", previousClose: 100, close: 110, dailyReturn: .1,
  categories: [], confidence: "unknown", overview: "Unknown", providerStatus: "empty", providerMessage: "Coverage unverified",
  fetchedAt: "2026-10-19T12:00:00Z", windowFrom: "2024-05-02", windowTo: "2024-05-04", articlesChecked: 0, usableForTraining: false, sources: []};
function context() {
  const rows = [];
  return {rows, db: {
    query: table => {assert.equal(table, "spikeResearch"); return {withIndex: (_name, select) => {
      let key; select({eq: (_field, value) => {key = value;}});
      return {unique: async () => rows.find(row => row.key === key) ?? null};
    }};},
    insert: async (table, value) => {assert.equal(table, "spikeResearch"); rows.push({...value, _id: "id"}); return "id";},
    replace: async (id, value) => {rows[rows.findIndex(row => row._id === id)] = {...value, _id: id};},
  }};
}
test("service secret protects shared research reads and writes", async () => {
  await assert.rejects(get._handler(context(), {secret: "wrong", key: record.key}), /Unauthorized/);
  await assert.rejects(save._handler(context(), {secret: "wrong", record}), /Unauthorized/);
});
test("dedicated research table upserts without creating prediction records", async () => {
  const ctx = context();
  assert.equal((await save._handler(ctx, {secret, record})).created, true);
  assert.equal((await save._handler(ctx, {secret, record: {...record, providerMessage: "Updated"}})).created, false);
  assert.equal(ctx.rows.length, 1);
  assert.equal((await get._handler(ctx, {secret, key: record.key})).providerMessage, "Updated");
});
test("inconsistent prices and unsafe sources cannot enter the research dataset", async () => {
  await assert.rejects(save._handler(context(), {secret, record: {...record, dailyReturn: .9}}), /Invalid/);
  await assert.rejects(save._handler(context(), {secret, record: {...record, sources: [{url: "javascript:alert(1)"}]}}), /Invalid/);
});
