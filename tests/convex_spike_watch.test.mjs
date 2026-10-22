import test from "node:test";
import assert from "node:assert/strict";
import {save} from "../convex/spikeWatch.js";

process.env.PREDICTION_STORE_SECRET = "watch-test-secret";
const secret = process.env.PREDICTION_STORE_SECRET;
const fingerprint = "a".repeat(64);
const record = {key: `AAPL:2026-10-19:price_volume_v2:${fingerprint}`, ticker: "AAPL", asOf: "2026-10-19", expectedSession: "2026-10-19",
  evaluatedAt: "2026-10-19T22:00:00Z", priceFingerprint: fingerprint, ruleVersion: "price_volume_v2", ruleApproved: false,
  status: "incomplete", validation: {sessions: 500, alerts: 20, hits: 4, falseAlerts: 16, precision: .2, baselineRate: .1, baselineTrials: 100, precisionLowerBound: .05},
  checks: ["volume", "momentum", "breakout", "insiders", "catalyst"].map(id => ({id, active: ["volume", "momentum", "breakout"].includes(id), status: "unknown", value: null, availableAt: null}))};
function context() {
  const rows = [];
  return {rows, db: {
    query: table => {assert.equal(table, "spikeWatch"); return {withIndex: (_name, select) => {
      let key; select({eq: (_field, value) => {key = value;}});
      return {unique: async () => rows.find(row => row.key === key) ?? null};
    }};},
    insert: async (table, value) => {assert.equal(table, "spikeWatch"); rows.push({...value, _id: "id"}); return "id";},
    replace: async (id, value) => {rows[rows.findIndex(row => row._id === id)] = {...value, _id: id};},
  }};
}
test("watch writes require service authorization", async () => {
  await assert.rejects(save._handler(context(), {secret: "wrong", record}), /Unauthorized/);
});
test("watch snapshots are deduplicated in a separate table", async () => {
  const ctx = context();
  assert.equal((await save._handler(ctx, {secret, record})).created, true);
  assert.equal((await save._handler(ctx, {secret, record})).created, false);
  assert.equal(ctx.rows.length, 1);
});
test("unapproved cautions, bad counts, future evidence, and unknown-as-met are rejected", async () => {
  for (const candidate of [
    {...record, status: "caution"}, {...record, ruleApproved: true}, {...record, key: "wrong"},
    {...record, validation: {...record.validation, falseAlerts: 0}},
    {...record, checks: [{...record.checks[0], availableAt: "2026-10-20"}, ...record.checks.slice(1)]},
    {...record, checks: [{...record.checks[0], status: "met"}, ...record.checks.slice(1)]},
  ]) await assert.rejects(save._handler(context(), {secret, record: candidate}), /Invalid/);
});
test("insider context is partial, dated, and cannot change the inactive checklist item", async () => {
  const evidenceFingerprint = "b".repeat(64);
  const evidence = {status: "buying_observed", coverage: "partial", from: "2026-09-19", to: "2026-10-19T20:00:00+00:00", retrievedAt: "2026-10-19T22:00:00+00:00", buyers: 1, transactions: 1, candidatesReturned: 1, filingsChecked: 1,
    records: [{key: "purchase", insiderName: "Buyer", transactionDate: "2026-10-16", acceptedAt: "2026-10-19T19:00:00+00:00", shares: 100, filingUrl: "https://www.sec.gov/Archives/edgar/data/320193/000089924315001800/0000899243-15-001800-index.html"}]};
  const candidate = {...record, key: record.key + ":" + evidenceFingerprint, evidenceFingerprint, insiderEvidence: evidence};
  assert.equal((await save._handler(context(), {secret, record: candidate})).created, true);
  await assert.rejects(save._handler(context(), {secret, record: {...candidate, insiderEvidence: {...evidence, records: [{...evidence.records[0], acceptedAt: "2026-10-19T22:00:00+00:00"}]}}}), /Invalid/);
  await assert.rejects(save._handler(context(), {secret, record: {...candidate, insiderEvidence: {...evidence, coverage: "complete"}}}), /Invalid/);
});
