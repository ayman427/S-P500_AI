import test from "node:test";
import assert from "node:assert/strict";
import {save, latest} from "../convex/marketWatch.js";

process.env.PREDICTION_STORE_SECRET = "scan-test";
const secret = "scan-test";
const universe = Array.from({length: 400}, (_, i) => "A" + String.fromCharCode(65 + Math.floor(i / 26), 65 + i % 26));
const scan = {expectedSession: "2026-10-19", startedAt: "2026-10-19T22:00:00Z", updatedAt: "2026-10-19T22:01:00Z",
  ruleVersion: "price_volume_v2", universe, processed: 1, complete: false, unavailable: [],
  rows: [{ticker: universe[0], asOf: "2026-10-19", status: "watch", met: 3,
    checks: [{id: "volume", value: 2, status: "met"}, {id: "momentum", value: .03, status: "met"}, {id: "breakout", value: .01, status: "met"}]}]};
const lease = {key: "market-checklist", owner: "worker"};
function context() {
  let record = null;
  return {db: {query: table => ({withIndex: () => ({unique: async () => table === "operationLocks" ? {owner: "worker", expiresAt: Date.now() + 10000} : record})}),
    insert: async (table, value) => {assert.equal(table, "marketScans"); record = {...value, _id: "scan-id"}; return record._id;},
    replace: async (id, value) => {record = {...value, _id: id};}}};
}
test("shared scan is stored separately and requires service authorization and a live lease", async () => {
  const ctx = context();
  await assert.rejects(latest._handler(ctx, {secret: "wrong"}), /Unauthorized/);
  await assert.rejects(save._handler(ctx, {secret, scan, lease: {...lease, owner: "other"}}), /lease/);
  assert.equal((await save._handler(ctx, {secret, scan, lease})).id, "scan-id");
  assert.deepEqual(await latest._handler(ctx, {secret}), scan);
});
test("invalid coverage, fake scores, future bars and incomplete current checks are rejected", async () => {
  for (const candidate of [
    {...scan, processed: 2}, {...scan, complete: true}, {...scan, unavailable: [universe[0]]},
    {...scan, rows: [{...scan.rows[0], met: 2}]},
    {...scan, rows: [{...scan.rows[0], asOf: "2026-10-20"}]},
    {...scan, rows: [{...scan.rows[0], checks: scan.rows[0].checks.map(c => ({...c, value: 0}))}]},
    {...scan, rows: [{...scan.rows[0], checks: scan.rows[0].checks.map(c => ({...c, value: null, status: "unknown"})), met: 0}]},
  ]) await assert.rejects(save._handler(context(), {secret, scan: candidate, lease}), /Invalid/);
});
