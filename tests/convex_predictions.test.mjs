import test from "node:test";
import assert from "node:assert/strict";
import { record, history, mine, hybridFeedback } from "../convex/predictions.js";

process.env.PREDICTION_STORE_SECRET = "isolated-test-secret";
const secret = process.env.PREDICTION_STORE_SECRET;
const selection = {ticker: "AMD", modelType: "ridge", targetKey: "open_to_close"};
const prediction = {...selection, key: "same-model-and-session", modelHash: "hash", signalDate: "2026-10-15",
  predictedReturn: 0.02, trainedThrough: "2026-10-14", signalClose: Date.now() - 1000,
  entryDate: "2026-10-16", exitDate: "2026-10-16", entryAt: Date.now() + 60000, exitAt: Date.now() + 120000};
function context(user, rows) {
  return {
    auth: {getUserIdentity: async () => user ? {tokenIdentifier: `https://clerk.example|${user}`} : null},
    db: {
      query: () => ({withIndex: (_name, select) => {
        const filters = [];
        const q = {eq: (key, value) => {filters.push([key, value]); return q;}};
        select(q);
        const matching = () => rows.filter(row => filters.every(([key, value]) => row[key] === value));
        return {unique: async () => matching()[0] ?? null, order: () => ({
          take: async n => matching().slice(0, n),
          paginate: async ({numItems, cursor}) => {
            const ordered = matching().sort((a,b) => b._creationTime - a._creationTime);
            const start = Number(cursor ?? 0), end = start + numItems;
            return {page: ordered.slice(start, end), isDone: end >= ordered.length, continueCursor: String(end)};
          },
        })};
      }}),
      insert: async (_table, value) => { const _id = `prediction-${rows.length}`; rows.push({...value, _id}); return _id; },
    },
  };
}
test("anonymous callers cannot save or read prediction history", async () => {
  await assert.rejects(record._handler(context(null, []), {secret, prediction}), /Sign in/);
  await assert.rejects(history._handler(context(null, []), {secret, ...selection}), /Sign in/);
  await assert.rejects(mine._handler(context(null, []), {secret, paginationOpts: {cursor: null, numItems: 20}}), /Sign in/);
});
test("two users get separate records and histories for the same forecast", async () => {
  const rows = [{...prediction, _id: "legacy"}];
  const alice = context("alice", rows), bob = context("bob", rows);
  const a = await record._handler(alice, {secret, prediction});
  const b = await record._handler(bob, {secret, prediction});
  assert.notEqual(a.id, b.id);
  assert.equal((await history._handler(alice, {secret, ...selection}))[0]._id, a.id);
  assert.equal((await history._handler(bob, {secret, ...selection}))[0]._id, b.id);
  assert.equal((await history._handler(alice, {secret, ...selection})).length, 1);
  assert.equal((await record._handler(alice, {secret, prediction})).created, false);
});
test("a submitted owner cannot override the verified identity", async () => {
  const rows = [];
  await record._handler(context("alice", rows), {secret, prediction: {...prediction, ownerId: "bob"}});
  assert.equal(rows[0].ownerId, "https://clerk.example|alice");
  assert.equal((await history._handler(context("bob", rows), {secret, ...selection})).length, 0);
});
test("a signed-in browser without the service credential cannot inject predictions", async () => {
  await assert.rejects(record._handler(context("alice", []), {secret: "incorrect", prediction}), /Unauthorized/);
});

test("personal history paginates across stocks and models without exposing another user or legacy data", async () => {
  const rows = [
    {...prediction, _id: "older", ownerId: "https://clerk.example|alice", _creationTime: 1},
    {...prediction, _id: "newer", ticker: "AAPL", modelType: "hist_gradient_boosting", targetKey: "open_to_close_5d", ownerId: "https://clerk.example|alice", _creationTime: 3},
    {...prediction, _id: "bob", ownerId: "https://clerk.example|bob", _creationTime: 4},
    {...prediction, _id: "legacy", _creationTime: 5},
  ];
  const alice = context("alice", rows);
  const first = await mine._handler(alice, {secret, paginationOpts: {cursor: null, numItems: 1}});
  assert.equal(first.page[0]._id, "newer");
  assert.equal(first.isDone, false);
  const next = await mine._handler(alice, {secret, paginationOpts: {cursor: first.continueCursor, numItems: 1}});
  assert.equal(next.page[0]._id, "older");
  assert.equal(next.isDone, true);
  await assert.rejects(mine._handler(alice, {secret: "incorrect", paginationOpts: {cursor: null, numItems: 20}}), /Unauthorized/);
});
test("hybrid learning reads only the owner's settled outcomes from the same model version", async () => {
  const hybrid = {...prediction, modelType: "hybrid", status: "settled", actualReturn: .01, exitAt: 50, settledAt: 100,
    hybrid: {ridgeReturn: .01, boostingReturn: .03, ridgeWeight: .5, feedbackSamples: 0, feedbackThrough: null, learningRule: "settled_error_v1"}};
  const rows = [
    {...hybrid, _id: "alice", ownerId: "https://clerk.example|alice"},
    {...hybrid, _id: "bob", ownerId: "https://clerk.example|bob"},
    {...hybrid, _id: "pending", ownerId: "https://clerk.example|alice", status: "pending"},
    {...hybrid, _id: "old-model", ownerId: "https://clerk.example|alice", modelHash: "old"},
    {...hybrid, _id: "future", ownerId: "https://clerk.example|alice", exitAt: Date.now() + 10000},
  ];
  const args = {secret, ticker: "AMD", targetKey: "open_to_close", modelHash: "hash", key: "new-session", signalClose: Date.now()};
  const result = await hybridFeedback._handler(context("alice", rows), args);
  assert.deepEqual(result.observations.map(row => row._id), ["alice"]);
  await assert.rejects(hybridFeedback._handler(context(null, rows), args), /Sign in/);
});
test("hybrid prediction records must preserve the blend and its component provenance", async () => {
  const hybrid = {...prediction, modelType: "hybrid", hybrid: {ridgeReturn: .01, boostingReturn: .03, ridgeWeight: .5, feedbackSamples: 0, feedbackThrough: null, learningRule: "settled_error_v1"}};
  assert.equal((await record._handler(context("alice", []), {secret, prediction: hybrid})).created, true);
  await assert.rejects(record._handler(context("alice", []), {secret, prediction: {...hybrid, predictedReturn: .5}}), /Invalid hybrid/);
  await assert.rejects(record._handler(context("alice", []), {secret, prediction: {...hybrid, hybrid: undefined}}), /Invalid hybrid/);
});
