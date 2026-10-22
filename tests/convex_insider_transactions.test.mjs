import test from "node:test";
import assert from "node:assert/strict";
import {save, recent} from "../convex/insiderTransactions.js";

process.env.PREDICTION_STORE_SECRET = "insider-test-secret";
const secret = process.env.PREDICTION_STORE_SECRET;
const accession = "0000899243-15-001800";
const base = "https://www.sec.gov/Archives/edgar/data/320193/000089924315001800/";
const record = {key: `AAPL:${accession}:0`, ticker: "AAPL", accession, cik: "320193", transactionIndex: 0,
  transactionDate: "2015-07-27", filingDate: "2015-07-28", acceptedAt: "2015-07-28T22:31:06+00:00", firstRetrievedAt: "2026-10-19T22:00:00+00:00",
  insiderIds: ["12345"], insiderName: "Example Buyer", position: "Director", shares: 100, price: 25, value: 2500,
  transactionCode: "P", form: "4", usableForTraining: false, filingUrl: base + accession + "-index.html", sourceUrl: base + "doc4.xml", filingFingerprint: "a".repeat(64)};
function context() {
  const rows = [];
  return {rows, db: {
    query: table => {assert.equal(table, "insiderTransactions"); return {withIndex: (name, select) => {
      const filters = {};
      const q = {eq: (field, value) => {filters[field] = value; return q;}, gte: (field, value) => {filters.min = value; return q;}, lte: (field, value) => {filters.max = value; return q;}};
      select(q);
      return {unique: async () => rows.find(row => row.key === filters.key) ?? null,
        order: () => ({take: async n => rows.filter(row => row.ticker === filters.ticker && row.acceptedAt >= filters.min && row.acceptedAt <= filters.max).slice(0, n)})};
    }};},
    insert: async (table, value) => {assert.equal(table, "insiderTransactions"); const id = `id-${rows.length}`; rows.push({...value, _id: id}); return id;},
    replace: async (id, value) => {rows[rows.findIndex(row => row._id === id)] = {...value, _id: id};},
  }};
}
test("insider reads and writes require service authorization", async () => {
  await assert.rejects(save._handler(context(), {secret: "wrong", records: [record]}), /Unauthorized/);
  await assert.rejects(recent._handler(context(), {secret: "wrong", ticker: "AAPL", from: "2015-07-01T00:00:00+00:00", to: "2015-07-29T20:00:00+00:00"}), /Unauthorized/);
});
test("verified rows deduplicate while preserving first retrieval and query cutoff", async () => {
  const ctx = context();
  const first = await save._handler(ctx, {secret, records: [record]});
  const second = await save._handler(ctx, {secret, records: [{...record, firstRetrievedAt: "2026-10-20T22:00:00+00:00"}]});
  assert.deepEqual(first.ids, second.ids);
  assert.equal(ctx.rows.length, 1);
  assert.equal(ctx.rows[0].firstRetrievedAt, record.firstRetrievedAt);
  assert.equal((await recent._handler(ctx, {secret, ticker: "AAPL", from: "2015-07-01T00:00:00+00:00", to: "2015-07-28T20:00:00+00:00"})).rows.length, 0);
  assert.equal((await recent._handler(ctx, {secret, ticker: "AAPL", from: "2015-07-01T00:00:00+00:00", to: "2015-07-29T20:00:00+00:00"})).rows.length, 1);
});
test("awards, amendments, invalid values and untrusted sources cannot be saved", async () => {
  for (const value of [{...record, transactionCode: "A"}, {...record, form: "4/A"}, {...record, value: 1}, {...record, sourceUrl: "http://127.0.0.1/private"}, {...record, acceptedAt: "2027-01-01T20:00:00+00:00"}])
    await assert.rejects(save._handler(context(), {secret, records: [value]}), /Invalid/);
});
