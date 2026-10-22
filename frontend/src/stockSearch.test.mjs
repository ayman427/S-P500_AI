import assert from "node:assert/strict";
import test from "node:test";
import { resolveStock, stockMatches } from "./stockSearch.ts";

const stocks = [
  { ticker: "AAPL", name: "Apple Inc." },
  { ticker: "NVDA", name: "NVIDIA CORP" },
  { ticker: "GOOG", name: "Alphabet Inc." },
  { ticker: "GOOGL", name: "Alphabet Inc." },
  { ticker: "BRK-B", name: "Berkshire Hathaway Inc." },
];

test("company names and symbols resolve to the same stock", () => {
  for (const query of ["apple", "APPLE", "Apple Inc.", "aapl", "Apple Inc. (AAPL)"]) {
    assert.equal(resolveStock(query, stocks)[0].ticker, "AAPL");
  }
  assert.equal(resolveStock("nvidia", stocks)[0].ticker, "NVDA");
  assert.equal(resolveStock("brk.b", stocks)[0].ticker, "BRK-B");
});

test("ambiguous company names keep all share classes for explicit choice", () => {
  assert.deepEqual(stockMatches("google", stocks).map(stock => stock.ticker), ["GOOG", "GOOGL"]);
  assert.equal(resolveStock("Alphabet", stocks).length, 2);
  assert.equal(resolveStock("GOOGL", stocks)[0].ticker, "GOOGL");
});

test("unknown names do not silently resolve to a different company", () => {
  assert.deepEqual(resolveStock("unknown business", stocks), []);
  assert.deepEqual(resolveStock("", stocks), []);
  assert.equal(resolveStock("ZZZZ", stocks)[0].ticker, "ZZZZ");
});
