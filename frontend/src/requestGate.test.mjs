import assert from "node:assert/strict";
import test from "node:test";
import { createRequestGate } from "./requestGate.ts";

test("a late ticker response cannot replace the newly selected ticker", async () => {
  const gate = createRequestGate();
  gate.select("AAPL:ridge:open_to_close");
  const oldResponse = gate.begin("AAPL:ridge:open_to_close");
  gate.select("AMD:ridge:open_to_close");
  const newResponse = gate.begin("AMD:ridge:open_to_close");
  let displayed = "";
  await Promise.resolve().then(() => { if (newResponse()) displayed = "AMD"; });
  await Promise.resolve().then(() => { if (oldResponse()) displayed = "AAPL"; });
  assert.equal(displayed, "AMD");
  assert.equal(gate.begin("AAPL:ridge:open_to_close"), null);
});

test("only the most recent refresh of the same selection can update results", () => {
  const gate = createRequestGate();
  gate.select("AMD:ridge:open_to_close");
  const first = gate.begin("AMD:ridge:open_to_close");
  const second = gate.begin("AMD:ridge:open_to_close");
  assert.equal(first(), false);
  assert.equal(second(), true);
  gate.select("AMD:hist_gradient_boosting:open_to_close_5d");
  assert.equal(second(), false);
});
