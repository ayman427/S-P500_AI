import test from "node:test";
import assert from "node:assert/strict";
import { stockResponse, stockSearchError } from "./stockResponse.ts";

test("market input failure remains visible with its failing stage", async () => {
  const detail = "Latest completed prices for XLRE are unavailable (2026-10-15). Please try again shortly.";
  await assert.rejects(stockResponse(Response.json({error: detail}, {status: 502})), error => {
    assert.equal(stockSearchError(error, "AAPL", "refresh market prices"), `Could not refresh market prices for AAPL. ${detail}`);
    return true;
  });
});
test("proxy errors and connection failures explain how to recover", async () => {
  await assert.rejects(stockResponse(new Response("Bad gateway", {status: 502})), /HTTP 502.*backend/);
  assert.match(stockSearchError(new TypeError("Failed to fetch"), "AAPL", "load results"), /backend is running/);
});
test("session errors are preserved and corrupt artifacts remain hidden", () => {
  assert.match(stockSearchError(new Error("Your session expired. Sign in again."), "AAPL", "load results"), /session expired/);
  assert.equal(stockSearchError(new Error("Checksum mismatch"), "AAPL", "load results"), "Saved results could not be verified. No figures have been displayed.");
});
