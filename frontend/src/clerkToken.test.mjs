import test from "node:test";
import assert from "node:assert/strict";
import { getConvexToken } from "./clerkToken.ts";

const token = claims => `header.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.signature`;
test("current Convex integration token is reused", async () => {
  const current = token({aud: "convex"});
  let count = 0;
  assert.equal(await getConvexToken(async () => {count++; return current;}), current);
  assert.equal(count, 1);
});
test("a cached pre-integration session gets refreshed before using a template", async () => {
  const calls = [];
  const refreshed = token({aud: "convex"});
  const actual = await getConvexToken(async options => { calls.push(options); return options.skipCache ? refreshed : token({sub: "user"}); });
  assert.equal(actual, refreshed);
  assert.deepEqual(calls, [{skipCache: false}, {skipCache: true}]);
});
test("legacy installations obtain the named convex JWT template", async () => {
  const templated = token({aud: "convex"});
  const actual = await getConvexToken(async options => options.template === "convex" ? templated : token({sub: "user"}));
  assert.equal(actual, templated);
});
test("missing sessions and unavailable integration fail without sending an invalid token", async () => {
  await assert.rejects(getConvexToken(async () => null), /sign in again/i);
  await assert.rejects(getConvexToken(async options => { if (options.template) throw new Error("missing"); return token({}); }), /Convex integration/);
});
