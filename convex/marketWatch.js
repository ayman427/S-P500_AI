import {mutationGeneric as mutation, queryGeneric as query} from "convex/server";
import {v} from "convex/values";
import {marketScanFields} from "./schema.js";

function authorize(secret) {
  if (!process.env.PREDICTION_STORE_SECRET || secret !== process.env.PREDICTION_STORE_SECRET)
    throw new Error("Unauthorized market scan access");
}
const current = ctx => ctx.db.query("marketScans").withIndex("by_key", q => q.eq("key", "latest")).unique();
export const latest = query({args: {secret: v.string()}, handler: async (ctx, {secret}) => {
  authorize(secret);
  return (await current(ctx))?.scan ?? null;
}});

export const save = mutation({args: {secret: v.string(), scan: v.object(marketScanFields),
  lease: v.object({key: v.string(), owner: v.string()})}, handler: async (ctx, {secret, scan, lease}) => {
  authorize(secret);
  const lock = await ctx.db.query("operationLocks").withIndex("by_key", q => q.eq("key", lease.key)).unique();
  if (lease.key !== "market-checklist" || !lock || lock.owner !== lease.owner || lock.expiresAt <= Date.now())
    throw new Error("Market scan lease expired");
  const day = value => /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
  const symbols = [...scan.rows.map(row => row.ticker), ...scan.unavailable];
  if (!day(scan.expectedSession) || !Number.isFinite(Date.parse(scan.startedAt)) || !Number.isFinite(Date.parse(scan.updatedAt))
      || Date.parse(scan.updatedAt) < Date.parse(scan.startedAt) || scan.universe.length < 400 || scan.universe.length > 600
      || new Set(scan.universe).size !== scan.universe.length || scan.universe.some(t => !/^[A-Z][A-Z0-9.-]{0,14}$/.test(t))
      || !Number.isSafeInteger(scan.processed) || scan.processed < 1 || scan.processed > scan.universe.length
      || scan.complete !== (scan.processed === scan.universe.length) || symbols.length !== scan.processed
      || new Set(symbols).size !== symbols.length || symbols.some(t => !scan.universe.slice(0, scan.processed).includes(t))
      || scan.rows.some(row => !day(row.asOf) || row.asOf > scan.expectedSession
        || row.checks.map(c => c.id).sort().join(",") !== "breakout,momentum,volume"
        || row.checks.some(c => c.value === null ? c.status !== "unknown" : !Number.isFinite(c.value)
          || c.status !== (c.id === "breakout" ? (c.value > 0 ? "met" : "not_met") : c.value >= (c.id === "volume" ? 1.5 : .02) ? "met" : "not_met"))
        || row.met !== row.checks.filter(c => c.status === "met").length
        || ((row.status === "watch" || row.status === "no_signal") && (row.asOf !== scan.expectedSession || row.checks.some(c => c.status === "unknown")))
        || (row.status === "watch" && row.met !== 3) || (row.status === "no_signal" && row.met === 3)))
    throw new Error("Invalid market scan");
  const old = await current(ctx);
  if (old && (old.scan.expectedSession > scan.expectedSession || Date.parse(old.scan.updatedAt) > Date.parse(scan.updatedAt)))
    throw new Error("Outdated market scan write");
  const value = {key: "latest", scan};
  const id = old ? (await ctx.db.replace(old._id, value), old._id) : await ctx.db.insert("marketScans", value);
  return {id};
}});
