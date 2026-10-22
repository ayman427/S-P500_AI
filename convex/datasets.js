import { mutationGeneric as mutation, queryGeneric as query, internalMutationGeneric } from "convex/server";
import { anyApi } from "convex/server";
import { v } from "convex/values";

function authorize(secret) {
  if (!process.env.PREDICTION_STORE_SECRET || secret !== process.env.PREDICTION_STORE_SECRET)
    throw new Error("Unauthorized dataset access");
}
function validateKey(key) {
  if ((!/^(prices|fundamentals|news|models|backtests|paper_trades|prediction_exports)\/[A-Za-z0-9^_./-]+$/.test(key) && key !== "companies.json") || key.split("/").includes(".."))
    throw new Error("Invalid dataset key");
}
export function tableForKey(key) {
  validateKey(key);
  if (key === "companies.json" || key === "fundamentals/company_tickers.json") return "companyMetadata";
  if (key.startsWith("models/")) return "modelArtifacts";
  if (key.startsWith("backtests/")) return "backtestReports";
  if (key.startsWith("paper_trades/")) return "paperLedgers";
  if (key.startsWith("prediction_exports/")) return "predictionExports";
  if (key.startsWith("prices/")) return key.endsWith(".full_refresh") ? "priceSync" : "prices";
  return key.startsWith("news/") ? "news" : "fundamentals";
}
export function datasetLabels(key) {
  const name = key.split("/").at(-1);
  const format = name.endsWith(".full_refresh") ? "timestamp" : name.includes(".") ? name.split(".").at(-1) : "text";
  return {format, ...(tableForKey(key) === "companyMetadata" ? {} : {ticker: name.replace(/\.(csv|json|full_refresh)$/, "")})};
}
export const uploadUrl = mutation({
  args: {secret: v.string()},
  handler: async (ctx, {secret}) => { authorize(secret); return await ctx.storage.generateUploadUrl(); },
});
export const get = query({
  args: {secret: v.string(), key: v.string()},
  handler: async (ctx, {secret, key}) => {
    authorize(secret); validateKey(key);
    const row = await ctx.db.query(tableForKey(key)).withIndex("by_key", q => q.eq("key", key)).unique();
    return row ? {...row, url: await ctx.storage.getUrl(row.storageId)} : null;
  },
});
export const list = query({
  args: {secret: v.string(), prefix: v.string(), table: v.union(v.literal("prices"), v.literal("fundamentals"), v.literal("news"), v.literal("companyMetadata"), v.literal("priceSync"), v.literal("modelArtifacts"), v.literal("backtestReports"), v.literal("paperLedgers"), v.literal("predictionExports")), cursor: v.union(v.null(), v.string())},
  handler: async (ctx, {secret, prefix, table, cursor}) => {
    authorize(secret);
    // Index range bounds select a directory without loading file contents.
    return await ctx.db.query(table).withIndex("by_key", q =>
      q.gte("key", prefix).lt("key", prefix + "\uffff")).paginate({numItems: 200, cursor});
  },
});
export const commit = mutation({
  args: {secret: v.string(), key: v.string(), storageId: v.id("_storage"),
    sha256: v.string(), contentSha256: v.string(), size: v.number(), modifiedAt: v.number(),
    expectedVersion: v.union(v.null(), v.number()),
    lease: v.optional(v.object({key: v.string(), owner: v.string()}))},
  handler: async (ctx, {secret, expectedVersion, lease, ...a}) => {
    authorize(secret); validateKey(a.key);
    if (lease) {
      const lock = await ctx.db.query("operationLocks").withIndex("by_key", q => q.eq("key", lease.key)).unique();
      if (!lock || lock.owner !== lease.owner || lock.expiresAt <= Date.now()) throw new Error("Cloud operation lock expired; write rejected");
    }
    const file = await ctx.db.system.get("_storage", a.storageId);
    const base64Hash = btoa(String.fromCharCode(...a.sha256.match(/../g).map(byte => parseInt(byte, 16))));
    if (!file || (file.sha256 !== a.sha256 && file.sha256 !== base64Hash)) throw new Error("Upload checksum mismatch");
    const table = tableForKey(a.key);
    const current = await ctx.db.query(table).withIndex("by_key", q => q.eq("key", a.key)).unique();
    if (current?.contentSha256 === a.contentSha256) {
      if (current.storageId !== a.storageId) await ctx.storage.delete(a.storageId);
      return current;
    }
    if ((current?.version ?? null) !== expectedVersion) {
      await ctx.storage.delete(a.storageId);
      throw new Error("Dataset changed concurrently; refresh and retry");
    }
    const value = {...a, ...datasetLabels(a.key), kind: table, version: (current?.version ?? 0) + 1};
    if (current) {
      await ctx.db.replace(current._id, value);
      await ctx.scheduler.runAfter(3600000, anyApi.datasets.removeUnused, {storageId: current.storageId, key: a.key});
    } else await ctx.db.insert(table, value);
    return value;
  },
});
export const acquireLock = mutation({
  args: {secret: v.string(), key: v.string(), owner: v.string()},
  handler: async (ctx, {secret, key, owner}) => {
    authorize(secret);
    const current = await ctx.db.query("operationLocks").withIndex("by_key", q => q.eq("key", key)).unique();
    const now = Date.now();
    // Legacy locks used a 15-minute TTL; recover abandoned scans without evicting live owners.
    const heartbeat = current?.heartbeatAt ?? (current ? current.expiresAt - 900000 : 0);
    if (current && current.expiresAt > now && (key !== "market-checklist" || now - heartbeat < 180000))
      throw new Error("Another operation holds this cloud lock; retry after it completes");
    const value = {key, owner, heartbeatAt: now, expiresAt: now + (key === "market-checklist" ? 180000 : 900000)};
    if (current) await ctx.db.replace(current._id, value); else await ctx.db.insert("operationLocks", value);
    return true;
  },
});
export const renewLock = mutation({
  args: {secret: v.string(), key: v.string(), owner: v.string()},
  handler: async (ctx, {secret, key, owner}) => {
    authorize(secret);
    const lock = await ctx.db.query("operationLocks").withIndex("by_key", q => q.eq("key", key)).unique();
    if (!lock || lock.owner !== owner || lock.expiresAt <= Date.now()) throw new Error("Cloud operation lock lost");
    const now = Date.now();
    await ctx.db.patch(lock._id, {heartbeatAt: now, expiresAt: now + (key === "market-checklist" ? 180000 : 900000)});
  },
});
export const releaseLock = mutation({
  args: {secret: v.string(), key: v.string(), owner: v.string()},
  handler: async (ctx, {secret, key, owner}) => {
    authorize(secret);
    const lock = await ctx.db.query("operationLocks").withIndex("by_key", q => q.eq("key", key)).unique();
    if (lock?.owner === owner) await ctx.db.delete(lock._id);
  },
});
export const removeUnused = internalMutationGeneric({
  args: {storageId: v.id("_storage"), key: v.string()},
  handler: async (ctx, {storageId, key}) => {
    const row = await ctx.db.query(tableForKey(key)).withIndex("by_key", q => q.eq("key", key)).unique();
    if (row?.storageId !== storageId) await ctx.storage.delete(storageId);
  },
});
