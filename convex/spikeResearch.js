import { mutationGeneric as mutation, queryGeneric as query } from "convex/server";
import { v } from "convex/values";
import { spikeResearchFields } from "./schema.js";

function authorize(secret) {
  if (!process.env.PREDICTION_STORE_SECRET || secret !== process.env.PREDICTION_STORE_SECRET)
    throw new Error("Unauthorized spike research access");
}
export const get = query({
  args: {secret: v.string(), key: v.string()},
  handler: async (ctx, {secret, key}) => {
    authorize(secret);
    return await ctx.db.query("spikeResearch").withIndex("by_key", q => q.eq("key", key)).unique();
  },
});
export const save = mutation({
  args: {secret: v.string(), record: v.object(spikeResearchFields)},
  handler: async (ctx, {secret, record}) => {
    authorize(secret);
    if (!/^[A-Z0-9^][A-Z0-9^.-]{0,14}$/.test(record.ticker)
        || !/^\d{4}-\d{2}-\d{2}$/.test(record.eventDate)
        || record.usableForTraining !== false || record.sources.length > 5 || record.categories.length > 7
        || ![record.previousClose, record.close, record.dailyReturn].every(Number.isFinite)
        || record.previousClose <= 0 || record.close <= record.previousClose
        || Math.abs(record.close / record.previousClose - 1 - record.dailyReturn) > 1e-8
        || !Number.isFinite(Date.parse(record.fetchedAt))) throw new Error("Invalid spike research record");
    for (const source of record.sources) {
      const url = new URL(source.url);
      if (!["https:", "http:"].includes(url.protocol) || url.username || url.password
          || !Number.isFinite(Date.parse(source.publishedAt)) || source.summary.length > 350)
        throw new Error("Invalid research source");
    }
    const existing = await ctx.db.query("spikeResearch").withIndex("by_key", q => q.eq("key", record.key)).unique();
    if (existing) {
      await ctx.db.replace(existing._id, record);
      return {id: existing._id, created: false};
    }
    return {id: await ctx.db.insert("spikeResearch", record), created: true};
  },
});
