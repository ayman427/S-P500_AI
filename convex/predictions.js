import { mutationGeneric as mutation, queryGeneric as query, paginationOptsValidator } from "convex/server";
import { v } from "convex/values";
import { predictionFields } from "./schema.js";

function authorize(secret) {
  if (!process.env.PREDICTION_STORE_SECRET || secret !== process.env.PREDICTION_STORE_SECRET)
    throw new Error("Unauthorized prediction store access");
}
const selection = {ticker: v.string(), modelType: v.string(), targetKey: v.string()};
async function owner(ctx) {
  const identity = await ctx.auth.getUserIdentity();
  if (!identity || typeof identity.tokenIdentifier !== "string" || !identity.tokenIdentifier)
    throw new Error("Sign in to access your predictions");
  return identity.tokenIdentifier;
}
export const record = mutation({
  args: {secret: v.string(), prediction: v.object(predictionFields)},
  handler: async (ctx, {secret, prediction: p}) => {
    authorize(secret);
    const ownerId = await owner(ctx);
    const existing = await ctx.db.query("predictions").withIndex("by_owner_key", q => q.eq("ownerId", ownerId).eq("key", p.key)).unique();
    if (existing) return { id: existing._id, created: false };
    const now = Date.now();
    if (!(p.signalClose <= now && now < p.entryAt && p.entryAt < p.exitAt)
        || !Number.isFinite(p.predictedReturn)) throw new Error("Prediction is outside its prospective recording window");
    if (p.modelType === "hybrid") {
      const h = p.hybrid;
      if (!h || ![h.ridgeReturn, h.boostingReturn, h.ridgeWeight].every(Number.isFinite) || h.ridgeWeight < 0 || h.ridgeWeight > 1
          || !Number.isSafeInteger(h.feedbackSamples) || h.feedbackSamples < 0 || h.feedbackSamples > 60
          || (h.feedbackSamples === 0) !== (h.feedbackThrough === null)
          || (h.feedbackThrough !== null && (!Number.isFinite(h.feedbackThrough) || h.feedbackThrough > now))
          || Math.abs(p.predictedReturn - h.ridgeWeight*h.ridgeReturn - (1-h.ridgeWeight)*h.boostingReturn) > 1e-10)
        throw new Error("Invalid hybrid prediction provenance");
    }
    const id = await ctx.db.insert("predictions", {...p, ownerId, status: "pending"});
    return { id, created: true };
  },
});
export const hybridFeedback = query({
  args: {secret: v.string(), ticker: v.string(), targetKey: v.string(), modelHash: v.string(), key: v.string(), signalClose: v.number()},
  handler: async (ctx, {secret, ticker, targetKey, modelHash, key, signalClose}) => {
    authorize(secret);
    const ownerId = await owner(ctx);
    const existing = await ctx.db.query("predictions").withIndex("by_owner_key", q => q.eq("ownerId", ownerId).eq("key", key)).unique();
    const recent = await ctx.db.query("predictions").withIndex("by_owner_selection", q =>
      q.eq("ownerId", ownerId).eq("ticker", ticker).eq("modelType", "hybrid").eq("targetKey", targetKey)).order("desc").take(200);
    const observations = recent.filter(row => row.modelHash === modelHash && row.status === "settled" && row.hybrid
      && Number.isFinite(row.actualReturn) && row.exitAt <= signalClose && row.settledAt <= Date.now());
    return {existing: existing?.modelType === "hybrid" && existing?.modelHash === modelHash && existing?.ticker === ticker && existing?.targetKey === targetKey ? existing : null, observations};
  },
});
export const history = query({
  args: {secret: v.string(), ...selection},
  handler: async (ctx, {secret, ticker, modelType, targetKey}) => {
    authorize(secret);
    const ownerId = await owner(ctx);
    return await ctx.db.query("predictions").withIndex("by_owner_selection", q =>
      q.eq("ownerId", ownerId).eq("ticker", ticker).eq("modelType", modelType).eq("targetKey", targetKey)).order("desc").take(30);
  },
});
export const mine = query({
  args: {secret: v.string(), paginationOpts: paginationOptsValidator},
  handler: async (ctx, {secret, paginationOpts}) => {
    authorize(secret);
    const ownerId = await owner(ctx);
    return await ctx.db.query("predictions").withIndex("by_owner", q =>
      q.eq("ownerId", ownerId)).order("desc").paginate(paginationOpts);
  },
});
export const due = query({
  args: {secret: v.string()},
  handler: async (ctx, {secret}) => {
    authorize(secret);
    return await ctx.db.query("predictions").withIndex("by_due", q =>
      q.eq("status", "pending").lte("exitAt", Date.now() - 3600000)).take(100);
  },
});
export const settle = mutation({
  args: {secret: v.string(), id: v.id("predictions"), entryDate: v.string(), exitDate: v.string(), entryOpen: v.number(), exitClose: v.number()},
  handler: async (ctx, a) => {
    authorize(a.secret);
    const p = await ctx.db.get(a.id);
    if (!p) throw new Error("Unknown prediction");
    if (p.status === "settled") return p._id;
    if (Date.now() < p.exitAt + 3600000 || p.entryDate !== a.entryDate || p.exitDate !== a.exitDate
        || !Number.isFinite(a.entryOpen) || !Number.isFinite(a.exitClose) || a.entryOpen <= 0 || a.exitClose <= 0)
      throw new Error("Invalid or premature actual prices");
    const actualReturn = a.exitClose / a.entryOpen - 1;
    await ctx.db.patch(a.id, {status: "settled", entryOpen: a.entryOpen, exitClose: a.exitClose,
      actualReturn, absoluteError: Math.abs(actualReturn - p.predictedReturn),
      directionCorrect: Math.sign(actualReturn) === Math.sign(p.predictedReturn),
      settledAt: Date.now(), source: "Yahoo Finance via yfinance; adjusted daily OHLC"});
    return p._id;
  },
});
