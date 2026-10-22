import {mutationGeneric as mutation, queryGeneric as query} from "convex/server";
import {v} from "convex/values";
import {anomalyObservationFields} from "./schema.js";

function authorize(secret) {
  if (!process.env.PREDICTION_STORE_SECRET || secret !== process.env.PREDICTION_STORE_SECRET) throw new Error("Unauthorized anomaly tracking access");
}
async function sha256(text) {
  const bytes = await crypto.subtle.digest("SHA-256",new TextEncoder().encode(text));
  return Array.from(new Uint8Array(bytes),x => x.toString(16).padStart(2,"0")).join("");
}
const hashOk = value => /^[a-f0-9]{64}$/.test(value);
const dateOk = value => /^\d{4}-\d{2}-\d{2}$/.test(value) && new Date(value).toISOString().slice(0,10) === value;
async function protocol(ctx, key) {
  return ctx.db.query("anomalyProtocols").withIndex("by_key", q => q.eq("key", key)).unique();
}
async function observation(ctx, key) {
  return ctx.db.query("anomalyObservations").withIndex("by_key", q => q.eq("key", key)).unique();
}
export const register = mutation({
  args: {secret:v.string(), key:v.string(), hash:v.string(), spec:v.string()},
  handler: async (ctx, {secret,key,hash,spec}) => {
    authorize(secret);
    if (key !== "price_volume_prospective_v1" || !hashOk(hash) || spec.length > 20000) throw new Error("Invalid protocol");
    const settings = JSON.parse(spec);
    if (!Array.isArray(settings.tickers) || !settings.tickers.length || settings.tickers.length > 500 || !settings.tickers.every(t => /^[A-Z0-9][A-Z0-9.-]{0,14}$/.test(t))
      || settings.method !== "price_volume_observations_v1" || settings.horizonSessions !== 20 || settings.declineThreshold !== -.1 || settings.multiSignalMinimum !== 2
      || settings.volumeThreshold !== 2 || settings.marketThreshold !== .03 || settings.accelerationThreshold !== .05) throw new Error("Invalid frozen settings");
    if (await sha256(spec) !== hash) throw new Error("Invalid protocol hash");
    const existing = await protocol(ctx,key);
    if (existing) {
      if (existing.hash !== hash || existing.spec !== spec) throw new Error("Frozen protocol mismatch; rules cannot silently change");
      return existing;
    }
    const row = {key,hash,spec,frozenAt:Date.now()};
    await ctx.db.insert("anomalyProtocols",row);
    return row;
  },
});
export const capture = mutation({
  args: {secret:v.string(), record:v.object(anomalyObservationFields)},
  handler: async (ctx, {secret,record:r}) => {
    authorize(secret);
    const p = await protocol(ctx,r.protocol);
    const now = Date.now();
    if (!p || p.hash !== r.protocolHash || !JSON.parse(p.spec).tickers.includes(r.ticker) || !/^[A-Z0-9][A-Z0-9.-]{0,14}$/.test(r.ticker)
      || !dateOk(r.eventDate) || !dateOk(r.dueDate) || r.key !== `${r.protocol}:${r.ticker}:${r.eventDate}`
      || ![r.closeAt,r.nextOpenAt,r.dueAt,r.eventClose].every(Number.isFinite) || r.eventClose <= 0
      || p.frozenAt > r.closeAt || now < r.closeAt || now >= r.nextOpenAt
      || r.dueAt < r.closeAt + 20*86400000 || r.dueAt > r.closeAt + 45*86400000
      || new Date(r.closeAt).toISOString().slice(0,10) !== r.eventDate
      || new Date(r.dueAt).toISOString().slice(0,10) !== r.dueDate
      || !hashOk(r.inputSha256) || r.inputSnapshot.length > 100000) throw new Error("Invalid or late prospective observation");
    if (await sha256(r.inputSnapshot) !== r.inputSha256) throw new Error("Invalid snapshot hash");
    const input = JSON.parse(r.inputSnapshot);
    if (!Array.isArray(input.stock) || input.stock.length < 21 || input.stock.length > 62 || !Array.isArray(input.market) || input.market.length !== 2
      || input.stock.at(-1).date !== r.eventDate || input.stock.at(-1).close !== r.eventClose
      || !input.stock.every((row,i) => dateOk(row.date) && row.date <= r.eventDate && Number.isFinite(row.close) && row.close > 0
        && (row.volume === null || (Number.isFinite(row.volume) && row.volume >= 0)) && (!i || input.stock[i-1].date < row.date))
      || !input.market.every(row => dateOk(row.date) && row.date <= r.eventDate && (row.close === null || (Number.isFinite(row.close) && row.close > 0)))) throw new Error("Invalid input snapshot");
    if (r.signals.length !== 4 || new Set(r.signals.map(s => s.id)).size !== 4 || !r.signals.every(s =>
      ["price","market","volume","acceleration"].includes(s.id) && [s.value,s.threshold].every(x => x === null || Number.isFinite(x))
      && s.status === (s.value === null || s.threshold === null ? "unavailable" : s.value >= s.threshold-1e-12 ? "elevated" : "within_rule"))
      || r.signals.find(s => s.id === "price").status !== "elevated") throw new Error("Invalid measured signals");
    if (input.market[0].date !== input.stock.at(-2).date || input.market[1].date !== r.eventDate) throw new Error("Unaligned market inputs");
    for (const [id,threshold] of [["market",.03],["volume",2],["acceleration",.05]]) {
      if (r.signals.find(s => s.id === id).threshold !== threshold) throw new Error("Changed frozen threshold");
    }
    const additional = r.signals.filter(s => s.id !== "price");
    const expected = additional.some(s => s.status === "unavailable") ? null : additional.filter(s => s.status === "elevated").length >= 2;
    if (r.flagged !== expected) throw new Error("Invalid multi-signal flag");
    const existing = await observation(ctx,r.key);
    if (existing) return {id:existing._id,created:false};
    return {id:await ctx.db.insert("anomalyObservations",{...r,observedAt:now,status:"pending"}),created:true};
  },
});
export const due = query({
  args:{secret:v.string(), cursor:v.union(v.string(),v.null())},
  handler: async (ctx,{secret,cursor}) => {
    authorize(secret);
    return ctx.db.query("anomalyObservations").withIndex("by_due",q => q.eq("status","pending").lte("dueAt",Date.now())).paginate({numItems:100,cursor});
  },
});
export const settle = mutation({
  args:{secret:v.string(),key:v.string(),outcomeSnapshot:v.string(),outcomeSha256:v.string()},
  handler: async (ctx,{secret,key,outcomeSnapshot,outcomeSha256}) => {
    authorize(secret);
    const r = await observation(ctx,key);
    if (!r) throw new Error("Observation not found");
    if (r.status === "settled") return {id:r._id,settled:false};
    if (Date.now() < r.dueAt || !hashOk(outcomeSha256) || outcomeSnapshot.length > 50000) throw new Error("Outcome is not due or invalid");
    if (await sha256(outcomeSnapshot) !== outcomeSha256) throw new Error("Invalid outcome hash");
    const rows = JSON.parse(outcomeSnapshot);
    if (!Array.isArray(rows) || rows.length !== 21 || rows[0].date !== r.eventDate || rows.at(-1).date !== r.dueDate
      || !rows.every((row,i) => dateOk(row.date) && Number.isFinite(row.close) && row.close > 0 && (!i || rows[i-1].date < row.date))) throw new Error("Incomplete outcome window");
    const anchor = rows[0].close;
    const largestDecline = Math.min(0,...rows.slice(1).map(row => row.close/anchor-1));
    await ctx.db.patch(r._id,{status:"settled",settledAt:Date.now(),outcomeSnapshot,outcomeSha256,
      return20:rows.at(-1).close/anchor-1,largestDecline,reversal:largestDecline <= -.1+1e-12});
    return {id:r._id,settled:true};
  },
});

export const checkpoint = mutation({
  args:{secret:v.string(),protocol:v.string(),ticker:v.string(),session:v.string(),status:v.string()},
  handler:async (ctx,{secret,protocol:version,ticker,session,status}) => {
    authorize(secret);
    const p = await protocol(ctx,version);
    if (!p || !JSON.parse(p.spec).tickers.includes(ticker) || !dateOk(session)
      || !["captured","no_spike","stale","missed_window","not_active","error"].includes(status)) throw new Error("Invalid tracking checkpoint");
    const key = `${version}:${ticker}:${session}`;
    if (status === "captured" && !await observation(ctx,key)) throw new Error("Capture was not confirmed");
    const row = {key,protocol:version,ticker,session,status,checkedAt:Date.now()};
    const existing = await ctx.db.query("anomalyTrackingChecks").withIndex("by_key",q => q.eq("key",key)).unique();
    if (existing) {
      if (existing.status !== "captured") await ctx.db.patch(existing._id,row);
      return {id:existing._id};
    }
    return {id:await ctx.db.insert("anomalyTrackingChecks",row)};
  },
});
