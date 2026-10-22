import { queryGeneric as query, mutationGeneric as mutation } from "convex/server";
import { v } from "convex/values";
import { insiderTransactionFields } from "./schema.js";

function authorize(secret) {
  if (!process.env.PREDICTION_STORE_SECRET || secret !== process.env.PREDICTION_STORE_SECRET)
    throw new Error("Unauthorized insider evidence access");
}
function date(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
}
export const recent = query({
  args: {secret: v.string(), ticker: v.string(), from: v.string(), to: v.string()},
  handler: async (ctx, {secret, ticker, from, to}) => {
    authorize(secret);
    if (!/^[A-Z0-9^][A-Z0-9^.-]{0,14}$/.test(ticker) || !Number.isFinite(Date.parse(from)) || !Number.isFinite(Date.parse(to)) || from > to)
      throw new Error("Invalid insider evidence window");
    const rows = await ctx.db.query("insiderTransactions").withIndex("by_ticker_accepted", q => q.eq("ticker", ticker).gte("acceptedAt", from).lte("acceptedAt", to)).order("desc").take(201);
    return {rows: rows.slice(0, 200), truncated: rows.length > 200};
  },
});
export const save = mutation({
  args: {secret: v.string(), records: v.array(v.object(insiderTransactionFields))},
  handler: async (ctx, {secret, records}) => {
    authorize(secret);
    if (records.length > 150) throw new Error("Insider batch exceeds limit");
    for (const row of records) {
      const base = `https://www.sec.gov/Archives/edgar/data/${row.cik}/${row.accession.replaceAll("-", "")}/`;
      if (!/^[A-Z0-9^][A-Z0-9^.-]{0,14}$/.test(row.ticker) || !/^\d{10}-\d{2}-\d{6}$/.test(row.accession)
          || !/^[1-9]\d{0,9}$/.test(row.cik) || !Number.isSafeInteger(row.transactionIndex) || row.transactionIndex < 0
          || row.key !== `${row.ticker}:${row.accession}:${row.transactionIndex}`
          || !date(row.transactionDate) || !date(row.filingDate) || !Number.isFinite(Date.parse(row.acceptedAt))
          || !Number.isFinite(Date.parse(row.firstRetrievedAt)) || Date.parse(row.acceptedAt) > Date.parse(row.firstRetrievedAt)
          || row.transactionDate > row.filingDate || row.transactionCode !== "P" || row.form !== "4" || row.usableForTraining !== false
          || !Number.isFinite(row.shares) || row.shares <= 0 || (row.price !== null && (!Number.isFinite(row.price) || row.price < 0))
          || (row.value === null) !== (row.price === null) || (row.value !== null && (!Number.isFinite(row.value) || Math.abs(row.value - row.price*row.shares) > Math.max(1e-6, Math.abs(row.value)*1e-10)))
          || !row.insiderIds.length || row.insiderIds.length > 20 || !row.insiderIds.every(id => /^[1-9]\d{0,9}$/.test(id))
          || !row.insiderName || row.insiderName.length > 500 || row.position.length > 500
          || row.filingUrl !== base + row.accession + "-index.html" || !row.sourceUrl.startsWith(base)
          || !/^[A-Za-z0-9_.-]+\.xml$/.test(row.sourceUrl.slice(base.length)) || !/^[a-f0-9]{64}$/.test(row.filingFingerprint))
        throw new Error("Invalid verified insider transaction");
    }
    const ids = [];
    for (const row of records) {
      const existing = await ctx.db.query("insiderTransactions").withIndex("by_key", q => q.eq("key", row.key)).unique();
      if (existing) {
        await ctx.db.replace(existing._id, {...row, firstRetrievedAt: existing.firstRetrievedAt});
        ids.push(existing._id);
      } else ids.push(await ctx.db.insert("insiderTransactions", row));
    }
    return {ids};
  },
});
