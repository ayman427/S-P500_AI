import { mutationGeneric as mutation } from "convex/server";
import { v } from "convex/values";
import { spikeWatchFields } from "./schema.js";

export const save = mutation({
  args: {secret: v.string(), record: v.object(spikeWatchFields)},
  handler: async (ctx, {secret, record}) => {
    if (!process.env.PREDICTION_STORE_SECRET || secret !== process.env.PREDICTION_STORE_SECRET)
      throw new Error("Unauthorized watch access");
    const date = value => /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
    const count = value => Number.isSafeInteger(value) && value >= 0;
    const rate = value => value === null || (Number.isFinite(value) && value >= 0 && value <= 1);
    const validation = record.validation;
    if (!/^[A-Z0-9^][A-Z0-9^.-]{0,14}$/.test(record.ticker) || !/^[a-f0-9]{64}$/.test(record.priceFingerprint)
        || !date(record.asOf) || !date(record.expectedSession) || record.asOf > record.expectedSession
        || !Number.isFinite(Date.parse(record.evaluatedAt)) || record.status === "caution" || record.ruleApproved !== false
        || (record.insiderEvidence !== undefined) !== (record.evidenceFingerprint !== undefined)
        || (record.evidenceFingerprint !== undefined && !/^[a-f0-9]{64}$/.test(record.evidenceFingerprint))
        || record.key !== `${record.ticker}:${record.expectedSession}:${record.ruleVersion}:${record.priceFingerprint}${record.evidenceFingerprint ? ":" + record.evidenceFingerprint : ""}`
        || record.checks.map(row => row.id).sort().join(",") !== "breakout,catalyst,insiders,momentum,volume"
        || ![validation.sessions, validation.alerts, validation.hits, validation.falseAlerts, validation.baselineTrials].every(count)
        || validation.hits + validation.falseAlerts !== validation.alerts || validation.alerts > validation.sessions
        || ![validation.precision, validation.baselineRate, validation.precisionLowerBound].every(rate)
        || (validation.precision === null) !== (validation.alerts === 0)
        || (validation.precision !== null && Math.abs(validation.precision - validation.hits / validation.alerts) > 1e-10))
      throw new Error("Invalid watch snapshot");
    for (const check of record.checks) {
      const active = ["volume", "momentum", "breakout"].includes(check.id);
      if ((check.value !== null && !Number.isFinite(check.value)) || (check.status === "unknown") !== (check.value === null)
          || check.active !== active || (!check.active && check.status !== "unknown") || (check.availableAt !== null && (!date(check.availableAt) || check.availableAt > record.asOf)))
        throw new Error("Invalid watch evidence");
      if (check.value !== null) {
        const met = check.id === "volume" ? check.value >= 1.5 : check.id === "momentum" ? check.value >= .02 : check.value > 0;
        if ((check.status === "met") !== met) throw new Error("Invalid watch evidence");
      }
    }
    if (record.insiderEvidence) {
      const evidence = record.insiderEvidence;
      if (!date(evidence.from) || !Number.isFinite(Date.parse(evidence.to)) || !Number.isFinite(Date.parse(evidence.retrievedAt))
          || evidence.records.length > 5 || evidence.coverage !== "partial"
          || ![evidence.buyers, evidence.transactions, evidence.candidatesReturned, evidence.filingsChecked].every(count)
          || evidence.transactions < evidence.records.length || evidence.filingsChecked > 3
          || (evidence.status === "buying_observed") !== (evidence.transactions > 0))
        throw new Error("Invalid insider watch evidence");
      for (const row of evidence.records) {
        if (!/^https:\/\/www\.sec\.gov\/Archives\/edgar\/data\/\d+\/\d{18}\/\d{10}-\d{2}-\d{6}-index\.html$/.test(row.filingUrl)
            || !date(row.transactionDate) || row.transactionDate > record.asOf
            || !Number.isFinite(Date.parse(row.acceptedAt)) || Date.parse(row.acceptedAt) > Date.parse(evidence.to)
            || !Number.isFinite(row.shares) || row.shares <= 0)
          throw new Error("Invalid insider watch source");
      }
    }
    const existing = await ctx.db.query("spikeWatch").withIndex("by_key", q => q.eq("key", record.key)).unique();
    if (existing) {
      await ctx.db.replace(existing._id, record);
      return {id: existing._id, created: false};
    }
    return {id: await ctx.db.insert("spikeWatch", record), created: true};
  },
});
