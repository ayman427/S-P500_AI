import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export const predictionFields = {
  key: v.string(), ticker: v.string(), modelType: v.string(), modelHash: v.string(),
  targetKey: v.string(), signalDate: v.string(), predictedReturn: v.number(),
  trainedThrough: v.string(), signalClose: v.number(), entryDate: v.string(),
  exitDate: v.string(), entryAt: v.number(), exitAt: v.number(),
  hybrid: v.optional(v.object({ridgeReturn: v.number(), boostingReturn: v.number(), ridgeWeight: v.number(),
    feedbackSamples: v.number(), feedbackThrough: v.union(v.number(), v.null()), learningRule: v.literal("settled_error_v1")})),
};
const datasetFields = {
  key: v.string(), kind: v.string(), ticker: v.optional(v.string()),
  format: v.string(), storageId: v.id("_storage"),
  sha256: v.string(), contentSha256: v.string(), size: v.number(),
  modifiedAt: v.number(), version: v.number(),
};
function datasetTable() {
  return defineTable(datasetFields).index("by_key", ["key"]).index("by_ticker", ["ticker"]);
}
export const spikeResearchFields = {
  key: v.string(), ticker: v.string(), eventDate: v.string(), method: v.union(v.literal("dated_topic_rules_v1"), v.literal("dated_topic_rules_v2"), v.literal("dated_topic_rules_v3"), v.literal("dated_topic_rules_v4")),
  priceFingerprint: v.string(), previousDate: v.string(), previousClose: v.number(), close: v.number(), dailyReturn: v.number(),
  categories: v.array(v.string()), confidence: v.union(v.literal("unknown"), v.literal("limited"), v.literal("moderate")),
  overview: v.string(), providerStatus: v.string(), providerMessage: v.string(), fetchedAt: v.string(),
  windowFrom: v.string(), windowTo: v.string(), articlesChecked: v.number(),
  usableForTraining: v.literal(false),
  documentsRetrieved: v.optional(v.number()), documentsMatched: v.optional(v.number()),
  evidenceStatus: v.optional(v.union(v.literal("possible_explanation"), v.literal("inconclusive"), v.literal("no_documents_found"), v.literal("sources_unavailable"))),
  coverage: v.optional(v.array(v.object({provider: v.string(), status: v.string(), message: v.string(), documentsChecked: v.number(), truncated: v.boolean()}))),
  sources: v.array(v.object({title: v.string(), url: v.string(), publisher: v.string(), publishedAt: v.string(),
    retrievedAt: v.string(), summary: v.string(), basis: v.union(v.literal("provider_excerpt"), v.literal("headline_only"), v.literal("sec_filing"), v.literal("company_announcement")),
    publicationPrecision: v.optional(v.union(v.literal("timestamp"), v.literal("date"))),
    timestampBasis: v.optional(v.union(v.literal("sec_acceptance"), v.literal("issuer_publication"))),
    contextPhase: v.optional(v.union(v.literal("earlier_context"), v.literal("move_window"), v.literal("later_reporting"))),
    categories: v.array(v.string()), reportedLink: v.boolean(),
    timing: v.union(v.literal("before_move"), v.literal("during_move"), v.literal("after_close"), v.literal("unknown_time")),
  })),
};
const trialFields = {trials: v.number(), hits: v.number(), falseAlerts: v.number(), precision: v.union(v.number(), v.null())};
export const insiderTransactionFields = {
  key: v.string(), ticker: v.string(), accession: v.string(), cik: v.string(), transactionIndex: v.number(),
  transactionDate: v.string(), filingDate: v.string(), acceptedAt: v.string(), firstRetrievedAt: v.string(),
  insiderIds: v.array(v.string()), insiderName: v.string(), position: v.string(), securityTitle: v.string(), ownership: v.string(),
  shares: v.number(), price: v.union(v.number(), v.null()), value: v.union(v.number(), v.null()), currency: v.literal("unverified"),
  transactionCode: v.literal("P"), form: v.literal("4"), source: v.literal("SEC EDGAR"), discoveredBy: v.literal("API Ninjas"),
  filingUrl: v.string(), sourceUrl: v.string(), filingFingerprint: v.string(), usableForTraining: v.literal(false),
};
const insiderEvidence = v.object({
  status: v.union(v.literal("buying_observed"), v.literal("coverage_incomplete")), coverage: v.literal("partial"), message: v.string(),
  from: v.string(), to: v.string(), retrievedAt: v.string(), providerStatus: v.string(), candidatesReturned: v.number(),
  filingsChecked: v.number(), truncated: v.boolean(), buyers: v.number(), transactions: v.number(),
  records: v.array(v.object({key: v.string(), insiderName: v.string(), securityTitle: v.optional(v.string()), transactionDate: v.string(), acceptedAt: v.string(),
    shares: v.number(), price: v.union(v.number(), v.null()), value: v.union(v.number(), v.null()), filingUrl: v.string()})),
});
export const spikeWatchFields = {
  key: v.string(), ticker: v.string(), evaluatedAt: v.string(), priceFingerprint: v.string(),
  asOf: v.string(), expectedSession: v.string(), ruleVersion: v.literal("price_volume_v2"), detectorVersion: v.literal("daily_5pct_3sigma_v1"),
  status: v.union(v.literal("stale"), v.literal("incomplete"), v.literal("caution"), v.literal("watch"), v.literal("no_signal")),
  horizonSessions: v.literal(5), usableForTraining: v.literal(false), ruleApproved: v.literal(false), missingSessions: v.array(v.string()),
  insiderEvidence: v.optional(insiderEvidence),
  evidenceFingerprint: v.optional(v.string()),
  checks: v.array(v.object({id: v.string(), label: v.string(), rule: v.string(), unit: v.string(), source: v.string(), active: v.boolean(),
    availableAt: v.union(v.string(), v.null()), value: v.union(v.number(), v.null()), status: v.union(v.literal("met"), v.literal("not_met"), v.literal("unknown"))})),
  validation: v.object({from: v.string(), sessions: v.number(), alerts: v.number(), hits: v.number(), falseAlerts: v.number(),
    precision: v.union(v.number(), v.null()), baselineRate: v.union(v.number(), v.null()), baselineTrials: v.number(),
    lift: v.union(v.number(), v.null()), precisionLowerBound: v.union(v.number(), v.null()), supported: v.boolean(),
    status: v.union(v.literal("insufficient"), v.literal("promising"), v.literal("failed")), folds: v.array(v.object(trialFields))}),
};
export const marketScanFields = {
  expectedSession: v.string(), startedAt: v.string(), updatedAt: v.string(), ruleVersion: v.literal("price_volume_v2"),
  universe: v.array(v.string()), processed: v.number(), complete: v.boolean(), unavailable: v.array(v.string()),
  rows: v.array(v.object({ticker: v.string(), asOf: v.string(),
    status: v.union(v.literal("stale"), v.literal("incomplete"), v.literal("watch"), v.literal("no_signal")), met: v.number(),
    checks: v.array(v.object({id: v.string(), status: v.union(v.literal("met"), v.literal("not_met"), v.literal("unknown")), value: v.union(v.number(), v.null())}))})),
};
export default defineSchema({
  marketScans: defineTable({key: v.string(), scan: v.object(marketScanFields)}).index("by_key", ["key"]),
  insiderTransactions: defineTable(insiderTransactionFields).index("by_key", ["key"]).index("by_ticker_accepted", ["ticker", "acceptedAt"]),
  spikeWatch: defineTable(spikeWatchFields).index("by_key", ["key"]).index("by_ticker", ["ticker"]),
  spikeResearch: defineTable(spikeResearchFields).index("by_key", ["key"]).index("by_ticker_date", ["ticker", "eventDate"]),
  prices: datasetTable(),
  fundamentals: datasetTable(),
  news: datasetTable(),
  companyMetadata: datasetTable(),
  priceSync: datasetTable(),
  modelArtifacts: datasetTable(),
  backtestReports: datasetTable(),
  paperLedgers: datasetTable(),
  predictionExports: datasetTable(),
  operationLocks: defineTable({key: v.string(), owner: v.string(), expiresAt: v.number(), heartbeatAt: v.optional(v.number())}).index("by_key", ["key"]),
  predictions: defineTable({
    // Existing ownerless records are retained as service-only legacy history.
    ownerId: v.optional(v.string()),
    ...predictionFields, status: v.union(v.literal("pending"), v.literal("settled")),
    actualReturn: v.optional(v.number()), absoluteError: v.optional(v.number()),
    directionCorrect: v.optional(v.boolean()), entryOpen: v.optional(v.number()),
    exitClose: v.optional(v.number()), settledAt: v.optional(v.number()),
    source: v.optional(v.string()),
  }).index("by_key", ["key"])
    .index("by_owner", ["ownerId"])
    .index("by_owner_key", ["ownerId", "key"])
    .index("by_owner_selection", ["ownerId", "ticker", "modelType", "targetKey"])
    .index("by_selection", ["ticker", "modelType", "targetKey"])
    .index("by_due", ["status", "exitAt"]),
});
