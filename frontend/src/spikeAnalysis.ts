export type AnomalyAnalysis = {
  method: string; asOf: string; title: string; summary: string;
  signals: {id: string; label: string; value: number | null; threshold: number | null; unit: string; status: string}[];
  baselineFrom: string | null; baselineThrough: string; baselineReturns: number; priorDailyVolatility: number | null; marketReturn: number | null;
  historicalOutcome: {sessions: number; through: string; return5: number | null; return20: number | null; largestDecline: number | null; gainReversed: number | null};
};
export function validAnalysis(a: AnomalyAnalysis | undefined, eventDate: string) {
  return !!a && a.method === "price_volume_observations_v1" && a.asOf === eventDate && typeof a.title === "string" && typeof a.summary === "string"
    && typeof a.baselineThrough === "string" && a.baselineThrough < eventDate
    && Array.isArray(a.signals) && a.signals.length === 4 && new Set(a.signals.map(s => s.id)).size === 4
    && a.signals.every(s => ["price", "market", "volume", "acceleration"].includes(s.id) && typeof s.label === "string"
      && ["return", "percentage_points", "multiple"].includes(s.unit) && [s.value, s.threshold].every(v => v === null || Number.isFinite(v))
      && (s.value === null || s.threshold === null ? s.status === "unavailable" : s.status === (s.value >= s.threshold - 1e-12 ? "elevated" : "within_rule")))
    && !!a.historicalOutcome && Number.isInteger(a.historicalOutcome.sessions) && a.historicalOutcome.sessions >= 0 && a.historicalOutcome.sessions <= 20
    && typeof a.historicalOutcome.through === "string" && a.historicalOutcome.through >= eventDate
    && [a.historicalOutcome.return5, a.historicalOutcome.return20, a.historicalOutcome.largestDecline, a.historicalOutcome.gainReversed, a.priorDailyVolatility, a.marketReturn].every(v => v === null || Number.isFinite(v));
}
export type OutcomeEvaluation = {eligible: number; flagged: number; flaggedReversals: number; baselineReversals: number; excluded: number};
