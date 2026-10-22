import type {AnomalyAnalysis, OutcomeEvaluation} from "./spikeAnalysis";
const pct = (value: number | null) => value === null ? "Unavailable" : `${(value * 100).toFixed(1)}%`;
export default function SpikeExplanation({analysis: a, evaluation}: {analysis: AnomalyAnalysis; evaluation: OutcomeEvaluation}) {
  const outcome = a.historicalOutcome;
  return <section className="spike-explanation" aria-label="Price and volume observations">
    <h4 className="anomaly-title">{a.title}</h4>
    <p className="spike-explanation-summary">{a.summary}</p>
    <dl className="anomaly-signals">{a.signals.map(s => <div key={s.id} data-status={s.status}>
      <dt>{s.label}</dt><dd>{s.value === null ? "Unavailable" : s.unit === "multiple" ? `${s.value.toFixed(2)}x` : s.unit === "percentage_points" ? `${(s.value * 100).toFixed(1)} pp` : pct(s.value)}</dd>
      <span>{s.status === "elevated" ? "Elevated" : s.status === "within_rule" ? "Within rule" : "Missing data"}</span>
    </div>)}</dl>
    <p className="risk-caption">Unusual trading patterns do not establish manipulation.</p>
    <details className="rise-details"><summary>Details</summary>
      <section><h4>Measurement rules</h4><ul>
        <li>Price: at least 5% and three standard deviations above the prior mean; 20-60 prior daily returns.</li>
        <li>Market: stock return minus SPY return on the same session; elevated at 3 percentage points.</li>
        <li>Volume: rise-day volume divided by the previous 20 sessions' mean; elevated at 2x.</li>
        <li>Acceleration: latest five-session return minus the preceding five-session return; elevated at 5 percentage points.</li>
      </ul><p className="risk-caption">Descriptive thresholds, not calibrated risk probabilities. Signals use data only through {a.asOf}.</p></section>
      <section><h4>Data</h4><p className="risk-caption">Price baseline: {a.baselineFrom ?? "Unavailable"} to {a.baselineThrough}; {a.baselineReturns} valid prior returns.</p>
        <p className="risk-caption">Prior daily volatility: {pct(a.priorDailyVolatility)}. SPY session return: {pct(a.marketReturn)}. Sector comparison unavailable.</p>
      </section>
      <section><h4>Historical check - selected period</h4>
        <p>{evaluation.flaggedReversals} of {evaluation.flagged} multi-signal rises fell at least 10% within 20 sessions.</p>
        <p>Price-spike baseline: {evaluation.baselineReversals} of {evaluation.eligible}.</p>
        <p className="risk-caption">Multi-signal means at least two additional elevated signals. Only complete, non-overlapping windows with all inputs qualify; {evaluation.excluded} excluded. Same-stock retrospective counts, not validated future risk.</p>
      </section>
      <section className="anomaly-outcome"><h4>After the rise - historical only</h4>
        <dl className="rise-evidence-grid"><div><dt>After 5 sessions</dt><dd>{pct(outcome.return5)}</dd></div>
          <div><dt>After 20 sessions</dt><dd>{pct(outcome.return20)}</dd></div>
          <div><dt>Largest fall from rise-day close</dt><dd>{pct(outcome.largestDecline)}</dd></div>
          <div><dt>Original gain reversed</dt><dd>{pct(outcome.gainReversed)}</dd></div></dl>
        <p className="risk-caption">{outcome.sessions}/20 subsequent sessions through {outcome.through}. Missing sessions stop the window. These outcomes do not affect the signals above.</p>
      </section>
    </details>
  </section>;
}
