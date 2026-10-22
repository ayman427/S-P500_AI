import { useEffect, useState } from "react";
import { useAuthenticatedFetch } from "./useAuthenticatedFetch";
import { stockResponse } from "./stockResponse";
import "./RiskPanel.css";

type Decline = {
  peakDate: string; peakPrice: number; startDate: string; startPrice: number;
  troughDate: string; troughPrice: number; decline: number; thresholdDate: string;
  recoveryDate: string | null; daysUnderwater: number; riseFromWindowStart: number;
  reboundNeededAtTrough: number;
};
type Risk = {
  ticker: string; from: string; asOf: string; availableFrom: string; sessions: number;
  startPrice: number; latestPrice: number; totalReturn: number; maxDrawdown: number;
  currentDrawdown: number; annualizedVolatility: number | null; worstDailyReturn: number;
  events: Decline[]; qualifyingDeclines: number;
  history: {date: string; close: number; drawdown: number}[];
  benchmark?: {ticker: string; from: string; asOf: string; sessions: number;
    totalReturn: number; maxDrawdown: number; currentDrawdown: number;
    annualizedVolatility: number | null;
    history: {date: string; close: number; drawdown: number}[]} | null;
  benchmarkError?: string;
};
const pct = (value: number | null) => value === null ? "Unavailable" : `${(value * 100).toFixed(1)}%`;
const usd = (value: number) => new Intl.NumberFormat("en-US", {style: "currency", currency: "USD", maximumFractionDigits: 2}).format(value);
const day = (value: string) => new Intl.DateTimeFormat("en-US", {month: "short", day: "numeric", year: "numeric"}).format(new Date(`${value}T12:00:00`));

export default function RiskPanel({ ticker }: {ticker: string}) {
  const apiFetch = useAuthenticatedFetch();
  const [years, setYears] = useState(5);
  const [threshold, setThreshold] = useState(0.20);
  const [selected, setSelected] = useState<string | null>(null);
  const [inspection, setInspection] = useState<{key: string; index: number} | null>(null);
  const [retry, setRetry] = useState(0);
  const [riskResult, setRiskResult] = useState<{key: string; data?: Risk; error?: string} | null>(null);
  const key = `${ticker}:${years}:${threshold}:${retry}`;
  const risk = riskResult?.key === key ? riskResult.data : undefined;
  const error = riskResult?.key === key ? riskResult.error : undefined;

  useEffect(() => {
    const controller = new AbortController();
    apiFetch(`/api/risk?ticker=${encodeURIComponent(ticker)}&years=${years}&threshold=${threshold}`, {signal: controller.signal})
      .then(stockResponse).then((data: Risk) => {
        if (data.ticker !== ticker) throw new Error("The risk results did not match your stock.");
        if (!controller.signal.aborted) setRiskResult({key, data});
      }).catch(reason => {
        if (!controller.signal.aborted) setRiskResult({key, error: reason instanceof Error ? reason.message : "Risk analysis is unavailable."});
      });
    return () => controller.abort();
  }, [apiFetch, ticker, years, threshold, key]);

  const event = risk?.events.find(item => item.startDate === selected) ?? risk?.events[0];
  const history = risk?.history ?? [];
  const min = Math.min(...history.map(row => row.close));
  const max = Math.max(...history.map(row => row.close));
  const firstTime = history.length ? Date.parse(history[0].date) : 0;
  const lastTime = history.length ? Date.parse(history[history.length - 1].date) : 1;
  const x = (date: string) => 20 + (Date.parse(date) - firstTime) / (lastTime - firstTime || 1) * 600;
  const y = (price: number) => 145 - (price - min) / (max - min || 1) * 120;
  const points = history.map(row => `${x(row.date)},${y(row.close)}`).join(" ");
  const benchmark = risk?.benchmark;
  const deepestDrawdown = Math.min(-0.01, ...history.map(row => row.drawdown), ...(benchmark?.history.map(row => row.drawdown) ?? []));
  const drawdownY = (value: number) => 25 + value / deepestDrawdown * 120;
  const drawdownPoints = history.map(row => `${x(row.date)},${drawdownY(row.drawdown)}`).join(" ");
  const marketDrawdownPoints = benchmark?.history.map(row => `${x(row.date)},${drawdownY(row.drawdown)}`).join(" ");
  const eventEnd = event?.recoveryDate ?? risk?.asOf;
  const selectedHistory = event && eventEnd ? history.filter(row => row.date >= event.peakDate && row.date <= eventEnd) : [];
  const selectedPricePoints = selectedHistory.map(row => `${x(row.date)},${y(row.close)}`).join(" ");
  const selectedDrawdownPoints = selectedHistory.map(row => `${x(row.date)},${drawdownY(row.drawdown)}`).join(" ");
  const recovery = event?.recoveryDate ? history.find(row => row.date === event.recoveryDate) : undefined;
  const inspectionKey = `${key}:${event?.startDate}`;
  const inspectedIndex = inspection?.key === inspectionKey ? Math.min(inspection.index, selectedHistory.length - 1) : 0;
  const inspected = selectedHistory[inspectedIndex];
  const inspectPointer = (e: React.PointerEvent<SVGRectElement>) => {
    const svg = e.currentTarget.ownerSVGElement;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix || !selectedHistory.length) return;
    const position = new DOMPoint(e.clientX, e.clientY).matrixTransform(matrix.inverse());
    const index = selectedHistory.reduce((nearest, row, i) => Math.abs(x(row.date) - position.x) < Math.abs(x(selectedHistory[nearest].date) - position.x) ? i : nearest, 0);
    setInspection({key: inspectionKey, index});
  };
  const selectEvent = (date: string) => setSelected(date);

  return <section className="risk-panel" aria-labelledby="risk-heading">
    <h3 id="risk-heading" className="sr-only">Declines</h3>
    <div className="risk-controls">
      <label>History <select value={years} onChange={e => {setYears(Number(e.target.value)); setSelected(null);}}>
        <option value={1}>1 year</option><option value={3}>3 years</option><option value={5}>5 years</option><option value={10}>10 years</option><option value={0}>All available</option>
      </select></label>
      <label>Min. decline <select value={threshold} onChange={e => {setThreshold(Number(e.target.value)); setSelected(null);}}>
        <option value={0.10}>10% or more</option><option value={0.20}>20% or more</option><option value={0.30}>30% or more</option>
      </select></label>
    </div>
    {error ? <p role="alert">{error} <button type="button" onClick={() => setRetry(retry + 1)}>Retry risk analysis</button></p> : !risk ? <p role="status">Comparing historical prices…</p> : <>
      <p className="risk-caption">{day(risk.from)} – {day(risk.asOf)} · Adjusted closing prices</p>
      <div className="risk-stats">
        <div><span>Largest decline</span><strong>{pct(risk.maxDrawdown)}</strong></div>
        <div><span>Below peak now</span><strong>{pct(risk.currentDrawdown)}</strong></div>
        <div><span>Annualized volatility</span><strong>{pct(risk.annualizedVolatility)}</strong></div>
        <div><span>Worst day</span><strong>{pct(risk.worstDailyReturn)}</strong></div>
      </div>
      <div className="risk-chart">
        <h4>Price history</h4>
        <svg viewBox="0 0 640 180" role="group" aria-label={`${ticker} historical adjusted closing prices. Inspect prices within the highlighted decline.`}>
          <title>Adjusted closing price with decline-onset pins</title>
          <text x="20" y="16">{usd(max)}</text><text x="20" y="172">{usd(min)}</text>
          {event && eventEnd && <rect className="selected-decline-range" x={x(event.peakDate)} y="25" width={x(eventEnd) - x(event.peakDate)} height="120" />}
          <polyline points={points} fill="none" stroke="var(--chart)" strokeWidth="2" />
          {event && <polyline className="selected-decline-line" points={selectedPricePoints} />}
          {event && <line x1={x(event.startDate)} x2={x(event.startDate)} y1="22" y2="150" stroke="var(--risk-event)" strokeDasharray="4 4" />}
          {event && eventEnd && inspected && <>
            <g className="risk-price-cursor" aria-hidden="true">
              <line x1={x(inspected.date)} x2={x(inspected.date)} y1="25" y2="145" />
              <circle cx={x(inspected.date)} cy={y(inspected.close)} r="5" />
            </g>
            <rect className="risk-price-inspector" x={x(event.peakDate)} y="25" width={x(eventEnd) - x(event.peakDate)} height="120"
              tabIndex={0} role="slider" aria-label="Inspect highlighted closing prices" aria-valuemin={0} aria-valuemax={selectedHistory.length - 1} aria-valuenow={inspectedIndex}
              aria-valuetext={`${day(inspected.date)}: ${usd(inspected.close)}`} onPointerMove={inspectPointer} onPointerDown={inspectPointer}
              onKeyDown={e => {
                const index = e.key === "Home" ? 0 : e.key === "End" ? selectedHistory.length - 1 : e.key === "ArrowRight" || e.key === "ArrowUp" ? inspectedIndex + 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? inspectedIndex - 1 : null;
                if (index !== null) {e.preventDefault(); setInspection({key: inspectionKey, index: Math.max(0, Math.min(index, selectedHistory.length - 1))});}
              }} />
          </>}
          {risk.events.map(item => <circle key={item.startDate} cx={x(item.startDate)} cy={y(item.startPrice)} r={item.startDate === event?.startDate ? 6 : 4} fill="var(--risk-event)" opacity={item.startDate === event?.startDate ? 1 : .4} tabIndex={0} role="button" aria-pressed={item.startDate === event?.startDate} aria-label={`Select decline starting ${day(item.startDate)}`} onClick={() => selectEvent(item.startDate)} onKeyDown={e => {if (e.key === "Enter" || e.key === " ") {e.preventDefault(); selectEvent(item.startDate);}}}>
            <title>{day(item.startDate)}: {usd(item.startPrice)}. Subsequent decline {pct(item.decline)}.</title>
          </circle>)}
          {recovery && <circle className="risk-recovery-point" cx={x(recovery.date)} cy={y(recovery.close)} r="6" role="img" aria-label={`Recovered on ${day(recovery.date)} at ${usd(recovery.close)}`}><title>Recovered {day(recovery.date)} at {usd(recovery.close)}</title></circle>}
        </svg>
        <div className="history-dates"><span>{day(risk.from)}</span><span>{day(risk.asOf)}</span></div>
        {inspected && <>
          <output className="risk-price-readout">{day(inspected.date)} · <strong>{usd(inspected.close)}</strong> adjusted close</output>
          <p className="risk-caption">Hover or tap to inspect. Sampled closes.</p>
        </>}
        {event && <div className="risk-event-legend"><span><i className="decline-legend" />Selected decline</span><span>{recovery ? <><i className="recovery-legend" />Recovered {day(recovery.date)}</> : "Not recovered"}</span></div>}
      </div>
      {risk.events.length === 0 ? <p>No declines of {pct(threshold)} or more in this history window.</p> : <>
        <div className="risk-event-list" aria-label="Select a historical decline">
          {risk.events.map(item => <button type="button" key={item.startDate} aria-pressed={item.startDate === event?.startDate} onClick={() => selectEvent(item.startDate)}>
            <span>{day(item.startDate)}</span><strong>{pct(item.decline)}</strong><small>{item.recoveryDate ? `Recovered ${day(item.recoveryDate)}` : "Not recovered"}</small>
          </button>)}
        </div>
        {risk.qualifyingDeclines > risk.events.length && <p className="risk-caption">Showing the {risk.events.length} deepest of {risk.qualifyingDeclines} declines.</p>}
      </>}
      {event && <div className="risk-event-detail">
        <h4>Selected decline: {day(event.startDate)}</h4>
        <dl className="risk-event-facts">
          <div><dt>Peak</dt><dd>{usd(event.peakPrice)} · {day(event.peakDate)}</dd></div>
          <div><dt>Lowest close</dt><dd>{usd(event.troughPrice)} · {day(event.troughDate)}</dd></div>
          <div><dt>{pct(threshold)} decline confirmed</dt><dd>{day(event.thresholdDate)}</dd></div>
          <div><dt>Recovery to peak</dt><dd>{event.recoveryDate ? day(event.recoveryDate) : "Not yet within this dataset"}</dd></div>
          <div><dt>Rise from period start to peak</dt><dd>{pct(event.riseFromWindowStart)}</dd></div>
          <div><dt>Time below peak</dt><dd>{event.daysUnderwater.toLocaleString()} days{event.recoveryDate ? "" : " so far"}</dd></div>
        </dl>
      </div>}
      <details className="risk-market-details">
        <summary>Market comparison (SPY)</summary>
      <div className="risk-comparison">
        <h4>Compared with the market</h4>
        {benchmark ? <>
          <table>
            <caption>Same dates: {day(benchmark.from)} – {day(benchmark.asOf)}</caption>
            <thead><tr><th scope="col">Measure</th><th scope="col">{ticker}</th><th scope="col">SPY</th></tr></thead>
            <tbody>
              <tr><th scope="row">Largest decline</th><td>{pct(risk.maxDrawdown)}</td><td>{pct(benchmark.maxDrawdown)}</td></tr>
              <tr><th scope="row">Volatility</th><td>{pct(risk.annualizedVolatility)}</td><td>{pct(benchmark.annualizedVolatility)}</td></tr>
              <tr><th scope="row">Period return</th><td>{pct(risk.totalReturn)}</td><td>{pct(benchmark.totalReturn)}</td></tr>
            </tbody>
          </table>
        </> : <p className="risk-caption" role="status">{risk.benchmarkError ?? "SPY comparison is unavailable for this period."}</p>}
      </div>
      <div className="risk-drawdown-chart">
        <h4>Decline from the previous peak</h4>
        <div className="risk-chart-legend"><span><i className="stock-legend" />{ticker}</span>{benchmark && <span><i className="market-legend" />SPY</span>}</div>
        <div className="risk-drawdown-scale"><span>0% · At peak</span><span>{pct(deepestDrawdown)}</span></div>
        <svg viewBox="0 0 640 165" role="img" aria-label={`${ticker} drawdown from previous peaks${benchmark ? " compared with SPY" : ""} from ${day(risk.from)} to ${day(risk.asOf)}`}>
          <title>Drawdown: the percentage below the running highest adjusted close</title>
          {event && eventEnd && <rect className="selected-decline-range" x={x(event.peakDate)} y="25" width={x(eventEnd) - x(event.peakDate)} height="120" />}
          <line x1="20" x2="620" y1="25" y2="25" stroke="var(--line)" />
          <polygon points={`20,25 ${drawdownPoints} 620,25`} fill="var(--risk-event)" opacity=".10" />
          {benchmark && <polyline points={marketDrawdownPoints} fill="none" stroke="var(--muted)" strokeWidth="2" strokeDasharray="6 4" />}
          <polyline points={drawdownPoints} fill="none" stroke="var(--risk-event)" strokeWidth="2" />
          {event && <polyline className="selected-decline-line" points={selectedDrawdownPoints} />}
          {recovery && <circle className="risk-recovery-point" cx={x(recovery.date)} cy={drawdownY(recovery.drawdown)} r="6" role="img" aria-label={`Recovered on ${day(recovery.date)}`}><title>Recovered {day(recovery.date)}: back at the previous peak</title></circle>}
        </svg>
        <div className="history-dates"><span>{day(risk.from)}</span><span>{day(risk.asOf)}</span></div>
      </div>
      </details>
      <details className="risk-methods">
        <summary>About these figures</summary>
        <p className="risk-caption">Prices account for splits and dividends. Peaks are measured within your selected period. A decline starts at the first lower close after a peak and qualifies once it reaches {pct(threshold)}. Events are identified retrospectively. Chart prices are sampled; focus the highlighted area and use arrow keys to inspect them. Volatility is daily-return standard deviation × √252.</p>
        <p className="risk-caption">Past price behavior does not predict future losses.</p>
      </details>
    </>}
  </section>;
}
