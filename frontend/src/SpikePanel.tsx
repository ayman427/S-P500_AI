import { useEffect, useState } from "react";
import { useAuthenticatedFetch } from "./useAuthenticatedFetch";
import { stockResponse } from "./stockResponse";
import SpikeExplanation from "./SpikeExplanation";
import {validAnalysis, type AnomalyAnalysis, type OutcomeEvaluation} from "./spikeAnalysis";
import "./RiskPanel.css";

type Price = {date: string; close: number; volume?: number | null};
type Activity = {sessions: number; priceReturn: number | null; meanAbsoluteReturn: number | null; volumeRatio: number | null};
type Spike = Price & {previousDate: string; previousClose: number; return: number; threshold: number; analysis?: AnomalyAnalysis; priorActivity?: {asOf: string; windows: Activity[]}};
type Results = {ticker: string; from: string; asOf: string; assessedSessions: number; events: Spike[]; history: Price[]; evaluation: OutcomeEvaluation};
const money = (value: number) => new Intl.NumberFormat("en-US", {style: "currency", currency: "USD"}).format(value);
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const volumeLabel = (value: number | null | undefined) => value == null ? "Unavailable" : new Intl.NumberFormat("en-US", {maximumFractionDigits: 0}).format(value);
const date = (value: string) => new Date(`${value}T12:00:00`).toLocaleDateString("en-US", {month: "short", day: "numeric", year: "numeric"});

export default function SpikePanel({ticker}: {ticker: string}) {
  const apiFetch = useAuthenticatedFetch();
  const [years, setYears] = useState(5);
  const [retry, setRetry] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [inspection, setInspection] = useState<{key: string; index: number} | null>(null);
  const [result, setResult] = useState<{key: string; data?: Results; error?: string} | null>(null);
  const key = `${ticker}:${years}:${retry}`;
  const data = result?.key === key ? result.data : undefined;
  const error = result?.key === key ? result.error : undefined;
  useEffect(() => {
    const controller = new AbortController();
    apiFetch(`/api/spikes?ticker=${encodeURIComponent(ticker)}&years=${years}`, {signal: controller.signal})
      .then(stockResponse).then((data: Results) => {
        if (data.ticker !== ticker) throw new Error("Spike results did not match your stock.");
        if (!Array.isArray(data.history) || !data.history.length || !Array.isArray(data.events)
          || !data.history.every((row, index) => /^\d{4}-\d{2}-\d{2}$/.test(row.date) && Number.isFinite(row.close) && row.close > 0
            && (index === 0 || data.history[index - 1].date < row.date)
            && (row.volume == null || (Number.isFinite(row.volume) && row.volume >= 0))))
          throw new Error("The saved price history could not be verified.");
        const closes = new Map(data.history.map(row => [row.date, row.close]));
        if (!data.events.every(row => closes.get(row.date) === row.close && Number.isFinite(row.previousClose) && row.previousClose > 0
          && Number.isFinite(row.return) && row.return > 0 && Math.abs(row.close / row.previousClose - 1 - row.return) < 1e-8))
          throw new Error("Spike results did not match the saved prices.");
        const check = data.evaluation;
        if (!check || ![check.eligible, check.flagged, check.flaggedReversals, check.baselineReversals, check.excluded].every(v => Number.isInteger(v) && v >= 0)
          || check.flagged > check.eligible || check.flaggedReversals > check.flagged || check.baselineReversals > check.eligible)
          throw new Error("Historical check could not be verified.");
        if (!data.events.every(row => validAnalysis(row.analysis, row.date))) throw new Error("Price/volume analysis could not be verified. Restart the backend and retry.");
        if (!controller.signal.aborted) setResult({key, data});
      }).catch(reason => {
        if (!controller.signal.aborted) setResult({key, error: reason instanceof Error ? reason.message : "Price spikes are unavailable."});
      });
    return () => controller.abort();
  }, [apiFetch, ticker, years, key]);
  const event = data?.events.find(row => row.date === selected) ?? data?.events[0];
  const position = data?.history.findIndex(row => row.date === event?.date) ?? -1;
  const context = data?.history.slice(Math.max(0, position - 10), position + 11) ?? [];
  const inspectionKey = `${key}:${event?.date}`;
  const inspectedIndex = inspection?.key === inspectionKey ? Math.min(inspection.index, context.length - 1) : Math.min(10, position);
  const inspected = context[inspectedIndex];
  const min = Math.min(...context.map(row => row.close));
  const max = Math.max(...context.map(row => row.close));
  const hasVolume = context.some(row => row.volume != null);
  const maxVolume = Math.max(0, ...context.map(row => row.volume ?? 0));
  const x = (index: number) => 20 + index / Math.max(1, context.length - 1) * 600;
  const y = (close: number) => 145 - (close - min) / (max - min || 1) * 120;
  const inspectPointer = (e: React.PointerEvent<SVGSVGElement>) => {
    const matrix = e.currentTarget.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(e.clientX, e.clientY).matrixTransform(matrix.inverse());
    setInspection({key: inspectionKey, index: Math.max(0, Math.min(context.length - 1, Math.round((point.x - 20) / 600 * (context.length - 1))))});
  };
  return <section className="risk-panel spike-panel" aria-labelledby="spike-heading">
    <div className="risk-controls"><h3 id="spike-heading" className="sr-only">Price spikes</h3>
      <label htmlFor="spike-history">History <select id="spike-history" value={years} onChange={e => {setYears(Number(e.target.value)); setSelected(null);}}>
        <option value={1}>1 year</option><option value={3}>3 years</option><option value={5}>5 years</option><option value={10}>10 years</option><option value={0}>All available</option>
      </select></label>
      {event && <label htmlFor="spike-event">Spike <select id="spike-event" value={event.date} onChange={e => setSelected(e.target.value)}>
        {data!.events.map(row => <option key={row.date} value={row.date}>{date(row.date)} (+{percent(row.return)})</option>)}
      </select></label>}
    </div>
    {error ? <p role="alert">{error} <button onClick={() => setRetry(retry + 1)}>Retry</button></p> : !data ? <p role="status">Checking historical prices…</p> : <>
      <p className="risk-caption">{data.events.length} spikes · {date(data.from)} – {date(data.asOf)}</p>
      {!event ? <p role="status">{data.assessedSessions ? "No spikes meet this rule in the selected period." : "Not enough history: at least 20 prior daily returns are needed."}</p> : <>
        <div className="risk-chart spike-chart">
          <h4>{date(event.date)} · +{percent(event.return)}</h4>
          <p className="risk-caption">{money(event.previousClose)} → {money(event.close)}</p>
          <svg viewBox={`0 0 640 ${hasVolume ? 275 : 180}`} role="slider" tabIndex={0} aria-label={`${ticker} closing prices and volume around the ${date(event.date)} spike`}
            aria-valuemin={0} aria-valuemax={context.length - 1} aria-valuenow={inspectedIndex}
            aria-valuetext={inspected ? `${date(inspected.date)}: ${money(inspected.close)}, volume ${volumeLabel(inspected.volume)}` : undefined}
            onKeyDown={e => {
              const index = e.key === "Home" ? 0 : e.key === "End" ? context.length - 1 : e.key === "ArrowRight" || e.key === "ArrowUp" ? inspectedIndex + 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? inspectedIndex - 1 : null;
              if (index !== null) {e.preventDefault(); setInspection({key: inspectionKey, index: Math.max(0, Math.min(index, context.length - 1))});}
            }}
            onPointerMove={inspectPointer} onPointerDown={inspectPointer}>
            <text x="20" y="16">{money(max)}</text><text x="20" y="172">{money(min)}</text>
            <polyline points={context.map((row, i) => `${x(i)},${y(row.close)}`).join(" ")} fill="none" stroke="var(--chart)" strokeWidth="2" />
            <line x1={x(Math.min(10, position))} x2={x(Math.min(10, position))} y1="25" y2="145" stroke="var(--risk-recovery)" strokeDasharray="4 4" />
            <circle cx={x(Math.min(10, position))} cy={y(event.close)} r="6" fill="var(--risk-recovery)"><title>Spike: {date(event.date)}, {money(event.close)}</title></circle>
            {hasVolume && <g className="spike-volume-bars">
              <text x="20" y="198">Volume</text><text x="620" y="198" textAnchor="end">{new Intl.NumberFormat("en-US", {notation: "compact"}).format(maxVolume)} max</text>
              <line x1="20" x2="620" y1="265" y2="265" stroke="var(--line)" />
              {context.map((row, i) => row.volume == null ? null : <rect key={row.date}
                x={x(i) - Math.min(9, 220 / context.length)} y={265 - row.volume / (maxVolume || 1) * 55}
                width={Math.min(18, 440 / context.length)} height={row.volume / (maxVolume || 1) * 55}
                fill={row.date === event.date ? "var(--risk-recovery)" : "var(--chart)"} opacity={i === inspectedIndex ? 1 : 0.55}>
                <title>{date(row.date)}: {volumeLabel(row.volume)} shares</title>
              </rect>)}
            </g>}
            {inspected && <g className="risk-price-cursor"><line x1={x(inspectedIndex)} x2={x(inspectedIndex)} y1="25" y2={hasVolume ? 265 : 145} /><circle cx={x(inspectedIndex)} cy={y(inspected.close)} r="5" /></g>}
          </svg>
          <div className="history-dates"><span>{date(context[0].date)}</span><span>{date(context[context.length - 1].date)}</span></div>
          {inspected && <output className="risk-price-readout">{date(inspected.date)} · <strong>{money(inspected.close)}</strong> adjusted close · Volume: {volumeLabel(inspected.volume)}{inspected.volume != null ? " shares" : ""}</output>}
        </div>
        <SpikeExplanation key={`${ticker}:${event.date}`} analysis={event.analysis!} evaluation={data.evaluation} />
      </>}
    </>}
  </section>;
}
