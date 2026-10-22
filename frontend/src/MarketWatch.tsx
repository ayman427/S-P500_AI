import {useEffect, useState} from "react";
import {useAuthenticatedFetch} from "./useAuthenticatedFetch";
import {stockResponse} from "./stockResponse";

type Check = {id: "volume" | "momentum" | "breakout"; status: string; value: number | null};
type Row = {ticker: string; asOf: string; status: string; met: number; checks: Check[]};
type Scan = {expectedSession: string; updatedAt: string; universe: string[]; processed: number; complete: boolean; rows: Row[]; unavailable: string[]};
type Payload = {scan: Scan | null; expectedSession: string; busy?: boolean};
const labels = {volume: "Volume buildup", momentum: "5-session momentum", breakout: "20-session breakout"};

export default function MarketWatch() {
  const apiFetch = useAuthenticatedFetch();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [limit, setLimit] = useState(20);
  const [restart, setRestart] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll(first: boolean) {
      if (controller.signal.aborted) return;
      if (document.hidden) {timer = setTimeout(() => void poll(first), 60000); return;}
      try {
        let data = await stockResponse(await apiFetch("/api/market-watch", {signal: controller.signal})) as Payload;
        if (!controller.signal.aborted) {setPayload(data); setError("");}
        if (!data.scan?.complete || data.scan.expectedSession !== data.expectedSession || (first && restart)) {
          data = await stockResponse(await apiFetch(`/api/market-watch/scan${first && restart ? "?restart=true" : ""}`,
            {method: "POST", signal: controller.signal})) as Payload;
          if (!controller.signal.aborted) setPayload(data);
        }
        if (!controller.signal.aborted) timer = setTimeout(() => void poll(false), data.scan?.complete ? 300000 : 15000);
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Could not load the market checklist.");
          timer = setTimeout(() => void poll(false), 60000);
        }
      }
    }
    void poll(true);
    return () => {controller.abort(); clearTimeout(timer);};
  }, [apiFetch, refresh, restart]);
  const scan = payload?.scan;
  const outdated = !!scan && scan.expectedSession !== payload?.expectedSession;
  const eligible = scan?.rows.filter(row => ["watch", "no_signal"].includes(row.status)) ?? [];
  const ranked = eligible.filter(row => row.met > 0).sort((a, b) => b.met - a.met || a.ticker.localeCompare(b.ticker));
  const excluded = (scan?.rows.length ?? 0) - eligible.length + (scan?.unavailable.length ?? 0);
  return <section className="market-watch" aria-labelledby="market-watch-heading">
    <header><h2 id="market-watch-heading">S&P 500 rise candidates</h2>
      <button type="button" disabled={!scan?.complete || outdated} onClick={() => {setRestart(true); setRefresh(n => n + 1);}}>Rescan</button>
    </header>
    <p>Price / volume discovery · 3 checks</p>
    {error && <p role="alert">{error} <button type="button" onClick={() => {setRestart(false); setRefresh(n => n + 1);}}>Retry</button></p>}
    {!payload && !error && <p role="status">Loading market scan…</p>}
    {payload && !scan && <p role="status">Starting the S&P 500 scan…</p>}
    {scan && <>
      <p role="status">{outdated ? "Previous session · updating" : scan.complete ? "Scan complete" : "Scanning"} · {scan.processed}/{scan.universe.length} stocks</p>
      <progress aria-label="Stocks scanned" value={scan.processed} max={scan.universe.length} />
      <p>{eligible.length} {outdated ? "from previous session" : "current"} · {excluded} unavailable / incomplete<br />Session {scan.expectedSession} · Updated {new Date(scan.updatedAt).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}</p>
      {!ranked.length && <p>{scan.complete ? "No stocks met a check in the available data." : "Rankings appear as stocks are checked."}</p>}
      <ol className="market-ranking">{ranked.slice(0, limit).map((row, index) => <li key={row.ticker}>
        <details><summary><span>{index + 1}. {row.ticker}</span><strong className={row.met === 3 && !outdated ? "checklist-caution" : ""}>{row.met}/3{row.met === 3 && !outdated ? " · Watch" : ""}</strong></summary>
          <ul>{row.checks.map(check => <li key={check.id}>
            <span>{check.status === "met" ? "✓" : check.status === "unknown" ? "?" : "—"} {labels[check.id]}</span>
            <strong>{check.value === null ? "Unavailable" : check.id === "volume" ? `${check.value.toFixed(2)}×` : `${(check.value * 100).toFixed(2)}%`}</strong>
          </li>)}</ul>
        </details>
      </li>)}</ol>
      {ranked.length > limit && <button type="button" className="my-predictions-more" onClick={() => setLimit(n => n + 20)}>Show more stocks</button>}
      {excluded > 0 && <details className="checklist-rules"><summary>Missing coverage ({excluded})</summary>
        <p>{scan.rows.filter(row => !["watch", "no_signal"].includes(row.status)).map(row => `${row.ticker}: ${row.status} (${row.asOf})`).join(" · ")}</p>
        {scan.unavailable.length > 0 && <p>Price refresh failed: {scan.unavailable.join(", ")}</p>}
      </details>}
    </>}
    <details className="checklist-rules"><summary>Checklist rules</summary>
      <p>Volume: last 3 sessions ≥ 1.5× the preceding 20.<br />Momentum: 5-session gain ≥ 2%.<br />Breakout: close above the preceding 20-session high.</p>
      <p>Discovery only. Use Historical risk → Rise investigation to examine dated evidence and subsequent falls. These checks do not assess manipulation.</p>
    </details>
  </section>;
}
