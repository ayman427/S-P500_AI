import { useEffect, useRef, useState } from "react";
import { useAuthenticatedFetch } from "./useAuthenticatedFetch";
import { stockResponse } from "./stockResponse";
import "./MyPredictions.css";

type Prediction = {
  _id: string; _creationTime: number; ticker: string; modelType: string; targetKey: string;
  signalDate: string; predictedReturn: number; status: "pending" | "settled";
  actualReturn?: number; exitDate: string; modelHash: string;
};
type Page = {rows: Prediction[]; cursor: string | null; hasMore: boolean};
const percent = (value?: number) => value == null || !Number.isFinite(value) ? "—" : `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
const date = (value: string) => new Intl.DateTimeFormat("en-US", {month: "short", day: "numeric", year: "numeric"}).format(new Date(`${value}T12:00:00`));

export default function MyPredictions({refreshKey}: {refreshKey: number}) {
  const apiFetch = useAuthenticatedFetch();
  const [refresh, setRefresh] = useState(0);
  const key = `${refreshKey}:${refresh}`;
  const [snapshot, setSnapshot] = useState<(Page & {key: string; error?: string}) | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const moreController = useRef<AbortController | null>(null);
  const page = snapshot?.key === key ? snapshot : null;
  useEffect(() => {
    const controller = new AbortController();
    apiFetch("/api/predictions", {signal: controller.signal}).then(stockResponse).then((data: Page) => {
      if (!controller.signal.aborted) {
        setSnapshot({...data, key});
        setLoadingMore(false);
      }
    }).catch(reason => {
      if (!controller.signal.aborted) setSnapshot({key, rows: [], cursor: null, hasMore: false,
        error: reason instanceof Error ? reason.message : "Could not load your predictions."});
    });
    return () => {controller.abort(); moreController.current?.abort();};
  }, [apiFetch, key]);

  async function loadMore() {
    if (!page?.hasMore || !page.cursor || loadingMore) return;
    const controller = new AbortController();
    moreController.current = controller;
    setLoadingMore(true);
    try {
      const next = await stockResponse(await apiFetch(`/api/predictions?cursor=${encodeURIComponent(page.cursor)}`, {signal: controller.signal})) as Page;
      if (!controller.signal.aborted) setSnapshot(current => current?.key === key ? {
        ...next, key, rows: [...new Map([...current.rows, ...next.rows].map(row => [row._id, row])).values()],
      } : current);
    } catch (reason) {
      if (!controller.signal.aborted) setSnapshot(current => current?.key === key ? {...current,
        error: reason instanceof Error ? reason.message : "Could not load older predictions."} : current);
    } finally {
      if (!controller.signal.aborted) setLoadingMore(false);
    }
  }

  return <section aria-labelledby="my-predictions-heading">
    <header><h2 id="my-predictions-heading">Predictions for you</h2>
      <button type="button" onClick={() => setRefresh(value => value + 1)} aria-label="Refresh your predictions">Refresh</button>
    </header>
    {!page ? <p role="status">Loading your predictions…</p> : <>
      {page.error && <p role="alert">{page.error}</p>}
      {!page.error && page.rows.length === 0 && <p>Your saved predictions will appear here.</p>}
      <ol className="my-prediction-list">{page.rows.map(row => <li key={row._id}>
        <div className="my-prediction-title"><h3>{row.ticker}</h3><span>{row.status === "settled" ? "Completed" : "Pending"}</span></div>
        <p>{row.modelType === "ridge" ? "Ridge regression" : row.modelType === "hist_gradient_boosting" ? "Gradient Boosting" : row.modelType === "hybrid" ? "Hybrid (experimental)" : row.modelType} · {row.targetKey === "open_to_close_5d" ? "5 sessions" : "1 session"}</p>
        <dl><div><dt>Predicted</dt><dd>{percent(row.predictedReturn)}</dd></div>
          <div><dt>Actual</dt><dd>{row.status === "settled" ? percent(row.actualReturn) : "Awaiting results"}</dd></div></dl>
        <p>Signal {date(row.signalDate)} · Ends {date(row.exitDate)}</p>
        <p className="my-prediction-created">Saved {new Intl.DateTimeFormat("en-US", {month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit"}).format(row._creationTime)}
          <span title={row.modelHash}> · Model {row.modelHash.slice(0, 8)}</span></p>
      </li>)}</ol>
      {page.hasMore && <button className="my-predictions-more" type="button" disabled={loadingMore} onClick={() => void loadMore()}>{loadingMore ? "Loading…" : "Load older predictions"}</button>}
    </>}
  </section>;
}
