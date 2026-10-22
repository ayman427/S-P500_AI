import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  ArrowDownRight,
  ArrowUpRight,
  CircleAlert,
  LoaderCircle,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { createRequestGate } from "./requestGate";
import "./App.css";
import { UserButton } from "@clerk/react";
import { useAuthenticatedFetch } from "./useAuthenticatedFetch";
import { stockResponse as responseData, stockSearchError } from "./stockResponse";
import { resolveStock } from "./stockSearch";
import type { Stock } from "./stockSearch";
import HistoricalRisk from "./HistoricalRisk";
import ResearchSidebar from "./ResearchSidebar";

type ModelType = "hist_gradient_boosting" | "ridge" | "hybrid";
const MODEL_LABELS: Record<ModelType, string> = {
  hist_gradient_boosting: "Gradient Boosting",
  ridge: "Ridge regression",
  hybrid: "Hybrid (experimental)",
};
type Period = "open_to_close" | "open_to_close_5d";
const PERIOD_LABELS: Record<Period, string> = {
  open_to_close: "1 trading session", open_to_close_5d: "5 trading sessions",
};

type TradingResults = {
  trades: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  cumulative_return: number;
};

type StockResults = {
  ticker: string;
  tickers: string[];
  modelType: ModelType;
  targetKey: Period;
  companyName?: string;
  liveHistory?: {saved?: boolean; created?: boolean; recordId?: string; message: string; rows: unknown[]};
  modelReady: boolean;
  artifactIntegrity?: "verified" | "legacy_unverified";
  quote: { close: number; dailyChange: number | null; date: string };
  history: { date: string; close: number }[];
  forecast: { as_of: string; predicted_return: number; hybrid?: {ridgeWeight: number; feedbackSamples: number} } | null;
  forecastError?: string | null;
  forecastNeedsRetraining?: boolean;
  metrics?: {
    development_walk_forward?: {
      trading?: TradingResults;
      mean_absolute_return_error?: number;
      zero_return_baseline_mean_absolute_return_error?: number;
      validation_folds?: { validation_start: string; validation_end: string }[];
    };
  } | null;
};

const money = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(value);
const percent = (value: number | null | undefined, signed = false) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : `${signed && value > 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
const date = (value?: string) =>
  value
    ? new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      }).format(new Date(`${value.slice(0, 10)}T12:00:00`))
    : "Not available";

function App({userId}: {userId: string}) {
  const apiFetch = useAuthenticatedFetch();
  const modelStorageKey = `northstar-model:${userId}`;
  const periodStorageKey = `northstar-target:${userId}`;
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      return localStorage.getItem("northstar-theme") === "dark"
        ? "dark"
        : "light";
    } catch {
      return "light";
    }
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#000000" : "#f4f6ef");
    try {
      localStorage.setItem("northstar-theme", theme);
    } catch {
      /* Keep the toggle usable when browser storage is unavailable. */
    }
  }, [theme]);

  useEffect(() => {
    const syncTheme = (event: StorageEvent) => {
      if (event.key === "northstar-theme")
        setTheme(event.newValue === "dark" ? "dark" : "light");
    };
    window.addEventListener("storage", syncTheme);
    return () => window.removeEventListener("storage", syncTheme);
  }, []);

  const [modelType, setModelType] = useState<ModelType>(() => {
    try {
      const saved = localStorage.getItem(modelStorageKey);
      return saved === "ridge" || saved === "hybrid" ? saved : "hist_gradient_boosting";
    } catch {
      return "hist_gradient_boosting";
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(modelStorageKey, modelType);
    } catch {
      /* The in-memory selection remains usable. */
    }
  }, [modelType, modelStorageKey]);

  const [period, setPeriod] = useState<Period>(() => {
    try { return localStorage.getItem(periodStorageKey) === "open_to_close_5d" ? "open_to_close_5d" : "open_to_close"; }
    catch { return "open_to_close"; }
  });
  useEffect(() => {
    try { localStorage.setItem(periodStorageKey, period); }
    catch { /* The in-memory selection remains usable. */ }
  }, [period, periodStorageKey]);

  const [query, setQuery] = useState("");
  const [result, setResult] = useState<StockResults | null>(null);
  const [activeTab, setActiveTab] = useState<"forecast" | "risk">("forecast");
  const [riskVisited, setRiskVisited] = useState(false);
  const [predictionsRefresh, setPredictionsRefresh] = useState(0);
  const [saveConfirmationVisible, setSaveConfirmationVisible] = useState(true);
  useEffect(() => {
    if (!result?.liveHistory?.saved || !result.liveHistory.recordId) return;
    const timer = window.setTimeout(() => setSaveConfirmationVisible(false), 5000);
    return () => window.clearTimeout(timer);
  }, [result?.liveHistory]);
  function selectTab(tab: "forecast" | "risk") {
    setActiveTab(tab);
    if (tab === "risk") setRiskVisited(true);
  }
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [choices, setChoices] = useState<Stock[]>([]);
  const directoryRef = useRef<Promise<Stock[]> | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    directoryRef.current = apiFetch("/api/stocks", {signal: controller.signal}).then(responseData).then((payload: {stocks: Stock[]}) => {
      if (!controller.signal.aborted) setStocks(payload.stocks);
      return payload.stocks;
    }).catch(() => []);
    return () => controller.abort();
  }, [apiFetch]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const gateRef = useRef(createRequestGate());
  const controllerRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const gate = gateRef.current;
    return () => {
      gate.select("");
      controllerRef.current?.abort();
    };
  }, []);

  async function search(symbol: string, selectedModel: ModelType = modelType, selectedPeriod: Period = period) {
    if (!symbol.trim()) { inputRef.current?.focus(); return; }
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const selection = `${symbol.trim().toUpperCase()}:${selectedModel}:${selectedPeriod}`;
    gateRef.current.select(selection);
    const isCurrent = gateRef.current.begin(selection)!;
    let ticker = symbol.trim().toUpperCase();
    let stage = "find the stock";
    setQuery(symbol);
    setResult(null);
    setActiveTab("forecast");
    setRiskVisited(false);
    setSaveConfirmationVisible(true);
    setChoices([]);
    setError("");
    setBusy(true);
    setStatus("Finding your stock...");
    const load = async (): Promise<StockResults> => {
      const response = await apiFetch(`/api/state?ticker=${encodeURIComponent(ticker)}&model_type=${selectedModel}&target_key=${selectedPeriod}`, {signal: controller.signal});
      const payload = await responseData(response) as StockResults;
      if (payload.ticker !== ticker || payload.modelType !== selectedModel || payload.targetKey !== selectedPeriod)
        throw new Error("The results did not match your search. Please try again.");
      return payload;
    };
    try {
      const directory = stocks.length ? stocks : await directoryRef.current ?? [];
      if (!isCurrent()) return;
      const matches = resolveStock(symbol, directory);
      if (matches.length !== 1) {
        setChoices(matches);
        setError(matches.length ? "Several stocks match that name. Choose one below, or refine your search." : "No matching company found. Try a company name or stock symbol.");
        return;
      }
      ticker = matches[0].ticker;
      const companyName = matches[0].name;
      setQuery(ticker);
      stage = "refresh market prices";
      setStatus(`Fetching the latest completed prices for ${ticker}...`);
      const refreshed = await responseData(await apiFetch("/api/prices/refresh", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ticker}), signal: controller.signal,
      }));
      if (!isCurrent()) return;
      stage = "load results";
      setStatus(`Generating ${ticker} results using prices through ${date(refreshed.asOf)}...`);
      let payload = await load();
      if (!isCurrent()) return;
      if (payload.forecastNeedsRetraining || !payload.metrics?.development_walk_forward || (!payload.modelReady && !payload.forecastError)) {
        stage = "train the model";
        setStatus(`Preparing ${ticker} with ${MODEL_LABELS[selectedModel]} (${PERIOD_LABELS[selectedPeriod]}). This may take a few minutes...`);
        await responseData(await apiFetch("/api/action/train", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ticker, model_type: selectedModel, target_key: selectedPeriod}), signal: controller.signal,
        }));
        if (!isCurrent()) return;
        stage = "load results";
        payload = await load();
        if (!payload.metrics?.development_walk_forward) throw new Error("Results could not be prepared. Please try again.");
      }
      if (!isCurrent()) return;
      if (payload.forecast && payload.forecast.as_of !== (refreshed.inputAsOf ?? refreshed.asOf))
        throw new Error("The model could not use the latest completed prices. Please try again shortly.");
      setResult({...payload, companyName});
      try {
        const liveHistory = await responseData(await apiFetch("/api/predictions", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ticker, model_type: selectedModel, target_key: selectedPeriod}), signal: controller.signal,
        }));
        if (selectedModel === "hybrid" && liveHistory.saved) {
          const confirmed = liveHistory.forecast;
          if (!confirmed || confirmed.ticker !== ticker || confirmed.target_key !== selectedPeriod || confirmed.as_of !== payload.forecast?.as_of
              || !Number.isFinite(confirmed.predicted_return) || !confirmed.hybrid || !Number.isFinite(confirmed.hybrid.ridgeWeight)
              || confirmed.hybrid.ridgeWeight < 0 || confirmed.hybrid.ridgeWeight > 1 || !Number.isSafeInteger(confirmed.hybrid.feedbackSamples) || confirmed.hybrid.feedbackSamples < 0)
            throw new Error("The saved hybrid forecast did not match this search.");
          payload = {...payload, forecast: confirmed};
        }
        if (isCurrent()) {
          setResult({...payload, companyName, liveHistory});
          if (liveHistory.saved) setPredictionsRefresh(value => value + 1);
        }
      } catch (reason) {
        if (isCurrent()) setResult({...payload, companyName, liveHistory: {
          message: reason instanceof Error ? reason.message : "Live prediction storage is unavailable. No save was confirmed.", rows: [],
        }});
      }
    } catch (reason) {
      if (!isCurrent() || controller.signal.aborted) return;
      setResult(null);
      setError(stockSearchError(reason, ticker, stage));
    } finally {
      if (isCurrent()) { setBusy(false); setStatus(""); }
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!busy) void search(query);
  }

  const validation = result?.metrics?.development_walk_forward;
  const trading = validation?.trading;
  const folds = validation?.validation_folds ?? [];
  const history =
    result?.history.filter((point) => Number.isFinite(point.close)) ?? [];
  const prices = history.map((point) => point.close);
  const low = prices.length ? Math.min(...prices) : 0;
  const range = prices.length ? Math.max(...prices) - low : 0;
  const points = prices
    .map(
      (price, index) =>
        `${(index / Math.max(prices.length - 1, 1)) * 640},${range ? 104 - ((price - low) / range) * 88 : 60}`,
    )
    .join(" ");
  const forecast = result?.modelReady ? result.forecast : null;
  const change = result?.quote.dailyChange;
  const forecastMessage = /SEC|fundamental/i.test(result?.forecastError ?? "")
    ? "This model needs company financial data that is currently unavailable. Historical test results are shown below."
    : /history|prices|market data|SPY/i.test(result?.forecastError ?? "")
      ? "The saved price or market data is insufficient for a forecast. Historical test results are shown below."
      : result?.forecastNeedsRetraining
        ? "The saved model is incompatible with this forecast target. Search again to prepare an updated model."
        : "The forecast could not be generated for this model. Historical test results are shown below.";

  const modelControls = (
        <div className="model-choice">
          <div className="selection-field">
            <label htmlFor="stock-model">Model</label>
            <select id="stock-model" value={modelType} disabled={busy} onChange={event => {
              const next = event.target.value as ModelType;
              setModelType(next);
              setError("");
              if (result) void search(result.ticker, next, period);
            }}>
              <option value="hist_gradient_boosting">Gradient Boosting</option>
              <option value="ridge">Ridge regression</option>
              <option value="hybrid">Hybrid (experimental)</option>
            </select>
          </div>
          <div className="selection-field">
            <label htmlFor="stock-period">Period</label>
            <select id="stock-period" value={period} disabled={busy} onChange={event => {
              const next = event.target.value as Period;
              setPeriod(next);
              setError("");
              if (result) void search(result.ticker, modelType, next);
            }}>
              <option value="open_to_close">1 trading session</option>
              <option value="open_to_close_5d">5 trading sessions</option>
            </select>
          </div>
        </div>
  );

  return (
    <main className="stock-page">
      <div className="stock-layout">
      <section className="stock-card" aria-label="Stock search">
        <header className="search-header">
          <div className="brand-row">
            <UserButton />
            <button
              className="theme-toggle"
              type="button"
              aria-label="Dark mode"
              aria-pressed={theme === "dark"}
              title={
                theme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
              onClick={() =>
                setTheme((current) => (current === "dark" ? "light" : "dark"))
              }
            >
              {theme === "dark" ? (
                <Sun size={18} aria-hidden="true" />
              ) : (
                <Moon size={18} aria-hidden="true" />
              )}
            </button>
          </div>
        </header>

        <form className="stock-search" onSubmit={submit} role="search">
          <label className="sr-only" htmlFor="stock-symbol">
            Company name or stock symbol
          </label>
          <Search className="search-icon" size={20} aria-hidden="true" />
          <input
            ref={inputRef}
            id="stock-symbol"
            list="stock-symbols"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search a company or symbol, e.g. Apple"
            autoComplete="off"
            autoCapitalize="characters"
            spellCheck={false}
            disabled={busy}
            required
          />
          <datalist id="stock-symbols">
            {stocks.map(stock => (
              <option key={stock.ticker} value={`${stock.name} (${stock.ticker})`} label={stock.ticker} />
            ))}
          </datalist>
          <button
            type="submit"
            aria-label={busy ? "Preparing stock results" : "Show stock results"}
            disabled={busy || !query.trim()}
          >
            {busy ? (
              <LoaderCircle className="spin" size={18} />
            ) : (
              <>
                <span>Show results</span>
                <ArrowUpRight size={17} />
              </>
            )}
          </button>
        </form>
        {!result && modelControls}

        {!result && !busy && !error && (
          <div className="search-examples">
            <span>Try</span>
            {["AAPL", "MSFT", "NVDA", "AMD"].map((ticker) => (
              <button
                type="button"
                key={ticker}
                onClick={() => void search(ticker)}
              >
                {ticker}
                <ArrowUpRight size={11} />
              </button>
            ))}
          </div>
        )}
        <div aria-live="polite" aria-atomic="true">
          {busy && (
            <div className="search-status" role="status">
              <LoaderCircle className="spin" size={24} />
              <p>{status}</p>
            </div>
          )}
        </div>
        {error && (
          <div className="search-error" role="alert">
            <CircleAlert size={18} />
            <p>{error}</p>
          </div>
        )}

        {choices.length > 0 && <ul className="stock-matches" aria-label="Choose a matching stock">
          {choices.slice(0, 8).map(stock => <li key={stock.ticker}><button type="button" onClick={() => void search(stock.ticker)}><span>{stock.name}</span><strong>{stock.ticker}</strong></button></li>)}
        </ul>}
        {result && (
          <div
            className="stock-result"
            key={`${result.ticker}:${result.modelType}:${result.targetKey}`}
          >
            <div className="result-heading">
              <div className="stock-identity">
                <span className="ticker-icon">{result.ticker.slice(0, 1)}</span>
                <div>
                  <h2>{result.ticker}</h2>
                  {result.companyName !== result.ticker && <p className="company-name">{result.companyName}</p>}
                  <p>Saved closing price · {date(result.quote.date)}</p>
                </div>
              </div>
              <div className="stock-price">
                <strong>{money(result.quote.close)}</strong>
                <span
                  className={
                    change != null && change < 0 ? "negative" : "positive"
                  }
                >
                  {change != null &&
                    (change < 0 ? (
                      <ArrowDownRight size={14} />
                    ) : (
                      <ArrowUpRight size={14} />
                    ))}
                  {percent(change, true)} <small>last session</small>
                </span>
              </div>
            </div>

            <div className="result-tabs" role="tablist" aria-label="Stock analysis">
              {(["forecast", "risk"] as const).map(tab => <button key={tab} type="button" role="tab" id={`stock-tab-${tab}`}
                aria-selected={activeTab === tab} aria-controls={`stock-panel-${tab}`} tabIndex={activeTab === tab ? 0 : -1}
                onClick={() => selectTab(tab)} onKeyDown={event => {
                  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
                  event.preventDefault();
                  const tabs = ["forecast", "risk"] as const;
                  const next = event.key === "Home" ? tabs[0] : event.key === "End" ? tabs[1] : tabs[(tabs.indexOf(tab) + 1) % tabs.length];
                  selectTab(next);
                  document.getElementById(`stock-tab-${next}`)?.focus();
                }}>{tab === "forecast" ? "Forecast" : "Historical risk"}</button>)}
            </div>
            <div className="result-panel" id="stock-panel-forecast" role="tabpanel" aria-labelledby="stock-tab-forecast" hidden={activeTab !== "forecast"} tabIndex={0}>
              {modelControls}
              {(!(result.liveHistory?.saved && result.liveHistory.recordId) || saveConfirmationVisible) &&
                <div className={`save-confirmation${result.liveHistory?.saved && result.liveHistory.recordId ? " save-confirmation-success" : result.liveHistory ? " save-confirmation-error" : ""}`} role="status" aria-live="polite">
                  <strong>{result.liveHistory?.saved && result.liveHistory.recordId ? result.liveHistory.created ? "Prediction saved to Convex" : "Prediction already saved in Convex" : result.liveHistory ? "Prediction save not confirmed" : "Saving prediction to Convex..."}</strong>
                  <span>{result.liveHistory?.message ?? "Waiting for the database confirmation."}</span>
                </div>}
              <p className="model-attribution">Results from <strong>{MODEL_LABELS[result.modelType]}</strong></p>
              {result.modelType === "hybrid" && <p className="risk-caption">Experimental · learns blend weights from your settled prediction errors.
                {result.forecast?.hybrid && <> Ridge {(result.forecast.hybrid.ridgeWeight * 100).toFixed(0)}% · Boosting {((1-result.forecast.hybrid.ridgeWeight) * 100).toFixed(0)}% · {result.forecast.hybrid.feedbackSamples} completed feedback outcomes.</>}
              </p>}
            {history.length > 1 && (
              <div className="price-history">
                <svg
                  viewBox="0 0 640 120"
                  preserveAspectRatio="none"
                  role="img"
                  aria-label={`${result.ticker} saved closing prices from ${date(history[0].date)} to ${date(history[history.length - 1].date)}`}
                >
                  <defs>
                    <linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="0%"
                        stopColor="var(--chart)"
                        stopOpacity=".13"
                      />
                      <stop
                        offset="100%"
                        stopColor="var(--chart)"
                        stopOpacity="0"
                      />
                    </linearGradient>
                  </defs>
                  <polygon
                    points={`0,120 ${points} 640,120`}
                    fill="url(#price-fill)"
                  />
                  <polyline
                    points={points}
                    fill="none"
                    stroke="var(--chart)"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
                <div className="history-dates">
                  <span>{date(history[0].date)}</span>
                  <span>{date(history[history.length - 1].date)}</span>
                </div>
              </div>
            )}

            <div className="forecast-result">
              <div>
                <span className="result-label">
                  {forecast
                    ? (result.targetKey === "open_to_close_5d" ? "ESTIMATED 5-SESSION RETURN" : "ESTIMATED NEXT-SESSION RETURN")
                    : "FORECAST UNAVAILABLE"}
                </span>
                <p>
                  {result.targetKey === "open_to_close_5d" ? "Next open ? fifth-session close ? Before costs" : "Next open ? close ? Before costs"}
                </p>
              </div>
              <strong
                className={
                  forecast && forecast.predicted_return < 0
                    ? "negative"
                    : "positive"
                }
              >
                {percent(forecast?.predicted_return, true)}
              </strong>
            </div>
            <p className="forecast-date">
              {forecast
                ? `Data through ${date(forecast.as_of)}${forecast.as_of < result.quote.date ? " ? Market inputs delayed" : ""}`
                : forecastMessage}
            </p>

            <details className="model-performance"><summary>Model performance</summary>
            <div className="past-heading">
              <h3>{MODEL_LABELS[result.modelType]} test results</h3>
              {result.modelType === "hybrid" && <p className="risk-caption">Historical tests evaluate the initial blend. Live error learning is not yet validated.</p>}
              <span>{PERIOD_LABELS[result.targetKey]}</span>
            </div>
            <div className="result-metrics">
              <div>
                <span>Win rate</span>
                <strong>{percent(trading?.win_rate)}</strong>
                <small>
                  {trading
                    ? `${trading.wins} wins · ${trading.losses} losses`
                    : "No trade results"}
                </small>
              </div>
              <div>
                <span>Return after costs</span>
                <strong
                  className={
                    trading && trading.cumulative_return < 0 ? "negative" : ""
                  }
                >
                  {percent(trading?.cumulative_return, true)}
                </strong>
                <small>Cumulative over test periods</small>
              </div>
              <div>
                <span>Trades tested</span>
                <strong>{trading?.trades.toLocaleString() ?? "—"}</strong>
                <small>
                  {trading?.trades === 0
                    ? "No entries met the signal rule"
                    : "Out-of-sample trades"}
                </small>
              </div>
              <div>
                <span>Typical prediction error</span>
                <strong>
                  {percent(validation?.mean_absolute_return_error)}
                </strong>
                <small>
                  No-change baseline:{" "}
                  {percent(
                    validation?.zero_return_baseline_mean_absolute_return_error,
                  )}
                </small>
              </div>
            </div>
            {folds.length > 0 && (
              <p className="validation-dates">
                Test dates: {date(folds[0].validation_start)} –{" "}
                {date(folds[folds.length - 1].validation_end)}.
              </p>
            )}
            {result.artifactIntegrity === "legacy_unverified" && (
              <p className="integrity-note">
                <CircleAlert size={15} />
                <span>
                  Older saved report. Its pairing with the model has not been
                  verified.
                </span>
              </p>
            )}
            <p className="result-footnote">
              Historical tests, not live trading performance.
            </p>
            </details>
            </div>
            <div className="result-panel" id="stock-panel-risk" role="tabpanel" aria-labelledby="stock-tab-risk" hidden={activeTab !== "risk"} tabIndex={0}>
              {riskVisited && <HistoricalRisk key={result.ticker} ticker={result.ticker} />}
            </div>
          </div>
        )}
      </section>
      <ResearchSidebar key={userId} refreshKey={predictionsRefresh} />
      </div>
    </main>
  );
}

export default App;
