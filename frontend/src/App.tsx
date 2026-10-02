import axios from "axios";
import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  BookOpenCheck,
  CalendarClock,
  Check,
  CircleHelp,
  Clock3,
  ExternalLink,
  FileClock,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  Newspaper,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  Moon,
  WalletCards,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

type Headline = {
  published_at_utc: string;
  publisher: string;
  title: string;
  url: string;
  relationships?: {
    predicate: string;
    counterparty_candidate: string;
    counterparty_context: {
      title: string;
      snippet: string;
      url: string;
      match: string;
    }[];
  }[];
};

type TradeSummary = {
  trades: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  cumulative_return: number;
  max_drawdown: number;
  annualized_sharpe: number | null;
  profit_factor: number | null;
  win_rate_95pct_wilson_interval: [number, number] | null;
  baselines: {
    no_trade_cumulative_return: number;
    always_long: { cumulative_return: number; win_rate: number | null };
  };
};

type IntradayCandidateMetrics = {
  target: string;
  validation_rows: number;
  mean_absolute_return_error: number;
  directional_accuracy: number;
  trading: TradeSummary;
};

type DashboardState = {
  ticker: string;
  tickers: string[];
  modelType: ModelType;
  ridgeAlpha: number;
  quote: {
    date: string;
    close: number;
    previousClose: number | null;
    dailyChange: number | null;
    volume: number;
  };
  history: { date: string; close: number }[];
  forecast: {
    as_of: string;
    last_close: number;
    predicted_next_close: number;
  } | null;
  forecastError: string | null;
  modelReady: boolean;
  holdout?: {
    period_start?: string;
    period_end?: string;
    training_through?: string;
    sessions?: number;
    mae?: number;
    mean_absolute_return_error?: number;
    previous_close_baseline_mean_absolute_return_error?: number;
    previous_close_baseline_mae?: number;
    trading?: TradeSummary;
  };
  metrics?: {
    development_walk_forward?: { trading?: TradeSummary };
    development_intraday_model_comparison?: Record<
      string,
      IntradayCandidateMetrics
    >;
  };
  intradayComparison?: Record<string, IntradayCandidateMetrics> | null;
  paper?: {
    signals_logged: number;
    settled_sessions: number;
    pending_signals: number;
    trades: number;
    wins: number;
    losses: number;
    win_rate: number | null;
    strategy_cumulative_return: number;
    always_long_cumulative_return: number;
  };
  paperRows: Record<string, string | number | boolean | null>[];
  headlines: Headline[];
};

type ActionName = "update" | "train" | "paper-trade" | "news" | "compare";
type LiveQuote = {
  ticker: string;
  price: number;
  previousClose: number | null;
  dailyChange: number | null;
  volume: number | null;
  fetchedAt: string;
};
const LIVE_QUOTE_INTERVAL_MS = 15_000;
type Theme = "light" | "dark";
type ModelType = "hist_gradient_boosting" | "ridge";

const money = (value?: number | null) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
      }).format(value);

const percent = (value?: number | null, digits = 2) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;

const paperForecastReturn = (
  row: Record<string, string | number | boolean | null>,
) => {
  const predictedReturn =
    row.predicted_return == null ? Number.NaN : Number(row.predicted_return);
  if (Number.isFinite(predictedReturn)) return predictedReturn;
  const lastClose = Number(row.last_close);
  const predictedClose = Number(row.predicted_next_close);
  return Number.isFinite(lastClose) &&
    lastClose !== 0 &&
    Number.isFinite(predictedClose)
    ? predictedClose / lastClose - 1
    : null;
};

const shortDate = (value?: string | null) => {
  if (!value) return "—";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(`${value.slice(0, 10)}T12:00:00`));
};

function App() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return localStorage.getItem("northstar-theme") === "dark"
        ? "dark"
        : "light";
    } catch {
      return "light";
    }
  });
  const [ticker, setTicker] = useState("AMD");
  const [modelType, setModelType] = useState<ModelType>(() => {
    try {
      return localStorage.getItem("northstar-model") === "ridge"
        ? "ridge"
        : "hist_gradient_boosting";
    } catch {
      return "hist_gradient_boosting";
    }
  });
  const [ridgeAlpha, setRidgeAlpha] = useState(() => {
    try {
      const saved = Number(localStorage.getItem("northstar-ridge-alpha"));
      return Number.isFinite(saved) && saved > 0 ? saved : 10;
    } catch {
      return 10;
    }
  });
  const [tickerQuery, setTickerQuery] = useState("AMD");
  const [tickerPickerOpen, setTickerPickerOpen] = useState(false);
  const [activeTickerIndex, setActiveTickerIndex] = useState(0);
  const [view, setView] = useState<"overview" | "paper">("overview");
  const [data, setData] = useState<DashboardState | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<ActionName | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [liveQuote, setLiveQuote] = useState<LiveQuote | null>(null);

  const loadState = useCallback(
    async (selectedTicker: string, selectedModel: ModelType) => {
      try {
        const response = await fetch(
          `/api/state?ticker=${encodeURIComponent(selectedTicker)}&model_type=${selectedModel}`,
        );
        const payload = await response.json();
        if (!response.ok)
          throw new Error(
            payload.error || payload.detail || "Could not load workspace data.",
          );
        setData(payload as DashboardState);
        setError("");
        if (
          payload.tickers?.length &&
          !payload.tickers.includes(selectedTicker)
        ) {
          setTicker(
            payload.tickers.includes("AMD") ? "AMD" : payload.tickers[0],
          );
        }
      } catch (reason) {
        setError(
          reason instanceof Error
            ? reason.message
            : "Dashboard API is unavailable.",
        );
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("northstar-theme", theme);
    } catch {
      // Keep the in-memory theme usable when browser storage is disabled.
    }
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem("northstar-model", modelType);
    } catch {
      // Keep model selection usable when browser storage is disabled.
    }
  }, [modelType]);

  useEffect(() => {
    try {
      localStorage.setItem("northstar-ridge-alpha", String(ridgeAlpha));
    } catch {
      // Keep the current alpha usable when browser storage is disabled.
    }
  }, [ridgeAlpha]);

  useEffect(() => {
    const task = window.setTimeout(() => void loadState(ticker, modelType), 0);
    return () => window.clearTimeout(task);
  }, [ticker, modelType, loadState]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    let cancelled = false;

    const refresh = async () => {
      try {
        const response = await axios.get<LiveQuote>("/api/quote", {
          params: { ticker },
          signal: controller.signal,
          timeout: 10_000,
        });
        if (!cancelled) setLiveQuote(response.data);
      } catch {
        // Keep the last good quote; the stored end-of-day price remains the fallback.
      }
      if (!cancelled)
        timer = window.setTimeout(refresh, LIVE_QUOTE_INTERVAL_MS);
    };

    void refresh();
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [ticker]);

  async function runAction(
    action: ActionName,
    selectedModel: ModelType = modelType,
  ) {
    setBusy(action);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`/api/action/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker,
          model_type: selectedModel,
          ridge_alpha: ridgeAlpha,
        }),
      });
      const payload = await response.json();
      if (!response.ok)
        throw new Error(payload.error || payload.detail || "Action failed.");
      setNotice(
        action === "update"
          ? "Price data updated"
          : action === "train"
            ? "Model training complete"
            : action === "paper-trade"
              ? "Paper signal recorded"
              : action === "news"
                ? "News refreshed for " + ticker
                : "Development comparison saved for " + ticker,
      );
      await loadState(ticker, selectedModel);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Could not complete this action.",
      );
    } finally {
      setBusy(null);
    }
  }

  const holdout = data?.holdout;
  const trading = holdout?.trading;
  const intradayComparison =
    data?.intradayComparison ??
    data?.metrics?.development_intraday_model_comparison;
  const forecastGap = data?.forecast
    ? data.forecast.predicted_next_close - data.forecast.last_close
    : null;
  const live = liveQuote?.ticker === ticker ? liveQuote : null;
  const displayClose = live?.price ?? data?.quote.close;
  const displayChange = live ? live.dailyChange : data?.quote.dailyChange;
  const displayPreviousClose = live?.previousClose ?? data?.quote.previousClose;
  const displayVolume = live?.volume ?? data?.quote.volume;
  const isUp = (displayChange ?? 0) >= 0;
  const availableTickers = data?.tickers?.length ? data.tickers : [ticker];
  const tickerMatches = availableTickers
    .filter((symbol) => symbol.includes(tickerQuery.trim().toUpperCase()))
    .slice(0, 8);

  function selectTicker(symbol: string) {
    setTicker(symbol);
    setTickerQuery(symbol);
    setTickerPickerOpen(false);
    setActiveTickerIndex(0);
    setLoading(true);
    setError("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="Northstar home">
          <span className="brand-mark">
            <Activity size={18} strokeWidth={2.5} />
          </span>
          <span className="brand-name">
            northstar<span> / quant desk</span>
          </span>
        </a>

        <div className="workspace-label">WORKSPACE</div>
        <button
          aria-label="Market overview"
          title="Market overview"
          className={`nav-item ${view === "overview" ? "active" : ""}`}
          onClick={() => setView("overview")}
        >
          <LayoutDashboard size={17} /> <span>Market overview</span>
          <span className="nav-dot" />
        </button>
        <button
          aria-label="Paper journal"
          title="Paper journal"
          className={`nav-item ${view === "paper" ? "active" : ""}`}
          onClick={() => setView("paper")}
        >
          <BookOpenCheck size={17} /> <span>Paper journal</span>
          {data?.paper?.pending_signals ? (
            <span className="nav-count">{data.paper.pending_signals}</span>
          ) : null}
        </button>
        <div className="workspace-label section-label">MODEL HEALTH</div>
        <div className="sidebar-health">
          <span className={`health-lamp ${data?.modelReady ? "ready" : ""}`} />
          <div>
            <strong>
              {data?.modelReady ? "Model ready" : "Model not trained"}
            </strong>
            <small>{ticker} · local artifact</small>
          </div>
        </div>

        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <div className="note-icon">
            <ShieldCheck size={16} />
          </div>
          <strong>Research mode</strong>
          <p>Forecasts are experimental. Paper-trade first.</p>
        </div>
        <div className="profile-row">
          <div className="avatar">SP</div>
          <div>
            <strong>Local workspace</strong>
            <small>Personal research</small>
          </div>
          <CircleHelp size={16} className="muted-icon" />
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="crumb-slash">/</span>
            <strong>
              {view === "overview" ? "Market overview" : "Paper journal"}
            </strong>
          </div>
          <div className="topbar-actions">
            <span className="local-indicator">
              <span /> Local data
            </span>
            <button
              className="icon-button notification-button"
              aria-label="Notifications"
            >
              <Bell size={17} />
              <i />
            </button>
            <button
              className="icon-button theme-toggle"
              type="button"
              aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
              aria-pressed={theme === "dark"}
              title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
              onClick={() =>
                setTheme((current) => (current === "light" ? "dark" : "light"))
              }
            >
              {theme === "light" ? <Moon size={17} /> : <Sun size={17} />}
            </button>
            <div className="top-avatar">A</div>
          </div>
        </header>

        <div className="page-wrap">
          <section className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" /> EQUITY RESEARCH{" "}
                <span className="eyebrow-date">
                  · {shortDate(data?.quote.date)}
                </span>
              </div>
              <h1>{view === "overview" ? "Market desk" : "Paper journal"}</h1>
              <p className="heading-copy">
                {view === "overview"
                  ? "A clear read on the model, the tape, and its latest signal."
                  : "Track prospective calls against real next-session prices."}
              </p>
            </div>
            <div className="heading-controls">
              <div className="ticker-picker">
                <label className="ticker-search-wrap" htmlFor="ticker-search">
                  <Search size={15} />
                  <input
                    id="ticker-search"
                    type="search"
                    role="combobox"
                    aria-label="Search stocks by ticker"
                    aria-autocomplete="list"
                    aria-expanded={tickerPickerOpen}
                    aria-controls="ticker-options"
                    aria-activedescendant={
                      tickerPickerOpen && tickerMatches.length
                        ? `ticker-option-${tickerMatches[activeTickerIndex]}`
                        : undefined
                    }
                    autoComplete="off"
                    placeholder="Search ticker"
                    value={tickerQuery}
                    onFocus={() => setTickerPickerOpen(true)}
                    onChange={(event) => {
                      setTickerQuery(event.target.value.toUpperCase());
                      setTickerPickerOpen(true);
                      setActiveTickerIndex(0);
                    }}
                    onBlur={() => {
                      window.setTimeout(() => {
                        setTickerPickerOpen(false);
                        setTickerQuery(ticker);
                      }, 120);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "ArrowDown" && tickerMatches.length) {
                        event.preventDefault();
                        setTickerPickerOpen(true);
                        setActiveTickerIndex((index) =>
                          Math.min(index + 1, tickerMatches.length - 1),
                        );
                      } else if (
                        event.key === "ArrowUp" &&
                        tickerMatches.length
                      ) {
                        event.preventDefault();
                        setActiveTickerIndex((index) => Math.max(index - 1, 0));
                      } else if (
                        event.key === "Enter" &&
                        tickerMatches.length
                      ) {
                        event.preventDefault();
                        const exactMatch = tickerMatches.find(
                          (symbol) => symbol === tickerQuery.trim(),
                        );
                        selectTicker(
                          exactMatch ?? tickerMatches[activeTickerIndex],
                        );
                      } else if (event.key === "Escape") {
                        setTickerPickerOpen(false);
                        setTickerQuery(ticker);
                      }
                    }}
                  />
                </label>
                {tickerPickerOpen ? (
                  <div
                    className="ticker-options"
                    id="ticker-options"
                    role="listbox"
                  >
                    {tickerMatches.length ? (
                      tickerMatches.map((symbol, index) => (
                        <button
                          className={`ticker-option ${index === activeTickerIndex ? "active" : ""}`}
                          id={`ticker-option-${symbol}`}
                          key={symbol}
                          type="button"
                          role="option"
                          aria-selected={symbol === ticker}
                          onMouseDown={(event) => event.preventDefault()}
                          onMouseEnter={() => setActiveTickerIndex(index)}
                          onClick={() => selectTicker(symbol)}
                        >
                          <span>{symbol}</span>
                          <small>
                            {symbol === ticker ? "Selected" : "S&P 500"}
                          </small>
                        </button>
                      ))
                    ) : (
                      <div className="ticker-no-results">
                        No matching ticker
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
              <label className="model-picker" aria-label="Choose model">
                <span>MODEL</span>
                <select
                  value={modelType}
                  onChange={(event) =>
                    setModelType(event.target.value as ModelType)
                  }
                >
                  <option value="hist_gradient_boosting">
                    HistGradientBoosting
                  </option>
                  <option value="ridge">Ridge</option>
                </select>
              </label>
              {modelType === "ridge" ? (
                <label
                  className="ridge-alpha-picker"
                  title="Ridge regularization strength"
                >
                  <span>α</span>
                  <input
                    aria-label="Ridge alpha"
                    type="number"
                    min="0.01"
                    step="0.5"
                    value={ridgeAlpha}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      if (Number.isFinite(value) && value > 0) {
                        setRidgeAlpha(value);
                      }
                    }}
                  />
                </label>
              ) : null}
              <button
                className="button button-secondary"
                onClick={() => void runAction("update")}
                disabled={busy !== null}
              >
                {busy === "update" ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <RefreshCw size={15} />
                )}{" "}
                Sync prices
              </button>
              <button
                className="button button-secondary"
                onClick={() => void runAction("train")}
                disabled={busy !== null}
                title={`Train ${modelType === "ridge" ? `Ridge (alpha ${ridgeAlpha})` : "HistGradientBoosting"} for ${ticker}`}
              >
                {busy === "train" ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <BarChart3 size={15} />
                )}{" "}
                Train selected
              </button>
              <button
                className="button button-primary"
                onClick={() => void runAction("paper-trade")}
                disabled={busy !== null || !data?.modelReady}
              >
                {busy === "paper-trade" ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <WalletCards size={15} />
                )}{" "}
                Log signal
              </button>
            </div>
          </section>

          {error ? (
            <div className="alert alert-error">
              {error}
              <button onClick={() => void loadState(ticker, modelType)}>
                Retry
              </button>
            </div>
          ) : null}
          {notice ? (
            <div className="alert alert-success">
              <Check size={15} />
              {notice}
              <button onClick={() => setNotice("")} aria-label="Dismiss">
                ×
              </button>
            </div>
          ) : null}

          {loading && !data ? (
            <div className="loading-state">
              <LoaderCircle className="spin" size={25} />
              <span>Loading local market data…</span>
            </div>
          ) : null}
          {data && view === "overview" ? (
            <>
              <section className="quote-strip">
                <div className="quote-identity">
                  <div className="ticker-monogram">{ticker.slice(0, 1)}</div>
                  <div>
                    <div className="ticker-name">
                      {ticker}
                      <span className="exchange-pill">NASDAQ</span>
                    </div>
                    <div className="quote-date">
                      {live
                        ? "Live · Yahoo Finance"
                        : `Adjusted close · ${shortDate(data.quote.date)}`}
                    </div>
                  </div>
                </div>
                <div className="quote-price">
                  {money(displayClose)}
                  <span
                    className={`change-pill ${isUp ? "positive" : "negative"}`}
                  >
                    {isUp ? (
                      <ArrowUpRight size={14} />
                    ) : (
                      <ArrowDownRight size={14} />
                    )}
                    {percent(displayChange)}
                  </span>
                </div>
                <div className="quote-meta">
                  <span>Prev close</span>
                  <strong>{money(displayPreviousClose)}</strong>
                </div>
                <div className="quote-meta volume-meta">
                  <span>Volume</span>
                  <strong>
                    {displayVolume == null
                      ? "—"
                      : new Intl.NumberFormat("en-US", {
                          notation: "compact",
                        }).format(displayVolume)}
                  </strong>
                </div>
                <div className="market-state">
                  <span className="market-live-dot" />{" "}
                  {live ? "LIVE" : "EOD DATA"}
                </div>
              </section>

              <section className="stat-grid">
                <article className="stat-card forecast-stat">
                  <div className="stat-top">
                    <span>MODEL FORECAST</span>
                    <Sparkles size={16} />
                  </div>
                  <div className="stat-main">
                    {money(data.forecast?.predicted_next_close)}
                  </div>
                  <div
                    className={`stat-foot ${forecastGap != null && forecastGap >= 0 ? "up" : "down"}`}
                  >
                    <span>
                      {forecastGap != null && forecastGap >= 0 ? (
                        <ArrowUpRight size={13} />
                      ) : (
                        <ArrowDownRight size={13} />
                      )}
                      {forecastGap == null
                        ? "No model forecast"
                        : `${money(forecastGap)} vs last close`}
                    </span>
                    <small>next session · experimental</small>
                  </div>
                </article>
                <article className="stat-card">
                  <div className="stat-top">
                    <span>FINAL HOLDOUT</span>
                    <CalendarClock size={16} />
                  </div>
                  <div className="stat-main">
                    {trading?.win_rate == null
                      ? "—"
                      : `${(trading.win_rate * 100).toFixed(1)}%`}
                    <small> win rate</small>
                  </div>
                  <div className="stat-foot">
                    <span>
                      {trading
                        ? `${trading.wins}W · ${trading.losses}L · ${trading.trades} trades`
                        : "Train to generate metrics"}
                    </span>
                    <small>
                      {holdout?.period_start
                        ? `${shortDate(holdout.period_start)} – ${shortDate(holdout.period_end)}`
                        : "252 sessions"}
                    </small>
                  </div>
                </article>
                <article className="stat-card">
                  <div className="stat-top">
                    <span>HOLDOUT RETURN</span>
                    <Gauge size={16} />
                  </div>
                  <div
                    className={`stat-main ${trading && trading.cumulative_return < 0 ? "text-negative" : "text-positive"}`}
                  >
                    {percent(trading?.cumulative_return)}
                  </div>
                  <div className="stat-foot">
                    <span>after estimated costs</span>
                    <small>
                      Always long{" "}
                      {percent(
                        trading?.baselines.always_long.cumulative_return,
                      )}
                    </small>
                  </div>
                </article>
                <article className="stat-card paper-stat">
                  <div className="stat-top">
                    <span>PAPER JOURNAL</span>
                    <FileClock size={16} />
                  </div>
                  <div className="stat-main">
                    {data.paper?.settled_sessions ?? 0}
                    <small> settled</small>
                  </div>
                  <div className="stat-foot">
                    <span>
                      {data.paper?.wins ?? 0}W · {data.paper?.losses ?? 0}L
                    </span>
                    <small>{data.paper?.pending_signals ?? 0} pending</small>
                  </div>
                </article>
              </section>

              <section className="content-grid">
                <div className="primary-column">
                  <article className="panel chart-panel">
                    <div className="panel-heading">
                      <div>
                        <div className="panel-kicker">PRICE HISTORY</div>
                        <h2>{ticker} adjusted close</h2>
                      </div>
                      <div className="chart-legend">
                        <span className="legend-line" /> Close{" "}
                        <span className="range-label">120 sessions</span>
                      </div>
                    </div>
                    <div className="chart-meta">
                      <strong>{money(displayClose)}</strong>
                      <span
                        className={isUp ? "text-positive" : "text-negative"}
                      >
                        {percent(displayChange)} <small>today</small>
                      </span>
                    </div>
                    <div className="price-chart">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart
                          data={data.history}
                          margin={{ top: 16, right: 8, left: 2, bottom: 0 }}
                        >
                          <defs>
                            <linearGradient
                              id="priceFill"
                              x1="0"
                              y1="0"
                              x2="0"
                              y2="1"
                            >
                              <stop
                                offset="0%"
                                stopColor="#2b7a62"
                                stopOpacity={0.19}
                              />
                              <stop
                                offset="100%"
                                stopColor="#2b7a62"
                                stopOpacity={0.015}
                              />
                            </linearGradient>
                          </defs>
                          <CartesianGrid
                            vertical={false}
                            stroke="#e8ece7"
                            strokeDasharray="3 5"
                          />
                          <XAxis
                            dataKey="date"
                            tickFormatter={(value: string) =>
                              new Date(`${value}T12:00:00`).toLocaleDateString(
                                "en-US",
                                { month: "short", day: "numeric" },
                              )
                            }
                            tickLine={false}
                            axisLine={false}
                            minTickGap={35}
                          />
                          <YAxis
                            orientation="right"
                            domain={["auto", "auto"]}
                            tickFormatter={(value: number) =>
                              `$${value.toFixed(0)}`
                            }
                            tickLine={false}
                            axisLine={false}
                            width={55}
                          />
                          <Tooltip content={<PriceTooltip />} />
                          <Area
                            type="monotone"
                            dataKey="close"
                            stroke="#27765f"
                            strokeWidth={2.25}
                            fill="url(#priceFill)"
                            activeDot={{
                              r: 4,
                              fill: "#e77754",
                              stroke: "#fff",
                              strokeWidth: 2,
                            }}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="chart-footer">
                      <span>
                        <span className="chart-dot" /> Adjusted close
                      </span>
                      <span>Source: Yahoo Finance · local cache</span>
                    </div>
                  </article>

                  <article className="panel holdout-panel">
                    <div className="panel-heading">
                      <div>
                        <div className="panel-kicker">MODEL VALIDATION</div>
                        <h2>Final holdout, not a promise</h2>
                      </div>
                      <span className="holdout-badge">
                        <ShieldCheck size={13} /> Untouched period
                      </span>
                    </div>
                    <div className="holdout-period">
                      <Clock3 size={14} /> {shortDate(holdout?.period_start)} –{" "}
                      {shortDate(holdout?.period_end)}{" "}
                      <span>
                        · trained through {shortDate(holdout?.training_through)}
                      </span>
                    </div>
                    <div className="benchmark-table">
                      <div className="benchmark-head">
                        <span>STRATEGY</span>
                        <span>NET RETURN</span>
                        <span>WIN RATE</span>
                        <span>TRADES</span>
                      </div>
                      <div className="benchmark-row selected">
                        <span>
                          <i className="strategy-mark model-mark" /> Model
                          signals
                        </span>
                        <strong
                          className={
                            trading && trading.cumulative_return < 0
                              ? "text-negative"
                              : "text-positive"
                          }
                        >
                          {percent(trading?.cumulative_return)}
                        </strong>
                        <span>{percent(trading?.win_rate)}</span>
                        <span>{trading?.trades ?? "—"}</span>
                      </div>
                      <div className="benchmark-row">
                        <span>
                          <i className="strategy-mark always-mark" /> Always
                          long
                        </span>
                        <strong>
                          {percent(
                            trading?.baselines.always_long.cumulative_return,
                          )}
                        </strong>
                        <span>
                          {percent(trading?.baselines.always_long.win_rate)}
                        </span>
                        <span>{holdout?.sessions ?? "—"}</span>
                      </div>
                      <div className="benchmark-row">
                        <span>
                          <i className="strategy-mark idle-mark" /> No trade
                        </span>
                        <strong>0.00%</strong>
                        <span>—</span>
                        <span>0</span>
                      </div>
                    </div>
                    <div className="holdout-foot">
                      <span>
                        Return MAE{" "}
                        <strong>
                          {holdout?.mean_absolute_return_error == null
                            ? money(holdout?.mae)
                            : percent(holdout.mean_absolute_return_error, 3)}
                        </strong>{" "}
                        <small>
                          vs{" "}
                          {holdout?.previous_close_baseline_mean_absolute_return_error ==
                          null
                            ? money(holdout?.previous_close_baseline_mae)
                            : percent(
                                holdout.previous_close_baseline_mean_absolute_return_error,
                                3,
                              )}{" "}
                          previous-close baseline
                        </small>
                      </span>
                      <span>
                        95% win interval{" "}
                        <strong>
                          {trading?.win_rate_95pct_wilson_interval
                            ? `${(trading.win_rate_95pct_wilson_interval[0] * 100).toFixed(1)}–${(trading.win_rate_95pct_wilson_interval[1] * 100).toFixed(1)}%`
                            : "—"}
                        </strong>
                      </span>
                    </div>
                  </article>

                  <article className="panel intraday-panel">
                    <div className="panel-heading">
                      <div>
                        <div className="panel-kicker">MODEL EXPERIMENT</div>
                        <h2>Open-to-close return comparison</h2>
                      </div>
                      <span className="holdout-badge">
                        <Activity size={13} /> Development only
                      </span>
                      <button
                        className="text-action"
                        onClick={() => void runAction("compare")}
                        disabled={busy !== null}
                      >
                        {busy === "compare" ? (
                          <LoaderCircle className="spin" size={14} />
                        ) : (
                          <RefreshCw size={14} />
                        )}
                        {intradayComparison ? "Run again" : "Compare models"}
                      </button>
                    </div>
                    <p className="intraday-caption">
                      Ridge and HistGradientBoosting predict next-session
                      open-to-close return. This development-only comparison
                      does not replace the selected production model or inspect
                      the final holdout.
                    </p>
                    {intradayComparison ? (
                      <div className="table-scroll intraday-scroll">
                        <div className="benchmark-table intraday-table">
                          <div className="benchmark-head">
                            <span>MODEL</span>
                            <span>RETURN MAE</span>
                            <span>DIRECTION</span>
                            <span>WIN RATE</span>
                            <span>TRADES</span>
                            <span>NET RETURN</span>
                          </div>
                          {Object.entries(intradayComparison).map(
                            ([name, result]) => (
                              <div className="benchmark-row" key={name}>
                                <span>
                                  <i
                                    className={`strategy-mark ${name === "ridge" ? "always-mark" : "model-mark"}`}
                                  />
                                  {name === "ridge"
                                    ? "Ridge"
                                    : "HistGradientBoosting"}
                                </span>
                                <span>
                                  {(
                                    result.mean_absolute_return_error * 100
                                  ).toFixed(2)}
                                  %
                                </span>
                                <span>
                                  {percent(result.directional_accuracy)}
                                </span>
                                <span>{percent(result.trading.win_rate)}</span>
                                <span>{result.trading.trades}</span>
                                <strong
                                  className={
                                    result.trading.cumulative_return < 0
                                      ? "text-negative"
                                      : "text-positive"
                                  }
                                >
                                  {percent(result.trading.cumulative_return)}
                                </strong>
                              </div>
                            ),
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="empty-state comparison-empty">
                        <Activity size={18} />
                        <span>
                          No development comparison yet. Run it to compare
                          models without changing the active stock model.
                        </span>
                      </div>
                    )}
                  </article>
                </div>

                <aside className="secondary-column">
                  <article className="forecast-card">
                    <div className="forecast-card-head">
                      <span className="forecast-icon">
                        <Sparkles size={16} />
                      </span>
                      <span>MODEL SIGNAL</span>
                      <span
                        className={`model-chip ${data.modelReady ? "chip-ready" : ""}`}
                      >
                        {data.modelReady ? "READY" : "MISSING"}
                      </span>
                    </div>
                    <div className="forecast-result">
                      {money(data.forecast?.predicted_next_close)}
                    </div>
                    <div className="forecast-caption">
                      Predicted next-session close
                    </div>
                    <div className="forecast-divider" />
                    <div className="forecast-detail">
                      <span>Last observed close</span>
                      <strong>{money(data.forecast?.last_close)}</strong>
                    </div>
                    <div className="forecast-detail">
                      <span>Forecast change</span>
                      <strong
                        className={
                          forecastGap != null && forecastGap >= 0
                            ? "text-positive"
                            : "text-negative"
                        }
                      >
                        {percent(
                          data.forecast && forecastGap != null
                            ? forecastGap / data.forecast.last_close
                            : null,
                        )}
                      </strong>
                    </div>
                    <div className="forecast-detail">
                      <span>Signal date</span>
                      <strong>{shortDate(data.forecast?.as_of)}</strong>
                    </div>
                    {data.forecastError ? (
                      <div className="inline-warning">{data.forecastError}</div>
                    ) : null}
                    <div className="forecast-disclaimer">
                      Experimental estimate · not a probability
                    </div>
                  </article>

                  <article className="panel journal-card">
                    <div className="panel-heading compact-heading">
                      <div>
                        <div className="panel-kicker">PAPER JOURNAL</div>
                        <h2>Recent signals</h2>
                      </div>
                      <button
                        className="text-action"
                        onClick={() => setView("paper")}
                      >
                        View all <ArrowRight size={14} />
                      </button>
                    </div>
                    {data.paperRows.length ? (
                      <div className="signal-list">
                        {data.paperRows.slice(0, 4).map((row, index) => (
                          <div
                            className="signal-row"
                            key={`${row.signal_date}-${index}`}
                          >
                            <span
                              className={`signal-state ${row.status === "pending" ? "pending" : row.enter_long ? "entered" : "skipped"}`}
                            />{" "}
                            <div className="signal-copy">
                              <strong>
                                {shortDate(String(row.signal_date ?? ""))}
                              </strong>
                              <small>
                                {row.status === "pending"
                                  ? "Awaiting next close"
                                  : row.enter_long
                                    ? "Long signal"
                                    : "No entry"}
                              </small>
                            </div>
                            <div className="signal-value">
                              {row.status === "pending"
                                ? percent(paperForecastReturn(row))
                                : percent(Number(row.strategy_daily_return))}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="empty-state">
                        <WalletCards size={19} />
                        <span>No paper signals logged</span>
                      </div>
                    )}
                    <button
                      className="button button-secondary full-button"
                      onClick={() => void runAction("paper-trade")}
                      disabled={busy !== null || !data.modelReady}
                    >
                      {busy === "paper-trade" ? (
                        <LoaderCircle className="spin" size={15} />
                      ) : (
                        <CalendarClock size={15} />
                      )}{" "}
                      Log today’s signal
                    </button>
                  </article>

                  <article className="panel news-panel">
                    <div className="panel-heading compact-heading">
                      <div>
                        <div className="panel-kicker">ENTITY CONTEXT</div>
                        <h2>In the news</h2>
                      </div>
                      <button
                        className="text-action news-refresh"
                        onClick={() => void runAction("news")}
                        disabled={busy !== null}
                      >
                        {busy === "news" ? (
                          <LoaderCircle className="spin" size={14} />
                        ) : (
                          <RefreshCw size={14} />
                        )}
                        {data.headlines.length ? "Refresh" : "Fetch news"}
                      </button>
                    </div>
                    {data.headlines.length ? (
                      <div className="news-list">
                        {data.headlines.slice(0, 3).map((headline, index) => (
                          <NewsItem
                            key={`${headline.url}-${index}`}
                            headline={headline}
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="news-empty">
                        <Newspaper size={18} />
                        <span>
                          No saved headlines for {ticker}. Fetch the latest
                          context without logging a paper signal.
                        </span>
                      </div>
                    )}
                    <div className="context-foot">
                      <ShieldCheck size={13} /> Headlines are context only, not
                      model inputs.
                    </div>
                  </article>
                </aside>
              </section>
            </>
          ) : null}

          {data && view === "paper" ? (
            <PaperView data={data} onBack={() => setView("overview")} />
          ) : null}

          <footer className="page-footer">
            <span>Northstar · Local research desk</span>
            <span>
              <span className="footer-dot" /> Workspace data{" "}
              {shortDate(data?.quote.date)}
            </span>
            <span>Signals are experimental; not investment advice.</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function PriceTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { value: number }[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;
  return (
    <div className="chart-tooltip">
      <span>{shortDate(label)}</span>
      <strong>{money(payload[0].value)}</strong>
    </div>
  );
}

function NewsItem({ headline }: { headline: Headline }) {
  const relations = headline.relationships ?? [];
  const primaryRelation = relations[0];
  const context = primaryRelation?.counterparty_context?.[0];
  return (
    <div className="news-item">
      <div className="news-meta">
        <span>{headline.publisher}</span>
        <time>
          {new Date(headline.published_at_utc).toLocaleTimeString("en-US", {
            hour: "numeric",
            minute: "2-digit",
            timeZoneName: "short",
          })}
        </time>
      </div>
      <a
        className="news-title"
        href={headline.url}
        target="_blank"
        rel="noreferrer"
      >
        {headline.title}
        <ExternalLink size={12} />
      </a>
      {primaryRelation ? (
        <div className="relationship-line">
          <span className="relation-node">
            {primaryRelation.predicate.replaceAll("_", " ")}
          </span>
          <ArrowRight size={12} />
          <strong>{primaryRelation.counterparty_candidate}</strong>
        </div>
      ) : null}
      {context ? (
        <p className="entity-snippet">
          {context.snippet}{" "}
          <a href={context.url} target="_blank" rel="noreferrer">
            Context <ExternalLink size={10} />
          </a>
        </p>
      ) : null}
    </div>
  );
}

function PaperView({
  data,
  onBack,
}: {
  data: DashboardState;
  onBack: () => void;
}) {
  const paper = data.paper;
  return (
    <section className="paper-page">
      <div className="paper-summary-grid">
        <article className="stat-card">
          <div className="stat-top">
            <span>SETTLED SESSIONS</span>
            <Clock3 size={16} />
          </div>
          <div className="stat-main">{paper?.settled_sessions ?? 0}</div>
          <div className="stat-foot">
            <span>{paper?.pending_signals ?? 0} awaiting outcome</span>
            <small>{paper?.signals_logged ?? 0} total signals</small>
          </div>
        </article>
        <article className="stat-card">
          <div className="stat-top">
            <span>WIN / LOSS</span>
            <Activity size={16} />
          </div>
          <div className="stat-main">
            {paper?.wins ?? 0}
            <small> W</small>
            <span className="slash-count">/</span>
            {paper?.losses ?? 0}
            <small> L</small>
          </div>
          <div className="stat-foot">
            <span>realized paper trades</span>
            <small>{percent(paper?.win_rate)}</small>
          </div>
        </article>
        <article className="stat-card">
          <div className="stat-top">
            <span>STRATEGY RETURN</span>
            <Gauge size={16} />
          </div>
          <div
            className={`stat-main ${paper && paper.strategy_cumulative_return < 0 ? "text-negative" : "text-positive"}`}
          >
            {percent(paper?.strategy_cumulative_return)}
          </div>
          <div className="stat-foot">
            <span>versus open-to-close</span>
            <small>
              long-only baseline {percent(paper?.always_long_cumulative_return)}
            </small>
          </div>
        </article>
      </div>
      <article className="panel paper-table-panel">
        <div className="panel-heading">
          <div>
            <div className="panel-kicker">PROSPECTIVE TRACK RECORD</div>
            <h2>{data.ticker} paper signals</h2>
          </div>
          <button className="text-action" onClick={onBack}>
            Back to overview <ArrowRight size={14} />
          </button>
        </div>
        {data.paperRows.length ? (
          <div className="table-scroll">
            <table className="paper-table">
              <thead>
                <tr>
                  <th>Signal date</th>
                  <th>Forecast return</th>
                  <th>Next open</th>
                  <th>Next close</th>
                  <th>Trade</th>
                  <th>Net return</th>
                  <th>State</th>
                </tr>
              </thead>
              <tbody>
                {data.paperRows.map((row, index) => (
                  <tr key={`${row.signal_date}-${index}`}>
                    <td>{shortDate(String(row.signal_date ?? ""))}</td>
                    <td>{percent(paperForecastReturn(row))}</td>
                    <td>
                      {row.next_open == null
                        ? "—"
                        : money(Number(row.next_open))}
                    </td>
                    <td>
                      {row.next_close == null
                        ? "—"
                        : money(Number(row.next_close))}
                    </td>
                    <td>
                      {row.status === "pending"
                        ? "Pending"
                        : row.enter_long
                          ? "Long"
                          : "Skipped"}
                    </td>
                    <td
                      className={
                        Number(row.strategy_daily_return) < 0
                          ? "text-negative"
                          : Number(row.strategy_daily_return) > 0
                            ? "text-positive"
                            : ""
                      }
                    >
                      {row.status === "pending"
                        ? "—"
                        : percent(Number(row.strategy_daily_return))}
                    </td>
                    <td>
                      <span
                        className={`status-pill ${row.status === "pending" ? "status-pending" : "status-settled"}`}
                      >
                        {String(row.status ?? "—")}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state large-empty">
            <WalletCards size={22} />
            <span>No paper signals yet</span>
            <small>
              Log a signal from the market overview to start the journal.
            </small>
          </div>
        )}
        <div className="paper-table-foot">
          <span>Per-side estimated cost: 10 bps</span>
          <span>
            One pending signal settles after next session prices arrive.
          </span>
        </div>
      </article>
      <div className="paper-note">
        <Sparkles size={15} />
        <span>
          Paper results are simulated and use cached daily prices. They do not
          account for every execution constraint.
        </span>
      </div>
    </section>
  );
}

export default App;
