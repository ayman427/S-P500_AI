import { useState } from "react";
import RiskPanel from "./RiskPanel";
import SpikePanel from "./SpikePanel";

export default function HistoricalRisk({ticker}: {ticker: string}) {
  const [view, setView] = useState<"declines" | "spikes">("declines");
  const [spikesVisited, setSpikesVisited] = useState(false);
  function selectView(next: "declines" | "spikes") {
    setView(next);
    if (next === "spikes") setSpikesVisited(true);
  }
  return <div className="historical-risk">
    <div className="risk-view-switch" role="tablist" aria-label="Historical risk views">
      {(["declines", "spikes"] as const).map(item => <button key={item} type="button" role="tab" id={`risk-view-${item}`}
        aria-selected={view === item} aria-controls={`risk-content-${item}`} tabIndex={view === item ? 0 : -1}
        onClick={() => selectView(item)} onKeyDown={e => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
          e.preventDefault();
          const next = e.key === "Home" ? "declines" : e.key === "End" ? "spikes" : item === "declines" ? "spikes" : "declines";
          selectView(next);
          document.getElementById(`risk-view-${next}`)?.focus();
        }}>{item === "declines" ? "Declines" : "Rise investigation"}</button>)}
    </div>
    <div id="risk-content-declines" role="tabpanel" aria-labelledby="risk-view-declines" hidden={view !== "declines"} tabIndex={0}>
      <RiskPanel ticker={ticker} />
    </div>
    <div id="risk-content-spikes" role="tabpanel" aria-labelledby="risk-view-spikes" hidden={view !== "spikes"} tabIndex={0}>
      {spikesVisited && <SpikePanel ticker={ticker} />}
    </div>
  </div>;
}
