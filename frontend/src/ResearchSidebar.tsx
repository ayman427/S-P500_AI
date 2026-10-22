import {useState} from "react";
import MyPredictions from "./MyPredictions";
import MarketWatch from "./MarketWatch";

export default function ResearchSidebar({refreshKey}: {refreshKey: number}) {
  const [tab, setTab] = useState("predictions");
  return <aside className="my-predictions" aria-label="Your predictions and rise candidates">
    <div className="sidebar-tabs" role="tablist" aria-label="Stock lists">
      {[["predictions", "Predictions for you"], ["watch", "Rise candidates"]].map(([id, label]) =>
        <button key={id} id={`sidebar-tab-${id}`} role="tab" type="button" aria-selected={tab === id}
          aria-controls={`sidebar-panel-${id}`} tabIndex={tab === id ? 0 : -1} onClick={() => setTab(id)}
          onKeyDown={event => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
            event.preventDefault();
            const next = event.key === "Home" ? "predictions" : event.key === "End" ? "watch" : tab === "watch" ? "predictions" : "watch";
            setTab(next);
            document.getElementById(`sidebar-tab-${next}`)?.focus();
          }}>{label}</button>)}
    </div>
    <div id="sidebar-panel-predictions" role="tabpanel" aria-labelledby="sidebar-tab-predictions" hidden={tab !== "predictions"}>
      <MyPredictions refreshKey={refreshKey} />
    </div>
    <div id="sidebar-panel-watch" role="tabpanel" aria-labelledby="sidebar-tab-watch" hidden={tab !== "watch"}>
      {tab === "watch" && <MarketWatch />}
    </div>
  </aside>;
}
