import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import type { HistoryPoint } from "../api/types";
import { colors } from "../theme/tokens";

export default function HistoryChart({ points }: { points: HistoryPoint[] }) {
  const withValue = points.filter((p) => p.value != null);
  if (withValue.length === 0) {
    return <div className="empty-state">No scored episodes yet.</div>;
  }

  const traces: Data[] = [
    {
      x: withValue.map((p) => p.episode ?? 0),
      y: withValue.map((p) => p.value as number),
      mode: "markers",
      type: "scatter",
      name: "episode reward",
      marker: { color: "#1f77b4", size: 6, opacity: 0.6 },
    },
    {
      x: withValue.map((p) => p.episode ?? 0),
      y: withValue.map((p) => p.running_max as number),
      mode: "lines",
      type: "scatter",
      name: "best-so-far",
      line: { color: colors.headerTo, width: 2 },
    },
  ];

  return (
    <Plot
      data={traces}
      layout={{
        title: { text: "Optimization History" },
        xaxis: { title: { text: "Episode" } },
        yaxis: { title: { text: "Reward" } },
        autosize: true,
        margin: { t: 40, r: 20, b: 50, l: 60 },
      }}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%", height: "360px" }}
      useResizeHandler
    />
  );
}
