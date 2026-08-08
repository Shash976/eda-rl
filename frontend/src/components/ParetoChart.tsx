import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import type { ParetoPoint } from "../api/types";
import { colors } from "../theme/tokens";

export default function ParetoChart({ points }: { points: ParetoPoint[] }) {
  if (points.length === 0) {
    return <div className="empty-state">No F3 builds yet.</div>;
  }

  const rest = points.filter((p) => !p.is_frontier);
  const frontier = points.filter((p) => p.is_frontier).sort((a, b) => a.area_um2 - b.area_um2);

  const traces: Data[] = [
    {
      x: rest.map((p) => p.area_um2),
      y: rest.map((p) => p.fmax_mhz),
      mode: "markers",
      type: "scatter",
      name: "F3 builds",
      marker: { color: "#1f77b4", size: 9, opacity: 0.7, line: { width: 1, color: "white" } },
    },
    {
      x: frontier.map((p) => p.area_um2),
      y: frontier.map((p) => p.fmax_mhz),
      mode: "lines+markers",
      type: "scatter",
      name: "Pareto frontier",
      line: { color: colors.headerTo, width: 2, dash: "dash" },
      marker: { color: colors.red, size: 10, line: { width: 1, color: "white" } },
    },
  ];

  return (
    <Plot
      data={traces}
      layout={{
        title: { text: "Pareto Frontier: Area vs Fmax" },
        xaxis: { title: { text: "Area (µm²)" } },
        yaxis: { title: { text: "Fmax (MHz)" } },
        autosize: true,
        margin: { t: 40, r: 20, b: 50, l: 60 },
      }}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%", height: "420px" }}
      useResizeHandler
    />
  );
}
