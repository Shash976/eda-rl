import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import type { FunnelCounts } from "../api/types";
import { fidelityColors } from "../theme/tokens";

const STAGES = ["F0", "F1", "F2", "F3"] as const;

export default function FunnelChart({ counts }: { counts: FunnelCounts }) {
  const values = STAGES.map((s) => counts[s]);

  const trace: Data = {
    x: STAGES as unknown as string[],
    y: values,
    type: "bar",
    marker: { color: STAGES.map((s) => fidelityColors[s]) },
    text: values.map(String),
    textposition: "outside",
  };

  return (
    <Plot
      data={[trace]}
      layout={{
        title: { text: "Fidelity Funnel: how many configs reached each gate" },
        xaxis: { title: { text: "Fidelity gate" } },
        yaxis: { title: { text: "Episodes" } },
        autosize: true,
        margin: { t: 40, r: 20, b: 50, l: 50 },
      }}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%", height: "360px" }}
      useResizeHandler
    />
  );
}
