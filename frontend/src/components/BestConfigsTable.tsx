import type { BestConfigOut } from "../api/types";

function fmt(v: unknown, digits = 0): string {
  return typeof v === "number" ? v.toFixed(digits) : "—";
}

export default function BestConfigsTable({ picks }: { picks: BestConfigOut[] }) {
  if (picks.length === 0) {
    return <div className="empty-state">No buildable F3 results yet.</div>;
  }

  return (
    <div className="table-wrap">
      <table className="best-configs">
        <thead>
          <tr>
            <th>Badge</th>
            <th>Why</th>
            <th>Area (µm²)</th>
            <th>Fmax (MHz)</th>
            <th>Power (mW)</th>
            <th>Timing met</th>
            <th>Config</th>
          </tr>
        </thead>
        <tbody>
          {picks.map((p) => (
            <tr key={p.variant}>
              <td>
                <span className="badge">{p.badge}</span>
              </td>
              <td className="muted">{p.sublabel}</td>
              <td>{fmt(p.obs.area_um2)}</td>
              <td>{fmt(p.obs.fmax_mhz)}</td>
              <td>{fmt(p.obs.power_mw)}</td>
              <td>{p.obs.timing_met ? "yes" : "no"}</td>
              <td className="config-cell">
                {Object.entries(p.config)
                  .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`)
                  .join(", ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {/* GDS paths in `obs.gds` are local filesystem paths from whatever
          machine ran the build — informational only, intentionally not
          rendered here as a download link (no file server behind them). */}
    </div>
  );
}
