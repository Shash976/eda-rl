import type { CampaignSummary } from "../api/types";

function Kpi({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "green" | "amber";
}) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-val${tone ? ` ${tone}` : ""}`}>{value}</div>
    </div>
  );
}

export default function KpiBanner({ summary }: { summary: CampaignSummary }) {
  const f3 = summary.fidelity_counts.F3;
  const total = summary.n_episodes;
  const yieldPct = total > 0 ? ((f3 / total) * 100).toFixed(0) : "0";

  return (
    <div className="kpi-banner">
      <Kpi label="Campaign" value={summary.campaign_id} />
      <Kpi label="Episodes" value={String(total)} />
      <Kpi label="Reached F3" value={`${f3} (${yieldPct}%)`} tone={f3 > 0 ? "green" : "amber"} />
      <Kpi
        label="Best score"
        value={summary.best_value != null ? summary.best_value.toFixed(3) : "—"}
      />
      <Kpi label="Params tracked" value={String(Object.keys(summary.params).length)} />
    </div>
  );
}
