import { Link, useParams } from "react-router-dom";
import {
  useCampaignBest,
  useCampaignFunnel,
  useCampaignHistory,
  useCampaignPareto,
  useCampaignSummary,
} from "../hooks/useCampaign";
import KpiBanner from "../components/KpiBanner";
import ParetoChart from "../components/ParetoChart";
import FunnelChart from "../components/FunnelChart";
import HistoryChart from "../components/HistoryChart";
import BestConfigsTable from "../components/BestConfigsTable";

export default function CampaignPage() {
  const { design = "", platform = "" } = useParams();

  const summary = useCampaignSummary(design, platform);
  const pareto = useCampaignPareto(design, platform);
  const funnel = useCampaignFunnel(design, platform);
  const history = useCampaignHistory(design, platform);
  const best = useCampaignBest(design, platform, 5);

  if (summary.isLoading) return <div className="empty-state">Loading {design}/{platform}…</div>;
  if (summary.error) {
    return (
      <div className="empty-state error">
        Failed to load {design}/{platform}: {String(summary.error)}
      </div>
    );
  }

  return (
    <div>
      <Link to="/" className="back-link">
        ← all campaigns
      </Link>
      <h2>
        {design} <span className="muted">/ {platform}</span>
      </h2>

      {summary.data && <KpiBanner summary={summary.data} />}

      <div className="card">{pareto.data && <ParetoChart points={pareto.data} />}</div>

      <div className="chart-row">
        <div className="card">{funnel.data && <FunnelChart counts={funnel.data} />}</div>
        <div className="card">{history.data && <HistoryChart points={history.data} />}</div>
      </div>

      <div className="card">
        <h3>Best configs</h3>
        {best.data && <BestConfigsTable picks={best.data} />}
      </div>
    </div>
  );
}
