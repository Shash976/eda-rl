import { Link } from "react-router-dom";
import { useDesigns } from "../hooks/useCampaign";

export default function DesignsPage() {
  const { data: designs, isLoading, error } = useDesigns();

  if (isLoading) return <div className="empty-state">Loading campaigns…</div>;
  if (error) return <div className="empty-state error">Failed to load: {String(error)}</div>;
  if (!designs || designs.length === 0) {
    return <div className="empty-state">No campaign logs found under eda_rl/campaigns/.</div>;
  }

  return (
    <div className="design-grid">
      {designs.map((d) => (
        <Link key={`${d.design}/${d.platform}`} to={`/${d.design}/${d.platform}`} className="card design-card">
          <div className="design-name">{d.design}</div>
          <div className="muted">{d.platform}</div>
        </Link>
      ))}
    </div>
  );
}
