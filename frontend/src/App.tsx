import { Route, Routes } from "react-router-dom";
import DesignsPage from "./pages/DesignsPage";
import CampaignPage from "./pages/CampaignPage";

export default function App() {
  return (
    <div>
      <header className="app-header">
        <h1>eda-rl campaign dashboard</h1>
        <div className="subtitle">
          multi-fidelity RTL→GDS design-space exploration — campaign results
        </div>
      </header>
      <Routes>
        <Route path="/" element={<DesignsPage />} />
        <Route path="/:design/:platform" element={<CampaignPage />} />
      </Routes>
    </div>
  );
}
