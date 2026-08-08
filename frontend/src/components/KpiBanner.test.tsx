import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { CampaignSummary } from "../api/types";
import KpiBanner from "./KpiBanner";

const summary: CampaignSummary = {
  design: "gcd",
  platform: "nangate45",
  campaign_id: "campaign_0_123",
  n_episodes: 36,
  fidelity_counts: { F0: 0, F1: 0, F2: 22, F3: 14 },
  best_value: -0.00625,
  params: {
    clock_period_ns: { name: "clock_period_ns", kind: "float", low: 0.3, high: 2.0, choices: [] },
  },
};

describe("KpiBanner", () => {
  it("renders episode count and F3 yield percentage", () => {
    render(<KpiBanner summary={summary} />);
    expect(screen.getByText("36")).toBeInTheDocument();
    expect(screen.getByText(/14 \(39%\)/)).toBeInTheDocument();
  });

  it("renders a placeholder when best_value is null", () => {
    render(<KpiBanner summary={{ ...summary, best_value: null }} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
