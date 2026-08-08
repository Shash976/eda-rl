import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ParetoPoint } from "../api/types";
import ParetoChart from "./ParetoChart";

// react-plotly.js needs browser canvas/WebGL APIs jsdom doesn't provide —
// stub it so this test exercises our own frontier/non-frontier trace-split
// logic, not Plotly's rendering internals.
vi.mock("react-plotly.js", () => ({
  default: (props: { data: unknown[] }) => (
    <div data-testid="plot" data-trace-count={props.data.length} />
  ),
}));

describe("ParetoChart", () => {
  it("shows an empty state with no F3 points", () => {
    render(<ParetoChart points={[]} />);
    expect(screen.getByText(/no f3 builds/i)).toBeInTheDocument();
  });

  it("splits points into a frontier trace and a non-frontier trace", () => {
    const points: ParetoPoint[] = [
      { area_um2: 100, fmax_mhz: 200, is_frontier: true, row: {} },
      { area_um2: 150, fmax_mhz: 180, is_frontier: false, row: {} },
    ];
    render(<ParetoChart points={points} />);
    expect(screen.getByTestId("plot")).toHaveAttribute("data-trace-count", "2");
  });
});
