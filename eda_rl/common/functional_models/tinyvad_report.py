"""tinyvad_report.py — the TinyVAD report extension (plugin-internal).

Owns every TinyMAC-specific piece of the static HTML report: the hand-picked
asap7 baseline, the mac_lanes colour palette, the L/A config label, the
Area-vs-inference-speedup figure, and the software-speedup KPI cards.  viz/report.py
resolves this via ``design.functional_model().report_extension()`` and renders
generically when it is absent — no TinyMAC literals live in the report core.

The software-baseline cycle model mirrors functional_models.tinyvad (kept as
plain literals here so the report has no import-time dependency on the plugin's
heavier internals).
"""

from __future__ import annotations

from typing import Any

# SW baseline latency (ns) and measured accel cycles per lane count — mirror of
# the TinyVAD cycle model, used only for the report's speedup axis.
_SW_LATENCY_NS = 11_196_638 * 10.0
_AVG_CYCLES = {1: 273_130, 2: 152_140, 4: 91_650, 8: 61_400, 16: 46_670, 32: 39_310}

# Qualitative palette for mac_lanes values.
_LANES_PALETTE = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c",
                  8: "#d62728", 16: "#9467bd", 32: "#8c564b"}

# Asap7 baseline: first documented GDS (hand-picked, Stage 6).
_ASAP7_BASELINE = {
    "area_um2": 1433,
    "fmax_mhz": 509,
    "wns_ns": -0.96,
    "timing_met": False,
    "config": {"mac_lanes": 4, "accumulator_width": 24,
               "clock_period_ns": 1.0, "abc_recipe": "orfs_speed"},
    "label": "Baseline (L4_A24 @ 1.0 ns)",
}


class TinyVADReport:
    """Report extension for the TinyVAD/TinyMAC composite design family."""

    label = "TinyMAC"

    #: The hand-picked reference build (a real asap7 GDS data point).
    baseline = _ASAP7_BASELINE
    #: mac_lanes → colour.
    palette = _LANES_PALETTE

    def cfg_label(self, cfg: dict) -> str:
        """L/A config label, e.g. 'L4_A24'."""
        return f"L{cfg.get('mac_lanes')}_A{cfg.get('accumulator_width')}"

    def _speedup(self, lanes: int, fmax_mhz: float) -> float:
        acyc = _AVG_CYCLES.get(lanes, 91_650)
        return _SW_LATENCY_NS / (acyc * (1000.0 / fmax_mhz))

    def best_speedup(self, f3_rows: list[dict]) -> float:
        """Peak inference speedup across F3 builds (SW-baseline cycle model)."""
        best = 0.0
        for r in f3_rows:
            lanes = r["config"].get("mac_lanes", 4)
            best = max(best, self._speedup(lanes, r["obs"]["fmax_mhz"]))
        return best

    def summary_kpi_cards(self, f3_rows: list[dict], best_fmax: float,
                          min_area: float) -> str:
        """The TinyMAC KPI cards (Fmax-vs-baseline, peak speedup, area-vs-baseline)."""
        b = self.baseline
        best_speedup = self.best_speedup(f3_rows)
        fmax_delta = (best_fmax - b["fmax_mhz"]) / b["fmax_mhz"] * 100
        area_delta = (min_area - b["area_um2"]) / b["area_um2"] * 100
        return f"""  <div class='kpi'>
    <div class='kpi-val green'>{best_fmax:.0f} MHz</div>
    <div class='kpi-label'>Best Fmax found &nbsp;·&nbsp; <b>+{fmax_delta:.1f}%</b> vs hand-picked baseline</div>
  </div>
  <div class='kpi'>
    <div class='kpi-val green'>{best_speedup:.0f}×</div>
    <div class='kpi-label'>Peak inference speedup vs software baseline (112 ms @ 100 MHz)</div>
  </div>
  <div class='kpi'>
    <div class='kpi-val'>{min_area:.0f} µm²</div>
    <div class='kpi-label'>Minimum area found &nbsp;·&nbsp; <b>{area_delta:+.1f}%</b> vs baseline</div>
  </div>
"""

    def speedup_figure(self, f3_rows: list[dict]) -> "tuple[str, Any]":
        """Area vs real-speedup-at-Fmax scatter for all F3 results."""
        import plotly.graph_objects as go

        fig = go.Figure()
        if not f3_rows:
            fig.add_annotation(text="No F3 data", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=18))
            fig.update_layout(title="Area vs Real Speedup")
            return ("speedup", fig)

        for lanes in sorted({r["config"].get("mac_lanes", 0) for r in f3_rows}):
            subset = [r for r in f3_rows if r["config"].get("mac_lanes") == lanes]
            xs, ys, hover = [], [], []
            for r in subset:
                fmax = r["obs"]["fmax_mhz"]
                area = r["obs"]["area_um2"]
                speedup = self._speedup(lanes, fmax)
                xs.append(area)
                ys.append(speedup)
                hover.append(
                    f"L{r['config'].get('mac_lanes')}_A{r['config'].get('accumulator_width')}<br>"
                    f"clk={r['config'].get('clock_period_ns', 0):.2f} ns · {r['config'].get('abc_recipe')}<br>"
                    f"Fmax={fmax:.0f} MHz · area={area:.0f} µm²<br>"
                    f"<b>Speedup = {speedup:.0f}×</b>"
                )
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=f"L={lanes}",
                marker=dict(color=self.palette.get(lanes, "#aaa"), size=10,
                            opacity=0.85, line=dict(width=1, color="white")),
                hovertext=hover, hoverinfo="text",
            ))

        b = self.baseline
        b_lanes = b["config"]["mac_lanes"]
        b_speedup = self._speedup(b_lanes, b["fmax_mhz"])
        fig.add_trace(go.Scatter(
            x=[b["area_um2"]], y=[b_speedup],
            mode="markers+text", name=b["label"],
            marker=dict(color="red", size=16, symbol="star",
                        line=dict(width=2, color="darkred")),
            text=["Baseline"], textposition="top right",
            hovertext=f"{b['label']}<br>area={b['area_um2']} µm² · fmax={b['fmax_mhz']} MHz<br>Speedup={b_speedup:.0f}×",
            hoverinfo="text",
        ))
        fig.update_layout(
            title=dict(text="Area vs Inference Speedup (at Fmax) — vs SW Baseline (112 ms @ 100 MHz)", font=dict(size=14)),
            xaxis_title="Area (µm²)",
            yaxis_title="Speedup over SW baseline",
            legend=dict(title="mac_lanes"),
            annotations=[dict(
                text="<i>Higher-left = smaller chip, faster inference</i>",
                xref="paper", yref="paper", x=0.01, y=0.99,
                showarrow=False, font=dict(size=11, color="#6b7280"),
                align="left",
            )],
        )
        return ("speedup", fig)
