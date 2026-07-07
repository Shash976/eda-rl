"""Golden-log regression tests for physical_runner's tool-output parsers (R5).

The third audit found two silent parser deaths — F3 (yosys `stat` format
change → cell counts None in 100% of rows) and F4 (`fmax = inf` for
combinational designs → constraint-echo fmax) — that PHYSICAL_MOCK self-tests
cannot catch, because mock mode fabricates exactly the fields these parsers
produce.  These tests feed short REAL tool-output fixtures (see
fixtures/README.md for provenance) through the actual parsing functions so a
toolchain upgrade that changes an output format fails loudly here.

Run directly (no pytest needed):   python3 tests/test_parsers.py
Or via pytest if installed:        python3 -m pytest tests/test_parsers.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from eda_rl.common.physical_runner import (  # noqa: E402
    _parse_metrics,
    _parse_sta_timing,
    _parse_synth_stat,
)

FIX = _HERE / "fixtures"


# ── yosys stat: cell count + cell area (audit F3) ──────────────────────────────

def test_stat_tabular_takes_last_top_module_total():
    """Installed-yosys tabular format: last stat block (post-opt_clean) wins,
    per-cell-type rows and the pre-map (no-area) block never shadow it."""
    cells, area = _parse_synth_stat((FIX / "yosys_stat_tabular.txt").read_text())
    assert cells == 36, f"expected post-map total 36, got {cells}"
    assert area == 231.472, f"expected Chip area 231.472, got {area}"


def test_stat_legacy_number_of_cells():
    """Pre-upgrade yosys 'Number of cells:  N' format still parses."""
    cells, area = _parse_synth_stat((FIX / "yosys_stat_legacy.txt").read_text())
    assert cells == 398, f"expected 398, got {cells}"
    assert area == 608.836, f"expected 608.836, got {area}"


def test_stat_per_cell_rows_never_match():
    """A per-cell-type row like '3  26.275  sky130_...xor2_1' must not parse as
    a total; with no total line at all the count is honestly None."""
    cells, area = _parse_synth_stat(
        "        3   26.275   sky130_fd_sc_hd__xor2_1\n"
        "        2   17.517   sky130_fd_sc_hd__a221oi_1\n"
    )
    assert cells is None and area is None


# ── OpenSTA report_clock_min_period (audit F4) ─────────────────────────────────

def test_sta_sequential_real_fmax():
    """A real clocked design: fmax/period are measured, no markers set."""
    t = _parse_sta_timing((FIX / "sta_sequential.txt").read_text(),
                          time_div=1.0, clk_ns=1.0)
    assert t["fmax_mhz"] == 1564.07, t
    assert t["period_min_ns"] == 0.64, t
    assert t["combinational"] is False and t["fmax_inferred"] is False, t
    assert t["wns_ns"] == 0.0 and t["timing_met"] is True, t


def test_sta_combinational_inf_is_none_not_echo():
    """'period_min = 0.00 fmax = inf' → fmax None + combinational marker,
    NEVER the 1000/clk constraint echo (the F4 bug)."""
    t = _parse_sta_timing((FIX / "sta_combinational.txt").read_text(),
                          time_div=1.0, clk_ns=6.5)
    assert t["fmax_mhz"] is None, f"constraint echo is back: {t}"
    assert t["period_min_ns"] is None, t
    assert t["combinational"] is True and t["fmax_inferred"] is False, t


def test_sta_asap7_native_ps_division():
    """asap7 reports in ps (time_div=1000): period converts to ns, fmax is
    already MHz and must NOT be divided."""
    out = "core_clock period_min = 45.2 fmax = 22123.89\nwns max -3.5\ntns max -3.5\n"
    t = _parse_sta_timing(out, time_div=1000.0, clk_ns=0.05)
    assert abs(t["period_min_ns"] - 0.0452) < 1e-9, t
    assert t["fmax_mhz"] == 22123.89, t
    assert abs(t["wns_ns"] - (-0.0035)) < 1e-9, t
    assert t["timing_met"] is False, t


def test_sta_slack_fallback_fires_only_on_violation_and_flags():
    """No period line + wns<0 → inferred fmax with the fmax_inferred flag."""
    t = _parse_sta_timing("wns max -1.72\ntns max -25.86\n",
                          time_div=1.0, clk_ns=4.0)
    assert t["fmax_inferred"] is True and t["combinational"] is False, t
    assert t["period_min_ns"] == 5.72 and t["fmax_mhz"] == round(1000.0 / 5.72, 2), t


def test_sta_slack_fallback_never_fires_when_timing_met():
    """No period line + wns>=0 → no fabricated fmax at all."""
    t = _parse_sta_timing("wns max 0.00\ntns max 0.00\n",
                          time_div=1.0, clk_ns=4.0)
    assert t["fmax_mhz"] is None and t["fmax_inferred"] is False, t
    assert t["timing_met"] is True, t


# ── _parse_metrics: 6_report.json / 6_report.log / 6_finish.rpt (F3 tail, F17) ─

def _metrics_tree(tmp: Path, *, sequential: bool) -> Path:
    """Build a minimal fake ORFS results tree for _parse_metrics."""
    plat, design, var = "nangate45", "gcd", "gcd_c1_x"
    (tmp / "reports" / plat / design / var).mkdir(parents=True)
    (tmp / "logs" / plat / design / var).mkdir(parents=True)
    (tmp / "results" / plat / design / var).mkdir(parents=True)
    (tmp / "reports" / plat / design / var / "6_finish.rpt").write_text(
        "wns max -1.72\ntns max -25.86\n"
        "core_clock period_min = 3.72 fmax = 268.64\n"
        "Total                  1.13e-02 3.16e-03 5.42e-05 1.45e-02 100.0%\n"
    )
    (tmp / "logs" / plat / design / var / "6_report.log").write_text(
        "Design area 19738 um^2 48% utilization.\n"
    )
    jd = {"finish__design__instance__count__stdcell": 812}
    if sequential:
        jd["finish__design__instance__count__class:sequential_cell"] = 34
    (tmp / "logs" / plat / design / var / "6_report.json").write_text(json.dumps(jd))
    return tmp


def test_parse_metrics_sequential_tree():
    with tempfile.TemporaryDirectory() as td:
        work = _metrics_tree(Path(td), sequential=True)
        out = _parse_metrics(work, "nangate45", "gcd_c1_x", 4.0, design_name="gcd")
    assert out["area_um2"] == 19738.0 and out["util_pct"] == 48.0, out
    assert out["cell_count"] == 812 and out["ff_count"] == 34, out
    assert out["wns_ns"] == -1.72 and out["fmax_mhz"] == 268.64, out


def test_parse_metrics_combinational_ff_none_is_legit():
    """A combinational design's 6_report.json omits the sequential-cell key —
    ff_count must stay None with no error (audit F17)."""
    with tempfile.TemporaryDirectory() as td:
        work = _metrics_tree(Path(td), sequential=False)
        out = _parse_metrics(work, "nangate45", "gcd_c1_x", 4.0, design_name="gcd")
    assert out["cell_count"] == 812 and out["ff_count"] is None, out


# ── plain-python runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} parser golden-log tests passed")
    sys.exit(1 if failed else 0)
