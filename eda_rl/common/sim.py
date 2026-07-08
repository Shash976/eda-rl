"""sim.py — mock-aware Verilator behavioral-sim wrappers for the F1 gate.

The funnel (env.py, build_table.py) needs the behavioral Verilator sim for the
F1 fidelity gate.  This module owns the mock-aware wrapper `_run_sim`; the real
subprocess driver lives in `common.verilator_sim` (rescued from the retired
gen1 tree — it is the only piece of gen1 the live system ever needed).  Both
entry points lazy-import it so the Verilator binary is only required on the
real (non-mock) path.  The measured constants and the acc_overflows predicate
live in `common.constants`.
"""

from __future__ import annotations

import os

from eda_rl.common.constants import (
    AVG_CYCLES as _AVG_CYCLES,
    behavioral_cycles as _behavioral_cycles,
)


def _run_sim(lanes: int, acc_w: int) -> dict:
    """Behavioural Verilator run → {accuracy, correct, n_total, avg_cycles}.

    Under PHYSICAL_MOCK, returns numbers shaped like the real measured sweep:
    acc_width < 24 loses accuracy (int16 ≈ 47/64), matching the empirical finding.
    Cycle counts come from the constants.py table (or the analytic fit for lane
    counts not in the table), keeping the mock faithful to the real sim.

    """
    if os.environ.get("PHYSICAL_MOCK"):
        if acc_w >= 24:
            acc = 1.0
        elif acc_w >= 20:
            acc = 0.92
        else:                       # int16 — overflow, real measured 47/64
            acc = 47.0 / 64.0
        # Use measured value from table when available; fall back to fit.
        cyc = float(_AVG_CYCLES[lanes]) if lanes in _AVG_CYCLES else _behavioral_cycles(lanes)
        return {"accuracy": acc, "correct": round(acc * 64), "n_total": 64,
                "avg_cycles": cyc}
    from eda_rl.common.verilator_sim import run_sim as _real_run_sim  # lazy: needs the Verilator binary
    return _real_run_sim(lanes, acc_w)


def run_sim(mac_lanes: int, acc_width: int = 32) -> dict:
    """Thin re-export of the real Verilator driver (`common.verilator_sim`).

    build_table.py calls this directly (it handles PHYSICAL_MOCK itself before
    reaching here), so this is a lazy passthrough — no mock branch.
    """
    from eda_rl.common.verilator_sim import run_sim as _real_run_sim  # lazy: needs the Verilator binary
    return _real_run_sim(mac_lanes, acc_width)
