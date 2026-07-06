"""sim.py — mock-aware Verilator behavioral-sim wrappers shared by gen2.

gen2 (funnel.py, build_table.py) needs the behavioral Verilator sim for the F1
fidelity gate, but importing it from gen1 (`gen1.cascade._run_sim`,
`gen1.runner.run_sim`) makes gen2 a live dependency of code documented as frozen
history (audit F16).  This module owns the mock-aware wrapper `_run_sim` (a real
copy of the former gen1.cascade._run_sim) so gen2's import sites point at
`common` only.

The heavy Verilator subprocess driver (`gen1.runner.run_sim`) genuinely lives in
gen1 — it drives the gen1 Verilator harness — and is out of scope to relocate.
Both entry points below lazy-import it for the real (non-mock) path; that single
lazy touchpoint is the actual Verilator binary, not a constant or wrapper, so it
stays where the harness is.  The measured constants and the acc_overflows
predicate that used to force the gen1 coupling now live in `common.constants`.
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

    (Moved verbatim from gen1.cascade._run_sim — audit F16.)
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
    from eda_rl.gen1.runner import run_sim as _gen1_run_sim  # lazy: needs the Verilator binary
    return _gen1_run_sim(lanes, acc_w)


def run_sim(mac_lanes: int, acc_width: int = 32) -> dict:
    """Thin re-export of the real gen1 Verilator driver (audit F16).

    build_table.py calls this directly (it handles PHYSICAL_MOCK itself before
    reaching here), so this is a lazy passthrough — no mock branch.  Routing it
    through common keeps gen2's import site off gen1 while leaving the Verilator
    harness in gen1 where it belongs.
    """
    from eda_rl.gen1.runner import run_sim as _gen1_run_sim  # lazy: needs the Verilator binary
    return _gen1_run_sim(mac_lanes, acc_width)
