#!/usr/bin/env python3
"""Property tests for the generic PPA reward (audit F18/F19).

Why this file exists
--------------------
Before it, `tests/` covered log parsing and nothing else — and a real reward-gaming
bug lived in production undetected for months.  On the sagar/sky130hd corpus
(n=178 successful F3 builds) the terminal reward correlated **+0.76 with the
`clock_period_ns` knob**, and every campaign's "best" config sat within 0.01 ns of
the top of its declared clock range.  The mechanism: OpenROAD reports power at the
*sampled* clock, so asking for a slower clock lowered reported dynamic power and
inflated the reward.  `corr(clk, power) = -0.69` while
`corr(clk, power x period) = -0.04` — the whole effect was the frequency confound.

The audit-F1 fix had already put timing on a fixed reference ruler; power was left
raw because it was assumed "physical".  Power measured at a *different operating
frequency* is not comparable, so the gaming simply migrated from the fmax term to
the power term.

The invariant these tests enforce: **the reward must measure the chip, not the
ruler.**  Holding the physical outcome fixed, no constraint knob may move it.

Run: python3 tests/test_reward.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from eda_rl.common.physical_reward import compute_generic_reward  # noqa: E402

# Anchors taken from the first successful F3 build of the real sagar campaign.
REFS = {"area_ref_um2": 487.0, "power_ref_mw": 0.0282, "fmax_ref_mhz": 505.05}


def _m(**over):
    """A baseline 'ok' metrics dict, overridable per test."""
    base = {
        "status": "ok",
        "area_um2": 487.0,
        "fmax_mhz": 505.05,
        "power_mw": 0.0282,
        "timing_met": True,
        "drc_count": 0,
    }
    base.update(over)
    return base


def _r(**over):
    return compute_generic_reward(_m(**over), refs=REFS)["reward"]


# ── monotonicity: the reward must actually track PPA ───────────────────────────

def test_reward_increases_with_fmax():
    assert _r(fmax_mhz=600.0) > _r(fmax_mhz=505.05) > _r(fmax_mhz=400.0)


def test_reward_decreases_with_area():
    assert _r(area_um2=400.0) > _r(area_um2=487.0) > _r(area_um2=600.0)


def test_reward_decreases_with_power():
    assert _r(power_mw=0.020) > _r(power_mw=0.0282) > _r(power_mw=0.040)


def test_timing_violation_is_penalised():
    assert _r(timing_met=False) < _r(timing_met=True)


# ── the anti-gaming invariants ────────────────────────────────────────────────

def test_reward_invariant_to_clock_knob_at_fixed_physics():
    """THE regression guard for audit F18.

    Two builds with identical physical outcomes must score identically no matter
    what clock period was requested.  `compute_generic_reward` never sees the
    clock knob, so this asserts the contract the caller relies on: the reward is
    a pure function of the (already ruler-corrected) physical metrics.
    """
    slow = _m()
    fast = _m()
    # A clock knob leaking into the metrics dict must not change the score.
    slow["clk_ns"] = 8.0
    fast["clk_ns"] = 5.5
    assert (compute_generic_reward(slow, refs=REFS)["reward"]
            == compute_generic_reward(fast, refs=REFS)["reward"])


def test_reward_invariant_to_constraint_knobs():
    """Audit F1 guard: SDC knobs must never enter the score."""
    base = compute_generic_reward(_m(), refs=REFS)["reward"]
    for knob, value in (("IO_DELAY", 0.9), ("CLOCK_UNCERTAINTY", 0.1),
                        ("GR_SEED", 77)):
        m = _m()
        m[knob] = value
        assert compute_generic_reward(m, refs=REFS)["reward"] == base, knob


def test_power_ruler_removes_the_frequency_confound():
    """The end-to-end property the F18 fix buys, at the reward's own level.

    Same silicon, measured at two clocks: dynamic power scales with frequency, so
    the RAW numbers differ and the slow build wins.  After normalising to the
    reference frequency (what FunnelEnv._power_at_ref_freq does), both score the
    same and the spurious preference is gone.
    """
    leakage, dynamic_at_ref = 0.002, 0.0262   # mW, at ref period 5.5 ns
    ref_ns = 5.5

    def raw_power(clk_ns):
        # What OpenROAD reports at that clock: dynamic scales with frequency.
        return leakage + dynamic_at_ref * (ref_ns / clk_ns)

    def normalised(clk_ns):
        p = raw_power(clk_ns)
        return leakage + (p - leakage) * (clk_ns / ref_ns)

    slow_raw = _r(power_mw=raw_power(8.0))
    fast_raw = _r(power_mw=raw_power(5.5))
    assert slow_raw > fast_raw, "precondition: raw power favours the slow clock"

    slow_norm = _r(power_mw=normalised(8.0))
    fast_norm = _r(power_mw=normalised(5.5))
    assert abs(slow_norm - fast_norm) < 1e-9, (slow_norm, fast_norm)


# ── DRC gate (audit F19) ──────────────────────────────────────────────────────

def test_drc_dirty_never_outscores_clean():
    clean = _r(drc_count=0)
    assert _r(drc_count=1) < clean
    assert _r(drc_count=500) < _r(drc_count=1) < clean


def test_drc_unmeasured_is_not_treated_as_clean():
    """drc_count None means the flow never routed.  It must not silently earn the
    clean-build score, and it must not crash."""
    out = compute_generic_reward(_m(drc_count=None), refs=REFS)
    assert out["drc_count"] is None and out["drc_violation"] is None, out


def test_drc_dirty_loses_to_a_worse_but_clean_build():
    """Manufacturability dominates: a clean build that is slower and bigger still
    beats a fast, small, DRC-dirty one."""
    dirty_but_good = _r(fmax_mhz=700.0, area_um2=400.0, drc_count=40)
    clean_but_worse = _r(fmax_mhz=450.0, area_um2=550.0, drc_count=0)
    assert clean_but_worse > dirty_but_good


# ── failure handling ──────────────────────────────────────────────────────────

def test_failed_status_is_infeasible():
    for st in ("FAIL", "TIMEOUT", "PARSE_FAIL"):
        out = compute_generic_reward(_m(status=st), refs=REFS)
        assert out["infeasible"] is True and out["reward"] == -100.0, (st, out)


def test_missing_physical_metrics_never_scores():
    """No area or no fmax => infeasible.  Never award from absent data."""
    for over in ({"area_um2": None}, {"fmax_mhz": None}):
        out = compute_generic_reward(_m(**over), refs=REFS)
        assert out["infeasible"] is True, (over, out)


def test_refs_are_deterministic():
    """Same metrics + same refs => identical score, every call."""
    vals = {compute_generic_reward(_m(), refs=REFS)["reward"] for _ in range(5)}
    assert len(vals) == 1, vals


# ── plain-python runner (matches tests/test_parsers.py) ───────────────────────

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
    print(f"\n{len(tests) - failed}/{len(tests)} reward property tests passed")
    sys.exit(1 if failed else 0)
