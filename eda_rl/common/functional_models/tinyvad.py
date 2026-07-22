"""tinyvad.py — the TinyVAD/TinyMAC functional model (a FunctionalModel plugin).

This is the ONE place the TinyVAD MAC-accelerator's design-specific machinery
lives.  A design opts into it by declaring ``functional_eval: {kind: tinyvad_sim}``
in its YAML (see ``designs/tinymac_accel.yaml``); nothing in the design-agnostic
core references TinyVAD/TinyMAC.  The plugin owns:

  * the measured behavioral cycle model + SW baseline (the F0/F1 speed model);
  * ``acc_overflows`` (the accumulator-too-narrow accuracy gate);
  * the composite F3 reward (speedup × accuracy × area × power);
  * the surrogate UCB mirror of that reward;
  * (lazily) the report extension with the hand-picked baseline and speedup fig.

All constants were MEASURED from the Verilator sim after the V13 saturation-order
fix (per-chunk, not per-MAC) landed in sim_main.cpp on 2026-06-10, aligning the
behavioral sim with the RTL FSM and the TB golden model.  Sweep that produced
them: ``python3 legacy/measure_real.py`` extended to LANES ∈ {1,2,4,8,16,32},
averaging all 64 inference vectors.  Do NOT re-hardcode these numbers elsewhere.
"""

from __future__ import annotations

import math
from typing import Any

from eda_rl.common.functional_models.base import FunctionalModel

# ── SW baseline (Stage-3, no accelerator) ────────────────────────────────────
# Measured on the Stage-3 PicoRV32 Verilator sim (100 MHz).
SW_BASELINE_CYCLES: int = 11_196_638
SW_BASELINE_CLOCK_NS: float = 10.0                              # 100 MHz
SW_BASELINE_LATENCY_NS: float = SW_BASELINE_CYCLES * SW_BASELINE_CLOCK_NS

# ── TinyVAD accumulator-overflow predicate ───────────────────────────────────
# TinyVAD worst-case signed accumulator magnitude:
#   Conv0: K=200 (in_ch=40 × kern=5), max |product| = 128×128 = 16384
#   Max |acc| = 200 × 16384 = 3,276,800
_TINYVAD_MAX_ACC = 3_276_800
_INT_MAX = {16: 32_767, 24: 8_388_607, 32: 2_147_483_647}


def acc_overflows(config: dict) -> bool:
    """Fast proxy check: True if acc_width is analytically too narrow for TinyVAD.

    The sim will also catch this (accuracy < 1.0), but this lets agents/rewards
    skip obviously bad configs without launching a subprocess.
    """
    acc_w = config.get("accumulator_width", 32)
    return _TINYVAD_MAX_ACC > _INT_MAX.get(acc_w, _INT_MAX[32])


# ── Measured accelerator cycles per inference ─────────────────────────────────
# Sweep: LANES ∈ {1,2,4,8,16,32}, ACC_W=32 (acc_w does not affect cycle count),
# 64 test vectors each, averaged.  Measured 2026-06-10 after V13 fix.
# Cycle model is output-stationary with ACCEL_CH_OVERHEAD=2 (bias load + requant):
#   latency = n_outputs × (ceil(K / LANES) + 2)
AVG_CYCLES: dict[int, int] = {
    1:  273_130,
    2:  152_140,
    4:   91_650,
    8:   61_400,
    16:  46_670,
    32:  39_310,
}

# ── behavioral_cycles(lanes): analytic fit to AVG_CYCLES (a + b/lanes) ────────
# Fit via least squares over all 6 lane counts; max abs residual ~309 cycles
# out of ~39K–273K (< 1%).
_CYCLE_OVERHEAD = 31_452.5
_CYCLE_MAC_WORK = 241_567.0


def behavioral_cycles(lanes: int) -> float:
    """Estimated accel cycles per TinyVAD inference for ``lanes`` MAC lanes."""
    return _CYCLE_OVERHEAD + _CYCLE_MAC_WORK / max(lanes, 1)


# ── max_speedup derivation ────────────────────────────────────────────────────
# GRID: for the 45-config space (5 lanes × 3 acc × 3 clk).  FULL: cascade/full
# space (lanes up to 32).  See git history for the frequency-aware derivation.
MAX_SPEEDUP_GRID: int = 576
MAX_SPEEDUP_FULL: int = 1024

# ── LANES=4 ACC_W=24 physical anchors (the first full GDS) ────────────────────
# area/power terms normalise to ~1.0 so the YAML weights carry over.
AREA_REF_UM2 = 19_738.0
POWER_REF_MW = 1_020.0

# ── F0 analytic accuracy table (V13: LANES-dependent at ACC_W<32) ─────────────
# (lanes, acc_w) → accuracy fraction from the 2026-06-10 measured sweep.
_ACC_TABLE: dict[tuple[int, int], float] = {
    (1,  32): 1.0, (1,  24): 1.0,  (1,  16): 47/64,
    (2,  32): 1.0, (2,  24): 1.0,  (2,  16): 48/64,
    (4,  32): 1.0, (4,  24): 1.0,  (4,  16): 48/64,
    (8,  32): 1.0, (8,  24): 1.0,  (8,  16): 48/64,
    (16, 32): 1.0, (16, 24): 1.0,  (16, 16): 48/64,
    (32, 32): 1.0, (32, 24): 1.0,  (32, 16): 58/64,
}


# ── Frequency-aware speedup over the Stage-3 SW baseline ──────────────────────

def achieved_period_ns(clk_ns: float, fmax_mhz: float | None) -> float:
    """Period the silicon actually runs at: the slower of the requested clock
    and the routed critical-path period (1000/Fmax).  An over-aggressive clock
    request gives no free speed.  If fmax_mhz is None (parse failed / no flow),
    return a very large period so the speedup term is essentially 0 — never
    award speed from zero physical data.
    """
    if fmax_mhz is None:
        return float("inf")
    crit = 1000.0 / fmax_mhz
    return max(float(clk_ns), crit)


def physical_real_speedup(metrics: dict, cycles: float | None = None) -> float:
    """Frequency-aware speedup over the Stage-3 SW baseline, using real Fmax."""
    cyc = cycles if cycles is not None else behavioral_cycles(metrics["lanes"])
    period = achieved_period_ns(metrics["clk_ns"], metrics.get("fmax_mhz"))
    latency_ns = max(cyc, 1.0) * period
    return SW_BASELINE_LATENCY_NS / max(latency_ns, 1e-9)


def compute_physical_reward(
    metrics: dict,
    weights: dict | None = None,
    cycles: float | None = None,
) -> dict:
    """Return {'reward': float, ...derived fields} for a physical metrics dict.

    TinyVAD composite: uses the behavioral cycle model, the LANES=4 area/power
    anchors, and the accumulator-overflow accuracy term.  Returns a dict (not a
    bare float) so the env can log the derived speedup / normalised terms.
    """
    w = weights or {}
    w_acc    = w.get("w_accuracy",          2.0)
    w_spd    = w.get("w_speedup",           3.0)
    w_area   = w.get("w_area",             -0.4)
    w_pwr    = w.get("w_power",            -0.4)
    w_tv     = w.get("w_timing_violation", -3.0)
    # MAX_SPEEDUP_FULL (1024) for the full lanes-up-to-32 space; must match the
    # surrogate UCB proxy (surrogate_reward below) or the two miscalibrate.
    max_spd  = w.get("max_speedup",        float(MAX_SPEEDUP_FULL))
    min_spd  = w.get("min_useful_speedup",  10.0)
    perf_pen = w.get("perf_floor_penalty",  -8.0)

    # ANY non-ok status is the worst-case outcome; never substitute reference
    # values for missing measurements (that would award reward from no data).
    status = metrics.get("status", "ok")
    if status not in ("ok", "mock", "mock-proxy"):
        return {"reward": -100.0, "real_speedup": 0.0, "norm_speedup": -1.0,
                "accuracy": 0.0, "timing_violation": True, "infeasible": True,
                "status": status}

    accuracy = 0.0 if acc_overflows({"accumulator_width": metrics["acc_w"]}) else 1.0
    spd      = physical_real_speedup(metrics, cycles)

    if spd > 0 and max_spd > 1:
        norm_spd = math.log2(max(spd, 1e-3)) / math.log2(max_spd)
        norm_spd = max(-1.0, min(1.0, norm_spd))
    else:
        norm_spd = -1.0

    area_um2 = metrics.get("area_um2")
    power_mw = metrics.get("power_mw")
    if area_um2 is None or power_mw is None:
        return {"reward": -100.0, "real_speedup": 0.0, "norm_speedup": -1.0,
                "accuracy": 0.0, "timing_violation": True, "infeasible": True,
                "status": "PARSE_FAIL"}

    area  = area_um2 / AREA_REF_UM2
    power = power_mw / POWER_REF_MW
    t_viol = 0.0 if metrics.get("timing_met", True) else 1.0

    floor_penalty = perf_pen if spd < min_spd else 0.0
    correctness   = -50.0 * (1.0 - accuracy)

    reward = (
        w_acc * accuracy
        + w_spd * norm_spd
        + w_area * area
        + w_pwr * power
        + w_tv * t_viol
        + correctness
        + floor_penalty
    )
    return {
        "reward":           round(reward, 4),
        "real_speedup":     round(spd, 3),
        "norm_speedup":     round(norm_spd, 4),
        "accuracy":         accuracy,
        "area_norm":        round(area, 4),
        "power_norm":       round(power, 4),
        "timing_violation": bool(t_viol),
        "infeasible":       False,
    }


# ── F0/F1 helpers ─────────────────────────────────────────────────────────────

def f0_cycles(config: dict) -> float:
    """Analytic cycle estimate for F0 (measured table when available)."""
    lanes = int(config.get("mac_lanes") or config.get("lanes") or 4)
    if lanes in AVG_CYCLES:
        return float(AVG_CYCLES[lanes])
    return behavioral_cycles(lanes)


def f0_accuracy(config: dict) -> float:
    """Analytic accuracy estimate for F0 from the measured (lanes, acc_w) table."""
    lanes = int(config.get("mac_lanes") or config.get("lanes") or 4)
    acc_w = int(config.get("accumulator_width") or config.get("acc_w") or 24)
    key = (lanes, acc_w)
    if key in _ACC_TABLE:
        return _ACC_TABLE[key]
    if acc_w >= 24:
        return 1.0
    return 47.0 / 64.0   # conservative lower bound


def run_f1(config: dict, *, mock: bool) -> dict:
    """Behavioral Verilator run → {accuracy, correct, n_total, avg_cycles}.

    Under ``mock``, returns numbers shaped like the real measured sweep:
    acc_width < 24 loses accuracy (int16 ≈ 47/64), matching the empirical finding.
    """
    lanes = int(config.get("mac_lanes") or config.get("lanes") or 4)
    acc_w = int(config.get("accumulator_width") or config.get("acc_w") or 24)
    if mock:
        if acc_w >= 24:
            acc = 1.0
        elif acc_w >= 20:
            acc = 0.92
        else:                       # int16 — overflow, real measured 47/64
            acc = 47.0 / 64.0
        cyc = float(AVG_CYCLES[lanes]) if lanes in AVG_CYCLES else behavioral_cycles(lanes)
        return {"accuracy": acc, "correct": round(acc * 64), "n_total": 64,
                "avg_cycles": cyc}
    from eda_rl.common.functional_models.tinyvad_verilator import run_sim
    return run_sim(lanes, acc_w)


# ── Canonical TinyVAD search space ────────────────────────────────────────────
# The 4-axis TinyVAD/tinymac design space (matches designs/tinymac_accel.yaml via
# KnobRegistry).  Used by the tinymac-shaped benchmark and self-tests that need a
# concrete space without loading the KnobRegistry.  The clock axis declares a
# _snap_step so table-mode lookups keep hitting stored rows.
def tinyvad_search_space() -> dict:
    """Return the canonical 4-axis TinyVAD search space (lanes/acc/clk/recipe)."""
    return {
        "mac_lanes": {
            "type": "categorical",
            "choices": [1, 2, 4, 8, 16, 32],
            "default": 4,
        },
        "accumulator_width": {
            "type": "categorical",
            "choices": [16, 24, 32],
            "default": 24,
        },
        "clock_period_ns": {
            "type": "float",
            "range": [3.0, 8.0],
            "default": 5.0,
            "_snap_step": 0.5,
        },
        "abc_recipe": {
            "type": "categorical",
            "choices": ["orfs_speed", "orfs_area", "plain"],
            "default": "plain",
        },
    }


# ── The plugin ─────────────────────────────────────────────────────────────────

class TinyVADModel(FunctionalModel):
    """TinyVAD MAC-accelerator functional model (speedup × accuracy composite)."""

    kind = "tinyvad_sim"
    reward_kind = "composite"
    enables_f1 = True
    sw_baseline_cycles = SW_BASELINE_CYCLES

    def f0_cycles(self, config: dict) -> float:
        return f0_cycles(config)

    def f0_accuracy(self, config: dict) -> float:
        return f0_accuracy(config)

    def run_f1(self, config: dict, *, mock: bool) -> dict:
        return run_f1(config, mock=mock)

    def terminal_reward(
        self, obs: dict, weights: dict | None = None, *, cycles: float | None = None
    ) -> dict:
        return compute_physical_reward(obs, weights, cycles)

    def surrogate_reward(self, x: dict, preds: dict) -> tuple[float, float]:
        """(mu, sigma) of the composite reward from surrogate metric predictions.

        Mirrors compute_physical_reward using the same constants so the UCB
        signal ranks configs the way the real objective scores them.
        """
        mu_area, sig_area = preds["area_um2"]
        mu_period, sig_period = preds["period_ns"]
        mu_power, sig_power = preds["power_mw"]
        mu_period = max(mu_period, 0.5)   # negative period predictions are unphysical

        lanes = int(x.get("mac_lanes") or x.get("lanes") or 4)
        acc_w = int(x.get("accumulator_width") or x.get("acc_w") or 24)

        accuracy = 0.0 if acc_w <= 16 else 1.0   # matches acc_overflows (A16 overflows)
        w_acc = 2.0
        correctness = -50.0 * (1.0 - accuracy)

        sw_baseline_ns = SW_BASELINE_CYCLES * SW_BASELINE_CLOCK_NS
        cyc = behavioral_cycles(lanes)
        mu_latency_ns = cyc * mu_period
        mu_speedup = sw_baseline_ns / max(mu_latency_ns, 1.0)
        max_spd = float(MAX_SPEEDUP_FULL)   # 1024 — matches the funnel YAML reward
        norm_spd = math.log2(max(mu_speedup, 1e-3)) / math.log2(max_spd) if max_spd > 1 else 0.0
        norm_spd = max(-1.0, min(1.0, norm_spd))
        sig_speedup = (sw_baseline_ns / mu_latency_ns**2) * cyc * sig_period
        sig_norm_spd = (sig_speedup / (mu_speedup * math.log(max_spd))) if mu_speedup > 0 else 0.1

        mu_reward = (
            w_acc * accuracy
            + 3.0 * norm_spd
            + (-0.4) * (mu_area / AREA_REF_UM2)
            + (-0.4) * (mu_power / POWER_REF_MW)
            + correctness
        )
        var = (
            (3.0 * sig_norm_spd) ** 2
            + (0.4 / AREA_REF_UM2 * sig_area) ** 2
            + (0.4 / POWER_REF_MW * sig_power) ** 2
        )
        return (float(mu_reward), float(math.sqrt(var)))

    def report_extension(self) -> "Any | None":
        from eda_rl.common.functional_models.tinyvad_report import TinyVADReport
        return TinyVADReport()
