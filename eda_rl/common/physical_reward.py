"""physical_reward.py — design-agnostic PPA reward over REAL ORFS metrics.

This module owns the generic, design-agnostic reward: higher Fmax is better,
larger area/power is worse, timing violations are penalised.  Each metric is
normalised by a per-design anchor supplied in ``refs`` (from the design's
``reward:`` YAML block or auto-anchored from its first F3 build), so the reward
is comparable across builds of the SAME design without any magic constants.

Design-family-specific composite rewards (e.g. the TinyVAD speedup × accuracy
reward) live behind the functional-model plugin registry
(``common.functional_models``): a design opts in via ``functional_eval.kind`` and
the env dispatches to ``FunctionalModel.terminal_reward`` instead of this
function.  Nothing here references a specific design family.
"""

from __future__ import annotations

import warnings


def compute_generic_reward(
    metrics: dict,
    weights: dict | None = None,
    refs: dict | None = None,
) -> dict:
    """Design-agnostic PPA reward for designs without a functional model.

    Pure physical objective: reward higher Fmax, penalise larger area and power,
    gate on timing.  There is NO speedup-vs-software-baseline term and NO
    accuracy term — those are functional-model-specific and meaningless for a
    generic block like gcd (audit C1).

    ``refs`` provides the per-design normalisation anchors
    {area_ref_um2, power_ref_mw, fmax_ref_mhz}; each metric is divided by its
    anchor so the three terms are O(1) and comparable for THIS design.  The
    caller (FunnelEnv) auto-anchors refs from the design's first successful F3
    build when the design YAML declares none, so no magic constants are needed.

    Weights default to: +1.0·(fmax/ref) − 1.0·(area/ref) − 0.4·(power/ref),
    with a timing-violation penalty; override via the design's ``reward:`` block.
    """
    w = weights or {}
    r = refs or {}
    w_fmax = w.get("w_fmax", 1.0)
    w_area = w.get("w_area", -1.0)
    w_pwr  = w.get("w_power", -0.4)
    w_tv   = w.get("w_timing_violation", -3.0)

    status = metrics.get("status", "ok")
    if status not in ("ok", "mock", "mock-proxy"):
        return {"reward": -100.0, "norm_fmax": 0.0, "timing_violation": True,
                "infeasible": True, "status": status}

    area_um2 = metrics.get("area_um2")
    fmax_mhz = metrics.get("fmax_mhz")
    if area_um2 is None or fmax_mhz is None:
        # No usable physical measurement → infeasible (never award from no data).
        return {"reward": -100.0, "norm_fmax": 0.0, "timing_violation": True,
                "infeasible": True, "status": "PARSE_FAIL"}

    if not r:
        # FunnelEnv always pre-resolves refs (design YAML or auto-anchored from
        # the first F3 build, see env.py._generic_reward_cfg) before calling
        # here, so this only fires for standalone callers that skip that step —
        # without it, norm_area/norm_fmax both collapse to 1.0 (self-normalised
        # against this call's own metrics), giving a constant, uninformative
        # reward regardless of actual PPA.
        warnings.warn(
            "compute_generic_reward called with no refs — norm_area/norm_fmax "
            "will self-normalise to 1.0 (constant reward). Pass refs from the "
            "design's reward: YAML block or an auto-anchored build.",
            stacklevel=2,
        )
    area_ref = float(r.get("area_ref_um2") or area_um2 or 1.0)
    fmax_ref = float(r.get("fmax_ref_mhz") or fmax_mhz or 1.0)
    power_mw = metrics.get("power_mw")
    power_ref = r.get("power_ref_mw")

    norm_area = area_um2 / max(area_ref, 1e-9)
    norm_fmax = fmax_mhz / max(fmax_ref, 1e-9)
    t_viol = 0.0 if metrics.get("timing_met", True) else 1.0

    reward = w_fmax * norm_fmax + w_area * norm_area + w_tv * t_viol
    power_term = 0.0
    norm_power = None
    if power_mw is not None and power_ref:
        norm_power = power_mw / max(float(power_ref), 1e-9)
        power_term = w_pwr * norm_power
        reward += power_term
    return {
        "reward":           round(reward, 4),
        "norm_fmax":        round(norm_fmax, 4),
        "norm_area":        round(norm_area, 4),
        "norm_power":       round(norm_power, 4) if norm_power is not None else None,
        "timing_violation": bool(t_viol),
        "infeasible":       False,
    }
