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

# Reward semantics version, stamped onto every campaign log row and summary.
#
#   1 — original.  Timing on the fixed reference SDC (audit F1), but power scored
#       RAW at the sampled clock and DRC ignored entirely.  Corpora produced under
#       v1 are contaminated by the clock-period confound (measured corr(reward,
#       clock_period_ns) = +0.76) and are NOT comparable with v2.
#   2 — power normalised to the design's reference frequency (audit F18) and a
#       DRC gate added (audit F19).
#
# Rows without a `reward_version` key are version 1 by definition.  Anything that
# aggregates across campaigns (fit_surrogate, benchmarks, cross-campaign plots)
# must refuse to mix versions rather than average them.
REWARD_VERSION = 2


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
    # audit F19: a build with routing DRC violations is not manufacturable, so it
    # must not outscore a clean one.  Weighted like the timing gate: any non-zero
    # count is a hard mark, with a small per-violation term so "nearly clean"
    # ranks above "hopeless".  drc_count is None when DRC was never measured —
    # that is NOT the same as zero and must not earn the clean-build score.
    w_drc  = w.get("w_drc", -3.0)

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

    # DRC gate (audit F19).  A dirty build is penalised as a flat mark plus a
    # saturating per-violation term, so 3 violations ranks above 3000 but both
    # rank below any clean build.  drc_count None => not measured => no term
    # (scoring it as clean would let an unrouted build win).
    drc_count = metrics.get("drc_count")
    drc_viol = None
    if drc_count is not None:
        try:
            drc_count = int(drc_count)
        except (TypeError, ValueError):
            drc_count = None
    if drc_count is not None:
        drc_viol = drc_count > 0
        if drc_viol:
            # 1.0 at the first violation, asymptotically 2.0 for many.
            severity = 1.0 + drc_count / (drc_count + 50.0)
            reward += w_drc * severity

    return {
        "reward":           round(reward, 4),
        "norm_fmax":        round(norm_fmax, 4),
        "norm_area":        round(norm_area, 4),
        "norm_power":       round(norm_power, 4) if norm_power is not None else None,
        "timing_violation": bool(t_viol),
        "drc_count":        drc_count,
        "drc_violation":    drc_viol,
        "infeasible":       False,
    }
