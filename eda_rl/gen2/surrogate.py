"""surrogate.py — multi-fidelity surrogate that predicts final (F3) physical
metrics from a config x, optionally conditioned on cheaper F2 observables.

Architecture
------------
Per-metric ensemble of three GradientBoostingRegressor instances trained on
three quantile loss values (q=0.16, q=0.50, q=0.84), following the quantile-
regression ensemble (QRE) approach.  Using quantile GBT rather than a bagged
ensemble avoids the n_estimators×3 memory cost while giving calibrated interval
estimates: sigma ≈ (q84 − q16) / 2, mu = q50 prediction.

Features
--------
Config features are DISCOVERED from the training corpus, not hardcoded (audit
F5). At fit() time an axis schema is built from the config axes present in the
rows: every numeric axis (sorted by name) becomes one feature; every small
categorical axis becomes a one-hot block. The ordered schema is stored in the
joblib payload so predict() builds features identically, and predict() REFUSES
(raises) when the incoming config does not cover the schema's required axes —
which also prevents a surrogate fitted on one design from silently
mispredicting another's configs. Notable transforms:
  - lanes              — featurized as log2(lanes) (area/cycles sub-linear)
  - abc (recipe)       — one-hot; the 'area' recipes ('area'/'orfs_area') are
                         distinguished from 'orfs_speed'/'plain'
  - util  ← CORE_UTILIZATION,  density ← PLACE_DENSITY (aliased so the funnel's
                         actual knob keys are seen, not the wrong 'util'/'density')
  - tier-2/3 knobs     — IO_DELAY, PLACE_DENSITY_LB_ADDON, CTS_*, … all featurized
  - platform, rtl_hash — one-hot context axes (not required by the coverage check)

Conditional F2 observables (present or missing — handled with indicator cols):
  - proxy_area_um2     — synth cell area × 1.35 inflation estimate
  - proxy_wns_ns       — pre-layout WNS from fast STA (sign of proxy timing)
  - ff_count           — sequential cell count from finish report JSON
  - cell_count         — total standard-cell count from finish report JSON
  - logic_levels       — not available from the current build artifacts; field
                         accepted but left always-missing (indicator = 0)

For each obs column, a paired "obs_<col>_present" indicator is appended so
a single model handles both the x-only and x+obs prediction modes.

Small-n fallback
----------------
If n < 10 training rows, fit is skipped.  predict() returns (mean, large_sigma)
where large_sigma = 2 × stddev of observed values (or a hard fallback if n==0).
This is documented behaviour, not a crash.

Composite-reward propagation
-----------------------------
predict_reward_stats() propagates the per-metric predictions through a crude
first-order uncertainty propagation of the physical_reward formula.  It is
intentionally approximate — its main use is as a UCB acquisition signal, not
a calibrated confidence interval.

Saving / loading
----------------
joblib.dump/load; the file stores (model dict, metadata).

Usage
-----
    s = Surrogate(seed=42)
    diag = s.fit(rows)          # rows: list[dict] in results_physical / funnel format
    mu_s, sig_s = s.predict({"mac_lanes": 4, "accumulator_width": 24,
                              "clock_period_ns": 5.0})["area_um2"]
    s.save("surrogate_n45.joblib")
    s2 = Surrogate.load("surrogate_n45.joblib")
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# ── Constants ──────────────────────────────────────────────────────────────────

METRICS = ["area_um2", "period_ns", "power_mw"]

# GBT hyperparameters.  Chosen to be stable on 40–50 rows:
# - n_estimators=200, max_depth=3, learning_rate=0.05: low-variance, avoids
#   overfit on ~10-fold-CV splits of ~32 rows.
# - subsample=0.8: standard stochastic GBT regularisation.
_GBT_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=2,
)

# Indicator that a model has NOT been fitted (too few rows)
_FALLBACK = "fallback"

# Obs columns that the surrogate accepts.  Order matters for feature vector.
_OBS_COLS = ["proxy_area_um2", "proxy_wns_ns", "ff_count", "cell_count", "logic_levels"]

# Obs-key aliases (audit C2): the live FunnelEnv F2 obs and build_table rows use
# the runner-facing keys area_um2 / wns_ns / cells, while the legacy report miner
# (fit_surrogate) emits proxy_area_um2 / proxy_wns_ns.  Without this map the two
# strongest proxy signals (area, WNS) were imputed as "missing" on every live
# query, silently disabling the multi-fidelity conditioning.  First present
# alias wins; canonical name first so explicit proxy_* values take priority.
_OBS_ALIASES: dict[str, list[str]] = {
    "proxy_area_um2": ["proxy_area_um2", "area_um2"],
    "proxy_wns_ns":   ["proxy_wns_ns", "wns_ns"],
    "ff_count":       ["ff_count"],
    "cell_count":     ["cell_count", "cells"],
    "logic_levels":   ["logic_levels"],
}


def _obs_value(obs: dict | None, col: str):
    """Return the value for surrogate column `col` from `obs`, honouring aliases."""
    if not obs:
        return None
    for key in _OBS_ALIASES.get(col, [col]):
        v = obs.get(key)
        if v is not None:
            return v
    return None


# ── Generalized config featurization (audit F5) ────────────────────────────────
#
# The old surrogate hardcoded an 8-element config vector (log2 lanes, acc_w, clk,
# abc_flag, plat_flag, hash_cat, util, density). That made it structurally blind
# to every tier-2/3 knob the campaigns actually vary (IO_DELAY, CTS_*,
# PLACE_DENSITY_LB_ADDON, …) and read util/density from the wrong keys. We now
# discover the config axes from the training corpus and featurize ALL of them:
# every numeric axis (sorted by name) becomes one feature; every small
# categorical axis becomes a one-hot block. The ordered axis schema is stored in
# the joblib payload so fit and predict agree, and predict refuses to run when
# the incoming config doesn't cover the schema's required axes (this also
# neutralizes the cross-design auto-load hazard: a tinymac-fitted surrogate can't
# silently mispredict a gcd/likith config).

# Max distinct values for an axis to be one-hot encoded (else it is dropped from
# the feature vector — e.g. a high-cardinality free axis; rtl_hash is context and
# handled separately).
_MAX_ONEHOT = 24

# Numeric axes featurized on a log2 scale (area/cycles grow sub-linearly in lanes).
_SPECIAL_LOG = {"lanes"}

# Categorical context axes: always featurized (missing → all-zero one-hot) but
# NOT required by the coverage check — a config being predicted need not carry
# them (platform is passed out-of-band; rtl_hash enables transfer to unseen RTL).
_CONTEXT_AXES = {"rtl_hash", "platform"}

# Row keys that are metrics / obs / bookkeeping, NOT config axes. Everything in a
# flattened row that is not one of these (and not an obs alias) is a config axis.
_NONCONFIG_KEYS = {
    # F3 / F2 metrics + obs
    "area_um2", "period_ns", "period_min_ns", "fmax_mhz", "power_mw",
    "wns_ns", "tns_ns", "setup_viol", "timing_met", "util_pct",
    "proxy_area_um2", "proxy_wns_ns", "ff_count", "cell_count", "cells",
    "logic_levels", "accuracy", "accuracy_flag", "avg_cycles", "cycles",
    # bookkeeping / structural
    "status", "variant", "design", "gds", "reward", "fidelity",
    "obs", "config", "metrics", "effective_abc_recipe",
}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _config_view(flat: dict) -> dict:
    """Extract the config-axis view of a flattened row for featurization.

    Canonicalizes the RTL/clock/recipe/util/density aliases to a single axis
    namespace and passes every other non-metric key (the tier-2/3 knobs) through
    unchanged. Only axes actually present are included — no defaults are injected,
    so the coverage check can tell a gcd config (no lanes/IO_DELAY) apart from a
    tinymac/likith one.
    """
    v: dict = {}

    def _first(*keys):
        for k in keys:
            if flat.get(k) is not None:
                return flat[k]
        return None

    lanes = _first("mac_lanes", "lanes")
    if lanes is not None:
        v["lanes"] = int(lanes)
    acc = _first("accumulator_width", "acc_w")
    if acc is not None:
        v["acc_w"] = int(acc)
    clk = _first("clock_period_ns", "clk_ns")
    if clk is not None:
        v["clk_ns"] = float(clk)
    abc = _first("abc_recipe", "abc")
    if abc is not None:
        v["abc"] = str(abc)
    util = _first("CORE_UTILIZATION", "util")
    if util is not None:
        v["util"] = float(util)
    density = _first("PLACE_DENSITY", "density")
    if density is not None:
        v["density"] = float(density)
    plat = _first("platform")
    if plat is not None:
        v["platform"] = str(plat)
    rtl_hash = _first("rtl_hash")
    if rtl_hash is not None:
        v["rtl_hash"] = str(rtl_hash)

    # Pass-through knob axes (IO_DELAY, PLACE_DENSITY_LB_ADDON, CTS_*, …). Skip
    # metric/obs/bookkeeping keys, the alias source keys already canonicalized,
    # and obs-alias keys.
    _consumed = {
        "mac_lanes", "lanes", "accumulator_width", "acc_w",
        "clock_period_ns", "clk_ns", "abc_recipe", "abc",
        "CORE_UTILIZATION", "util", "PLACE_DENSITY", "density",
        "platform", "rtl_hash",
    }
    _obs_alias_keys = {a for aliases in _OBS_ALIASES.values() for a in aliases}
    for k, val in flat.items():
        if k in _consumed or k in _NONCONFIG_KEYS or k in _obs_alias_keys:
            continue
        if val is None:
            continue
        v[k] = val
    return v


# ── Feature engineering ────────────────────────────────────────────────────────


def _encode_config(x: dict) -> list[float]:
    """LEGACY, unused: the fixed 8-element config feature vector.

    Kept only for reference — the live path is Surrogate._build_feature_row,
    which featurizes the full discovered axis schema (see _config_view). This
    helper returns 8 elements (log2 lanes, acc_w, clk, abc_flag, plat_flag,
    hash placeholder, util, density); the old docstring's "6-element" claim was
    stale (audit F17).

    Accepts either the optimizer-facing keys (mac_lanes / accumulator_width /
    clock_period_ns / abc_recipe / platform) or the runner-facing keys
    (lanes / acc_w / clk_ns / abc / platform).  Both styles are used in the
    corpus so we normalise here.
    """
    lanes = int(x.get("mac_lanes") or x.get("lanes") or 4)
    acc_w = int(x.get("accumulator_width") or x.get("acc_w") or 24)
    clk   = float(x.get("clock_period_ns") or x.get("clk_ns") or 5.0)

    # abc_recipe: any 'area' recipe → 1 (catches 'area' and 'orfs_area'); else 0
    abc_raw = x.get("abc_recipe") or x.get("abc") or ""
    abc_flag = 1.0 if "area" in str(abc_raw).lower() else 0.0

    # platform: 'asap7' → 1; everything else (nangate45/sky130…) → 0
    plat_raw = x.get("platform") or "nangate45"
    plat_flag = 1.0 if str(plat_raw).lower() == "asap7" else 0.0

    # util/density: floorplan/placement axes (frozen at 40/0.60 in the funnel
    # space, but explicit features so matched builds don't alias — see EXP-F3).
    util    = float(x.get("util", 40) or 40)
    density = float(x.get("density", 0.60) or 0.60)

    # rtl_hash: treated as a categorical context feature (integer-encoded later
    # by the Surrogate which holds the hash→int mapping built during fit)
    # We return a placeholder 0.0 here; _build_feature_row replaces it.
    return [math.log2(max(lanes, 1)), float(acc_w), clk, abc_flag, plat_flag, 0.0,
            util, density]


def _encode_obs(obs: dict | None, means: dict) -> list[float]:
    """Return the 2×|_OBS_COLS| obs feature vector (value + present indicator).

    Missing values are replaced with the training-set column mean (from `means`)
    and their indicator is set to 0.  Present values get indicator 1.
    """
    feats: list[float] = []
    for col in _OBS_COLS:
        val = _obs_value(obs, col)
        if val is not None:
            feats.append(float(val))
            feats.append(1.0)
        else:
            feats.append(float(means.get(col, 0.0)))
            feats.append(0.0)
    return feats


# ── Row normalisation ──────────────────────────────────────────────────────────


def _flatten_row(row: dict) -> dict:
    """Accept both flat and nested row formats and return a flat dict.

    Supported shapes:
      - results_physical.jsonl flat format: {config:{mac_lanes,...}, metrics:{...}}
      - runner flat format: {lanes:4, acc_w:24, clk_ns:5, area_um2:..., ...}
      - funnel format: {config:{...}, fidelity:..., obs:{...}, metrics:{...}}
    """
    flat: dict = {}

    # nested → flatten config
    if "config" in row and isinstance(row["config"], dict):
        flat.update(row["config"])
    # nested → flatten metrics
    if "metrics" in row and isinstance(row["metrics"], dict):
        flat.update(row["metrics"])
    # nested → flatten obs
    if "obs" in row and isinstance(row["obs"], dict):
        flat.update(row["obs"])
    # Merge remaining top-level keys (flat format)
    for k, v in row.items():
        if k not in ("config", "metrics", "obs") and k not in flat:
            flat[k] = v

    # Normalise key aliases so downstream code sees one canonical set
    for src, dst in [
        ("mac_lanes", "lanes"),
        ("accumulator_width", "acc_w"),
        ("clock_period_ns", "clk_ns"),
        ("abc_recipe", "abc"),
    ]:
        if src in flat and dst not in flat:
            flat[dst] = flat[src]

    # period_ns comes from period_min_ns in ORFS records
    if "period_ns" not in flat and "period_min_ns" in flat:
        flat["period_ns"] = flat["period_min_ns"]
    # fmax → period_ns if still missing
    if "period_ns" not in flat and "fmax_mhz" in flat and flat["fmax_mhz"]:
        flat["period_ns"] = 1000.0 / float(flat["fmax_mhz"])

    return flat


def _is_f3_row(flat: dict) -> bool:
    """Return True if this row carries at least the F3 area metric."""
    return flat.get("area_um2") is not None and flat.get("status", "ok") in (
        "ok", "mock"
    )


# ── Surrogate class ────────────────────────────────────────────────────────────


class Surrogate:
    """Multi-fidelity surrogate predicting F3 physical metrics from (x, obs)."""

    METRICS = METRICS

    def __init__(self, seed: int = 0):
        self.seed = seed
        # Per-metric: either {q: GBT} or _FALLBACK
        self._models: dict[str, Any] = {m: _FALLBACK for m in METRICS}
        # Obs column means for missing-value imputation
        self._obs_means: dict[str, float] = {c: 0.0 for c in _OBS_COLS}
        # RTL hash → integer encoding for the context feature
        self._hash_map: dict[str, int] = {}
        # Discovered config-axis schema (audit F5): built at fit(), stored in the
        # joblib payload, and enforced at predict(). Shape:
        #   {"numeric": [axis, ...], "categorical": {axis: [cat, ...]}}
        self._schema: dict[str, Any] = {"numeric": [], "categorical": {}}
        # Per-numeric-axis training means, for imputing an absent CONTEXT axis
        # (required axes are enforced, not imputed).
        self._numeric_means: dict[str, float] = {}
        # Per-metric target statistics for the fallback regime
        self._target_stats: dict[str, tuple[float, float]] = {
            m: (0.0, 1.0) for m in METRICS
        }
        # Whether a real fit has been done
        self._fitted: bool = False
        self._n_rows: int = 0
        # Metadata (stored in the joblib file alongside the model)
        self.meta: dict = {}

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, rows: list[dict]) -> dict:
        """Fit the surrogate on F3 rows.

        Parameters
        ----------
        rows : list[dict]
            Each element may be flat, nested (results_physical.jsonl style),
            or funnel style.  Only rows with F3-level area_um2 are used as
            training targets; rows without area_um2 but with F2 obs are joined
            (same config key) to enrich the feature vector with proxy data.

        Returns
        -------
        dict
            Fit diagnostics: {"n_f3": int, "n_obs": int,
                              "cv_rho_<metric>": float, "cv_n_<metric>": int,
                              "fallback": bool}
        """
        flat_rows = [_flatten_row(r) for r in rows]

        # Split into F3 rows (have area_um2) and F2-only rows (have obs but no area)
        f3_rows = [r for r in flat_rows if _is_f3_row(r)]
        obs_index: dict[tuple, dict] = {}  # (lanes, acc_w, clk_ns) → obs dict
        for r in flat_rows:
            if not _is_f3_row(r):
                key = (
                    int(r.get("lanes", 0) or 0),
                    int(r.get("acc_w", 0) or 0),
                    float(r.get("clk_ns", 0.0) or 0.0),
                )
                if any(_obs_value(r, c) is not None for c in _OBS_COLS):
                    obs_index[key] = r

        n_f3 = len(f3_rows)
        self._n_rows = n_f3
        diag: dict = {"n_f3": n_f3, "n_obs": len(obs_index), "fallback": False}

        if n_f3 == 0:
            diag["fallback"] = True
            return diag

        # Build RTL hash encoding (context feature)
        all_hashes = list({str(r.get("rtl_hash", "unknown")) for r in f3_rows})
        self._hash_map = {h: i for i, h in enumerate(sorted(all_hashes))}

        # Discover the config-axis schema from the training configs (audit F5).
        self._build_schema([_config_view(r) for r in f3_rows])

        # Compute obs means for imputation (from F3 rows that carry obs columns)
        obs_cols_present = {c: [] for c in _OBS_COLS}
        for r in f3_rows:
            for c in _OBS_COLS:
                v = _obs_value(r, c)
                if v is not None:
                    obs_cols_present[c].append(float(v))
        for c in _OBS_COLS:
            vals = obs_cols_present[c]
            if vals:
                self._obs_means[c] = float(np.mean(vals))

        # Build feature matrix X, target vectors Y, and per-row group keys.
        # Group key = build identity (RTL axes + recipe + platform); used by
        # GroupKFold so identical/near-identical configs never span train/val.
        X_list, targets, group_keys = [], {m: [] for m in METRICS}, []
        for r in f3_rows:
            fv = self._build_feature_row(r, obs_index)
            X_list.append(fv)
            group_keys.append(self._group_key(r))
            for m in METRICS:
                targets[m].append(r.get(m))

        X = np.array(X_list, dtype=float)
        groups_all = np.array(group_keys)

        # Per-metric: compute target stats then fit quantile models
        for m in METRICS:
            y_raw = targets[m]
            y_valid_idx = [i for i, v in enumerate(y_raw) if v is not None]
            y = np.array([float(y_raw[i]) for i in y_valid_idx])
            X_m = X[y_valid_idx]
            groups_m = groups_all[y_valid_idx]
            n = len(y)
            diag[f"cv_n_{m}"] = n

            if n == 0:
                self._target_stats[m] = (0.0, 1.0)
                diag[f"cv_rho_{m}"] = float("nan")
                continue

            mu_y, std_y = float(np.mean(y)), float(np.std(y))
            self._target_stats[m] = (mu_y, max(std_y, 1e-6))

            if n < 10:
                # Too few rows: fall back to mean prediction
                self._models[m] = _FALLBACK
                diag[f"cv_rho_{m}"] = float("nan")
                continue

            # Fit three quantile-GBT models
            self._models[m] = {}
            for q in (0.16, 0.50, 0.84):
                gbt = GradientBoostingRegressor(
                    loss="quantile", alpha=q,
                    random_state=self.seed,
                    **_GBT_PARAMS,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gbt.fit(X_m, y)
                self._models[m][q] = gbt

            # Grouped 5-fold CV for Spearman rho (no config spans train/val)
            rho = self._cv_spearman(X_m, y, m, groups=groups_m, n_splits=5)
            diag[f"cv_rho_{m}"] = round(rho, 4)

        self._fitted = True
        self.meta.update({"n_f3": n_f3, "n_obs": len(obs_index),
                          "seed": self.seed})
        return diag

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        x: dict,
        obs: dict | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Predict (mu, sigma) for each metric given config x and optional F2 obs.

        Parameters
        ----------
        x : dict
            Config with keys: mac_lanes / lanes, accumulator_width / acc_w,
            clock_period_ns / clk_ns, abc_recipe / abc (opt), platform (opt),
            rtl_hash (opt).
        obs : dict | None
            Optional F2 observables (any subset of proxy_area_um2, proxy_wns_ns,
            ff_count, cell_count, logic_levels).

        Returns
        -------
        dict[str, tuple[float, float]]
            {metric: (mu, sigma)} for each metric in METRICS.
            If not fitted (too few rows), returns (training_mean, 2*std).
        """
        # Cross-design guard (audit F5): a fitted surrogate refuses to predict a
        # config that does not cover its fitted axis schema. Checked once, up
        # front, so callers get a clear error instead of a silent misprediction
        # (this is also what neutralizes the surrogate_n45 auto-load hazard —
        # its callers wrap predict in try/except and degrade to no-surrogate).
        if self._fitted and self._required_axes():
            self._check_coverage(_config_view(x))

        result = {}
        for m in METRICS:
            if self._models[m] is _FALLBACK or not self._fitted:
                mu, std = self._target_stats[m]
                result[m] = (float(mu), float(2.0 * std))
                continue

            fv = self._build_feature_row(x, obs_dict=obs)
            X_pred = np.array([fv], dtype=float)

            q16 = float(self._models[m][0.16].predict(X_pred)[0])
            q50 = float(self._models[m][0.50].predict(X_pred)[0])
            q84 = float(self._models[m][0.84].predict(X_pred)[0])

            mu = q50
            # sigma ≈ half-IQR of the 68% interval (like one-sigma for Gaussian)
            sigma = max((q84 - q16) / 2.0, 1e-6)
            result[m] = (mu, sigma)

        return result

    def predict_reward_stats(
        self,
        x: dict,
        obs: dict | None = None,
        reward_kind: str = "tinyvad",
        refs: dict | None = None,
    ) -> tuple[float, float]:
        """Return (mu, sigma) of the final composite reward for config x.

        The reward proxy mirrors physical_reward so the UCB acquisition signal
        ranks configs the same way the real objective scores them.  All constants
        are imported from the single sources of truth (constants.py /
        physical_reward.py) rather than re-hardcoded — previously max_speedup was
        576 here while the funnel's actual reward used 1024 from the YAML, so the
        UCB signal was miscalibrated against the objective it ranks (audit H2).

        reward_kind:
          "tinyvad" (default) — speedup/accuracy/area composite (TinyMAC).
          "generic"           — pure PPA proxy (higher Fmax, lower area/power),
                                 matching compute_generic_reward.  `refs` supplies
                                 the per-design anchors {area_ref_um2,
                                 fmax_ref_mhz, power_ref_mw}; falls back to the
                                 predicted values (self-normalised) when absent.
        """
        preds = self.predict(x, obs)
        mu_area, sig_area = preds["area_um2"]
        mu_period, sig_period = preds["period_ns"]
        mu_power, sig_power = preds["power_mw"]
        mu_period = max(mu_period, 0.5)   # negative period predictions are unphysical
        mu_fmax = 1000.0 / mu_period

        if reward_kind == "generic":
            r = refs or {}
            area_ref = float(r.get("area_ref_um2") or mu_area or 1.0)
            fmax_ref = float(r.get("fmax_ref_mhz") or mu_fmax or 1.0)
            power_ref = r.get("power_ref_mw")
            w_fmax, w_area, w_pwr = 1.0, -1.0, -0.4
            mu_reward = w_fmax * (mu_fmax / max(fmax_ref, 1e-9)) \
                + w_area * (mu_area / max(area_ref, 1e-9))
            # ∂fmax/∂period = -1000/period² → sigma on fmax
            sig_fmax = 1000.0 / (mu_period ** 2) * sig_period
            var = (w_fmax / max(fmax_ref, 1e-9) * sig_fmax) ** 2 \
                + (w_area / max(area_ref, 1e-9) * sig_area) ** 2
            if mu_power is not None and power_ref:
                mu_reward += w_pwr * (mu_power / max(float(power_ref), 1e-9))
                var += (w_pwr / max(float(power_ref), 1e-9) * sig_power) ** 2
            return (float(mu_reward), float(math.sqrt(var)))

        # ── TinyVAD composite (mirrors compute_physical_reward) ───────────────
        from eda_rl.common.constants import (
            SW_BASELINE_CYCLES, MAX_SPEEDUP_FULL, behavioral_cycles,
        )
        from eda_rl.gen1.reward import SW_BASELINE_CLOCK_NS
        from eda_rl.common.physical_reward import AREA_REF_UM2, POWER_REF_MW

        lanes = int(x.get("mac_lanes") or x.get("lanes") or 4)
        acc_w = int(x.get("accumulator_width") or x.get("acc_w") or 24)

        accuracy = 0.0 if acc_w <= 16 else 1.0   # matches acc_overflows (A16 overflows)
        w_acc = 2.0
        correctness = -50.0 * (1.0 - accuracy)

        SW_BASELINE_NS = SW_BASELINE_CYCLES * SW_BASELINE_CLOCK_NS
        cyc = behavioral_cycles(lanes)
        mu_latency_ns = cyc * mu_period
        mu_speedup = SW_BASELINE_NS / max(mu_latency_ns, 1.0)
        max_spd = float(MAX_SPEEDUP_FULL)   # 1024 — matches the funnel YAML reward
        norm_spd = math.log2(max(mu_speedup, 1e-3)) / math.log2(max_spd) if max_spd > 1 else 0.0
        norm_spd = max(-1.0, min(1.0, norm_spd))
        sig_speedup = (SW_BASELINE_NS / mu_latency_ns**2) * cyc * sig_period
        sig_norm_spd = (sig_speedup / (mu_speedup * math.log(max_spd))) if mu_speedup > 0 else 0.1

        mu_reward = (
            w_acc * accuracy
            + 3.0 * norm_spd
            + (-0.4) * (mu_area / AREA_REF_UM2)
            + (-0.4) * (mu_power / POWER_REF_MW)
            + correctness
        )
        sig_reward = math.sqrt(
            (3.0 * sig_norm_spd) ** 2
            + (0.4 / AREA_REF_UM2 * sig_area) ** 2
            + (0.4 / POWER_REF_MW * sig_power) ** 2
        )
        return (float(mu_reward), float(sig_reward))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """Save the fitted surrogate to a joblib file.

        The discovered config-axis schema (audit F5) is part of the payload —
        it is the contract that makes fit-time and predict-time feature rows
        agree, and what lets a loaded surrogate refuse configs from a
        different design's space.
        """
        payload = {
            "models": self._models,
            "obs_means": self._obs_means,
            "hash_map": self._hash_map,
            "target_stats": self._target_stats,
            "fitted": self._fitted,
            "n_rows": self._n_rows,
            "meta": self.meta,
            "seed": self.seed,
            "schema": self._schema,
            "numeric_means": self._numeric_means,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path) -> "Surrogate":
        """Load a previously saved surrogate from a joblib file.

        Raises ValueError for a fitted pre-schema (pre-audit-F5) payload: its
        models were trained on the old fixed 8-element feature layout, which
        the current feature builder no longer produces, so predictions would
        be silently wrong. Refit from the campaign corpus instead
        (eda-rl fit-surrogate / python -m eda_rl.gen2.fit_surrogate).
        """
        payload = joblib.load(path)
        s = cls(seed=payload.get("seed", 0))
        s._models = payload["models"]
        s._obs_means = payload["obs_means"]
        s._hash_map = payload["hash_map"]
        s._target_stats = payload["target_stats"]
        s._fitted = payload["fitted"]
        s._n_rows = payload["n_rows"]
        s.meta = payload.get("meta", {})
        if "schema" in payload:
            s._schema = payload["schema"]
            s._numeric_means = payload.get("numeric_means", {})
        elif s._fitted:
            raise ValueError(
                f"Surrogate.load({path}): fitted payload has no config-axis "
                "schema — it predates the audit-F5 featurization change and "
                "its models are incompatible with the current feature layout. "
                "Refit the surrogate from the campaign corpus "
                "(python -m eda_rl.gen2.fit_surrogate)."
            )
        return s

    # ── Internal helpers ──────────────────────────────────────────────────────

    # ── Schema construction / featurization (audit F5) ─────────────────────────

    def _build_schema(self, config_views: list[dict]) -> None:
        """Discover the config-axis schema from training config views.

        A key that is numeric in every row where it appears → numeric axis.
        A key that is ever non-numeric → categorical axis (one-hot), kept only
        if it has ≤ _MAX_ONEHOT distinct values (rtl_hash/platform are context
        categoricals and always kept). Numeric means are recorded for imputing
        an absent context axis at predict time.
        """
        from collections import defaultdict

        numeric_vals: dict[str, list[float]] = defaultdict(list)
        cat_vals: dict[str, set] = defaultdict(set)
        ever_nonnumeric: set[str] = set()
        for cv in config_views:
            for k, val in cv.items():
                if _is_number(val):
                    numeric_vals[k].append(float(val))
                else:
                    ever_nonnumeric.add(k)
                    cat_vals[k].add(str(val))

        numeric = sorted(k for k in numeric_vals if k not in ever_nonnumeric)
        categorical: dict[str, list[str]] = {}
        for k in sorted(cat_vals):
            if k in _CONTEXT_AXES or len(cat_vals[k]) <= _MAX_ONEHOT:
                categorical[k] = sorted(cat_vals[k])

        self._schema = {"numeric": numeric, "categorical": categorical}
        self._numeric_means = {
            k: float(np.mean(v)) for k, v in numeric_vals.items() if k in set(numeric)
        }

    def _required_axes(self) -> list[str]:
        """Axes a config MUST provide to be predictable (context axes excluded)."""
        req = list(self._schema.get("numeric", []))
        req += [k for k in self._schema.get("categorical", {}) if k not in _CONTEXT_AXES]
        return req

    def _featurize_config(self, cview: dict) -> list[float]:
        """Build the config feature block from a config view using the schema."""
        feats: list[float] = []
        for axis in self._schema.get("numeric", []):
            val = cview.get(axis)
            if val is None:
                val = self._numeric_means.get(axis, 0.0)   # context axis only
            x = float(val)
            if axis in _SPECIAL_LOG:
                x = math.log2(max(x, 1.0))
            feats.append(x)
        for axis in sorted(self._schema.get("categorical", {})):
            cats = self._schema["categorical"][axis]
            cur = str(cview.get(axis, ""))
            feats.extend(1.0 if cur == c else 0.0 for c in cats)
        return feats

    def _check_coverage(self, cview: dict) -> None:
        """Raise if the config does not cover the schema's required axes.

        This is the cross-design guard (audit F5): a surrogate fitted on one
        design's space refuses to predict for a config from a different space
        (e.g. a tinymac-fitted model asked about a gcd config lacking lanes, or
        a likith-fitted model asked about a config lacking IO_DELAY).
        """
        missing = [a for a in self._required_axes() if cview.get(a) is None]
        if missing:
            raise ValueError(
                "Surrogate.predict: incoming config does not cover the fitted "
                f"axis schema — missing required axes {missing}. This surrogate "
                f"was fit on axes {self._required_axes()}; predicting a config "
                "from a different design/space would be a silent misprediction. "
                "Refusing. Fit a surrogate on this design's corpus instead."
            )

    def _build_feature_row(
        self,
        row_or_x: dict,
        obs_index: dict | None = None,
        obs_dict: dict | None = None,
    ) -> list[float]:
        """Build the full feature vector for one row or (x, obs) pair.

        When called with a training row, `obs_index` is the {config_key: obs}
        lookup built during fit(); when called at prediction time, `obs_dict`
        is the optional F2 observable dict passed by the caller.
        """
        flat = row_or_x  # already flat at this point
        cview = _config_view(flat)

        # Config features from the discovered schema (audit F5).
        cfg_feats = self._featurize_config(cview)

        # For the obs_index training-time join we still key on (lanes, acc_w, clk).
        lanes = int(cview.get("lanes") or 4)
        acc_w = int(cview.get("acc_w") or 24)
        clk   = float(cview.get("clk_ns") or 5.0)

        # Resolve obs: priority = explicit obs_dict > obs from the row itself >
        # obs looked up from obs_index (training time join)
        obs: dict | None = None
        if obs_dict is not None:
            obs = obs_dict
        elif obs_index is not None:
            key = (lanes, acc_w, clk)
            obs = obs_index.get(key)
        # Also pick up obs columns that are present directly in the row
        # (honouring aliases so live area_um2/wns_ns feed proxy_* columns).
        row_obs: dict = {}
        for c in _OBS_COLS:
            v = _obs_value(flat, c)
            if v is not None:
                row_obs[c] = v
        if row_obs:
            if obs is None:
                obs = row_obs
            else:
                obs = {**row_obs, **obs}  # row_obs provides defaults, obs overrides

        obs_feats = _encode_obs(obs, self._obs_means)
        return cfg_feats + obs_feats

    @staticmethod
    def _group_key(flat: dict) -> str:
        """Build identity for grouped CV: RTL axes + recipe + platform.

        Rows sharing this key (e.g. the same config rebuilt, or matched builds
        differing only in floorplan knobs) are kept in the same CV fold so the
        reported Spearman ρ is a genuine generalization estimate (EXP-F3 fix).
        """
        lanes = int(flat.get("lanes") or flat.get("mac_lanes") or 0)
        acc   = int(flat.get("acc_w") or flat.get("accumulator_width") or 0)
        clk   = round(float(flat.get("clk_ns") or flat.get("clock_period_ns") or 0.0), 4)
        abc   = str(flat.get("abc") or flat.get("abc_recipe") or "")
        plat  = str(flat.get("platform") or "nangate45")
        return f"{lanes}|{acc}|{clk}|{abc}|{plat}"

    def _cv_spearman(
        self, X: np.ndarray, y: np.ndarray, metric: str,
        groups: np.ndarray | None = None, n_splits: int = 5,
    ) -> float:
        """Mean Spearman ρ via *grouped* K-fold cross-validation.

        Uses GroupKFold so that no build identity (see _group_key) appears in
        both train and validation — this removes the matched-build leakage that
        inflated the previous contiguous-slice estimate (EXP-F3). Uses only the
        q=0.50 (median) model for the rank correlation. Returns NaN when there
        are too few distinct groups to form ≥2 folds.
        """
        from scipy.stats import spearmanr
        from sklearn.model_selection import GroupKFold, KFold

        n = len(y)
        if groups is None:
            groups = np.arange(n)   # degrade to plain KFold-by-row
        n_groups = len(set(groups.tolist()))
        # Need at least 2 groups to cross-validate; cap splits by group count.
        if n_groups < 2:
            return float("nan")
        n_splits = max(2, min(n_splits, n_groups))

        if n_groups < n:
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X, y, groups)
        else:
            # All rows are unique groups → GroupKFold == KFold; shuffle for a
            # less order-dependent estimate.
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            split_iter = splitter.split(X)

        rhos = []
        for train_idx, val_idx in split_iter:
            if len(train_idx) < 3 or len(val_idx) < 2:
                continue
            gbt = GradientBoostingRegressor(
                loss="quantile", alpha=0.5,
                random_state=self.seed,
                **_GBT_PARAMS,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gbt.fit(X[train_idx], y[train_idx])
            y_pred = gbt.predict(X[val_idx])
            rho, _ = spearmanr(y[val_idx], y_pred)
            if not math.isnan(rho):
                rhos.append(rho)

        return float(np.mean(rhos)) if rhos else float("nan")


# ── Self-test / data mining entrypoint ────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Run the fit script if invoked directly
    script = Path(__file__).parent / "fit_surrogate.py"
    if script.exists():
        exec(compile(script.read_text(), str(script), "exec"))
    else:
        print("fit_surrogate.py not found — run it directly for data mining + CP3 validation")
        sys.exit(1)
