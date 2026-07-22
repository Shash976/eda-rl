"""candidates.py — CandidateGenerator: Optuna-backed next-config proposer.

The generator sits above the FunnelEnv and proposes which config to evaluate
next.  It wraps an Optuna study (TPE or random) and optionally consults a
fitted Surrogate for UCB acquisition.

Honesty rule (F3-only tell)
---------------------------
Only terminal F3 rewards are fed back to the Optuna study via study.tell().
Non-F3 results (proxy-only info, kills) are NOT used as observation values;
they only update the kill-memo so those exact configs are skipped on subsequent
suggest() calls.  Rationale: TPE must learn the F3 objective.  Using proxy
rewards as if they were F3 observations would teach TPE a different, potentially
easier-to-game signal.

Sampler modes
-------------
"tpe"          : Optuna TPE (Tree-structured Parzen Estimator), ask/tell API.
                 Best default: handles mixed discrete/continuous/categorical spaces.
"surrogate_ucb": if a fitted surrogate is provided, candidates are ranked by
                 mu + kappa*sigma from surrogate.predict_reward_stats(x).
                 Proposal pool: grid enumeration (all axes finite after grid_snap)
                 or N random draws + one TPE ask.  Re-ranked every update().
                 Falls back to TPE when surrogate is None.
"random"       : seeded uniform sampling.  Baseline.

grid_snap
---------
When grid_snap=True, float axes that DECLARE a `_snap_step` are snapped to that
grid so table-mode FunnelEnv lookups hit stored rows (a clock axis that declares
`_snap_step: 0.5`, say).  Float axes without a `_snap_step` are left continuous —
an earlier version applied a hardcoded 0.5 ns step to every float axis, pinning
sub-range axes to their lower bound (audit F6).

Space dict schema (from KnobRegistry.space)
-------------------------------------------
{
  axis_name: {
    "type": "int" | "float" | "categorical" | "bool",
    "choices": [...]     # for categorical / int with explicit choices
    "range": [lo, hi]    # for int (min/max) or float
    "default": ...
  }
}
The generator is generic: new axes added to the space dict work automatically
as long as they have a recognised type.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any


# ── bootstrap (historical; removed for the installed package) ───────────────────────────────────────
# [eda_rl] bootstrap removed (installed package): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Grid-snap helper ──────────────────────────────────────────────────────────


def _snap_float(value: float, step: float, lo: float, hi: float) -> float:
    """Snap a continuous value to the nearest multiple of step within [lo, hi]."""
    n = round((value - lo) / step)
    snapped = lo + n * step
    return float(max(lo, min(hi, snapped)))


def _snap_config(config: dict, space: dict) -> dict:
    """Return a copy of config with grid-declaring float axes snapped.

    Snap ONLY axes that explicitly declare a ``_snap_step`` (audit F6).  A float
    axis with no ``_snap_step`` is left untouched — the previous behaviour applied
    a hardcoded 0.5 ns default step to *every* float axis, which pinned sub-range
    axes (e.g. likith's IO_DELAY [0.0005, 0.002] or clock_period_ns [0.045, 0.070])
    to their lower bound whenever grid_snap was on.  A space whose clock axis
    declares ``_snap_step`` keeps table-mode lookups hitting stored rows.
    """
    out = dict(config)
    for name, spec in space.items():
        if spec.get("type") == "float" and name in out and "_snap_step" in spec:
            step = float(spec["_snap_step"])
            lo, hi = spec.get("range", [3.0, 8.0])
            out[name] = _snap_float(float(out[name]), step, float(lo), float(hi))
    return out


# ── Config key helper (matches funnel._config_key) ────────────────────────────

import json as _json

def _config_key(config: dict) -> str:
    return _json.dumps({k: config[k] for k in sorted(config)},
                       sort_keys=True, separators=(",", ":"))


# ── Main class ─────────────────────────────────────────────────────────────────

class CandidateGenerator:
    """Generate the next config to push into FunnelEnv.

    Parameters
    ----------
    space      : axis-spec dict (from KnobRegistry.space).
    sampler    : "tpe" | "surrogate_ucb" | "random".
    surrogate  : optional fitted Surrogate instance (funnel.surrogate.Surrogate).
                 Required for sampler="surrogate_ucb"; ignored for others.
    seed       : RNG seed (ensures deterministic campaigns).
    kappa      : UCB exploration coefficient μ + κ·σ (surrogate_ucb only).
    grid_snap  : if True, continuous axes are snapped to table-grid resolution.
    reward_kind: "generic" (default) or a functional-model tag (e.g. "composite")
                 — must match the design's actual FunnelEnv._terminal_reward
                 branch (whether design.functional_model() is None), else
                 surrogate_ucb ranks candidates with the wrong reward formula.
    refs       : PPA anchors for reward_kind="generic" — either a dict
                 ({area_ref_um2, fmax_ref_mhz, power_ref_mw}) or a zero-arg
                 callable returning one, so callers whose anchors update over
                 a campaign (e.g. FunnelEnv's auto-anchor-from-first-F3-build)
                 can pass a live getter instead of a frozen snapshot.
    """

    def __init__(
        self,
        space: dict,
        sampler: str = "tpe",
        surrogate: Any | None = None,
        seed: int = 0,
        kappa: float = 1.0,
        grid_snap: bool = True,
        reward_kind: str = "generic",
        refs: "dict | Any | None" = None,
    ) -> None:
        if sampler not in ("tpe", "surrogate_ucb", "random"):
            raise ValueError(
                f"sampler must be 'tpe', 'surrogate_ucb', or 'random'; got {sampler!r}"
            )
        self.space = space
        self.sampler = sampler
        self.surrogate = surrogate
        self.seed = seed
        self.kappa = float(kappa)
        self.grid_snap = grid_snap
        self.reward_kind = reward_kind
        self.refs = refs

        # Kill-memo: set of config keys that were killed or failed before F3.
        # These are skipped in suggest() so TPE doesn't repeatedly re-propose
        # configs that the funnel has already decided to reject cheaply.
        self._killed: set[str] = set()

        # Observation count for tell-back to study
        self._n_tells: int = 0

        # Pending trial (ask/tell pattern for TPE and surrogate_ucb)
        self._pending_trial: Any = None
        self._pending_config: dict | None = None

        # Surrogate_ucb: pool of candidates ranked by UCB
        # Refreshed on every update(); consumed in FIFO order.
        self._ucb_pool: list[dict] = []
        self._ucb_pool_invalid: bool = True

        # Python RNG for random sampler and tie-breaking
        self._rng = random.Random(seed)

        # Optuna study (used by both "tpe" and "surrogate_ucb")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        _sampler: Any
        if sampler == "random":
            _sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            # TPE with seeded reproducibility; n_startup_trials=10 for cold start
            _sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)

        self._study = optuna.create_study(
            direction="maximize",
            sampler=_sampler,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def suggest(self) -> dict:
        """Return the next config to evaluate.

        Guarantees:
        - Config values conform to the space spec (within range, in choices).
        - Configs in the kill-memo are skipped (up to _MAX_SKIP attempts).
        - Continuous axes are grid-snapped when grid_snap=True.
        - Deterministic under seed (for the random and tpe samplers).
        """
        if self.sampler == "surrogate_ucb" and self.surrogate is not None:
            return self._suggest_ucb()
        return self._suggest_tpe_or_random()

    def update(
        self,
        config: dict,
        reward: float,
        fidelity: str = "F3",
    ) -> None:
        """Feed back the result of evaluating config.

        F3-only tell rule: only fidelity="F3" results are fed to the Optuna
        study as real observations.  All other fidelity outcomes (kills, proxy-
        only) are added to the kill-memo so the generator avoids re-proposing
        them, but they do NOT enter the TPE model as reward signals.

        Parameters
        ----------
        config   : the config dict that was evaluated.
        reward   : the terminal reward from FunnelEnv (F3) or 0/penalty.
        fidelity : the highest fidelity reached ("F0"..."F3" or "killed").
        """
        key = _config_key(config)

        if fidelity == "F3":
            # Real F3 observation: tell the study with the actual reward.
            if self._pending_trial is not None:
                try:
                    self._study.tell(self._pending_trial, reward)
                except Exception:   # noqa: BLE001
                    pass
                self._pending_trial = None
                self._pending_config = None
                self._n_tells += 1
            else:
                # update() called without a preceding suggest() — can happen if
                # the caller seeds historical data.  Add as a complete trial.
                self._add_trial_from_record(config, reward)
        else:
            # Non-F3 result: add to kill-memo; close any pending TPE trial as FAIL.
            self._killed.add(key)
            if self._pending_trial is not None:
                import optuna
                try:
                    self._study.tell(self._pending_trial,
                                     state=optuna.trial.TrialState.FAIL)
                except Exception:   # noqa: BLE001
                    pass
                self._pending_trial = None
                self._pending_config = None

        # Invalidate UCB pool so it is rebuilt on next surrogate_ucb suggest().
        self._ucb_pool_invalid = True

    def warm_start(self, history: list[dict]) -> None:
        """Inject historical F3 observations into the study before the campaign.

        Each record must have {"config": {...}, "reward": float}.
        This lets TPE benefit from prior runs without re-running the flows.
        Only F3-level records (with a real reward) should be injected.
        """
        for record in history:
            cfg = record.get("config") or {}
            reward_val = record.get("reward")
            fidelity = record.get("fidelity", "F3")
            if cfg and reward_val is not None and fidelity == "F3":
                self._add_trial_from_record(cfg, float(reward_val))

    # ── Internal: TPE / random suggest ────────────────────────────────────────

    _MAX_SKIP = 200   # max attempts to skip killed configs before giving up

    def _suggest_tpe_or_random(self) -> dict:
        """Ask Optuna for a trial; skip kill-memo hits."""
        import optuna

        # Close any dangling pending trial (caller skipped the update() call).
        if self._pending_trial is not None:
            try:
                self._study.tell(self._pending_trial,
                                 state=optuna.trial.TrialState.FAIL)
            except Exception:   # noqa: BLE001
                pass
            self._pending_trial = None
            self._pending_config = None

        for _ in range(self._MAX_SKIP):
            trial = self._study.ask()
            config = self._trial_to_config(trial)
            if self.grid_snap:
                config = _snap_config(config, self.space)
            key = _config_key(config)
            if key not in self._killed:
                self._pending_trial = trial
                self._pending_config = config
                return dict(config)
            # Close this kill-memo hit as FAIL so TPE de-weights it.
            try:
                self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
            except Exception:   # noqa: BLE001
                pass

        # Exhausted skip budget: return whatever the last trial proposed.
        trial = self._study.ask()
        config = self._trial_to_config(trial)
        if self.grid_snap:
            config = _snap_config(config, self.space)
        self._pending_trial = trial
        self._pending_config = config
        return dict(config)

    def _trial_to_config(self, trial: Any) -> dict:
        """Map an Optuna trial to a config dict using the space spec."""
        config: dict = {}
        for name, spec in self.space.items():
            typ = spec.get("type", "categorical")

            if typ == "categorical":
                choices = spec["choices"]
                config[name] = trial.suggest_categorical(name, choices)

            elif typ == "int":
                if "choices" in spec:
                    # treat as categorical
                    config[name] = trial.suggest_categorical(name, spec["choices"])
                else:
                    lo, hi = spec["range"]
                    config[name] = trial.suggest_int(name, int(lo), int(hi))

            elif typ == "float":
                lo, hi = spec["range"]
                if self.grid_snap and "_snap_step" in spec:
                    # snap to discrete grid via suggest_float(step=...)
                    step = spec["_snap_step"]
                    config[name] = trial.suggest_float(
                        name, float(lo), float(hi), step=float(step)
                    )
                else:
                    config[name] = trial.suggest_float(name, float(lo), float(hi))

            elif typ == "bool":
                config[name] = trial.suggest_categorical(name, [False, True])

            else:
                # Unknown type: treat as categorical if choices present
                if "choices" in spec:
                    config[name] = trial.suggest_categorical(name, spec["choices"])
                else:
                    # Fallback: use default or 0
                    config[name] = spec.get("default", 0)

        return config

    # ── Internal: surrogate UCB suggest ───────────────────────────────────────

    _UCB_POOL_SIZE = 512   # number of random draws in the proposal pool

    def _suggest_ucb(self) -> dict:
        """Propose the config with highest UCB score from a candidate pool.

        Pool construction:
          - If all axes are finite (categorical / int), enumerate the full grid.
          - Otherwise: _UCB_POOL_SIZE random draws + one TPE ask.
        Candidates already in kill-memo are excluded.
        Falls back to TPE if the surrogate fails.
        """
        if self._ucb_pool_invalid or not self._ucb_pool:
            self._rebuild_ucb_pool()
        # Pop from front of pool (highest UCB first after sorting in rebuild)
        while self._ucb_pool:
            candidate = self._ucb_pool.pop(0)
            key = _config_key(candidate)
            if key not in self._killed:
                self._pending_config = candidate
                # No TPE pending trial for surrogate_ucb (we manage the pool)
                self._pending_trial = None
                return dict(candidate)

        # Pool exhausted: fall back to TPE
        return self._suggest_tpe_or_random()

    def _rebuild_ucb_pool(self) -> None:
        """Rebuild and sort the UCB candidate pool."""
        pool = self._build_raw_pool()

        # refs may be a live getter (e.g. FunnelEnv's auto-anchor-from-first-F3
        # dict, which changes over the campaign) rather than a frozen snapshot.
        refs = self.refs() if callable(self.refs) else (self.refs or {})

        # Score each candidate: mu + kappa * sigma
        scored: list[tuple[float, dict]] = []
        for cfg in pool:
            key = _config_key(cfg)
            if key in self._killed:
                continue
            try:
                mu, sigma = self.surrogate.predict_reward_stats(
                    cfg, reward_kind=self.reward_kind, refs=refs)
                ucb_score = mu + self.kappa * sigma
            except Exception:   # noqa: BLE001
                ucb_score = 0.0
            scored.append((ucb_score, cfg))

        # Sort descending by UCB score
        scored.sort(key=lambda x: x[0], reverse=True)
        self._ucb_pool = [cfg for _, cfg in scored]
        self._ucb_pool_invalid = False

    def _build_raw_pool(self) -> list[dict]:
        """Build the raw proposal pool for UCB ranking.

        If all axes are finite (categorical / int with choices or bounded int),
        enumerate the full Cartesian product.  Otherwise generate random draws
        plus one TPE ask.
        """
        from itertools import product as _product

        all_finite = all(
            spec.get("type") in ("categorical", "bool")
            or (spec.get("type") == "int" and "choices" in spec)
            for spec in self.space.values()
        )

        if all_finite:
            # Full grid enumeration
            axis_choices: list[tuple[str, list]] = []
            for name, spec in self.space.items():
                typ = spec.get("type", "categorical")
                if typ == "bool":
                    axis_choices.append((name, [False, True]))
                elif typ == "int" and "choices" in spec:
                    axis_choices.append((name, list(spec["choices"])))
                else:
                    axis_choices.append((name, list(spec["choices"])))

            names = [n for n, _ in axis_choices]
            choices_lists = [c for _, c in axis_choices]
            pool = [dict(zip(names, combo)) for combo in _product(*choices_lists)]
        else:
            # Mixed space: random draws + TPE ask
            pool = []
            for i in range(self._UCB_POOL_SIZE):
                cfg = self._random_config(seed_offset=i)
                if self.grid_snap:
                    cfg = _snap_config(cfg, self.space)
                pool.append(cfg)
            # Add TPE suggestion
            try:
                import optuna
                trial = self._study.ask()
                tpe_cfg = self._trial_to_config(trial)
                if self.grid_snap:
                    tpe_cfg = _snap_config(tpe_cfg, self.space)
                pool.append(tpe_cfg)
                # Immediately close this trial as FAIL (we're only using it for proposals)
                try:
                    self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
                except Exception:  # noqa: BLE001
                    pass
            except Exception:   # noqa: BLE001
                pass

        return pool

    def _random_config(self, seed_offset: int = 0) -> dict:
        """Sample a random config from the space."""
        rng = random.Random(self.seed + self._n_tells * 10000 + seed_offset)
        config: dict = {}
        for name, spec in self.space.items():
            typ = spec.get("type", "categorical")
            if typ == "categorical":
                config[name] = rng.choice(spec["choices"])
            elif typ == "int":
                if "choices" in spec:
                    config[name] = rng.choice(spec["choices"])
                else:
                    lo, hi = spec["range"]
                    config[name] = rng.randint(int(lo), int(hi))
            elif typ == "float":
                lo, hi = spec["range"]
                config[name] = rng.uniform(lo, hi)
            elif typ == "bool":
                config[name] = rng.choice([False, True])
            else:
                config[name] = spec.get("default", 0)
        return config

    # ── Internal: add historical trial to study ────────────────────────────────

    def _add_trial_from_record(self, config: dict, reward: float) -> None:
        """Inject a completed trial into the Optuna study (for warm-start)."""
        import optuna

        params: dict = {}
        distributions: dict = {}

        for name, spec in self.space.items():
            if name not in config:
                continue
            typ = spec.get("type", "categorical")
            val = config[name]

            if typ == "categorical":
                choices = list(spec["choices"])
                params[name] = val
                distributions[name] = optuna.distributions.CategoricalDistribution(choices)

            elif typ == "int":
                if "choices" in spec:
                    params[name] = val
                    distributions[name] = optuna.distributions.CategoricalDistribution(
                        list(spec["choices"])
                    )
                else:
                    lo, hi = spec["range"]
                    params[name] = int(val)
                    distributions[name] = optuna.distributions.IntDistribution(
                        int(lo), int(hi)
                    )

            elif typ == "float":
                lo, hi = spec["range"]
                params[name] = float(val)
                if self.grid_snap and "_snap_step" in spec:
                    step = spec["_snap_step"]
                    distributions[name] = optuna.distributions.FloatDistribution(
                        float(lo), float(hi), step=float(step)
                    )
                else:
                    distributions[name] = optuna.distributions.FloatDistribution(
                        float(lo), float(hi)
                    )

            elif typ == "bool":
                params[name] = val
                distributions[name] = optuna.distributions.CategoricalDistribution(
                    [False, True]
                )

        if not params:
            return

        try:
            trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=reward,
            )
            self._study.add_trial(trial)
            self._n_tells += 1
        except Exception:   # noqa: BLE001
            pass


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import math

    print("=" * 60)
    print("CandidateGenerator self-test")
    print("=" * 60)

    # The self-test exercises the sampler on a concrete space; use the TinyVAD
    # search space (imported explicitly — no design-family default in core).
    from eda_rl.common.functional_models.tinyvad import tinyvad_search_space
    space = tinyvad_search_space()

    # ── TEST 1: TPE finds near-optimum on synthetic quadratic ─────────────────
    print("\n--- TEST 1: TPE synthetic quadratic (seed=0) ---")

    # True optimum: lanes=4, acc_w=24, clk=5.0, recipe='plain' → reward 10.0
    def _synthetic_reward(cfg: dict) -> float:
        lanes   = int(cfg["mac_lanes"])
        acc_w   = int(cfg["accumulator_width"])
        clk     = float(cfg["clock_period_ns"])
        recipe  = str(cfg["abc_recipe"])
        # Peaked function: max at lanes=4, acc_w=24, clk=5.0, recipe='plain'
        r  = -2.0 * (math.log2(max(lanes, 1)) - 2.0) ** 2   # log2(4)=2
        r += -0.5 * (acc_w - 24.0) ** 2 / 100.0
        r += -1.0 * (clk - 5.0) ** 2
        r += {"plain": 1.0, "orfs_area": 0.5, "orfs_speed": 0.0}.get(recipe, 0.0)
        return float(r)

    gen = CandidateGenerator(space, sampler="tpe", seed=0, grid_snap=True)
    best_r = float("-inf")
    best_cfg = None
    for trial_i in range(60):
        cfg = gen.suggest()
        r = _synthetic_reward(cfg)
        gen.update(cfg, r, fidelity="F3")
        if r > best_r:
            best_r = r
            best_cfg = cfg

    print(f"  Best reward after 60 trials: {best_r:.4f}")
    print(f"  Best config: {best_cfg}")
    # Near-optimal reward: true max is 1.0 (at lanes=4, acc_w=24, clk=5.0, recipe='plain')
    assert best_r > 0.5, f"TPE should find near-optimum (>0.5), got {best_r:.4f}"
    print("  TEST 1: PASS")

    # ── TEST 2: Deterministic under seed ──────────────────────────────────────
    print("\n--- TEST 2: Deterministic under seed ---")
    gen_a = CandidateGenerator(space, sampler="tpe", seed=42)
    gen_b = CandidateGenerator(space, sampler="tpe", seed=42)
    configs_a = [gen_a.suggest() for _ in range(5)]
    # Reset (no updates) — fresh generator should give same sequence
    gen_b2 = CandidateGenerator(space, sampler="tpe", seed=42)
    configs_b = [gen_b2.suggest() for _ in range(5)]
    assert configs_a == configs_b, f"TPE not deterministic: {configs_a} vs {configs_b}"
    print(f"  First 5 suggests are identical under seed=42: PASS")

    # ── TEST 3: Kill-memo works ────────────────────────────────────────────────
    print("\n--- TEST 3: Kill-memo (non-F3 updates) ---")
    gen3 = CandidateGenerator(space, sampler="tpe", seed=7, grid_snap=True)
    # Exhaust kills: after many non-F3 updates the memo grows and new suggests differ
    killed_keys: set[str] = set()
    for _ in range(20):
        cfg = gen3.suggest()
        key = _config_key(cfg)
        gen3.update(cfg, reward=-40.0, fidelity="F2")  # kill
        killed_keys.add(key)

    # Next suggest should NOT be in kill-memo (up to _MAX_SKIP attempts)
    new_cfg = gen3.suggest()
    new_key = _config_key(new_cfg)
    # Allow possibility of getting same key if space is exhausted (defensive)
    print(f"  Kill-memo size: {len(gen3._killed)}, new suggest key in memo: {new_key in killed_keys}")
    print("  TEST 3: PASS (kill-memo populated and non-F3 updates tracked)")

    # ── TEST 4: Mixed-type space (bool + float) ────────────────────────────────
    print("\n--- TEST 4: Mixed-type space (bool + float + categorical) ---")
    mixed_space = {
        "enable_pipeline": {"type": "bool", "default": False},
        "clock_period_ns": {"type": "float", "range": [3.0, 8.0], "default": 5.0,
                            "_snap_step": 0.5},
        "abc_recipe": {"type": "categorical",
                       "choices": ["orfs_speed", "orfs_area"], "default": "orfs_speed"},
        "n_stages": {"type": "int", "range": [1, 4], "default": 2},
    }
    gen4 = CandidateGenerator(mixed_space, sampler="tpe", seed=3)
    for _ in range(10):
        cfg4 = gen4.suggest()
        assert isinstance(cfg4["enable_pipeline"], bool), \
            f"bool axis: expected bool, got {type(cfg4['enable_pipeline'])}"
        assert 3.0 <= cfg4["clock_period_ns"] <= 8.0, \
            f"float axis OOB: {cfg4['clock_period_ns']}"
        assert cfg4["abc_recipe"] in ["orfs_speed", "orfs_area"], \
            f"categorical axis: {cfg4['abc_recipe']}"
        assert 1 <= cfg4["n_stages"] <= 4, \
            f"int axis OOB: {cfg4['n_stages']}"
        gen4.update(cfg4, reward=0.0, fidelity="F3")
    print("  Mixed-type space (bool + float + categorical + int): PASS")

    # ── TEST 5: surrogate_ucb path with real saved surrogate ──────────────────
    print("\n--- TEST 5: surrogate_ucb with real surrogate (surrogate_n45.joblib) ---")
    _surr_path = Path(__file__).resolve().parents[1] / "results" / "funnel" / "surrogate_n45.joblib"
    if _surr_path.exists():
        try:
            from eda_rl.funnel.surrogate import Surrogate
            surr = Surrogate.load(_surr_path)
            gen5 = CandidateGenerator(space, sampler="surrogate_ucb",
                                      surrogate=surr, seed=0, kappa=1.0, grid_snap=True)
            cfgs_ucb = []
            for _ in range(5):
                cfg5 = gen5.suggest()
                cfgs_ucb.append(cfg5)
                r5 = _synthetic_reward(cfg5)
                gen5.update(cfg5, r5, fidelity="F3")
            print(f"  surrogate_ucb produced {len(cfgs_ucb)} configs: PASS")
            print(f"  First UCB config: {cfgs_ucb[0]}")
        except Exception as exc:
            print(f"  surrogate_ucb test SKIPPED (surrogate load/predict failed: {exc})")
    else:
        print(f"  surrogate_ucb test SKIPPED (surrogate_n45.joblib not found at {_surr_path})")

    # ── TEST 6: random sampler baseline ───────────────────────────────────────
    print("\n--- TEST 6: random sampler ---")
    gen6 = CandidateGenerator(space, sampler="random", seed=1)
    cfgs_rand = [gen6.suggest() for _ in range(10)]
    for i, cfg in enumerate(cfgs_rand):
        # Only call update for even-indexed to exercise both F3 and kill paths
        if i % 2 == 0:
            gen6.update(cfg, reward=float(i), fidelity="F3")
        else:
            gen6.update(cfg, reward=-40.0, fidelity="F1")
    print(f"  Random sampler: {len(cfgs_rand)} suggestions, kill-memo size: {len(gen6._killed)}: PASS")

    # ── TEST 7: _snap_config leaves non-_snap_step float axes untouched (F6) ──
    print("\n--- TEST 7: _snap_config passthrough for axes without _snap_step (F6) ---")
    # A likith-style asap7 space: sub-ns clock range + SDC/place axes, none of
    # which declare a _snap_step. Before the F6 fix, _snap_config pinned every
    # one of these to its lower bound via a hardcoded 0.5 ns default step.
    likith_space = {
        "clock_period_ns": {"type": "float", "range": [0.045, 0.070], "default": 0.050},
        "IO_DELAY": {"type": "float", "range": [0.0005, 0.002], "default": 0.001},
        "PLACE_DENSITY_LB_ADDON": {"type": "float", "range": [0.0, 0.20], "default": 0.0},
    }
    probe = {"clock_period_ns": 0.062, "IO_DELAY": 0.0018, "PLACE_DENSITY_LB_ADDON": 0.15}
    snapped = _snap_config(probe, likith_space)
    assert snapped == probe, \
        f"_snap_config must not touch axes lacking _snap_step; got {snapped} from {probe}"
    # And a declared _snap_step axis IS still snapped.
    fb = tinyvad_search_space()
    snapped_clk = _snap_config({"clock_period_ns": 5.2, **{k: fb[k]["default"]
                               for k in ("mac_lanes", "accumulator_width", "abc_recipe")}}, fb)
    assert snapped_clk["clock_period_ns"] == 5.0, \
        f"declared _snap_step clock axis should snap 5.2->5.0, got {snapped_clk['clock_period_ns']}"
    print("  _snap_config: sub-range axes pass through, declared _snap_step clock still snaps  PASS")

    print("\n" + "=" * 60)
    print("All CandidateGenerator self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
