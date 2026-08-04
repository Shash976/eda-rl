#!/usr/bin/env python3
"""run_funnel_optimizer.py — live campaign driver for the funnel optimizer.

Gen2 counterpart of gen1/run_physical_optimizer.py.  Drives a loop of:
  1. CandidateGenerator.suggest() → next config
  2. FunnelEnv.reset(config)      → run F0, get initial state
  3. PromotionAgent.act(state) → step(action) → repeat until done=True
  4. CandidateGenerator.update(config, terminal_reward, fidelity)
  5. PromotionAgent.update per step (online LinUCB update)

Logs one JSONL line per episode to eda_rl/campaigns/<design>/<platform>/results_funnel_campaigns.jsonl.
Prints a running incumbent line and a summary on exit.

CLI
---
  python3 eda-rl optimize \\
      --design tinymac_accel --platform nangate45 \\
      --budget-hours 4 \\
      --max-tier 1 \\
      --sampler tpe|surrogate_ucb|random \\
      --promotion fixed|linucb|random \\
      --seed 0 \\
      --table eda_rl/results/funnel/results_funnel.jsonl  # omit for live mode
      --surrogate eda_rl/results/funnel/surrogate_n45.joblib   # default: auto-detect

Table mode (--table given): replays logged observations, charges recorded cost
against a simulated wall-clock budget.  PHYSICAL_MOCK=1 activates mock metrics
for live mode (no real ORFS calls).

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ── bootstrap (historical; package is installed) ───────────────────────────────
# [eda_rl] bootstrap removed (installed package): sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

# Force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except Exception:
    pass

# ── imports (all defensive with clear error messages) ─────────────────────────

try:
    from eda_rl.funnel.env import CampaignState, FunnelEnv, load_table
    _FUNNEL_OK = True
except Exception as _e:
    _FUNNEL_OK = False
    _FUNNEL_ERR = str(_e)

# Reward semantics version — stamped on every log row (audit F18/F19).  Imported
# defensively so a broken reward module can't take down the whole driver.
try:
    from eda_rl.common.physical_reward import REWARD_VERSION
except Exception:   # noqa: BLE001
    REWARD_VERSION = None

from eda_rl.funnel.budget import Budget

try:
    from eda_rl.funnel.candidates import CandidateGenerator
    _CAND_OK = True
except Exception as _e:
    _CAND_OK = False
    _CAND_ERR = str(_e)

try:
    from eda_rl.funnel.promotion_agent import (
        FixedGateAgent,
        PromotionAgent,
        RandomPromotionAgent,
    )
    _PROMO_OK = True
except Exception as _e:
    _PROMO_OK = False
    _PROMO_ERR = str(_e)


# ── constants ─────────────────────────────────────────────────────────────────

_OPT_ROOT = Path(__file__).resolve().parents[1]
_CAMPAIGNS_ROOT = _OPT_ROOT / "campaigns"
_DEFAULT_SURROGATE = _OPT_ROOT / "results" / "funnel" / "surrogate_n45.joblib"
_DEFAULT_SPACE_YAML = Path(__file__).resolve().parent / "search_space_funnel.yaml"

# Fidelity labels in promotion order
_FIDELITY_ORDER = ["F0", "F1", "F2", "F3"]


# ── helper: load surrogate ────────────────────────────────────────────────────

def _load_surrogate(path: str | Path | None) -> Any | None:
    """Try to load a Surrogate from path; return None on failure."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        from eda_rl.funnel.surrogate import Surrogate
        s = Surrogate.load(p)
        return s
    except Exception as exc:   # noqa: BLE001
        print(f"  [WARNING] could not load surrogate from {p}: {exc}")
        return None


def _surrogate_covers(surrogate: Any, space: dict) -> bool:
    """True if the fitted surrogate's config-axis schema covers this space.

    Probes with one representative config drawn from the space (defaults /
    first choices / range lows). An unfitted surrogate passes — it predicts
    nothing either way and stays useful as a shaping no-op.
    """
    if not getattr(surrogate, "_fitted", False):
        return True
    cfg: dict = {}
    for axis, spec in space.items():
        if "default" in spec:
            cfg[axis] = spec["default"]
        elif spec.get("choices"):
            cfg[axis] = spec["choices"][0]
        elif spec.get("range"):
            cfg[axis] = spec["range"][0]
    try:
        surrogate.predict_reward_stats(cfg, None, reward_kind="generic", refs={})
        return True
    except Exception:  # noqa: BLE001 — schema-coverage refusal (or any predict failure)
        return False


# ── helper: build space dict ──────────────────────────────────────────────────

def _build_space(
    design: str | None,
    platform: str,
    max_tier: int,
    space_yaml: "str | Path | None" = None,
) -> dict:
    """Return the design's knob space from KnobRegistry+DesignSpec.

    KnobRegistry.space() normalizes str designs via DesignSpec.load() and emits
    design.params axes under their canonical names (mac_lanes, accumulator_width
    for tinymac; empty for designs like gcd that have no RTL params).

    Knob fixing/exclusion/overrides are design-authoritative: they come from the
    design YAML's ``knobs:`` block (applied inside ``KnobRegistry.space()``), so a
    design is fully described by its own YAML and the user never edits
    search_space_funnel.yaml.  A design with no ``knobs:`` block optimizes every
    knob up to ``--max-tier``.  ``space_yaml`` is retained for signature
    compatibility but no longer governs the live knob space.

    Raises RuntimeError if the space can't be resolved — there is no silent
    design-shaped fallback (that used to substitute a tinymac space for any
    design whose registry lookup failed).
    """
    try:
        from eda_rl.common.knobs import KnobRegistry

        reg = KnobRegistry.load()
        # reg.space() accepts str and normalizes via DesignSpec.load() internally,
        # and applies the design's knobs.fix/exclude/override block.
        sp = reg.space(max_tier=max_tier, design=design, platform=platform)
        if sp:
            return sp
        raise RuntimeError("KnobRegistry.space() returned an empty space")
    except Exception as exc:
        raise RuntimeError(
            f"could not resolve the knob space for design={design!r} "
            f"platform={platform!r}: {exc}"
        ) from exc


_SAFE_SLUG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _design_slug(design: str | None) -> str:
    """Canonical campaign-directory name for a --design argument.

    `--design` takes either a bare name (`sagar`) or a YAML path
    (`eda_rl/designs/sagar.yaml`); both must map to the SAME campaign directory.
    Before this, the raw string became the path, so passing a path produced
    `campaigns/eda_rl/designs/sagar.yaml/<platform>/` — four such ghost trees are
    committed in this repo (see campaigns/eda_rl/README.md).

    The slug is the file STEM, deliberately not `DesignSpec.name`: sagar.yaml
    declares name "alu4b" and likith.yaml declares "id", but their campaign
    directories are `sagar/` and `likith/`. Switching to `.name` would relocate
    every future log away from its existing history.
    """
    raw = (design or "").strip()
    if not raw:
        raise ValueError("--design is required; got an empty value")
    # Path(...).name strips every separator, so a traversal attempt collapses to
    # a plain filename and cannot escape campaigns/ — and a design that does not
    # actually exist fails earlier, in _build_space's DesignSpec.load().
    stem = Path(raw).name
    for suffix in (".yaml", ".yml"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # The slug becomes a filesystem path; refuse anything that could escape it.
    if not stem or not _SAFE_SLUG_RE.match(stem):
        raise ValueError(
            f"--design {design!r} does not yield a safe campaign directory name "
            f"(got {stem!r}); expected a bare design name or a path to its YAML."
        )
    return stem


# ── helper: build promotion agent ─────────────────────────────────────────────

def _make_promotion_agent(name: str, seed: int) -> Any:
    """Construct promotion agent by name."""
    if not _PROMO_OK:
        raise RuntimeError(f"promotion_agent.py not available: {_PROMO_ERR}")
    if name == "fixed":
        return FixedGateAgent(seed=seed)
    elif name == "linucb":
        return PromotionAgent(seed=seed)
    elif name == "random":
        return RandomPromotionAgent(seed=seed)
    else:
        raise ValueError(f"Unknown promotion agent: {name!r}; use fixed|linucb|random")


# ── campaign loop ─────────────────────────────────────────────────────────────

def run_campaign(
    *,
    design: str | None,
    platform: str,
    budget: "Budget",
    max_tier: int,
    sampler: str,
    promotion: str,
    seed: int,
    table_path: str | Path | None,
    surrogate_path: str | Path | None,
    results_path: str | Path,
    space_yaml: str | Path,
    verbose: bool = True,
    jobs: int = 1,
) -> dict:
    """Run one funnel optimizer campaign.

    Returns a summary dict: {best_config, best_reward, n_episodes, elapsed_s, ...}
    """
    if not _FUNNEL_OK:
        raise RuntimeError(f"FunnelEnv not available: {_FUNNEL_ERR}")
    if not _CAND_OK:
        raise RuntimeError(f"CandidateGenerator not available: {_CAND_ERR}")

    t0 = time.time()
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_id = f"campaign_{seed}_{int(t0)}"

    # ── build space ────────────────────────────────────────────────────────────
    space = _build_space(design, platform, max_tier, space_yaml=space_yaml)
    if verbose:
        print(f"  Space: {len(space)} axes: {list(space.keys())}")

    # ── load surrogate ─────────────────────────────────────────────────────────
    # Auto-detect default surrogate if not specified
    if surrogate_path is None and _DEFAULT_SURROGATE.exists():
        surrogate_path = _DEFAULT_SURROGATE
    surrogate = _load_surrogate(surrogate_path)
    if surrogate is not None and not _surrogate_covers(surrogate, space):
        # A fitted surrogate whose config-axis schema doesn't cover this
        # design's space would raise on every predict — the coverage guard
        # catches each call, so it silently contributes 0.0 everywhere and
        # surrogate_ucb degenerates to arbitrary ordering. Drop it loudly
        # instead of announcing "Surrogate loaded".
        if verbose:
            print(f"  [WARNING] surrogate at {surrogate_path} does not cover this "
                  f"design's config axes — ignoring it. Refit with: eda-rl "
                  f"fit-surrogate on campaigns of this design.")
        surrogate = None
    if verbose and surrogate is not None:
        print(f"  Surrogate loaded from {surrogate_path} "
              f"(fitted={surrogate._fitted}, n_rows={surrogate._n_rows})")
    elif verbose:
        print(f"  No surrogate (UCB scoring disabled)")

    # ── load table (if given) ──────────────────────────────────────────────────
    table = None
    if table_path is not None:
        tp = Path(table_path)
        if tp.exists():
            table = load_table(tp)
            if verbose:
                f3_count = sum(1 for e in table.values() if "F3" in e)
                print(f"  Table loaded: {len(table)} configs, {f3_count} with F3 rows")
                if f3_count == 0:
                    print(f"  [INFO] Table has no F3 rows: commit actions will return "
                          f"the failure-ladder penalty (-20) per documented honesty rule.")
        else:
            print(f"  [WARNING] Table not found at {tp}; running in live mode.")

    # ── build FunnelEnv(s) ─────────────────────────────────────────────────────
    # With --jobs W there are W envs running episodes concurrently, all sharing
    # ONE CampaignState so they share a budget, an incumbent, the reward anchors
    # and the log file.  W=1 is exactly the historical single-env setup.
    jobs = max(1, int(jobs))
    campaign_state = CampaignState(budget, results_path.parent / f"funnel_{campaign_id}.jsonl")

    def _make_env() -> FunnelEnv:
        return FunnelEnv(
            space_yaml=space_yaml,
            platform=platform,
            budget=budget,
            surrogate=surrogate,
            table=table,
            results_path=results_path.parent / f"funnel_{campaign_id}.jsonl",
            seed=seed,
            design=design,
            max_tier=max_tier,
            active_space=space,  # pass the KnobRegistry space so validation uses its bounds
            campaign_state=campaign_state,
        )

    envs = [_make_env() for _ in range(jobs)]
    env = envs[0]   # any env answers campaign-level questions; they share state

    # ── build agents ───────────────────────────────────────────────────────────
    # surrogate_ucb must rank candidates with the same reward formula
    # FunnelEnv._terminal_reward will actually use for this design (the env
    # derives the reward_kind from the design's functional model, or "generic").
    _ucb_kind, _ = env._surrogate_reward_kind()
    gen = CandidateGenerator(
        space=space,
        sampler=sampler,
        surrogate=surrogate,
        seed=seed,
        kappa=1.0,
        grid_snap=(table is not None),   # snap to table grid in table mode
        reward_kind=_ucb_kind,
        # refs is a live getter: FunnelEnv auto-anchors generic-design refs
        # from the first F3 build, so a frozen dict here would go stale.
        refs=(lambda: env._surrogate_reward_kind()[1]) if _ucb_kind == "generic" else None,
    )
    promo = _make_promotion_agent(promotion, seed=seed)

    # ── campaign tracking ──────────────────────────────────────────────────────
    best_reward = float("-inf")
    best_config: dict | None = None
    n_episodes = 0
    n_killed = 0
    n_f3 = 0
    per_fidelity_counts: dict[str, int] = {f: 0 for f in _FIDELITY_ORDER}

    # audit F15: a reset() failure (invalid config / constraint violation) costs
    # zero budget, so a generator whose space systematically mismatches the
    # table/constraints busy-loops forever with no output.  Abort after N
    # consecutive failures.  Reset to 0 on any successful reset.
    consecutive_reset_failures = 0
    _MAX_CONSECUTIVE_RESET_FAILURES = 25

    # audit F15: campaign metadata stamped onto every episode row so a log can be
    # attributed to a (design, platform, sampler, promotion, tier, seed) after the
    # fact — the two existing overnight logs could not be.
    campaign_meta = {
        "design":    design,
        "platform":  platform,
        "sampler":   sampler,
        "promotion": promotion,
        "max_tier":  max_tier,
        "seed":      seed,
        # audit F18/F19: reward semantics are versioned so corpora produced under
        # different rulers are never silently averaged together.  Absent key on an
        # old row == version 1.
        "reward_version": REWARD_VERSION,
    }

    if verbose:
        print(f"\n  Campaign {campaign_id}")
        print(f"  sampler={sampler} promotion={promotion} seed={seed} "
              f"table={'yes' if table else 'no'}")
        print(f"  budget: {budget.summary_line()}")
        print(f"  {'Episode':>8} {'Fidelity':>8} {'Reward':>9} {'Best':>9} "
              f"{'Spent(h)':>9} {'Config'}")
        print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*40}")

    log_rows: list[dict] = []

    # Episodes since the last F3 attempt — feeds the Budget's no-progress guard.
    # Without it a count-bounded campaign whose promotion policy kills everything
    # spins to the wall cap and produces nothing (observed: 7,884 consecutive
    # kills in a real likith campaign).
    episodes_since_f3 = 0

    # ── concurrency ────────────────────────────────────────────────────────────
    # Only the *stages* run in parallel; the learners stay serialised.  Optuna's
    # in-memory study is not concurrency-safe, and a bandit updated from two
    # threads at once corrupts its covariance matrix.  Both are microseconds of
    # work against minutes of build, so the lock costs nothing.
    gen_lock = threading.Lock()     # CandidateGenerator ask/tell
    promo_lock = threading.Lock()   # promotion agent act/update
    drv_lock = threading.Lock()     # campaign counters + best-so-far
    log_lock = threading.Lock()     # episode JSONL append
    stop_flag = threading.Event()

    def _worker(env: "FunnelEnv") -> None:
        nonlocal best_reward, best_config, n_episodes, n_killed, n_f3
        nonlocal consecutive_reset_failures, episodes_since_f3

        # The single stop test.  FunnelEnv owns every counter the Budget needs
        # (wall clock, tool seconds, successful/attempted F3 builds), so both this
        # loop and the mid-episode check below ask the same question.
        while not stop_flag.is_set() and not env.budget_exhausted(episodes_since_f3):
            # ── generate candidate ─────────────────────────────────────────────
            try:
                with gen_lock:
                    config = gen.suggest()
            except Exception as exc:   # noqa: BLE001
                print(f"  [ERROR] CandidateGenerator.suggest() failed: {exc}")
                stop_flag.set()
                return

            # ── run episode ────────────────────────────────────────────────────
            try:
                state = env.reset(config)
            except (ValueError, KeyError) as exc:
                # Invalid config (not in table, or constraint violation)
                with gen_lock:
                    gen.update(config, reward=-100.0, fidelity="invalid")
                with drv_lock:
                    consecutive_reset_failures += 1
                    too_many = consecutive_reset_failures >= _MAX_CONSECUTIVE_RESET_FAILURES
                if too_many:
                    print(f"  [ABORT] {consecutive_reset_failures} consecutive "
                          f"env.reset failures (last: {exc!r}) — the candidate space "
                          f"likely mismatches the table/constraints; stopping campaign.")
                    stop_flag.set()
                    return
                continue
            except Exception as exc:   # noqa: BLE001
                with gen_lock:
                    gen.update(config, reward=-100.0, fidelity="invalid")
                print(f"  [WARN] env.reset failed: {exc}")
                with drv_lock:
                    consecutive_reset_failures += 1
                    too_many = consecutive_reset_failures >= _MAX_CONSECUTIVE_RESET_FAILURES
                if too_many:
                    print(f"  [ABORT] {consecutive_reset_failures} consecutive "
                          f"env.reset failures (last: {exc!r}) — stopping campaign.")
                    stop_flag.set()
                    return
                continue
            with drv_lock:
                consecutive_reset_failures = 0   # a good reset breaks any failure streak

            episode_reward_acc = 0.0
            episode_done = False
            fidelity_reached = "F0"
            episode_actions: list[str] = []
            episode_step_rewards: list[float] = []
            episode_terminal_reward: float | None = None   # pure F3 PPA reward (no shaping)
            episode_table_miss = False
            episode_t0 = time.time()

            while not episode_done:
                if stop_flag.is_set() or env.budget_exhausted(episodes_since_f3):
                    # Over budget mid-episode.  env.spent_s already includes this
                    # episode's accumulated cost (FunnelEnv._charge adds every cost to
                    # both _spent_s and _episode_spent_s), so adding _episode_spent_s
                    # here double-counted the episode spend and stopped campaigns early
                    # (audit F15).  In count mode this also stops the moment the last
                    # required F3 build lands, rather than starting another episode.
                    break

                with promo_lock:
                    action = promo.act(state)
                episode_actions.append(action)

                try:
                    # The expensive part, deliberately OUTSIDE every lock: this is
                    # the ORFS build, and it is the only thing --jobs parallelises.
                    next_state, reward, episode_done, info = env.step(action)
                    if verbose:
                        print(f"    └── [STEP] Gate: {fidelity_reached} -> Action Chosen: "
                              f"{action.upper()} -> Step Reward: {reward:+.3f}")
                except Exception as exc:   # noqa: BLE001
                    print(f"  [WARN] env.step({action!r}) failed: {exc}")
                    episode_done = True
                    reward = 0.0
                    next_state = state
                    info = {"fidelity": fidelity_reached, "action": action}

                with promo_lock:
                    promo.update(state, action, reward)
                state = next_state
                episode_reward_acc += reward
                episode_step_rewards.append(reward)

                # Capture the pure terminal PPA reward (no shaping) when F3 completes
                # so TPE / best / the log record the real physical score, not the
                # shaped accumulator (audit H0).
                if info.get("terminal_reward") is not None:
                    episode_terminal_reward = float(info["terminal_reward"])
                if info.get("table_miss"):
                    episode_table_miss = True

                fid = info.get("fidelity", fidelity_reached)
                if fid in _FIDELITY_ORDER:
                    if _FIDELITY_ORDER.index(fid) > _FIDELITY_ORDER.index(fidelity_reached):
                        fidelity_reached = fid

                if episode_done:
                    act = info.get("action", action)
                    if act == "kill":
                        with drv_lock:
                            n_killed += 1

            # ── episode complete ───────────────────────────────────────────────
            # F3 terminal reward: use the PURE terminal PPA reward (no shaping), and
            # only count a real F3 commit (status ok, not a table_miss) as an F3
            # observation for TPE / the incumbent (audit H0).
            is_real_f3 = (
                episode_done and fidelity_reached == "F3"
                and not episode_table_miss and episode_terminal_reward is not None
            )
            f3_reward: float | None = episode_terminal_reward if is_real_f3 else None

            with drv_lock:
                n_episodes += 1
                my_episode = n_episodes
                per_fidelity_counts[fidelity_reached] = (
                    per_fidelity_counts.get(fidelity_reached, 0) + 1
                )
                # Reset on any episode that actually entered F3, successful or not —
                # the guard is about the policy refusing to promote, not about build
                # quality.
                episodes_since_f3 = (0 if fidelity_reached == "F3"
                                     else episodes_since_f3 + 1)
                if f3_reward is not None:
                    n_f3 += 1
                    if f3_reward > best_reward:
                        best_reward = f3_reward
                        best_config = dict(config)
                best_snapshot = best_reward

            if f3_reward is not None:
                with gen_lock:
                    gen.update(config, f3_reward, fidelity="F3")
            else:
                # table_miss carries no real terminal data — route it through the
                # kill-memo (not fidelity="F3", which CandidateGenerator.update()
                # treats as a genuine observation and tells to the Optuna study).
                not_f3_fidelity = "table_miss" if episode_table_miss else fidelity_reached
                with gen_lock:
                    gen.update(config, episode_reward_acc, fidelity=not_f3_fidelity)

            # ── logging ────────────────────────────────────────────────────────
            spent_h = env.spent_s / 3600.0
            log_row = {
                "ts":            time.time(),
                "campaign_id":   campaign_id,
                **campaign_meta,   # audit F15: design/platform/sampler/promotion/max_tier/seed
                "episode":       my_episode,
                "config":        config,
                "actions":       episode_actions,
                "fidelity":      fidelity_reached,
                "step_rewards":  episode_step_rewards,
                "episode_reward": episode_reward_acc,
                "shaped_episode_reward": episode_reward_acc,  # incl. per-step shaping
                "f3_reward":     f3_reward,                    # pure terminal PPA reward
                "best_reward":   best_snapshot if best_snapshot != float("-inf") else None,
                # Terminal-fidelity observation: real physical metrics + the 6_final.gds
                # path for F3 episodes, so reporting and `eda-rl collect` can locate the
                # actual layouts without re-deriving variant names.
                "obs":           env.terminal_obs,
                "spent_s":       round(env.spent_s, 2),
                "episode_s":     round(time.time() - episode_t0, 3),
            }

            # Serialised: concurrent appends interleave and corrupt JSONL lines.
            with log_lock:
                log_rows.append(log_row)
                try:
                    with open(results_path, "a", encoding="utf-8") as fout:
                        fout.write(json.dumps(log_row) + "\n")
                except OSError:
                    pass

            if verbose:
                cfg_items = []
                for axis_name in space.keys():
                    val = config.get(axis_name, '?')
                    # If it's a float, format it cleanly so it doesn't clutter the screen
                    if isinstance(val, float):
                        cfg_items.append(f"{axis_name}={val:.3f}")
                    else:
                        cfg_items.append(f"{axis_name}={val}")
                cfg_str = " | ".join(cfg_items)
                r_str = (f"{episode_reward_acc:+.3f}" if f3_reward is None
                         else f"{f3_reward:+.3f}(F3)")
                best_str = (f"{best_snapshot:+.3f}" if best_snapshot != float("-inf")
                            else "     —")
                print(f"  {my_episode:>8d} {fidelity_reached:>8} {r_str:>9} "
                      f"{best_str:>9} {spent_h:>8.3f}h  {cfg_str}")

    if jobs == 1:
        # Identical control flow to the pre-concurrency driver: no threads, no
        # scheduling nondeterminism, so a seeded campaign stays reproducible.
        _worker(envs[0])
    else:
        threads = [threading.Thread(target=_worker, args=(e,), daemon=True,
                                    name=f"funnel-w{i}")
                   for i, e in enumerate(envs)]
        for th in threads:
            th.start()
        try:
            for th in threads:
                th.join()
        except KeyboardInterrupt:
            stop_flag.set()
            for th in threads:
                th.join(timeout=30)
            raise

    # ── summary ────────────────────────────────────────────────────────────────
    elapsed_s = time.time() - t0

    # A campaign that stopped because nothing was ever promoted has produced no
    # data.  Say so loudly rather than leaving a silent empty log: this is the
    # observed LinUCB-collapse / mis-tuned-gate failure, and it looks identical
    # to a normal finish in the summary row alone.
    _stop = env.budget_stop_reason(episodes_since_f3)
    if _stop == "no_f3_progress":
        print(f"\n  [ABORT] {episodes_since_f3} consecutive episodes without a single "
              f"F3 build — stopping.")
        print(f"          The promotion policy (--promotion {promotion}) is killing "
              f"every candidate before F3,")
        print(f"          so this campaign cannot make progress toward its budget. "
              f"Check the gate thresholds")
        print(f"          against this design's real F2 metrics "
              f"(`eda-rl doctor --design {design} --platform {platform}`).")
    elif budget.is_count_mode and env.n_f3_ok < (budget.max_f3_ok or 0):
        print(f"\n  [WARNING] stopped at {env.n_f3_ok}/{budget.max_f3_ok} successful "
              f"F3 builds (reason: {_stop}); "
              f"{env.n_f3_attempts} attempts were made.")

    summary = {
        "campaign_id":        campaign_id,
        "best_config":        best_config,
        "best_reward":        best_reward if best_reward != float("-inf") else None,
        "n_episodes":         n_episodes,
        "n_f3":               n_f3,
        "n_killed":           n_killed,
        "per_fidelity":       per_fidelity_counts,
        "elapsed_s":          round(elapsed_s, 1),
        "simulated_s":        round(env.spent_s, 1),
        "sampler":            sampler,
        "promotion":          promotion,
        "seed":               seed,
        "platform":           platform,
        "design":             design,
        "max_tier":           max_tier,
        "reward_version":     REWARD_VERSION,
        # What bounded this campaign, and which limit actually ended it — so a
        # log can be compared against another only when the bounds match.
        "budget":             budget.describe(),
        "stop_reason":        env.budget_stop_reason(episodes_since_f3),
        "wall_elapsed_s":     round(env.wall_elapsed_s, 1),
        "n_f3_ok":            env.n_f3_ok,
        "n_f3_attempts":      env.n_f3_attempts,
    }

    # audit F15: persist the summary as a final self-describing row so a campaign
    # log is complete on its own.  It is wrapped under "campaign_summary" (no
    # top-level "config" key) so campaign_data.load_campaign_rows — which keeps
    # only dict rows containing "config" — skips it, keeping report/dashboard
    # backward-compatible with logs that lack it.
    try:
        with open(results_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps({"campaign_summary": summary}) + "\n")
    except OSError:
        pass

    if verbose:
        print(f"\n  Summary")
        print(f"  {'Episodes':>15}: {n_episodes}")
        print(f"  {'F3 commits':>15}: {n_f3}")
        print(f"  {'Killed':>15}: {n_killed}")
        print(f"  {'Per fidelity':>15}: {per_fidelity_counts}")
        print(f"  {'Best reward':>15}: "
              f"{best_reward:.4f}" if best_reward != float("-inf") else "  (none)")
        print(f"  {'Best config':>15}: {best_config}")
        print(f"  {'Elapsed':>15}: {elapsed_s:.1f}s real, "
              f"{env.spent_s/3600:.3f}h simulated")
        print(f"  Results → {results_path}")
        per_ep = elapsed_s / max(n_episodes, 1)
        per_f3 = elapsed_s / max(n_f3, 1)
        print(f"  Throughput: {per_ep:.1f}s/episode, {per_f3:.1f}s/F3")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Gen2 funnel optimizer campaign driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--design", required=True,
                   help="Design name or YAML path (REQUIRED — no default design)")
    p.add_argument("--platform", default="nangate45",
                   help="Target platform (nangate45 or asap7)")
    p.add_argument("--budget-hours", type=float, default=None, dest="budget_hours",
                   help="Time budget in hours. Alone: the historical tool-time "
                        "budget (default 4.0). With --max-f3: a wall-clock safety "
                        "cap (default 24h)")
    p.add_argument("--max-f3", type=int, default=None, dest="max_f3",
                   help="Stop after this many SUCCESSFUL full (F3) builds. Failed, "
                        "timed-out and aborted builds do not consume the quota, so "
                        "--max-f3 50 yields 50 usable data points. Combines with "
                        "--budget-hours; whichever limit trips first ends the run")
    p.add_argument("--max-f3-attempts", type=int, default=None, dest="max_f3_attempts",
                   help="Hard cap on F3 build attempts regardless of outcome "
                        "(default: 3x --max-f3), so a design that fails every build "
                        "still terminates")
    p.add_argument("--max-tier", type=int, default=1, dest="max_tier",
                   help="Maximum knob tier from KnobRegistry (1 = core axes only)")
    p.add_argument("--sampler", choices=["tpe", "surrogate_ucb", "random"],
                   default="tpe",
                   help="Candidate generator sampler")
    p.add_argument("--promotion", choices=["fixed", "linucb", "random"],
                   default="fixed",
                   help="Promotion policy agent")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")
    p.add_argument("--table", default=None, dest="table",
                   help="Path to results_funnel.jsonl (table mode); omit for live mode")
    p.add_argument("--surrogate", default=None, dest="surrogate",
                   help="Path to surrogate .joblib (default: auto-detect surrogate_n45.joblib)")
    p.add_argument("--out", default=None, dest="out",
                   help="JSONL output path for campaign summary log "
                        "(default: campaigns/<design>/<platform>/results_funnel_campaigns.jsonl)")
    p.add_argument("--space-yaml", default=str(_DEFAULT_SPACE_YAML), dest="space_yaml",
                   help="Path to search_space_funnel.yaml")
    p.add_argument("--jobs", type=int, default=1, dest="jobs",
                   help="Concurrent episodes (parallel F3 builds). Only the tool "
                        "runs overlap; the sampler and promotion policy stay "
                        "serialised. NOTE: >1 is NOT bit-reproducible for a given "
                        "--seed, both learners then see delayed feedback, and "
                        "--max-f3 may overshoot by up to --jobs-1 builds already "
                        "in flight when the quota is reached")
    p.add_argument("--openroad-threads", type=int, default=None,
                   dest="openroad_threads",
                   help="Threads per ORFS build (NUM_CORES). Default: all cores "
                        "divided by --jobs, so W builds x T threads fits the box")
    p.add_argument("--memory-limit-gb", type=float, default=None,
                   dest="memory_limit_gb",
                   help="Per-build address-space cap (ulimit -v). Unset = no cap. "
                        "Worth setting with --jobs so one greedy build fails "
                        "cleanly instead of OOM-killing its peers")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-episode output")

    args = p.parse_args()

    if not _FUNNEL_OK:
        print(f"ERROR: FunnelEnv not available: {_FUNNEL_ERR}")
        sys.exit(1)
    if not _CAND_OK:
        print(f"ERROR: CandidateGenerator not available: {_CAND_ERR}")
        sys.exit(1)

    if args.out is not None:
        out_path = Path(args.out)
    else:
        # Normalise the design identity before it becomes a path.  --design
        # accepts a bare name OR a YAML path, and the raw string used to be
        # pasted straight into the log path — so
        # `--design eda_rl/designs/sagar.yaml` created
        # campaigns/eda_rl/designs/sagar.yaml/sky130hd/.  Four such ghost trees
        # (~31 MB, two holding real 7-hour campaigns) are committed here.
        #
        # The slug is the YAML STEM, not DesignSpec.name: sagar.yaml declares
        # name "alu4b" and likith.yaml declares "id", while their campaign
        # directories have always been campaigns/sagar/ and campaigns/likith/.
        # Using .name here would silently relocate every future log and orphan
        # every existing one.
        design_slug = _design_slug(args.design)
        out_path = _CAMPAIGNS_ROOT / design_slug / args.platform / "results_funnel_campaigns.jsonl"

    # Neither limit given → the historical default (4 h of tool time), so bare
    # `eda-rl optimize` behaves exactly as before.
    budget_hours = args.budget_hours
    if budget_hours is None and args.max_f3 is None:
        budget_hours = 4.0
    try:
        budget = Budget.from_args(budget_hours=budget_hours, max_f3=args.max_f3,
                                  max_f3_attempts=args.max_f3_attempts)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    is_mock = os.environ.get("PHYSICAL_MOCK", "").strip() in ("1", "true", "True", "yes")
    if is_mock:
        print(f"  [MOCK MODE] PHYSICAL_MOCK=1 — using mock metrics")

    # Resource split across concurrent builds.  Set via the environment because
    # physical_runner reads its defaults at import time and run_physical is
    # reached through several call paths (env, build_table, doctor).
    jobs = max(1, int(args.jobs))
    threads = args.openroad_threads
    if threads is None:
        threads = max(1, (os.cpu_count() or 1) // jobs)
    os.environ["EDA_RL_NUM_CORES"] = str(int(threads))
    if args.memory_limit_gb is not None:
        os.environ["EDA_RL_MEMORY_LIMIT_GB"] = str(float(args.memory_limit_gb))
    if jobs > 1:
        print(f"  [PARALLEL] {jobs} concurrent builds x {threads} threads each"
              + (f", {args.memory_limit_gb} GB cap per build"
                 if args.memory_limit_gb else ""))
        print(f"  [PARALLEL] --seed {args.seed} is NOT reproducible at --jobs>1: "
              f"episode interleaving varies, and the sampler and bandit see "
              f"delayed, out-of-order feedback.")

    run_campaign(
        design=args.design,
        platform=args.platform,
        budget=budget,
        max_tier=args.max_tier,
        sampler=args.sampler,
        promotion=args.promotion,
        seed=args.seed,
        table_path=args.table,
        surrogate_path=args.surrogate,
        results_path=out_path,
        space_yaml=Path(args.space_yaml),
        verbose=not args.quiet,
        jobs=jobs,
    )


if __name__ == "__main__":
    main()
