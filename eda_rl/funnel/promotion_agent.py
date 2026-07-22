"""promotion_agent.py — promotion policy agents for the multi-fidelity funnel.

Three agents implement the same interface for the FunnelEnv action space
{kill, re-proxy, promote, commit}:

1. PromotionAgent — LinUCB contextual bandit (the primary RL component).
   Per-action linear models: A_a = I * lambda, b_a = 0.  At each step the
   agent picks argmax_{a} theta_a^T s + alpha * sqrt(s^T A_a^{-1} s), then
   updates the chosen arm on receiving the scalar reward.

2. FixedGateAgent — mirrors the hard-coded cascade.py gate logic as a
   deterministic policy over the same 22-dim state vector.  Useful baseline:
   if LinUCB cannot beat fixed gates in the table-sim benchmark it is not
   worth deploying.

3. RandomPromotionAgent — uniform-random action selection.  Sanity check.

State vector layout: see eda_rl/funnel/state_spec.py — the single source of truth
(imported below).  The agents below read it through the IDX_* constants so they
can never drift from what FunnelEnv._build_state actually emits (audit H3/H4).
Key slots the gates use: [6] F0 accuracy, [8] F1 accuracy, [10] F2 wns_norm
(clip[-2,2] of wns/clk for generic designs, wns_ns/5 for tinymac/legacy — see
state_spec.py [10]), [18..21] depth one-hot.  The unrun convention is 0.0.

FixedGateAgent mapping to the original cascade gates (legacy/gen1/cascade.py):
    The cascade uses three hard thresholds (derived from search_space_full.yaml
    gates: block + cascade.py _run_sim / proxy checks):

    Depth F0 (validate+analytic):
      - if accuracy < 0.95 (state[6] < 0.95 or effectively == 0.0): "kill"
        maps to cascade gate: sim min_accuracy=0.95 (acc_width too narrow)
      - else: "promote" to F1

    Depth F1 (behavioral sim):
      - if accuracy < 0.95 (state[8] < 0.95): "kill"
        maps to cascade gate: sim min_accuracy=0.95
      - else: "promote" to F2

    Depth F2 (synth+STA proxy):
      - if proxy_wns < -0.5 (state[10] < -0.5 in FunnelEnv normalised units)
        → "kill"
        Clock-relative gate (F11): kill when wns < -0.5·clock_period.  FunnelEnv
        stores d10 = clip(wns/clk, -2, 2) for generic designs (F10), so the
        threshold -0.5 == wns < -0.5·clk and fires on sub-ns platforms where the
        old absolute -2.5 ns gate was inert.  For tinymac/legacy FunnelEnv keeps
        d10 = clip(wns_ns/5, -2, 2), so -0.5 == raw -2.5 ns (Phase 5 Exp 3),
        unchanged.  proxy_wns_kill_threshold is checked in the NORMALISED units.
        For callers that feed RAW WNS (ns) in state[10] instead of FunnelEnv's
        normalised value, create FixedGateAgent(proxy_wns_kill_threshold=-2.5).
      - else: "promote" to F3

    Depth F3 (full flow result available):
      - always "commit" — the full flow result IS the ground truth; the agent
        commits and the FunnelEnv terminates the episode.

    Note: cascade.py's proxy block also has max_area_um2=80000, require_timing_met=false.
    FixedGateAgent only gates on timing (the well-calibrated signal per Phase 5).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# ── type alias for clarity ────────────────────────────────────────────────────
_ActionTuple = tuple[str, ...]
_DEFAULT_ACTIONS: _ActionTuple = ("kill", "re-proxy", "promote", "commit")

# ── state vector indices — re-exported from the canonical state_spec module ───
from eda_rl.funnel.state_spec import (  # noqa: E402,F401
    STATE_DIM,
    IDX_LANES_NORM, IDX_ACCW_NORM, IDX_CLK_NORM,
    IDX_RECIPE, IDX_PLATFORM, IDX_RECIPE_SPD, IDX_RECIPE_AREA,
    IDX_F0_CYCLES, IDX_F0_ACC, IDX_F1_CYCLES, IDX_F1_ACC,
    IDX_F2_AREA, IDX_F2_WNS, IDX_F2_FF, IDX_F2_CELLS, IDX_F2_LEVELS,
    IDX_SURR_MU, IDX_SURR_SIG, IDX_INCUMBENT, IDX_BUDGET_FRAC,
    IDX_DEPTH_F0, IDX_DEPTH_F1, IDX_DEPTH_F2, IDX_DEPTH_F3,
)


# ── LinUCB contextual bandit ──────────────────────────────────────────────────

_N_DEPTHS = 4   # F0, F1, F2, F3 — see IDX_DEPTH_F0..F3 in state_spec.py


class PromotionAgent:
    """LinUCB contextual bandit over the 4 funnel actions.

    Standard disjoint LinUCB (one linear model per arm):
        theta_a = A_a^{-1} b_a
        UCB(a, s) = theta_a^T s + alpha * sqrt(s^T A_a^{-1} s)
        action = argmax_a UCB(a, s)

    Per-arm update on reward r, context s, arm a:
        A_a <- A_a + s s^T
        b_a <- b_a + r * s

    Depth-conditioned arms (fixes Audit M6): each base action gets one linear
    model PER DEPTH (F0/F1/F2/F3), keyed off the state's depth one-hot
    (IDX_DEPTH_F0..F3). Without this, a single "promote" arm absorbed both the
    tiny per-step shaping reward at F0->F1/F1->F2 AND the rare, huge-magnitude
    F2->F3 terminal payoff — a real campaign (likith/asap7, 2026-07-10) showed
    this collapse in practice: two early F2->F3 promotions that hit genuine
    reference-SDC timing violations (reward ~-3.4) permanently zeroed the
    "promote" arm's F2-depth estimate for the remaining 7884 episodes of an
    8-hour budget (F2->F3 rate: 0.56% -> a hard 0.00%, never recovering),
    while the same arm's F0-depth behaviour (rich, frequent, small-reward
    data) stayed unaffected. Depth-conditioning gives F2->F3 its own linear
    model so a couple of unlucky terminal outcomes can't poison F0->F2
    shaping decisions or vice versa — each depth transition now has to earn
    its own bad reputation from its own (still small) sample of outcomes.

    Reward clipping (also Audit M6): terminal rewards span roughly
    [-100, +4] (full/parse failures vs a good PPA score) while per-step
    shaping rewards are O(1e-4). Feeding raw magnitudes into a ridge-
    regularised linear model means a single outlier sample can dominate
    theta for an arm that has seen only a handful of updates. Clipping the
    reward fed to `update()` to [-reward_clip, +reward_clip] bounds any one
    sample's influence while preserving its sign and rank among "bad"
    outcomes.

    Epsilon-greedy exploration floor (found necessary, not just theorised,
    by stress-testing depth-conditioning against likith's actual reward
    regime before shipping this): a combinational design's F2 proxy WNS
    feature is pinned at exactly 0.0 (the real timing-margin signal only
    exists after F3 runs), so the F2-depth "promote" arm sees a near-
    identical context on every decision. Pure LinUCB is *rationally*
    correct to converge fast there — under a stress test mirroring
    likith's ~95%-catastrophic F2->F3 outcome rate, depth-conditioning
    alone still locked to permanent "always kill" after just 1-3 samples,
    and raising alpha only delayed that by a few samples, it didn't fix
    the structural collapse. With probability `epsilon`, `act()` ignores
    the UCB argmax and returns a uniformly random action instead,
    guaranteeing the bandit keeps periodically re-checking every
    (action, depth) arm for the whole campaign budget instead of
    permanently writing one off from a handful of early unlucky draws.

    Parameters
    ----------
    dim   : context dimension (must match state vector; default 22)
    alpha : exploration coefficient (UCB width; default 1.0)
    seed  : RNG seed for tie-breaking
    actions : tuple of action strings; must be a superset of the FunnelEnv actions
    lam   : ridge regularisation for initial A (A_a = lam * I); prevents
            singular A before observations arrive; default 1.0
    depth_conditioned : maintain one linear model per (action, depth) pair
            instead of one per action (default True; see class docstring).
    reward_clip : clip |reward| fed to update() to this bound before the A/b
            update (default 5.0; None disables clipping).
    epsilon : probability of overriding the UCB argmax with a uniformly
            random action in act() (default 0.03; 0.0 disables the floor).
    """

    def __init__(
        self,
        dim: int = STATE_DIM,
        alpha: float = 1.0,
        seed: int = 0,
        actions: _ActionTuple = _DEFAULT_ACTIONS,
        lam: float = 1.0,
        depth_conditioned: bool = True,
        reward_clip: float | None = 5.0,
        epsilon: float = 0.03,
    ) -> None:
        self.dim = dim
        self.alpha = float(alpha)
        self.actions = tuple(actions)
        self.lam = float(lam)
        self.depth_conditioned = bool(depth_conditioned)
        self.reward_clip = float(reward_clip) if reward_clip is not None else None
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        n_actions = len(self.actions)
        self._n_depths = _N_DEPTHS if self.depth_conditioned else 1
        n_arms = n_actions * self._n_depths
        # Per-arm precision matrix A_a (dim × dim) and reward vector b_a (dim,)
        # A_a starts as lam * I; b_a starts at zero. Arm index = depth * n_actions
        # + action_idx when depth-conditioned, else just action_idx.
        self._A: list[np.ndarray] = [np.eye(dim) * lam for _ in range(n_arms)]
        self._b: list[np.ndarray] = [np.zeros(dim) for _ in range(n_arms)]
        # Cached inverse (invalidated on update)
        self._A_inv: list[np.ndarray | None] = [None] * n_arms
        # Track update counts for logging
        self._n_updates: list[int] = [0] * n_arms

    # ── core interface ─────────────────────────────────────────────────────────

    def _depth_of(self, s: np.ndarray) -> int:
        """Read the current depth (0=F0..3=F3) off the state's one-hot slots.

        Returns 0 (F0) if no depth bit is set (shouldn't happen in practice —
        FunnelEnv always sets exactly one — but stay defensive rather than
        index out of range).
        """
        if not self.depth_conditioned:
            return 0
        depth_idxs = (IDX_DEPTH_F0, IDX_DEPTH_F1, IDX_DEPTH_F2, IDX_DEPTH_F3)
        for depth, idx in enumerate(depth_idxs):
            if idx < len(s) and s[idx] > 0.5:
                return depth
        return 0

    def _arm(self, action_idx: int, depth: int) -> int:
        return depth * len(self.actions) + action_idx if self.depth_conditioned else action_idx

    def _to_state_vec(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float).reshape(-1)
        if len(s) < self.dim:
            # Pad with zeros if state is shorter than expected (defensive)
            s = np.pad(s, (0, self.dim - len(s)))
        elif len(s) > self.dim:
            s = s[: self.dim]
        return s

    def act(self, state: np.ndarray) -> str:
        """Select an action given the 22-dim state vector.

        Returns the action string with the highest UCB score, breaking ties
        randomly (seeded) to ensure reproducibility. With probability
        `epsilon`, overrides the UCB argmax with a uniformly random action
        (forced-exploration floor — see class docstring: pure LinUCB can
        rationally converge to permanently avoiding an arm within a handful
        of samples when its context is near-featureless, e.g. a
        combinational design's always-zero F2 proxy WNS).
        """
        if self.epsilon > 0.0 and self._py_rng.random() < self.epsilon:
            return self._py_rng.choice(self.actions)

        s = self._to_state_vec(state)
        depth = self._depth_of(s)

        ucb_scores = []
        for i, action in enumerate(self.actions):
            arm = self._arm(i, depth)
            A_inv = self._get_A_inv(arm)
            theta = A_inv @ self._b[arm]
            # UCB bonus: alpha * sqrt(s^T A_inv s)
            val = s @ A_inv @ s
            bonus = self.alpha * np.sqrt(max(float(val), 0.0))
            ucb_scores.append(float(theta @ s) + bonus)

        # Argmax with random tie-breaking
        best_val = max(ucb_scores)
        best_actions = [i for i, v in enumerate(ucb_scores) if abs(v - best_val) < 1e-12]
        chosen_idx = self._py_rng.choice(best_actions)
        return self.actions[chosen_idx]

    def update(self, state: np.ndarray, action: str, reward: float) -> None:
        """Update the chosen arm's linear model with (state, reward).

        Only the arm corresponding to `(action, depth)` is updated (disjoint
        LinUCB; depth read off the state one-hot, see `_depth_of`). Invalid
        action strings are silently ignored (defensive). `reward` is clipped
        to `[-reward_clip, +reward_clip]` before the A/b update.

        Audit M6 (now fixed, not just an experiment): the "promote" arm used
        to absorb both the big terminal F3 payoff (at depth F2→F3) and tiny
        per-step shaping (at F0→F1, F1→F2), with rewards spanning ~[-100, +4]
        unscaled. A real campaign (likith/asap7, 2026-07-10) showed the
        predicted failure mode in practice: two early F2→F3 promotions that
        hit genuine reference-SDC timing violations (reward ~-3.4) permanently
        zeroed the shared "promote" arm's F2-depth estimate for the remaining
        7884 episodes of an 8-hour budget. `depth_conditioned` arms (see
        `_arm`) and `reward_clip` are the fix: F2→F3 now has its own linear
        model, and no single terminal outcome can dominate it unboundedly.
        """
        if action not in self.actions:
            return
        action_idx = self.actions.index(action)
        s = self._to_state_vec(state)
        depth = self._depth_of(s)
        arm = self._arm(action_idx, depth)

        r = float(reward)
        if self.reward_clip is not None:
            r = max(-self.reward_clip, min(self.reward_clip, r))

        self._A[arm] += np.outer(s, s)
        self._b[arm] += r * s
        self._A_inv[arm] = None   # invalidate cached inverse
        self._n_updates[arm] += 1

    # ── persistence ───────────────────────────────────────────────────────────

    def _arm_key(self, action: str, depth: int) -> str:
        key = action.replace("-", "_")   # "re-proxy" → "re_proxy"
        return f"{key}_d{depth}" if self.depth_conditioned else key

    def save(self, path: str | Path) -> None:
        """Save agent parameters to a .npz file."""
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        for depth in range(self._n_depths):
            for i, action in enumerate(self.actions):
                key = self._arm_key(action, depth)
                arm = self._arm(i, depth)
                arrays[f"A_{key}"] = self._A[arm]
                arrays[f"b_{key}"] = self._b[arm]
        # Metadata as 0-d arrays
        arrays["dim"] = np.array(self.dim)
        arrays["alpha"] = np.array(self.alpha)
        arrays["lam"] = np.array(self.lam)
        arrays["depth_conditioned"] = np.array(self.depth_conditioned)
        arrays["reward_clip"] = np.array(
            self.reward_clip if self.reward_clip is not None else np.nan)
        arrays["epsilon"] = np.array(self.epsilon)
        arrays["n_updates"] = np.array(self._n_updates)
        # Persist the action tuple itself — without this, load() always
        # reconstructs with _DEFAULT_ACTIONS regardless of what actions this
        # agent was actually trained with, silently mis-keying the A_*/b_*
        # arrays for any non-default actions tuple.
        arrays["actions"] = np.array(self.actions)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "PromotionAgent":
        """Load agent from a .npz file produced by save()."""
        data = np.load(str(path))
        dim = int(data["dim"])
        alpha = float(data["alpha"])
        lam = float(data["lam"])
        # Older .npz files saved before depth-conditioning have no
        # "depth_conditioned"/"reward_clip" keys — fall back to the pre-fix
        # behaviour (single arm per action, no clipping) so old artifacts
        # still load, rather than silently mis-keying A_*/b_* lookups.
        depth_conditioned = bool(data["depth_conditioned"]) if "depth_conditioned" in data else False
        if "reward_clip" in data:
            rc = float(data["reward_clip"])
            reward_clip = None if np.isnan(rc) else rc
        else:
            reward_clip = None
        epsilon = float(data["epsilon"]) if "epsilon" in data else 0.0
        # Older .npz files saved before this fix have no "actions" key —
        # fall back to the default tuple for those (matches their save()-time
        # behaviour, since only the default tuple was ever used pre-fix).
        actions = tuple(data["actions"].tolist()) if "actions" in data else _DEFAULT_ACTIONS
        agent = cls(dim=dim, alpha=alpha, seed=seed, actions=actions, lam=lam,
                    depth_conditioned=depth_conditioned, reward_clip=reward_clip,
                    epsilon=epsilon)
        for depth in range(agent._n_depths):
            for i, action in enumerate(agent.actions):
                key = agent._arm_key(action, depth)
                arm = agent._arm(i, depth)
                agent._A[arm] = data[f"A_{key}"]
                agent._b[arm] = data[f"b_{key}"]
                agent._A_inv[arm] = None
        if "n_updates" in data:
            agent._n_updates = list(data["n_updates"].astype(int))
        return agent

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_A_inv(self, idx: int) -> np.ndarray:
        """Return cached A_inv, recomputing if invalidated."""
        if self._A_inv[idx] is None:
            try:
                self._A_inv[idx] = np.linalg.inv(self._A[idx])
            except np.linalg.LinAlgError:
                self._A_inv[idx] = np.linalg.pinv(self._A[idx])
        return self._A_inv[idx]

    def __repr__(self) -> str:
        if self.depth_conditioned:
            updates = {
                f"{action}@F{depth}": self._n_updates[self._arm(i, depth)]
                for depth in range(self._n_depths)
                for i, action in enumerate(self.actions)
            }
        else:
            updates = dict(zip(self.actions, self._n_updates))
        return (f"PromotionAgent(dim={self.dim}, alpha={self.alpha}, "
                f"lam={self.lam}, depth_conditioned={self.depth_conditioned}, "
                f"reward_clip={self.reward_clip}, epsilon={self.epsilon}, "
                f"updates={updates})")


# ── FixedGateAgent ────────────────────────────────────────────────────────────

class FixedGateAgent:
    """Deterministic policy mirroring the hard-coded cascade.py gate thresholds.

    This is the primary baseline: LinUCB must beat it to justify deployment.

    Gate mapping (from cascade.py + search_space_full.yaml gates block):

    Depth F0 (validate + analytic, state[18]=1):
        state[6] (F0 accuracy flag) < 0.95 → "kill"
            (cascade: sim gate min_accuracy=0.95; acc_width<24 → accuracy≈0.73)
        otherwise → "promote"

    Depth F1 (behavioral sim, state[19]=1):
        state[8] (F1 accuracy) < 0.95 → "kill"
            (cascade: same sim gate on exact measured accuracy)
        otherwise → "promote"

    Depth F2 (synth+STA proxy, state[20]=1):
        state[10] (F2 proxy_wns_norm) < -_WNS_KILL_CLK_FRACTION (= -0.5) → "kill"
            Clock-relative gate (F11): kill when wns < -0.5·clock_period.  This is
            read straight off the normalised state[10]:
              • generic designs — env.py sets state[10] = wns/clk (F10), so
                -0.5 == wns < -0.5·clk.  Fires correctly on sub-ns platforms
                (asap7/sky130hd) where the old absolute -2.5 ns gate was inert.
              • tinymac/legacy — env.py keeps state[10] = wns_ns/5, so -0.5 ==
                raw wns < -2.5 ns (Phase-5 Exp-3 calibrated), unchanged.
        otherwise → "promote"

    Depth F3 (full flow, state[21]=1):
        always → "commit"
            (we have the full measurement; no benefit to killing or re-proxying)

    Unknown depth (all depth bits zero):
        "promote"  (default: keep moving forward)
    """

    # F2 WNS kill rule, expressed as a fraction of the clock period (F11).
    # The gate kills a candidate at F2 when its worst slack is more negative than
    # this fraction of the clock period below zero:
    #       wns < -_WNS_KILL_CLK_FRACTION * clock_period.
    # This is computed directly from the normalised state slot [10] WITHOUT the
    # agent needing the raw clock, because env.py normalises WNS by the
    # design's own clock period for designs that expose a clock range (F10), so
    # for those designs state[10] = wns/clk and the rule is exactly
    #       state[10] < -_WNS_KILL_CLK_FRACTION   (= -0.5).
    # The previous threshold was an ABSOLUTE -2.5 ns (state[10] = wns_ns/5), which
    # is inert on sub-ns platforms (asap7/likith WNS is O(±0.003 ns)): the gate
    # could never fire and "fixed gates" degenerated to always-promote (F11).
    #
    # Legacy/TinyVAD path: env.py keeps the fixed /5-ns WNS ruler (state[10] =
    # wns_ns/5) for bit-compat with saved LinUCB agents and the benchmark tables.
    # The SAME normalised threshold -0.5 then corresponds to raw wns < -2.5 ns —
    # the Phase-5 Exp-3 calibrated tinymac/nangate45 value — so tinymac behaviour
    # is unchanged.  (For 3–8 ns tinymac clocks -2.5 ns is not literally
    # -0.5·clk; the /5 ruler is a fixed nangate45-scale ruler, preserved as-is.)
    _WNS_KILL_CLK_FRACTION: float = 0.5   # kill at wns < -0.5 * clock_period
    # Normalised threshold in FunnelEnv state[10] units (both normalisations):
    _NORM_WNS_KILL: float = -_WNS_KILL_CLK_FRACTION   # = -0.5
    # Raw-ns equivalent under the legacy /5 ruler, for callers that feed RAW WNS
    # (ns) into state[10] instead of the normalised value:
    _RAW_WNS_KILL_NS: float = _NORM_WNS_KILL * 5.0   # = -2.5

    def __init__(
        self,
        actions: _ActionTuple = _DEFAULT_ACTIONS,
        seed: int = 0,
        proxy_wns_kill_threshold: float = _NORM_WNS_KILL,  # -0.5 (normalised)
        accuracy_kill_threshold: float = 0.95,
    ) -> None:
        self.actions = tuple(actions)
        self._py_rng = random.Random(seed)
        # proxy_wns_kill_threshold is in the SAME units as state[10]:
        #   FunnelEnv: normalised by /5.0, so default is -0.5
        #              (equivalent to raw -2.5 ns).
        #   raw-WNS callers: pass -2.5 explicitly (state[10] carries raw ns).
        self.proxy_wns_kill_threshold = float(proxy_wns_kill_threshold)
        self.accuracy_kill_threshold = float(accuracy_kill_threshold)

    def act(self, state: np.ndarray) -> str:
        """Apply fixed gate logic.  State slots map per IDX_* constants above."""
        s = np.asarray(state, dtype=float).reshape(-1)

        def _get(idx: int, default: float = 0.0) -> float:
            return float(s[idx]) if idx < len(s) else default

        depth_f0 = _get(IDX_DEPTH_F0)
        depth_f1 = _get(IDX_DEPTH_F1)
        depth_f2 = _get(IDX_DEPTH_F2)
        depth_f3 = _get(IDX_DEPTH_F3)

        if depth_f3 > 0.5:
            # Full flow result available — commit unconditionally
            return "commit"

        if depth_f2 > 0.5:
            # After synth+STA proxy: gate on proxy WNS.
            # state[10] is in the SAME units as proxy_wns_kill_threshold.
            # With FunnelEnv: state[10] = clip(wns_ns/5, -2, 2) — normalised.
            # With raw-WNS callers: state[10] carries raw ns.
            proxy_wns = _get(IDX_F2_WNS, default=0.0)
            # At depth F2 the proxy HAS run (depth one-hot guarantees it), so
            # there is no "unrun" value to guard against — kill whenever the
            # proxy WNS is below the calibrated threshold.  The previous code used
            # a -1.9 (normalised) / -4.9 (raw) sentinel floor to skip "unrun"
            # configs, but that floor also let *catastrophically* late configs
            # escape: FunnelEnv clips state[10] to [-2, 2], so any raw WNS <= -9.5
            # ns clipped to -2.0, fell below the -1.9 floor, and was PROMOTED to a
            # full 7-min F3 build instead of killed (audit H1).  The unrun case is
            # encoded as 0.0 in FunnelEnv (> threshold → promote), so dropping the
            # floor is safe.  Keep a generous guard against the legacy raw -1
            # "unrun" sentinel only when it sits ABOVE the kill threshold.
            if proxy_wns < self.proxy_wns_kill_threshold:
                return "kill"
            return "promote"

        if depth_f1 > 0.5:
            # After behavioral sim: gate on accuracy
            f1_acc = _get(IDX_F1_ACC, default=-1.0)
            if f1_acc >= 0.0 and f1_acc < self.accuracy_kill_threshold:
                return "kill"
            return "promote"

        if depth_f0 > 0.5:
            # After analytic F0: gate on accuracy flag.
            # F0 accuracy is 0.0 for generic designs (no functional eval).
            # Treat exactly 0.0 as the "no-data sentinel" and promote rather than
            # kill — killing on sentinel value would wedge all generic (non-tinyvad)
            # designs at F0 regardless of their actual merit.
            # Only kill when accuracy is in the range (0.0, accuracy_kill_threshold):
            # i.e. a real measured sub-threshold accuracy, not the no-data sentinel.
            f0_acc = _get(IDX_F0_ACC, default=1.0)
            if f0_acc > 0.0 and f0_acc < self.accuracy_kill_threshold:
                return "kill"
            return "promote"

        # No depth bit set — default: promote
        return "promote"

    def update(self, state: np.ndarray, action: str, reward: float) -> None:
        """No-op: FixedGateAgent is deterministic and does not learn."""

    def save(self, path: str | Path) -> None:
        """Save threshold configuration."""
        np.savez(str(path),
                 proxy_wns_kill_threshold=np.array(self.proxy_wns_kill_threshold),
                 accuracy_kill_threshold=np.array(self.accuracy_kill_threshold))

    @classmethod
    def load(cls, path: str | Path) -> "FixedGateAgent":
        data = np.load(str(path))
        return cls(
            proxy_wns_kill_threshold=float(data.get("proxy_wns_kill_threshold", -2.5)),
            accuracy_kill_threshold=float(data.get("accuracy_kill_threshold", 0.95)),
        )

    def __repr__(self) -> str:
        return (f"FixedGateAgent(proxy_wns_kill={self.proxy_wns_kill_threshold}, "
                f"acc_kill={self.accuracy_kill_threshold})")


# ── RandomPromotionAgent ──────────────────────────────────────────────────────

class RandomPromotionAgent:
    """Uniform-random action selection over the funnel action space.

    Sanity check: any agent that cannot beat this is worthless.
    Seeded for reproducibility.
    """

    def __init__(
        self,
        seed: int = 0,
        actions: _ActionTuple = _DEFAULT_ACTIONS,
    ) -> None:
        self.actions = tuple(actions)
        self._py_rng = random.Random(seed)

    def act(self, state: np.ndarray) -> str:  # noqa: ARG002
        """Ignore state, return a uniform-random action."""
        return self._py_rng.choice(self.actions)

    def update(self, state: np.ndarray, action: str, reward: float) -> None:  # noqa: ARG002
        """No-op."""

    def save(self, path: str | Path) -> None:
        np.savez(str(path), actions=np.array(list(self.actions)))

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "RandomPromotionAgent":
        data = np.load(str(path), allow_pickle=True)
        actions = tuple(str(a) for a in data["actions"])
        return cls(seed=seed, actions=actions)

    def __repr__(self) -> str:
        return f"RandomPromotionAgent(actions={self.actions})"


# ── self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Quick smoke test — runs in < 1 s, no external deps."""
    import tempfile, os

    rng = np.random.default_rng(42)
    dim = STATE_DIM
    actions = _DEFAULT_ACTIONS

    # PromotionAgent
    agent = PromotionAgent(dim=dim, alpha=1.0, seed=0)
    for _ in range(50):
        s = rng.standard_normal(dim)
        a = agent.act(s)
        assert a in actions, f"invalid action: {a!r}"
        r = rng.standard_normal()
        agent.update(s, a, r)

    # save/load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "test_agent.npz")
        agent.save(p)
        agent2 = PromotionAgent.load(p)
        assert agent2.dim == dim
        assert agent2.depth_conditioned == agent.depth_conditioned
        assert agent2.reward_clip == agent.reward_clip
        assert agent2.epsilon == agent.epsilon
        s_test = rng.standard_normal(dim)
        # After load, the same state + same RNG state must produce the same
        # action. Re-seed both agents' tie-break/epsilon RNGs identically
        # first — `agent` has already consumed draws from its 50-iteration
        # training loop above (including epsilon-floor draws), while `agent2`
        # starts fresh from load(), so comparing without re-seeding would
        # spuriously fail on nothing but RNG-position drift, not a real
        # save/load bug.
        agent._py_rng = random.Random(123)
        agent2._py_rng = random.Random(123)
        a1 = agent.act(s_test)
        a2 = agent2.act(s_test)
        assert a1 == a2, f"save/load mismatch: {a1!r} vs {a2!r}"

    # ── Regression: depth-conditioned arms isolate F2->F3 catastrophes ────────
    # Reproduces the likith/asap7 2026-07-10 campaign collapse: 2 catastrophic
    # F2->F3 terminal rewards (~-3.4, real reference-SDC timing violations)
    # permanently zeroed the shared "promote" arm and killed F2->F3 promotion
    # for the remaining 7884 episodes of an 8-hour budget, while F0->F2
    # promotion (same shared arm, pre-fix) was untouched only by luck of
    # feature separation. Assert the fix makes that isolation exact, not lucky.
    dc_agent = PromotionAgent(dim=dim, alpha=1.0, seed=1)  # depth_conditioned=True default
    action_idx_promote = dc_agent.actions.index("promote")
    arm_f0_promote = dc_agent._arm(action_idx_promote, 0)
    arm_f2_promote = dc_agent._arm(action_idx_promote, 2)
    assert arm_f0_promote != arm_f2_promote, "F0/F2 promote must be distinct arms"

    b_f0_before = dc_agent._b[arm_f0_promote].copy()
    A_f0_before = dc_agent._A[arm_f0_promote].copy()

    s_f2 = rng.standard_normal(dim).astype(float)
    s_f2[IDX_DEPTH_F0] = s_f2[IDX_DEPTH_F1] = s_f2[IDX_DEPTH_F3] = 0.0
    s_f2[IDX_DEPTH_F2] = 1.0
    for catastrophic_reward in (-3.4, -3.49, -3.4, -3.49, -3.4):
        dc_agent.update(s_f2, "promote", catastrophic_reward)

    # F2's promote arm must have absorbed the damage...
    assert dc_agent._n_updates[arm_f2_promote] == 5
    assert not np.allclose(dc_agent._b[arm_f2_promote], 0.0), \
        "F2 promote arm should have been updated by the catastrophic rewards"
    # ...while F0's promote arm is untouched byte-for-byte (real isolation,
    # not just "small effect" — these are disjoint parameter arrays).
    assert np.array_equal(dc_agent._b[arm_f0_promote], b_f0_before), \
        "F0 promote arm must be exactly unaffected by F2 promote updates"
    assert np.array_equal(dc_agent._A[arm_f0_promote], A_f0_before), \
        "F0 promote arm's A matrix must be exactly unaffected by F2 promote updates"
    assert dc_agent._n_updates[arm_f0_promote] == 0

    # ── Regression: reward clipping bounds a single outlier's influence ───────
    clip_agent = PromotionAgent(dim=dim, alpha=1.0, seed=2, reward_clip=5.0)
    s_clip = rng.standard_normal(dim).astype(float)
    arm0 = clip_agent._arm(clip_agent.actions.index("kill"), 0)
    clip_agent.update(s_clip, "kill", 1000.0)   # full-fail-scale outlier
    assert np.allclose(clip_agent._b[arm0], 5.0 * s_clip), \
        "reward_clip=5.0 must clip a +1000 reward to +5.0 before the b-update"
    clip_agent2 = PromotionAgent(dim=dim, alpha=1.0, seed=2, reward_clip=None)
    arm0b = clip_agent2._arm(clip_agent2.actions.index("kill"), 0)
    clip_agent2.update(s_clip, "kill", 1000.0)
    assert np.allclose(clip_agent2._b[arm0b], 1000.0 * s_clip), \
        "reward_clip=None must disable clipping"

    # ── Regression: epsilon floor prevents permanent collapse under a
    # likith-like near-featureless, ~95%-catastrophic F2->F3 reward regime.
    # Depth-conditioning + clipping alone were verified (above) to isolate
    # the damage, but a stress test before shipping showed pure LinUCB still
    # rationally locks the isolated F2 arm to "always kill" after 1-3 samples
    # when every F2-depth decision sees the same near-identical context
    # (true for combinational designs: the F2 proxy WNS feature is pinned at
    # 0.0, so the real timing-margin signal doesn't exist until F3 runs).
    # epsilon>0 must keep a nonzero long-run promotion rate regardless.
    def _run_likith_like(epsilon: float, seed: int, n_episodes: int = 800) -> float:
        a = PromotionAgent(dim=dim, alpha=1.0, seed=seed, epsilon=epsilon)
        er = np.random.default_rng(seed + 500)
        s_f2 = np.zeros(dim, dtype=float)
        s_f2[IDX_DEPTH_F2] = 1.0
        promotes_2nd_half = []
        for ep in range(n_episodes):
            act_choice = a.act(s_f2)
            if act_choice == "promote":
                r = er.normal(-3.4, 0.05) if er.random() < 0.95 else er.normal(1.0, 0.2)
            else:
                r = 0.0
            a.update(s_f2, act_choice, r)
            if ep >= n_episodes // 2:
                promotes_2nd_half.append(1 if act_choice == "promote" else 0)
        return float(np.mean(promotes_2nd_half))

    no_floor_rate = _run_likith_like(epsilon=0.0, seed=7)
    with_floor_rate = _run_likith_like(epsilon=0.03, seed=7)
    assert no_floor_rate == 0.0, \
        (f"expected epsilon=0.0 to reproduce the collapse (0% 2nd-half "
         f"promote rate) under the likith-like stress regime, got {no_floor_rate}")
    assert with_floor_rate > 0.0, \
        (f"epsilon=0.03 must keep a nonzero F2->F3 promote rate under the "
         f"same stress regime that collapses without it, got {with_floor_rate}")

    # epsilon=0 must never override the UCB argmax: once one arm is trained to
    # a clear, unambiguous winner (no tie), repeated act() calls on the same
    # state must all return it. (An *untrained* agent's arms all start tied
    # at theta=0 with identical A_inv, so ties — and their random
    # tie-break — are expected there; that's unrelated to epsilon.)
    det_agent = PromotionAgent(dim=dim, alpha=0.01, seed=0, epsilon=0.0)
    s_det = np.zeros(dim, dtype=float); s_det[IDX_DEPTH_F0] = 1.0
    for _ in range(10):
        det_agent.update(s_det, "promote", 10.0)   # make "promote" the clear winner
    votes = {det_agent.act(s_det) for _ in range(30)}
    assert votes == {"promote"}, \
        f"epsilon=0.0 with an unambiguous winner must always return it, got {votes}"

    # ── Legacy (non-depth-conditioned) mode still selectable and functional ──
    legacy_agent = PromotionAgent(dim=dim, alpha=1.0, seed=3,
                                   depth_conditioned=False, reward_clip=None)
    assert len(legacy_agent._A) == len(legacy_agent.actions), \
        "legacy mode should have exactly one arm per action, no depth split"
    for _ in range(20):
        s = rng.standard_normal(dim)
        a = legacy_agent.act(s)
        assert a in actions
        legacy_agent.update(s, a, rng.standard_normal())

    # FixedGateAgent — one state per depth level
    # Default threshold is -0.5 (normalised, matching FunnelEnv state[10]=wns_ns/5)
    fg = FixedGateAgent()
    assert fg.proxy_wns_kill_threshold == FixedGateAgent._NORM_WNS_KILL, \
        f"default threshold should be {FixedGateAgent._NORM_WNS_KILL}, got {fg.proxy_wns_kill_threshold}"

    # F0 depth, real sub-threshold accuracy (e.g. tinymac acc_w=16 → 0.73) → kill.
    # NOTE: 0.0 is the *no-data sentinel* (generic designs have no F0 accuracy);
    # it must PROMOTE, not kill — using 0.0 here was a pre-existing test bug.
    s = np.zeros(dim); s[IDX_DEPTH_F0] = 1.0; s[IDX_F0_ACC] = 47.0 / 64.0
    assert fg.act(s) == "kill", "F0 measured-low acc (0.73) should kill"

    # F0 depth, no-data sentinel (0.0) → promote (generic designs)
    s[IDX_F0_ACC] = 0.0
    assert fg.act(s) == "promote", "F0 no-data sentinel (0.0) should promote"

    # F0 depth, high accuracy → promote
    s[IDX_F0_ACC] = 1.0
    assert fg.act(s) == "promote", "F0 high acc should promote"

    # F1 depth, low accuracy → kill
    s = np.zeros(dim); s[IDX_DEPTH_F1] = 1.0; s[IDX_F1_ACC] = 0.5
    assert fg.act(s) == "kill", "F1 low acc should kill"

    # F2 depth: normalised WNS test (FunnelEnv state = wns_ns/5)
    # raw -3.0 ns → normalised -3.0/5 = -0.6 < -0.5 threshold → kill
    s = np.zeros(dim); s[IDX_DEPTH_F2] = 1.0; s[IDX_F2_WNS] = -0.6   # norm: -3.0ns/5
    assert fg.act(s) == "kill", "F2 normalised WNS -0.6 should kill (raw -3.0 ns)"

    # raw +0.5 ns → normalised +0.1 > -0.5 → promote
    s[IDX_F2_WNS] = 0.1
    assert fg.act(s) == "promote", "F2 normalised WNS 0.1 should promote (raw +0.5 ns)"

    # H1 regression: catastrophic timing (raw WNS <= -10 ns) clips to the -2.0
    # normalised floor — it MUST kill, not escape to a full F3 build.
    s = np.zeros(dim); s[IDX_DEPTH_F2] = 1.0; s[IDX_F2_WNS] = -2.0   # clipped from raw <= -10 ns
    assert fg.act(s) == "kill", "F2 clipped WNS -2.0 (raw <= -10 ns) must kill, not escape (H1)"
    # F2 just-run, timing met (norm 0.0) must promote (not a false kill).
    s[IDX_F2_WNS] = 0.0
    assert fg.act(s) == "promote", "F2 WNS 0.0 (timing met) should promote"

    # F11 regression: sub-ns-platform-shaped state (likith/asap7, clk O(0.05 ns)).
    # env.py normalises WNS by the design's clock period for generic designs
    # (F10), so state[10] = wns/clk — the raw O(±0.003 ns) magnitudes that made
    # the old absolute -2.5 ns gate inert become clock-relative fractions here.
    # wns = -0.6·clk → state[10] = -0.6 < -0.5 → MUST kill (was always-promote).
    s = np.zeros(dim); s[IDX_DEPTH_F2] = 1.0; s[IDX_F2_WNS] = -0.6
    assert fg.act(s) == "kill", \
        "F11: sub-ns wns = -0.6*clk (clock-normalised state) must kill"
    # wns = -0.1·clk → state[10] = -0.1 > -0.5 → MUST promote (real slack margin).
    s[IDX_F2_WNS] = -0.1
    assert fg.act(s) == "promote", \
        "F11: sub-ns wns = -0.1*clk (clock-normalised state) must promote"
    # Guard the constant wiring: the clock-fraction and normalised threshold agree.
    assert FixedGateAgent._NORM_WNS_KILL == -FixedGateAgent._WNS_KILL_CLK_FRACTION
    assert FixedGateAgent._RAW_WNS_KILL_NS == FixedGateAgent._NORM_WNS_KILL * 5.0

    # Also verify with raw-WNS mode (state[10] carries raw ns)
    fg_raw = FixedGateAgent(proxy_wns_kill_threshold=-2.5)  # raw ns
    s_raw = np.zeros(dim); s_raw[IDX_DEPTH_F2] = 1.0; s_raw[IDX_F2_WNS] = -3.0  # raw ns
    assert fg_raw.act(s_raw) == "kill", "F2 raw WNS -3.0 ns should kill (raw mode)"
    s_raw[IDX_F2_WNS] = 0.5
    assert fg_raw.act(s_raw) == "promote", "F2 raw WNS +0.5 ns should promote (raw mode)"

    # F3 depth → commit
    s = np.zeros(dim); s[IDX_DEPTH_F3] = 1.0
    assert fg.act(s) == "commit", "F3 depth should commit"

    # RandomPromotionAgent
    ra = RandomPromotionAgent(seed=99)
    seen = set()
    for _ in range(200):
        a = ra.act(np.zeros(dim))
        assert a in actions
        seen.add(a)
    assert len(seen) > 1, "random agent should try multiple actions"

    print("promotion_agent.py self-test: PASS")


if __name__ == "__main__":
    _selftest()
