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

State vector layout: see eda_rl/gen2/state_spec.py — the single source of truth
(imported below).  The agents below read it through the IDX_* constants so they
can never drift from what FunnelEnv._build_state actually emits (audit H3/H4).
Key slots the gates use: [6] F0 accuracy, [8] F1 accuracy, [10] F2 wns_norm
(= clip(wns_ns/5, -2, 2)), [18..21] depth one-hot.  The unrun convention is 0.0.

FixedGateAgent mapping to cascade.py / cascade_reward.py gates:
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
        FunnelEnv stores d10 = clip(wns_ns/5, -2, 2), so the calibrated raw
        threshold of -2.5 ns (Phase 5 Exp 3) maps to -2.5/5 = -0.5.
        proxy_wns_kill_threshold is stored and checked in the NORMALISED
        units FunnelEnv produces (-0.5, not -2.5).
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

import os
import random
from pathlib import Path

import numpy as np

# ── type alias for clarity ────────────────────────────────────────────────────
_ActionTuple = tuple[str, ...]
_DEFAULT_ACTIONS: _ActionTuple = ("kill", "re-proxy", "promote", "commit")

# ── state vector indices — re-exported from the canonical state_spec module ───
from eda_rl.gen2.state_spec import (  # noqa: E402,F401
    STATE_DIM,
    IDX_LANES_NORM, IDX_ACCW_NORM, IDX_CLK_NORM,
    IDX_RECIPE, IDX_PLATFORM, IDX_RECIPE_SPD, IDX_RECIPE_AREA,
    IDX_F0_CYCLES, IDX_F0_ACC, IDX_F1_CYCLES, IDX_F1_ACC,
    IDX_F2_AREA, IDX_F2_WNS, IDX_F2_FF, IDX_F2_CELLS, IDX_F2_LEVELS,
    IDX_SURR_MU, IDX_SURR_SIG, IDX_INCUMBENT, IDX_BUDGET_FRAC,
    IDX_DEPTH_F0, IDX_DEPTH_F1, IDX_DEPTH_F2, IDX_DEPTH_F3,
)


# ── LinUCB contextual bandit ──────────────────────────────────────────────────

class PromotionAgent:
    """LinUCB contextual bandit over the 4 funnel actions.

    Standard disjoint LinUCB (one linear model per arm):
        theta_a = A_a^{-1} b_a
        UCB(a, s) = theta_a^T s + alpha * sqrt(s^T A_a^{-1} s)
        action = argmax_a UCB(a, s)

    Per-arm update on reward r, context s, arm a:
        A_a <- A_a + s s^T
        b_a <- b_a + r * s

    Parameters
    ----------
    dim   : context dimension (must match state vector; default 22)
    alpha : exploration coefficient (UCB width; default 1.0)
    seed  : RNG seed for tie-breaking
    actions : tuple of action strings; must be a superset of the FunnelEnv actions
    lam   : ridge regularisation for initial A (A_a = lam * I); prevents
            singular A before observations arrive; default 1.0
    """

    def __init__(
        self,
        dim: int = STATE_DIM,
        alpha: float = 1.0,
        seed: int = 0,
        actions: _ActionTuple = _DEFAULT_ACTIONS,
        lam: float = 1.0,
    ) -> None:
        self.dim = dim
        self.alpha = float(alpha)
        self.actions = tuple(actions)
        self.lam = float(lam)
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        n_actions = len(self.actions)
        # Per-arm precision matrix A_a (dim × dim) and reward vector b_a (dim,)
        # A_a starts as lam * I; b_a starts at zero.
        self._A: list[np.ndarray] = [np.eye(dim) * lam for _ in range(n_actions)]
        self._b: list[np.ndarray] = [np.zeros(dim) for _ in range(n_actions)]
        # Cached inverse (invalidated on update)
        self._A_inv: list[np.ndarray | None] = [None] * n_actions
        # Track update counts for logging
        self._n_updates: list[int] = [0] * n_actions

    # ── core interface ─────────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> str:
        """Select an action given the 22-dim state vector.

        Returns the action string with the highest UCB score, breaking ties
        randomly (seeded) to ensure reproducibility.
        """
        s = np.asarray(state, dtype=float).reshape(-1)
        if len(s) < self.dim:
            # Pad with zeros if state is shorter than expected (defensive)
            s = np.pad(s, (0, self.dim - len(s)))
        elif len(s) > self.dim:
            s = s[: self.dim]

        ucb_scores = []
        for i, action in enumerate(self.actions):
            A_inv = self._get_A_inv(i)
            theta = A_inv @ self._b[i]
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

        Only the arm corresponding to `action` is updated (disjoint LinUCB).
        Invalid action strings are silently ignored (defensive).

        Audit M6 (experiment, not a fix): the "promote" arm absorbs both the big
        terminal F3 payoff (at depth F2→F3) and tiny per-step shaping (at F0→F1,
        F1→F2), and rewards span ~[-100, +4] unscaled.  If the benchmark shows the
        bandit's myopia/mixed-regime estimate loses to fixed gates, try
        depth-conditioned arms or reward normalisation here — but measure first.
        """
        if action not in self.actions:
            return
        idx = self.actions.index(action)
        s = np.asarray(state, dtype=float).reshape(-1)
        if len(s) < self.dim:
            s = np.pad(s, (0, self.dim - len(s)))
        elif len(s) > self.dim:
            s = s[: self.dim]

        self._A[idx] += np.outer(s, s)
        self._b[idx] += float(reward) * s
        self._A_inv[idx] = None   # invalidate cached inverse
        self._n_updates[idx] += 1

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save agent parameters to a .npz file."""
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        for i, action in enumerate(self.actions):
            key = action.replace("-", "_")   # "re-proxy" → "re_proxy"
            arrays[f"A_{key}"] = self._A[i]
            arrays[f"b_{key}"] = self._b[i]
        # Metadata as 0-d arrays
        arrays["dim"] = np.array(self.dim)
        arrays["alpha"] = np.array(self.alpha)
        arrays["lam"] = np.array(self.lam)
        arrays["n_updates"] = np.array(self._n_updates)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "PromotionAgent":
        """Load agent from a .npz file produced by save()."""
        data = np.load(str(path))
        dim = int(data["dim"])
        alpha = float(data["alpha"])
        lam = float(data["lam"])
        agent = cls(dim=dim, alpha=alpha, seed=seed, lam=lam)
        for i, action in enumerate(agent.actions):
            key = action.replace("-", "_")
            agent._A[i] = data[f"A_{key}"]
            agent._b[i] = data[f"b_{key}"]
            agent._A_inv[i] = None
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
        updates = dict(zip(self.actions, self._n_updates))
        return (f"PromotionAgent(dim={self.dim}, alpha={self.alpha}, "
                f"lam={self.lam}, updates={updates})")


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
        state[10] (F2 proxy_wns_ns, raw value) < -2.5 → "kill"
            (Phase 5 Exp 3 calibrated gate: proxy_wns < -2.5 loses no true
             positives; all 3 timing-miss cases are proxy-pessimistic, meaning
             proxy says fail but full flow meets → safe to kill on proxy timing)
        otherwise → "promote"

    Depth F3 (full flow, state[21]=1):
        always → "commit"
            (we have the full measurement; no benefit to killing or re-proxying)

    Unknown depth (all depth bits zero):
        "promote"  (default: keep moving forward)
    """

    # Raw WNS kill threshold (nanoseconds).  Used to compute the normalised
    # threshold for FunnelEnv (state[10] = clip(wns_ns/5, -2, 2)).
    _RAW_WNS_KILL_NS: float = -2.5   # Phase 5 Exp 3 calibrated value
    # Normalised threshold for FunnelEnv state: raw / 5.0 = -0.5
    _NORM_WNS_KILL: float = _RAW_WNS_KILL_NS / 5.0  # = -0.5

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
    import py_compile, tempfile, os

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
        s_test = rng.standard_normal(dim)
        # after load, the same state must produce the same action
        a1 = agent.act(s_test)
        a2 = agent2.act(s_test)
        assert a1 == a2, f"save/load mismatch: {a1!r} vs {a2!r}"

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
