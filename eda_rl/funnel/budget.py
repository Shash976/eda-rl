"""budget.py — what bounds a campaign, and how that bound is exposed to the RL.

Why this module exists
----------------------
Before it, a campaign had exactly one stop condition: ``env.spent_s < budget_s``.
That single float was doing three unrelated jobs at once:

1. **the stop condition** — when to end the campaign;
2. **the shaping normaliser** — the per-step cost penalty is
   ``-lambda * cost_s / budget_s`` (``env.py`` ``_run_stage``);
3. **state slot [17]** — the remaining-budget fraction the promotion policy sees
   (``state_spec.IDX_BUDGET_FRAC``).

Overloading one number for all three is why "run 50 full builds" was not
expressible: you cannot set a build count without also rescaling every step
reward and the policy's budget signal.  This module separates the three roles so
a campaign can be bounded by **wall-clock time**, by **successful F3 build
count** (AutoTuner's ``--samples``), or by both — whichever trips first.

It also fixes a smaller honesty bug.  ``env.spent_s`` accumulates only *stage*
costs; candidate generation, Optuna ask/tell, logging and Python overhead are
never charged.  So ``--budget-hours 4`` was a **tool-time** budget that ran
longer than four hours of wall clock (a real campaign logged ``elapsed_s``
26123.9 against ``simulated_s`` 28804.2).  ``wall_s`` is genuine elapsed time;
``tool_s`` preserves the historical tool-time meaning.

Bit-compatibility
-----------------
Saved LinUCB agents and offline tables depend on the exact state layout and
reward scale, so **a time-only budget must reproduce the old numbers exactly**:
``remaining_fraction`` returns ``max(0, 1 - tool_spent_s / max(tool_s, 1.0))``
and ``shaping_normalizer_s`` returns ``tool_s``, unchanged.  The self-test at the
bottom asserts this.
"""

from __future__ import annotations

from dataclasses import dataclass

# Nominal cost of one F3 build, used to give count-mode a seconds-scale for the
# shaping term before any real build has been timed.  Mirrors
# env.FIDELITY_COST_S["F3"]; duplicated rather than imported to keep this module
# dependency-free (env imports budget, not the other way round).
DEFAULT_F3_COST_S = 420.0

# Safety multiplier: with --max-f3 N and no explicit attempt cap, stop after
# 3*N attempts so a design that fails every build cannot run forever.
_ATTEMPT_SAFETY_FACTOR = 3

# Safety wall-clock cap applied when the user asks for a build count but names no
# time limit at all.
DEFAULT_COUNT_MODE_WALL_S = 24 * 3600.0

# No-progress guard: stop after this many consecutive episodes that never reached
# F3 at all.  A count-bounded campaign whose promotion policy kills everything
# would otherwise spin until the wall cap and produce nothing — the exact shape of
# the observed LinUCB collapse (7,884 consecutive kills in a real likith run).
# The attempt cap cannot catch it because no attempt is ever made.
#
# Sized from real campaign logs: the longest legitimate kill streak observed
# across all committed corpora is 1,202 episodes (and that campaign was itself
# pathological); a healthy heavy-screening run peaks around 230.  2,000 is
# therefore well clear of normal operation.
DEFAULT_MAX_EPISODES_WITHOUT_F3 = 2000


@dataclass
class Budget:
    """The bound on a campaign, in whichever terms the user expressed it.

    Every field is optional; an unset field is simply not a limit.  At least one
    must be set (``__post_init__`` enforces it) or the campaign would never stop.

    wall_s
        Real elapsed seconds since the campaign started.
    tool_s
        Accumulated *stage* seconds (the historical ``budget_s``).  Set this — and
        only this — for behaviour identical to pre-Budget releases.
    max_f3_ok
        Stop after this many **successful** F3 builds (status ok/mock).  Failed,
        timed-out and aborted builds do not consume the quota, so "give me 50
        data points" reliably yields 50.
    max_f3_attempts
        Hard cap on F3 *entries* regardless of outcome.  Defaults to
        ``3 * max_f3_ok`` when a success quota is set, so a 100%-failing design
        terminates instead of retrying forever.
    """

    wall_s: float | None = None
    tool_s: float | None = None
    max_f3_ok: int | None = None
    max_f3_attempts: int | None = None
    max_episodes_without_f3: int | None = DEFAULT_MAX_EPISODES_WITHOUT_F3

    def __post_init__(self) -> None:
        for name in ("wall_s", "tool_s"):
            v = getattr(self, name)
            if v is not None:
                v = float(v)
                if v <= 0.0:
                    raise ValueError(f"{name} must be > 0, got {v}")
                setattr(self, name, v)
        for name in ("max_f3_ok", "max_f3_attempts", "max_episodes_without_f3"):
            v = getattr(self, name)
            if v is not None:
                v = int(v)
                if v <= 0:
                    raise ValueError(f"{name} must be > 0, got {v}")
                setattr(self, name, v)

        if self.max_f3_ok is not None and self.max_f3_attempts is None:
            self.max_f3_attempts = self.max_f3_ok * _ATTEMPT_SAFETY_FACTOR
        if self.max_f3_ok is not None and self.wall_s is None and self.tool_s is None:
            # A pure count budget with no clock is an unbounded-time promise.
            self.wall_s = DEFAULT_COUNT_MODE_WALL_S

        if not self._active():
            raise ValueError(
                "Budget has no limits — a campaign would never stop. Set at least "
                "one of wall_s / tool_s / max_f3_ok."
            )

    # ── introspection ─────────────────────────────────────────────────────────

    def _active(self) -> bool:
        # max_episodes_without_f3 is a safety net, not a budget: a Budget that has
        # only that is still unbounded in the dimension the user cares about.
        return any(v is not None for v in
                   (self.wall_s, self.tool_s, self.max_f3_ok, self.max_f3_attempts))

    @property
    def is_count_mode(self) -> bool:
        """True when a build count is the headline limit (time is a safety cap)."""
        return self.max_f3_ok is not None

    @property
    def is_legacy_time_only(self) -> bool:
        """True for the historical tool-time-only budget (the bit-compatible path)."""
        return (self.tool_s is not None and self.wall_s is None
                and self.max_f3_ok is None and self.max_f3_attempts is None)

    # ── the three roles ───────────────────────────────────────────────────────

    def exhausted(self, *, wall_elapsed_s: float = 0.0, tool_spent_s: float = 0.0,
                  n_f3_ok: int = 0, n_f3_attempts: int = 0,
                  episodes_since_f3: int = 0) -> bool:
        """Role 1: has any active limit been reached?"""
        return self.stop_reason(
            wall_elapsed_s=wall_elapsed_s, tool_spent_s=tool_spent_s,
            n_f3_ok=n_f3_ok, n_f3_attempts=n_f3_attempts,
            episodes_since_f3=episodes_since_f3,
        ) is not None

    def stop_reason(self, *, wall_elapsed_s: float = 0.0, tool_spent_s: float = 0.0,
                    n_f3_ok: int = 0, n_f3_attempts: int = 0,
                    episodes_since_f3: int = 0) -> str | None:
        """Which limit ended the campaign (for the summary row and the console)."""
        if self.tool_s is not None and tool_spent_s >= self.tool_s:
            return "tool_time"
        if self.wall_s is not None and wall_elapsed_s >= self.wall_s:
            return "wall_time"
        if self.max_f3_ok is not None and n_f3_ok >= self.max_f3_ok:
            return "f3_count"
        if self.max_f3_attempts is not None and n_f3_attempts >= self.max_f3_attempts:
            return "f3_attempts"
        if (self.max_episodes_without_f3 is not None
                and episodes_since_f3 >= self.max_episodes_without_f3):
            return "no_f3_progress"
        return None

    def remaining_fraction(self, *, wall_elapsed_s: float = 0.0,
                           tool_spent_s: float = 0.0, n_f3_ok: int = 0,
                           n_f3_attempts: int = 0) -> float:
        """Role 3: what state slot [17] sees — fraction of the budget left.

        With several active limits the **most binding** one wins, so the policy
        always sees the constraint that will actually end the run.
        """
        fracs: list[float] = []
        if self.tool_s is not None:
            fracs.append(1.0 - tool_spent_s / max(self.tool_s, 1.0))
        if self.wall_s is not None:
            fracs.append(1.0 - wall_elapsed_s / max(self.wall_s, 1.0))
        if self.max_f3_ok is not None:
            fracs.append(1.0 - n_f3_ok / float(self.max_f3_ok))
        if self.max_f3_attempts is not None:
            fracs.append(1.0 - n_f3_attempts / float(self.max_f3_attempts))
        if not fracs:
            return 1.0
        return max(0.0, min(fracs))

    def shaping_normalizer_s(self, ema_f3_cost_s: float | None = None) -> float:
        """Role 2: the seconds-scale dividing the per-step cost penalty.

        Time mode uses the budget itself (unchanged).  Count mode has no seconds
        budget, so it uses the *expected* total tool time — the build quota times
        the running mean cost of an F3 build — which keeps the shaping term on
        the same order of magnitude as before rather than silently rescaling
        every step reward.
        """
        if self.tool_s is not None:
            return self.tool_s
        if self.max_f3_ok is not None:
            per = ema_f3_cost_s if (ema_f3_cost_s and ema_f3_cost_s > 0) else DEFAULT_F3_COST_S
            return max(1.0, self.max_f3_ok * float(per))
        if self.wall_s is not None:
            return self.wall_s
        return 1.0

    # ── serialisation for the self-describing campaign log ────────────────────

    def describe(self) -> dict:
        return {
            "wall_s": self.wall_s,
            "tool_s": self.tool_s,
            "max_f3_ok": self.max_f3_ok,
            "max_f3_attempts": self.max_f3_attempts,
            "mode": "f3_count" if self.is_count_mode else "time",
        }

    def summary_line(self) -> str:
        bits = []
        if self.max_f3_ok is not None:
            bits.append(f"max_f3={self.max_f3_ok} ok builds "
                        f"(attempt cap {self.max_f3_attempts})")
        if self.wall_s is not None:
            bits.append(f"wall<={self.wall_s / 3600:.2f}h")
        if self.tool_s is not None:
            bits.append(f"tool<={self.tool_s / 3600:.2f}h")
        return ", ".join(bits)

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, *, budget_hours: float | None = None,
                  max_f3: int | None = None,
                  max_f3_attempts: int | None = None,
                  legacy_tool_time: bool = True) -> "Budget":
        """Build from CLI arguments.

        legacy_tool_time
            When True (the default) ``--budget-hours`` keeps its historical
            meaning of *tool* seconds, so existing invocations and saved agents
            behave identically.  In count mode the same flag becomes a wall-clock
            safety cap, which is what a user asking for "50 builds, but stop after
            8 hours" actually means.
        """
        if budget_hours is None and max_f3 is None:
            raise ValueError("specify at least one of --budget-hours / --max-f3")
        secs = float(budget_hours) * 3600.0 if budget_hours is not None else None
        if max_f3 is not None:
            # Count mode: the time limit is a wall-clock safety cap.
            return cls(wall_s=secs, max_f3_ok=max_f3,
                       max_f3_attempts=max_f3_attempts)
        if legacy_tool_time:
            return cls(tool_s=secs)
        return cls(wall_s=secs)


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    fails = 0

    def check(label: str, cond: bool, extra: str = "") -> None:
        global fails
        if cond:
            print(f"  {label}  PASS")
        else:
            fails += 1
            print(f"  {label}  FAIL {extra}")

    print("--- A: legacy time-only mode is bit-compatible ---")
    B = 14400.0
    b = Budget.from_args(budget_hours=4.0)
    check("A1 is_legacy_time_only", b.is_legacy_time_only)
    check("A2 shaping normalizer == budget_s",
          b.shaping_normalizer_s() == B, f"{b.shaping_normalizer_s()} != {B}")
    for spent in (0.0, 1.0, 7200.0, 14399.0, 14400.0, 99999.0):
        old = max(0.0, 1.0 - spent / max(B, 1.0))
        new = b.remaining_fraction(tool_spent_s=spent)
        check(f"A3 remaining_fraction(spent={spent:.0f}) == old formula",
              new == old, f"{new} != {old}")
    check("A4 stops exactly at the budget",
          (not b.exhausted(tool_spent_s=14399.9)) and b.exhausted(tool_spent_s=14400.0))

    print("--- B: count mode counts successes, not attempts ---")
    b = Budget.from_args(max_f3=3)
    check("B1 is_count_mode", b.is_count_mode)
    check("B2 attempt cap defaults to 3x", b.max_f3_attempts == 9)
    check("B3 wall safety cap applied", b.wall_s == DEFAULT_COUNT_MODE_WALL_S)
    check("B4 failures do not consume quota",
          not b.exhausted(n_f3_ok=2, n_f3_attempts=8))
    check("B5 stops at the 3rd success", b.exhausted(n_f3_ok=3, n_f3_attempts=3))
    check("B6 attempt cap terminates a hopeless design",
          b.exhausted(n_f3_ok=0, n_f3_attempts=9))
    check("B7 stop_reason distinguishes them",
          b.stop_reason(n_f3_ok=3) == "f3_count"
          and b.stop_reason(n_f3_attempts=9) == "f3_attempts")

    print("--- C: count-mode state slot [17] and shaping scale ---")
    b = Budget.from_args(max_f3=10)
    check("C1 fraction falls with successful builds",
          b.remaining_fraction(n_f3_ok=0) == 1.0
          and abs(b.remaining_fraction(n_f3_ok=5) - 0.5) < 1e-12
          and b.remaining_fraction(n_f3_ok=10) == 0.0)
    check("C2 fraction never negative", b.remaining_fraction(n_f3_ok=99) == 0.0)
    check("C3 shaping scale seeds from the nominal F3 cost",
          b.shaping_normalizer_s() == 10 * DEFAULT_F3_COST_S)
    check("C4 shaping scale tracks measured cost",
          b.shaping_normalizer_s(ema_f3_cost_s=90.0) == 900.0)

    print("--- D: combined limits take the most binding ---")
    b = Budget(wall_s=1000.0, max_f3_ok=10)
    check("D1 wall binds when time is nearly gone",
          abs(b.remaining_fraction(wall_elapsed_s=900.0, n_f3_ok=1) - 0.1) < 1e-12)
    check("D2 count binds when builds are nearly gone",
          abs(b.remaining_fraction(wall_elapsed_s=100.0, n_f3_ok=9) - 0.1) < 1e-12)
    check("D3 either limit stops the run",
          b.exhausted(wall_elapsed_s=1000.0) and b.exhausted(n_f3_ok=10))

    print("--- E: rejects nonsense ---")
    for kwargs in ({}, {"wall_s": 0.0}, {"tool_s": -5.0}, {"max_f3_ok": 0}):
        try:
            Budget(**kwargs)
            check(f"E rejects {kwargs}", False, "no ValueError raised")
        except ValueError:
            check(f"E rejects {kwargs}", True)
    try:
        Budget.from_args()
        check("E from_args needs a limit", False, "no ValueError raised")
    except ValueError:
        check("E from_args needs a limit", True)

    print()
    if fails:
        print(f"=== budget.py self-test FAILED ({fails} failures) ===")
        sys.exit(1)
    print("=== budget.py self-test PASSED ===")
