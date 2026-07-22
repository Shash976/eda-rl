"""functional_models/base.py — the FunctionalModel plugin interface.

A design opts into a functional model by declaring ``functional_eval.kind`` in
its YAML.  The registry (``functional_models/__init__.py``) maps that kind to a
``FunctionalModel`` instance; core code (env, surrogate, report, build_table)
dispatches through this interface and never names a specific design family.

Designs with no functional model (kind ``"none"`` or absent) resolve to ``None``
and take the generic PPA path everywhere (``compute_generic_reward``, F1 skipped,
default report rendering).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FunctionalModel(ABC):
    """Plugin describing a design family's behavioral model + composite reward.

    Subclasses set ``kind`` (== the YAML ``functional_eval.kind``), ``reward_kind``
    (the tag stored in logs / surrogate joblib schemas), and ``enables_f1``.
    """

    #: Registry key — must equal the design YAML's ``functional_eval["kind"]``.
    kind: str = ""
    #: Reward tag stored in logs and surrogate joblib schemas (e.g. "composite").
    reward_kind: str = ""
    #: Whether the F1 behavioral-sim fidelity gate runs for this model.
    enables_f1: bool = True
    #: Optional reference cycle count for a "speedup vs software" report field
    #: (None when the model has no software baseline).
    sw_baseline_cycles: "int | None" = None

    # ── F0 analytic model ────────────────────────────────────────────────────
    def f0_cycles(self, config: dict) -> float:
        """Analytic cycle estimate for the F0 gate (0.0 == no analytic model)."""
        return 0.0

    def f0_accuracy(self, config: dict) -> float:
        """Analytic accuracy estimate for the F0 gate.

        Returns 0.0 as the "no-data" sentinel when the model has no analytic
        accuracy table (the promotion agent treats 0.0 as promote, not kill).
        """
        return 0.0

    # ── F1 behavioral sim ────────────────────────────────────────────────────
    @abstractmethod
    def run_f1(self, config: dict, *, mock: bool) -> dict:
        """Run the behavioral sim for ``config``.

        Returns an obs dict with at least ``avg_cycles`` and ``accuracy`` (and,
        for the composite reward, ``correct``/``n_total``).  ``mock`` selects the
        synthetic path (no external binary required).
        """

    # ── Terminal (F3) reward ─────────────────────────────────────────────────
    @abstractmethod
    def terminal_reward(
        self, obs: dict, weights: dict | None = None, *, cycles: float | None = None
    ) -> dict:
        """Composite F3 reward dict (``{"reward": float, ...}``) for this model."""

    # ── Surrogate UCB mirror ─────────────────────────────────────────────────
    @abstractmethod
    def surrogate_reward(self, x: dict, preds: dict) -> tuple[float, float]:
        """(mu, sigma) of the composite reward from surrogate metric predictions.

        ``preds`` maps metric name → (mu, sigma) for ``area_um2`` / ``period_ns``
        / ``power_mw`` (the surrogate's per-metric heads).
        """

    # ── Report rendering (optional) ──────────────────────────────────────────
    def report_extension(self) -> "Any | None":
        """Return a report extension object for design-specific rendering, or
        ``None`` to use the generic PPA report.  Loaded lazily so the registry
        never pulls in plotting dependencies at import time.
        """
        return None
