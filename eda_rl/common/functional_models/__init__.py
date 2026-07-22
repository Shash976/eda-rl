"""functional_models — plugin registry for design functional-evaluation models.

A design opts into a functional model via its YAML ``functional_eval.kind``; the
registry maps that kind to a ``FunctionalModel`` instance.  Designs with no
functional model (kind ``"none"`` or absent) resolve to ``None`` and take the
generic PPA path everywhere.

Public API:
    register(model)          — add a plugin (built-ins register at import time)
    for_kind(kind)           — model by its registry key (== functional_eval.kind)
    for_reward_kind(tag)     — model by its ``reward_kind`` tag (surrogate/logs)
    for_design(design)       — model for a DesignSpec (reads functional_eval.kind)
    available()              — sorted list of registered kinds
"""

from __future__ import annotations

from typing import Any

from eda_rl.common.functional_models.base import FunctionalModel

_REGISTRY: dict[str, FunctionalModel] = {}


def register(model: FunctionalModel) -> FunctionalModel:
    """Register a FunctionalModel under its ``kind``.  Returns the model."""
    if not model.kind:
        raise ValueError("FunctionalModel.kind must be non-empty to register")
    _REGISTRY[model.kind] = model
    return model


def for_kind(kind: str | None) -> FunctionalModel | None:
    """Return the model registered for ``kind`` (the YAML functional_eval.kind)."""
    if not kind:
        return None
    return _REGISTRY.get(kind)


def for_reward_kind(reward_kind: str | None) -> FunctionalModel | None:
    """Return the model whose ``reward_kind`` tag matches (surrogate/log lookup)."""
    if not reward_kind:
        return None
    for model in _REGISTRY.values():
        if model.reward_kind == reward_kind:
            return model
    return None


def for_design(design: Any) -> FunctionalModel | None:
    """Return the functional model a DesignSpec opts into, or None (generic)."""
    fe = getattr(design, "functional_eval", None) or {}
    return for_kind(fe.get("kind"))


def available() -> list[str]:
    """Sorted list of registered functional-model kinds."""
    return sorted(_REGISTRY)


# ── Built-in plugins (register on import) ─────────────────────────────────────
from eda_rl.common.functional_models.tinyvad import TinyVADModel  # noqa: E402

register(TinyVADModel())
