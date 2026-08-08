"""schemas.py — Pydantic response models for the eda-rl REST API.

Strict models only for shapes this API owns (summaries, funnel counts, param
specs, best-config picks). Per-episode rows and their ``obs`` blocks are
intentionally passed through as ``dict[str, Any]`` — their shape is owned by
whichever design/functional-model plugin logged them (see
``eda_rl/common/functional_models/``), and forcing a fixed schema on that
would either drop design-specific fields or require one schema per design.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DesignPlatformPair(BaseModel):
    design: str
    platform: str


class ParamSpecOut(BaseModel):
    name: str
    kind: str
    low: float
    high: float
    choices: list[str]


class FunnelCounts(BaseModel):
    F0: int
    F1: int
    F2: int
    F3: int


class CampaignSummary(BaseModel):
    design: str
    platform: str
    campaign_id: str
    n_episodes: int
    fidelity_counts: FunnelCounts
    best_value: float | None
    params: dict[str, ParamSpecOut]


class ParetoPoint(BaseModel):
    area_um2: float
    fmax_mhz: float
    is_frontier: bool
    row: dict[str, Any]


class HistoryPoint(BaseModel):
    episode: int | None
    ts: float | None
    value: float | None
    running_max: float | None


class BestConfigOut(BaseModel):
    badge: str
    sublabel: str
    variant: str
    config: dict[str, Any]
    obs: dict[str, Any]
    score: float | None


class BaselineOut(BaseModel):
    earliest_f3: dict[str, Any] | None
    hand_picked_baseline: dict[str, Any] | None
