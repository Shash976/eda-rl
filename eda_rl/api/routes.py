"""routes.py — the eda-rl dashboard REST API.

Purely read-only over already-committed campaign logs (``eda_rl/campaigns/
<design>/<platform>/results_funnel_campaigns.jsonl``): nothing here triggers
a new build, an F3 run, or an ``eda-rl collect`` baseline build — those are
real ORFS/mock tool invocations that take minutes, far too heavy for a
request/response API.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Query

from eda_rl.funnel.collect_best import select_best
from eda_rl.viz.campaign_data import (
    CampaignData,
    earliest_f3,
    episode_value,
    list_design_platform_pairs,
    pareto_front,
    pareto_points,
    report_extension_for,
    resolve_log_path,
    running_max,
)

from .cache import load_cached
from .schemas import (
    BaselineOut,
    BestConfigOut,
    CampaignSummary,
    DesignPlatformPair,
    FunnelCounts,
    HistoryPoint,
    ParamSpecOut,
    ParetoPoint,
)

router = APIRouter(prefix="/api")

# design/platform come straight off the URL path — they double as filesystem
# path segments in resolve_log_path, so a value like ".." must never reach it
# (Starlette's default path converter already excludes "/", but not "..").
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")


def _resolve(design: str, platform: str, campaign: str | None) -> CampaignData:
    if not (_SAFE_SEGMENT.match(design) and _SAFE_SEGMENT.match(platform)):
        raise HTTPException(status_code=400, detail="invalid design/platform name")
    try:
        log_path = resolve_log_path(None, design, platform)
    except SystemExit as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not log_path.exists():
        raise HTTPException(
            status_code=404, detail=f"no campaign log for {design}/{platform}"
        )
    return load_cached(log_path, campaign)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/designs", response_model=list[DesignPlatformPair])
def designs() -> list[DesignPlatformPair]:
    return [DesignPlatformPair(design=d, platform=p) for d, p in list_design_platform_pairs()]


@router.get("/campaigns/{design}/{platform}", response_model=CampaignSummary)
def campaign_summary(
    design: str, platform: str, campaign: str | None = Query(default=None)
) -> CampaignSummary:
    data = _resolve(design, platform, campaign)
    values = [v for v in data.values() if v is not None]
    return CampaignSummary(
        design=design,
        platform=platform,
        campaign_id=data.campaign_id,
        n_episodes=len(data.rows),
        fidelity_counts=FunnelCounts(**data.fidelity_counts()),
        best_value=max(values) if values else None,
        params={
            name: ParamSpecOut(
                name=spec.name,
                kind=spec.kind,
                low=spec.low,
                high=spec.high,
                choices=[str(c) for c in spec.choices],
            )
            for name, spec in data.specs.items()
        },
    )


@router.get("/campaigns/{design}/{platform}/episodes")
def campaign_episodes(
    design: str, platform: str, campaign: str | None = Query(default=None)
) -> list[dict]:
    return _resolve(design, platform, campaign).rows


@router.get("/campaigns/{design}/{platform}/pareto", response_model=list[ParetoPoint])
def campaign_pareto(
    design: str, platform: str, campaign: str | None = Query(default=None)
) -> list[ParetoPoint]:
    rows = _resolve(design, platform, campaign).rows
    pts = pareto_points(rows)
    frontier_ids = {id(r) for _, _, r in pareto_front(pts)}
    return [
        ParetoPoint(area_um2=a, fmax_mhz=f, is_frontier=id(r) in frontier_ids, row=r)
        for a, f, r in pts
    ]


@router.get("/campaigns/{design}/{platform}/funnel", response_model=FunnelCounts)
def campaign_funnel(
    design: str, platform: str, campaign: str | None = Query(default=None)
) -> FunnelCounts:
    return FunnelCounts(**_resolve(design, platform, campaign).fidelity_counts())


@router.get("/campaigns/{design}/{platform}/history", response_model=list[HistoryPoint])
def campaign_history(
    design: str, platform: str, campaign: str | None = Query(default=None)
) -> list[HistoryPoint]:
    data = _resolve(design, platform, campaign)
    values = data.values()
    bests = running_max(values)
    return [
        HistoryPoint(episode=r.get("episode"), ts=r.get("ts"), value=v, running_max=b)
        for r, v, b in zip(data.rows, values, bests)
    ]


@router.get("/campaigns/{design}/{platform}/best", response_model=list[BestConfigOut])
def campaign_best(
    design: str,
    platform: str,
    campaign: str | None = Query(default=None),
    top: int = Query(default=3, ge=1, le=20),
) -> list[BestConfigOut]:
    rows = _resolve(design, platform, campaign).rows
    picks = select_best(rows, top=top)
    return [
        BestConfigOut(
            badge=r["_badge"],
            sublabel=r["_sublabel"],
            variant=r["_variant"],
            config=r.get("config") or {},
            obs=r.get("obs") or {},
            score=episode_value(r),
        )
        for r in picks
    ]


@router.get("/campaigns/{design}/{platform}/baseline", response_model=BaselineOut)
def campaign_baseline(
    design: str, platform: str, campaign: str | None = Query(default=None)
) -> BaselineOut:
    rows = _resolve(design, platform, campaign).rows
    ext = report_extension_for(rows)
    return BaselineOut(
        earliest_f3=earliest_f3(rows),
        hand_picked_baseline=getattr(ext, "baseline", None) if ext is not None else None,
    )
