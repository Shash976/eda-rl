"""cache.py — mtime-keyed cache over CampaignData.load.

Campaign logs (e.g. sagar/sky130hd) run several MB; reparsing one on every
request is wasteful for read-only data that only changes when a new campaign
appends to it. Keying the cache on the file's mtime (not just its path) means
a live campaign still being appended to is picked up on its next request
without needing an explicit invalidation call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from eda_rl.viz.campaign_data import CampaignData


@lru_cache(maxsize=32)
def _load_at(path_str: str, campaign: str | None, mtime: float) -> CampaignData:
    return CampaignData.load(path_str, campaign)


def load_cached(log_path: Path, campaign: str | None = None) -> CampaignData:
    mtime = log_path.stat().st_mtime
    return _load_at(str(log_path), campaign, mtime)
