#!/usr/bin/env python3
"""Tests for eda_rl.api — the read-only REST API over committed campaign logs.

Exercised against the real, committed ``campaigns/gcd/nangate45/
results_funnel_campaigns.jsonl`` (smallest corpus, fastest) — same
"test against committed corpora, no mocking" convention as
tests/test_log_schema.py: this is the exact data a deployed instance would
serve, not a synthetic stand-in that could pass while the real thing 404s.

Run: pytest tests/test_api.py -v   (or  python3 tests/test_api.py)
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from fastapi.testclient import TestClient  # noqa: E402

from eda_rl.api.server import create_app  # noqa: E402

_CAMPAIGNS = _HERE.parent / "eda_rl" / "campaigns"
_DESIGN, _PLATFORM = "gcd", "nangate45"
_HAS_GCD_LOG = (_CAMPAIGNS / _DESIGN / _PLATFORM / "results_funnel_campaigns.jsonl").exists()

client = TestClient(create_app())


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_designs_lists_gcd():
    r = client.get("/api/designs")
    assert r.status_code == 200
    pairs = {(d["design"], d["platform"]) for d in r.json()}
    assert not _HAS_GCD_LOG or (_DESIGN, _PLATFORM) in pairs


def test_campaign_summary_shape():
    if not _HAS_GCD_LOG:
        return
    r = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}")
    assert r.status_code == 200
    body = r.json()
    assert body["design"] == _DESIGN
    assert body["platform"] == _PLATFORM
    assert body["n_episodes"] > 0
    assert set(body["fidelity_counts"]) == {"F0", "F1", "F2", "F3"}
    assert sum(body["fidelity_counts"].values()) == body["n_episodes"]


def test_pareto_frontier_is_subset_and_non_dominated():
    if not _HAS_GCD_LOG:
        return
    r = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}/pareto")
    assert r.status_code == 200
    points = r.json()
    assert points, "gcd/nangate45 has F3 rows — pareto endpoint must return points"
    frontier = [p for p in points if p["is_frontier"]]
    assert frontier
    # Non-domination: no frontier point is dominated by another point (smaller
    # area AND larger fmax) in the full set.
    for p in frontier:
        dominated_by = [
            q for q in points
            if q is not p and q["area_um2"] <= p["area_um2"] and q["fmax_mhz"] >= p["fmax_mhz"]
            and (q["area_um2"] < p["area_um2"] or q["fmax_mhz"] > p["fmax_mhz"])
        ]
        assert not dominated_by, f"frontier point {p} is dominated"


def test_funnel_counts_match_summary():
    if not _HAS_GCD_LOG:
        return
    r = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}/funnel")
    assert r.status_code == 200
    summary = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}").json()
    assert r.json() == summary["fidelity_counts"]


def test_history_running_max_is_monotonic():
    if not _HAS_GCD_LOG:
        return
    r = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}/history")
    assert r.status_code == 200
    points = r.json()
    assert len(points) > 0
    running = [p["running_max"] for p in points if p["running_max"] is not None]
    assert running == sorted(running)


def test_best_configs_respects_top_and_never_triggers_a_build():
    if not _HAS_GCD_LOG:
        return
    r = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}/best?top=2")
    assert r.status_code == 200
    picks = r.json()
    assert picks
    for p in picks:
        assert p["badge"]
        assert "config" in p and "obs" in p


def test_baseline_is_read_only_over_logged_rows():
    if not _HAS_GCD_LOG:
        return
    r = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}/baseline")
    assert r.status_code == 200
    body = r.json()
    assert "earliest_f3" in body and "hand_picked_baseline" in body


def test_episodes_are_json_serializable_rows():
    if not _HAS_GCD_LOG:
        return
    r = client.get(f"/api/campaigns/{_DESIGN}/{_PLATFORM}/episodes")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    assert all("config" in row for row in rows)


def test_path_traversal_segment_is_rejected():
    """design/platform double as filesystem path segments — ".." must never
    reach resolve_log_path (Starlette's path converter already excludes "/")."""
    r = client.get("/api/campaigns/%2E%2E/%2E%2E")
    assert r.status_code == 400


def test_unknown_design_is_404_not_500():
    r = client.get("/api/campaigns/this-design-does-not-exist/nowhere")
    assert r.status_code == 404


# ── plain-python runner (matches tests/test_parsers.py / test_log_schema.py) ──

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} API tests passed")
    sys.exit(1 if failed else 0)
