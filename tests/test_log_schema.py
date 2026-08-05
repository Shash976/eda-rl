#!/usr/bin/env python3
"""Schema tests for committed campaign logs (audit A8).

Why this file exists
--------------------
The campaign JSONL corpora are the repo's evidence base — every claim about
whether a policy works is computed from them — and nothing validated them.
Three concrete consequences, all found in committed data:

* `campaigns/sagar/sky130hd/results_funnel_campaigns.jsonl` was **not valid
  JSONL**: one `campaign_summary` had been written pretty-printed across four
  lines, so a reader doing `json.loads` per line either crashed or silently
  dropped it. Repaired; this test stops it recurring.
* 6 of 16 committed campaigns predate the audit-F15 metadata stamping and carry
  no design/platform/sampler/seed, so their settings are unrecoverable. That is
  history and cannot be fixed retroactively — but a *new* log missing them is a
  regression, so the test only requires provenance where it is present.
* Reward semantics are now versioned (audit F18/F19). A log that mixes
  `reward_version` values inside one campaign would be scored under two
  different rulers.

These are cheap invariants that protect an expensive-to-regenerate asset.

Run: python3 tests/test_log_schema.py
"""

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CAMPAIGNS = _HERE.parent / "eda_rl" / "campaigns"


def _logs() -> list[Path]:
    return sorted(_CAMPAIGNS.rglob("results_funnel_campaigns.jsonl"))


def _rows(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def test_every_committed_log_is_valid_jsonl():
    """One object per line, no exceptions.  A multi-line record silently
    truncates any per-line reader."""
    problems = []
    for p in _logs():
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{p.relative_to(_CAMPAIGNS)}:{i}: {e}")
    assert not problems, "invalid JSONL:\n  " + "\n  ".join(problems[:10])


def test_rows_are_objects_and_typed():
    """Every row is a dict, and is either an episode (has "config") or the
    trailing summary (has "campaign_summary") — the discriminator
    campaign_data.load_campaign_rows relies on to skip the summary."""
    problems = []
    for p in _logs():
        for i, row in enumerate(_rows(p), 1):
            if not isinstance(row, dict):
                problems.append(f"{p.name}:{i}: row is {type(row).__name__}, not dict")
            elif "config" not in row and "campaign_summary" not in row:
                problems.append(f"{p.name}:{i}: neither an episode nor a summary")
    assert not problems, "malformed rows:\n  " + "\n  ".join(problems[:10])


def test_episode_rows_carry_the_fields_readers_use():
    """report/collect/fit-surrogate all index these; a missing one is a silent
    None downstream rather than an error."""
    required = ("campaign_id", "episode", "config", "fidelity")
    problems = []
    for p in _logs():
        for i, row in enumerate(_rows(p), 1):
            if "config" not in row:
                continue
            missing = [k for k in required if k not in row]
            if missing:
                problems.append(f"{p.name}:{i}: missing {missing}")
    assert not problems, "incomplete episode rows:\n  " + "\n  ".join(problems[:10])


def test_provenance_is_self_consistent_where_present():
    """Audit F15 stamps design/platform/sampler/promotion/max_tier/seed on every
    row so a log is attributable after the fact.  Older logs predate it — but
    where the keys ARE present they must not disagree within one campaign."""
    problems = []
    for p in _logs():
        seen: dict[tuple, set] = {}
        for row in _rows(p):
            if "config" not in row:
                continue
            cid = row.get("campaign_id")
            for key in ("design", "platform", "sampler", "promotion", "max_tier", "seed"):
                if key in row:
                    seen.setdefault((cid, key), set()).add(json.dumps(row[key]))
        for (cid, key), vals in seen.items():
            if len(vals) > 1:
                problems.append(f"{p.name}: campaign {cid} has {len(vals)} values "
                                f"for {key}: {sorted(vals)[:4]}")
    assert not problems, "inconsistent provenance:\n  " + "\n  ".join(problems[:10])


def test_reward_version_never_mixed_within_a_campaign():
    """A single campaign is scored by one reward implementation.  Two versions
    under one campaign_id would mean two rulers in one dataset (audit F18/F19).
    Rows with no key are version 1 by definition."""
    problems = []
    for p in _logs():
        vers: dict = {}
        for row in _rows(p):
            if "config" not in row:
                continue
            vers.setdefault(row.get("campaign_id"), set()).add(
                row.get("reward_version", 1))
        for cid, vs in vers.items():
            if len(vs) > 1:
                problems.append(f"{p.name}: campaign {cid} mixes reward_version {sorted(vs)}")
    assert not problems, "mixed reward semantics:\n  " + "\n  ".join(problems[:10])


def test_f3_rows_have_usable_or_explained_metrics():
    """An episode that reached F3 either carries real metrics or an explicit
    non-ok status.  Silent nulls with status ok would mean the reward scored
    absent data — the PARSE_FAIL guard exists precisely to prevent that."""
    problems = []
    for p in _logs():
        for i, row in enumerate(_rows(p), 1):
            if row.get("fidelity") != "F3":
                continue
            obs = row.get("obs") or {}
            if obs.get("status") in ("ok", "mock", "mock-proxy"):
                if obs.get("area_um2") is None:
                    problems.append(f"{p.name}:{i}: status ok but area_um2 is null")
    assert not problems, "ok-status F3 rows with no metrics:\n  " + "\n  ".join(problems[:10])


# ── plain-python runner (matches tests/test_parsers.py) ───────────────────────

if __name__ == "__main__":
    if not _CAMPAIGNS.is_dir():
        print(f"  SKIP  no campaigns directory at {_CAMPAIGNS}")
        sys.exit(0)
    print(f"  ({len(_logs())} committed campaign logs)")
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
    print(f"\n{len(tests) - failed}/{len(tests)} log-schema tests passed")
    sys.exit(1 if failed else 0)
