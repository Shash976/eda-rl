#!/usr/bin/env python3
"""Controlled clock sweep — the REAL-TOOLS validation of the audit-F18 power ruler.

This is NOT part of the mock self-test suite and will refuse to run under
PHYSICAL_MOCK.  It exists because of the single most important lesson in this
repo's history: mock metrics fabricate exactly the fields the parsers produce, so
no mock test can validate a measurement change.

What it does
------------
Builds the SAME design at several clock periods with every other knob fixed, then
scores each build twice — once with raw power (reward v1) and once with power
normalised to the design's reference frequency (reward v2) — and reports how
strongly each correlates with the clock request.

Why a logged campaign cannot answer this
----------------------------------------
The bug was found as corr(reward, clock_period_ns) = +0.76 on a 178-build sagar
corpus.  You cannot then validate the fix by re-scoring that corpus: TPE selected
those points *under the biased reward*, so the sample is contaminated by the very
effect under test.  Only a controlled sweep works.

Reference result (sagar/sky130hd, 8 clocks over [5.5, 8.0], all timing-clean).
The chip is identical at every point — area 492.0 um^2 and fmax_ref ~505 MHz
constant — so only the clock request differs:

    raw power   spread 46.0 %   corr(clk, power)  = -1.000
    power @ref  spread  1.4 %   corr(clk, power)  = +0.071
    reward v1   corr(clk, reward) = +1.000   best clock 8.000 (the range CEILING)
    reward v2   corr(clk, reward) = +0.143   best clock 7.286
    reward spread 0.126 -> 0.0084  (93 % smaller)

Usage
-----
    export ORFS_DIR=/path/to/OpenROAD-flow-scripts
    python3 tests/verify_reward_ruler_real.py --design sagar --platform sky130hd
    python3 tests/verify_reward_ruler_real.py --design gcd --platform nangate45 --points 4

Each point is a full RTL->GDS build: budget minutes per point, not seconds.
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _spearman(x, y):
    """Rank correlation with average ranks for ties (ties are common here —
    fmax_ref often reads identically across builds)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    a, b = rank(x), rank(y)
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((v - ma) ** 2 for v in a) * sum((v - mb) ** 2 for v in b))
    return num / den if den else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", default="sagar")
    p.add_argument("--platform", default="sky130hd")
    p.add_argument("--points", type=int, default=8)
    p.add_argument("--util", type=int, default=None,
                   help="CORE_UTILIZATION (default: the design's own override, "
                        "else 40). Tiny designs need a low value — see PDN-0185.")
    args = p.parse_args()

    if os.environ.get("PHYSICAL_MOCK", "").strip() in ("1", "true", "True", "yes"):
        print("REFUSING: PHYSICAL_MOCK is set. Mock metrics are design-independent "
              "and fabricate the very fields under test — this check is meaningless "
              "without real tools.")
        return 2

    from eda_rl.common.designs import DesignSpec
    from eda_rl.common.physical_reward import compute_generic_reward
    from eda_rl.common.physical_runner import run_physical

    design = DesignSpec.load(args.design)
    plat = design.platforms.get(args.platform)
    if not plat:
        print(f"ERROR: design {args.design!r} does not declare platform {args.platform!r}")
        return 2
    ref = float(plat.get("default_clock_ns") or 0)
    lo, hi = (float(v) for v in plat["clock_range_ns"])
    n = max(2, args.points)
    clks = [round(lo + i * (hi - lo) / (n - 1), 4) for i in range(n)]

    util = args.util
    if util is None:
        ov = ((getattr(design, "knobs", None) or {}).get("override") or {}) \
            .get("CORE_UTILIZATION") or {}
        util = int(ov.get("default") or (ov.get("range") or [40])[0])

    print(f"controlled clock sweep — {args.design}/{args.platform}")
    print(f"  ref_period={ref} ns  util={util}  points={clks}")
    print(f"  every other knob fixed; each point is a full build\n")

    rows = []
    for clk in clks:
        r = run_physical(0, 0, clk, args.platform, util=util, density=0.60,
                         abc_recipe="orfs_speed", design=design, knob_values={})
        if r.get("status") != "ok":
            print(f"  clk={clk:7.4f}  status={r.get('status')} — skipped")
            continue
        if r.get("power_internal_mw") is None:
            print(f"  clk={clk:7.4f}  no power decomposition — cannot test the ruler")
            continue
        r["_clk"] = clk
        rows.append(r)
        print(f"  clk={clk:7.4f} timing_met={str(r.get('timing_met')):5s} "
              f"area={r['area_um2']:8.1f} raw_pw={r['power_mw']:.5f} "
              f"fmax_ref={r.get('fmax_ref_mhz')} drc={r.get('drc_count')}")

    clean = [r for r in rows if r.get("timing_met")]
    if len(clean) < 3:
        print(f"\n  only {len(clean)} timing-clean build(s) — need >=3 for a verdict.")
        print("  Timing-failing builds carry the -3.0 penalty and swamp the "
              "comparison; narrow the clock range to the feasible region.")
        return 1

    base = clean[0]
    refs = {"area_ref_um2": base["area_um2"], "power_ref_mw": base["power_mw"],
            "fmax_ref_mhz": base.get("fmax_ref_mhz") or base["fmax_mhz"]}

    def p_at_ref(r):
        dyn = r["power_internal_mw"] + r["power_switching_mw"]
        return r["power_leakage_mw"] + dyn * (r["_clk"] / ref)

    def score(r, normalised):
        m = dict(r)
        m["fmax_mhz"] = r.get("fmax_ref_mhz") or r["fmax_mhz"]
        if normalised:
            m["power_mw"] = p_at_ref(r)
        return compute_generic_reward(m, refs=refs)["reward"]

    clk = [r["_clk"] for r in clean]
    raw = [r["power_mw"] for r in clean]
    nor = [p_at_ref(r) for r in clean]
    v1 = [score(r, False) for r in clean]
    v2 = [score(r, True) for r in clean]
    area = [r["area_um2"] for r in clean]

    print(f"\n  {len(clean)} timing-clean builds")
    print(f"  area spread          {100 * (max(area) / min(area) - 1):6.1f}% "
          f"{'(identical chip)' if max(area) == min(area) else ''}")
    print(f"  raw power spread     {100 * (max(raw) / min(raw) - 1):6.1f}%   "
          f"corr(clk,·) = {_spearman(clk, raw):+.3f}")
    print(f"  power@ref spread     {100 * (max(nor) / min(nor) - 1):6.1f}%   "
          f"corr(clk,·) = {_spearman(clk, nor):+.3f}")
    r1, r2 = _spearman(clk, v1), _spearman(clk, v2)
    print(f"\n  corr(clk, reward v1) = {r1:+.3f}   best clock "
          f"{clk[v1.index(max(v1))]:.4f}  (range ceiling {max(clk)})")
    print(f"  corr(clk, reward v2) = {r2:+.3f}   best clock "
          f"{clk[v2.index(max(v2))]:.4f}")
    s1, s2 = max(v1) - min(v1), max(v2) - min(v2)
    print(f"  reward spread  v1={s1:.5f}  v2={s2:.5f}"
          + (f"  ({100 * (1 - s2 / s1):.0f}% smaller)" if s1 else ""))

    ok = abs(r2) < 0.5 and abs(r2) < abs(r1)
    print(f"\n  {'PASS' if ok else 'FAIL'}: the ruler "
          f"{'neutralises' if ok else 'does NOT neutralise'} the clock knob "
          f"(|{r2:+.3f}| vs |{r1:+.3f}|)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
