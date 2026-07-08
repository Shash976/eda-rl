"""doctor.py — per-design physics-sanity preflight (`eda-rl doctor`, audit R4).

Turns the failure modes new designs have actually hit — silently dead F2
parsers (audit F3/F4), PDN-0185 aborts from tool-default CORE_UTILIZATION
ranges on tiny floorplans, ps-vs-ns knob-range incoherence, tier flags that
silently exclude a design's own overridden knobs — into an automated check
that runs in seconds (plus an optional real F3 probe), instead of tribal
knowledge frozen in YAML comments.

Usage:
    eda-rl doctor --design likith --platform asap7
    eda-rl doctor --design sagar  --platform sky130hd --probe-f3
    PHYSICAL_MOCK=1 eda-rl doctor --design gcd --platform nangate45   # no tools

Checks:
  static   — YAML loads; platform declared; RTL files exist; clock range sane;
             knob overrides inside (or deliberately wider than) registry
             ranges; constraint-knob (IO_DELAY/CLOCK_UNCERTAINTY) ranges
             coherent with the platform clock period; minimum --max-tier that
             activates every knob the YAML declares.
  f2       — one real synth+STA proxy at the design's default config: area
             parseable, cell count non-None (F3 regression guard), fmax either
             a real number or None-with-combinational marker (F4 semantics).
  f3 probe — (--probe-f3 only) one real full build at the design's default
             utilization; on failure, bisect utilization downward to report
             the working floor and a suggested knobs.override block.

Exit code: 0 = all PASS/WARN, 1 = any FAIL.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

_PASS, _WARN, _FAIL = "PASS", "WARN", "FAIL"


class _Report:
    def __init__(self) -> None:
        self.failed = False

    def line(self, status: str, check: str, detail: str = "") -> None:
        if status == _FAIL:
            self.failed = True
        print(f"  {status:4s}  {check}" + (f" — {detail}" if detail else ""))


# ── static checks ───────────────────────────────────────────────────────────────

def _static_checks(rep: _Report, design: Any, platform: str) -> dict:
    """YAML/registry coherence. Returns context used by later checks."""
    from pathlib import Path

    from eda_rl.common.knobs import KnobRegistry

    ctx: dict = {"default_clk": None, "clk_range": None}

    # RTL files exist
    missing = [p for p in design.rtl_files if not Path(p).exists()]
    rep.line(_FAIL if missing else _PASS, "RTL files resolve",
             f"missing: {missing}" if missing else f"{len(design.rtl_files)} file(s)")

    # platform declared + clock range sane
    plat = (design.platforms or {}).get(platform)
    if plat is None:
        rep.line(_FAIL, f"platform '{platform}' declared in design YAML",
                 f"declared: {sorted(design.platforms or {})}")
        return ctx
    rep.line(_PASS, f"platform '{platform}' declared in design YAML")
    lo, hi = (plat.get("clock_range_ns") or [None, None])[:2]
    dflt = plat.get("default_clock_ns")
    ok = (isinstance(lo, (int, float)) and isinstance(hi, (int, float))
          and 0 < lo < hi and (dflt is None or lo <= dflt <= hi))
    rep.line(_PASS if ok else _FAIL, "clock_range_ns sane (0 < lo < hi, default inside)",
             f"[{lo}, {hi}] default={dflt}")
    if ok:
        ctx["clk_range"] = (float(lo), float(hi))
        ctx["default_clk"] = float(dflt if dflt is not None else (lo + hi) / 2)

    # knob overrides vs registry ranges + minimum activating tier
    reg = KnobRegistry.load()
    kcfg = getattr(design, "knobs", None) or {}
    declared = set((kcfg.get("override") or {})) | set(kcfg.get("enable") or [])
    max_needed_tier = 1
    for name in sorted(declared):
        knob = reg.get(name)
        if knob is None:
            rep.line(_WARN, f"knobs.override/enable names unknown knob '{name}'",
                     "not in the registry — it will never be sampled")
            continue
        max_needed_tier = max(max_needed_tier, knob.tier)
        ov = (kcfg.get("override") or {}).get(name) or {}
        if "range" in ov and knob.range is not None:
            olo, ohi = float(ov["range"][0]), float(ov["range"][1])
            klo, khi = float(knob.range[0]), float(knob.range[1])
            if olo > ohi:
                rep.line(_FAIL, f"{name} override range ordered", f"[{olo}, {ohi}]")
            elif olo < klo or ohi > khi:
                rep.line(_WARN, f"{name} override wider than registry range",
                         f"override [{olo}, {ohi}] vs registry [{klo}, {khi}] — "
                         "deliberate widening is allowed but unvalidated territory")
            else:
                rep.line(_PASS, f"{name} override inside registry range",
                         f"[{olo}, {ohi}] ⊆ [{klo}, {khi}]")

    # constraint knobs vs the platform clock period (ps-vs-ns mistakes land here)
    if ctx["clk_range"]:
        clk_lo = ctx["clk_range"][0]
        for name in ("IO_DELAY", "CLOCK_UNCERTAINTY"):
            if name not in declared:
                continue
            ov = (kcfg.get("override") or {}).get(name) or {}
            knob = reg.get(name)
            hi_v = float((ov.get("range") or (knob.range if knob else (0, 0)))[1])
            if hi_v >= clk_lo:
                rep.line(_FAIL, f"{name} range coherent with clock period",
                         f"max {hi_v} ns >= min clock {clk_lo} ns — the constraint "
                         "can exceed the whole period (ps-vs-ns conversion missing?)")
            elif name == "IO_DELAY" and 2 * hi_v >= clk_lo:
                rep.line(_WARN, f"{name} range coherent with clock period",
                         f"2x max {hi_v} ns >= min clock {clk_lo} ns — input+output "
                         "delay can consume the whole period")
            else:
                rep.line(_PASS, f"{name} range coherent with clock period",
                         f"max {hi_v} ns vs min clock {clk_lo} ns")

    # tier reachability: does the declared knob set need a higher --max-tier?
    if declared:
        rep.line(_PASS if max_needed_tier == 1 else _WARN,
                 "minimum --max-tier activating all declared knobs",
                 f"--max-tier {max_needed_tier}"
                 + ("" if max_needed_tier == 1 else
                    " (campaigns at a lower tier silently drop some declared knobs)"))
    return ctx


# ── F2 proxy check ──────────────────────────────────────────────────────────────

def _f2_check(rep: _Report, design: Any, platform: str, clk_ns: float) -> None:
    from eda_rl.common.physical_runner import run_synth_sta

    try:
        r = run_synth_sta(0, 0, clk_ns, platform, design=design,
                          abc_recipe="orfs_speed")
    except Exception as exc:   # noqa: BLE001 — a preflight reports, never crashes
        rep.line(_FAIL, "F2 proxy runs", f"{type(exc).__name__}: {exc}")
        return
    if r.get("status") not in ("ok", "mock-proxy"):
        rep.line(_FAIL, "F2 proxy runs", f"status={r.get('status')} (see {r.get('report')})")
        return
    rep.line(_PASS, "F2 proxy runs", f"status={r['status']}")

    rep.line(_PASS if r.get("area_um2") is not None else _FAIL,
             "F2 area parseable", f"area_um2={r.get('area_um2')}")
    rep.line(_PASS if r.get("cells") is not None else _FAIL,
             "F2 cell count non-None (F3 parser guard)", f"cells={r.get('cells')}")

    fmax = r.get("fmax_mhz")
    if fmax is not None:
        rep.line(_PASS, "F2 fmax semantics (F4 guard)",
                 f"measured fmax={fmax}"
                 + (" [inferred-from-slack]" if r.get("fmax_inferred") else ""))
    elif r.get("combinational"):
        rep.line(_PASS, "F2 fmax semantics (F4 guard)",
                 "fmax=None with combinational=True (no clocked path — legitimate)")
    else:
        rep.line(_FAIL, "F2 fmax semantics (F4 guard)",
                 "fmax=None WITHOUT the combinational marker — parser regression")


# ── F3 probe (optional, real builds) ───────────────────────────────────────────

def _f3_probe(rep: _Report, design: Any, platform: str, clk_ns: float) -> None:
    from eda_rl.common.knobs import KnobRegistry
    from eda_rl.common.physical_runner import run_physical

    reg = KnobRegistry.load()
    space = reg.space(max_tier=1, design=design, platform=platform)
    util0 = int(round(float(space.get("CORE_UTILIZATION", {}).get("default", 40))))

    tried: list[tuple[int, str]] = []
    util = util0
    while True:
        try:
            r = run_physical(0, 0, clk_ns, platform, util=util, design=design,
                             abc_recipe="orfs_speed")
            status = r.get("status", "?")
        except Exception as exc:   # noqa: BLE001
            status = f"{type(exc).__name__}"
        tried.append((util, status))
        if status == "ok":
            if util == util0:
                rep.line(_PASS, "F3 build at default utilization", f"util={util} ok")
            else:
                rep.line(_WARN, "F3 utilization floor found below default",
                         f"default util={util0} fails; util={util} builds. Suggested "
                         "YAML block:\n"
                         "          knobs:\n"
                         "            override:\n"
                         "              CORE_UTILIZATION:\n"
                         f"                range: [{max(util - 2, 1)}.0, {util + 2}.0]\n"
                         f"                default: {util}.0")
            return
        # bisect downward: halve toward 1 (PDN-0185-class failures need a
        # smaller footprint → lower utilization = bigger floorplan)
        nxt = max(util // 2, 1)
        if nxt == util:
            rep.line(_FAIL, "F3 build (utilization bisect exhausted)",
                     f"tried {tried} — failure is not utilization-shaped; "
                     "check the build log")
            return
        util = nxt


# ── entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-design physics-sanity preflight (audit R4). "
                    "Exit 0 = all PASS/WARN, 1 = any FAIL.")
    ap.add_argument("--design", required=True, help="design name or YAML path")
    ap.add_argument("--platform", required=True,
                    help="ORFS platform (nangate45 / sky130hd / asap7)")
    ap.add_argument("--probe-f3", action="store_true",
                    help="also run real F3 build(s) to find the utilization "
                         "floor (minutes per probe)")
    args = ap.parse_args()

    rep = _Report()
    print(f"eda-rl doctor — design={args.design} platform={args.platform}"
          + (" [PHYSICAL_MOCK]" if os.environ.get("PHYSICAL_MOCK") else ""))

    from eda_rl.common.designs import DesignSpec
    try:
        design = DesignSpec.load(args.design)
        rep.line(_PASS, "DesignSpec loads")
    except (FileNotFoundError, ValueError) as exc:
        rep.line(_FAIL, "DesignSpec loads", str(exc))
        sys.exit(1)

    ctx = _static_checks(rep, design, args.platform)

    if ctx["default_clk"] is not None:
        _f2_check(rep, design, args.platform, ctx["default_clk"])
        if args.probe_f3:
            if os.environ.get("PHYSICAL_MOCK"):
                rep.line(_WARN, "F3 probe skipped", "PHYSICAL_MOCK metrics are "
                         "TinyMAC-shaped; a mock probe proves nothing")
            else:
                _f3_probe(rep, design, args.platform, ctx["default_clk"])
    else:
        rep.line(_WARN, "F2/F3 checks skipped", "no usable clock range")

    print(f"\ndoctor: {'FAIL' if rep.failed else 'OK'}")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
