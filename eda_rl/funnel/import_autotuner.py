#!/usr/bin/env python3
"""import_autotuner.py — convert an ORFS AutoTuner config into eda-rl YAML.

Why
---
`eda_rl/designs/{likith,sagar}/autotuner.json` are kept purely as provenance
records: nothing reads them, and the knob ranges they describe were mirrored into
the design YAMLs **by hand**. The eda-rl/AutoTuner head-to-head in
`docs/eda_rl_vs_autotuner.md` went further and pinned both tools to the same
space via YAML edits it records as *uncommitted* — i.e. that experiment is
currently not reproducible.

This turns that hand work into a command, so "run both optimizers over the same
space" is a repeatable setup step rather than a careful manual edit.

The two traps it exists to handle
---------------------------------
1. **Units.** AutoTuner's `_SDC_CLK_PERIOD` is in the platform's native SDC unit
   — picoseconds on asap7, nanoseconds on sky130hd/nangate45 — while every eda-rl
   `clock_range_ns` is nanoseconds and `PLATFORM_TIME_UNIT` converts downstream.
   Getting this wrong is precisely what `eda-rl doctor` FAILs on.
2. **The clock is not a knob override.** In eda-rl the clock axis comes from
   `platforms.<platform>.clock_range_ns`, not from the `knobs:` block, so
   `_SDC_CLK_PERIOD` must be emitted into a different part of the YAML than every
   other parameter. Emitting it as a knob override would silently no-op.

Usage
-----
    eda-rl import-autotuner --config eda_rl/designs/sagar/autotuner.json \\
                            --platform sky130hd
    eda-rl import-autotuner --config .../autotuner.json --platform asap7 --diff sagar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# AutoTuner pseudo-parameters → eda-rl knob names.  The `_SDC_`/`_FR_` prefixes
# mean "rewrite the SDC / fastroute.tcl" in AutoTuner (utils.py write_sdc /
# write_fast_route); eda-rl models the same things as pseudo_sdc /
# pseudo_fastroute knobs.
_PSEUDO_MAP = {
    "_SDC_UNCERTAINTY": "CLOCK_UNCERTAINTY",
    "_SDC_IO_DELAY":    "IO_DELAY",
    "_FR_LAYER_ADJUST": "ROUTING_LAYER_ADJUSTMENT",
    "_FR_GR_SEED":      "GR_SEED",
}

# TIME-valued AutoTuner parameters.  These are expressed in the platform's native
# SDC unit — picoseconds on asap7 — and every one must be divided by
# PLATFORM_TIME_UNIT, not just the clock.  Missing this is the exact ps-vs-ns
# mistake `eda-rl doctor` FAILs on: likith's AutoTuner IO_DELAY of [0.5, 2.0]
# taken literally as ns would be an I/O delay ~30x the whole 0.045-0.07 ns clock
# period.  likith.yaml's hand-mirrored [0.0005, 0.002] is the converted form.
_TIME_VALUED = {"_SDC_CLK_PERIOD", "_SDC_UNCERTAINTY", "_SDC_IO_DELAY"}

# Handled specially (not a knob) or deliberately unsupported.
_CLOCK_KEY = "_SDC_CLK_PERIOD"
_IGNORED = {"_SDC_FILE_PATH", "_FR_FILE_PATH", "best_result"}
_UNSUPPORTED = {
    "_PINS_DISTANCE": "maps to ORFS PLACE_PINS_ARGS, which eda-rl has no knob for",
    "_SYNTH_FLATTEN": "AutoTuner itself ignores this (TUN-0013)",
}

# eda-rl treats SDC/fastroute-touching knobs as opt-in per design, because they
# rewrite the very constraints the reward is measured against (audit F1/R6).
_OPT_IN = {"CLOCK_UNCERTAINTY", "IO_DELAY", "GR_SEED"}

# ns per native SDC time unit, mirroring physical_runner.PLATFORM_TIME_UNIT.
_PLATFORM_TIME_UNIT = {"nangate45": 1.0, "sky130hd": 1.0, "asap7": 1000.0}


def _minmax(spec: Any) -> tuple[float, float] | None:
    """Extract [min, max] from either AutoTuner schema.

    The classic form is {"minmax": [lo, hi]}; asap7's *_new.json files use
    {"type": "range_float", "min": lo, "max": hi}.
    """
    if not isinstance(spec, dict):
        return None
    if "minmax" in spec and isinstance(spec["minmax"], (list, tuple)) \
            and len(spec["minmax"]) == 2:
        lo, hi = spec["minmax"]
    elif "min" in spec and "max" in spec:
        lo, hi = spec["min"], spec["max"]
    else:
        return None
    try:
        return float(lo), float(hi)
    except (TypeError, ValueError):
        return None


def convert(config: dict, platform: str,
            known_knobs: set[str] | None = None) -> tuple[dict, list[str]]:
    """Convert a parsed autotuner.json into {clock_range_ns, knobs, ...}.

    Returns (result, notes).  `result` has:
        clock_range_ns   : [lo, hi] in NANOSECONDS, or None if not swept
        default_clock_ns : midpoint, a reasonable per-design default
        enable           : opt-in knob names to list under knobs.enable
        override         : {knob: {range: [lo, hi]}} for knobs.override
    `notes` collects everything a human must check.
    """
    notes: list[str] = []
    unit = _PLATFORM_TIME_UNIT.get(platform)
    if unit is None:
        notes.append(f"WARNING: unknown platform {platform!r}; assuming SDC times "
                     f"are already in ns (no conversion applied)")
        unit = 1.0

    out: dict = {"clock_range_ns": None, "default_clock_ns": None,
                 "enable": [], "override": {}}

    for key, spec in config.items():
        if key in _IGNORED:
            continue
        if key in _UNSUPPORTED:
            notes.append(f"SKIPPED {key}: {_UNSUPPORTED[key]}")
            continue
        if not isinstance(spec, dict):
            continue   # bare scalars are file paths / metadata, not axes

        rng = _minmax(spec)
        if rng is None:
            if spec.get("type") == "string" and spec.get("values"):
                notes.append(f"SKIPPED {key}: categorical sweeps are not "
                             f"expressible as a knob range override")
            else:
                notes.append(f"SKIPPED {key}: no minmax/min-max in {spec!r}")
            continue
        lo, hi = rng

        step = spec.get("step", 0)
        if step:
            notes.append(
                f"NOTE {key}: AutoTuner declares step={step}, but eda-rl samples "
                f"this axis continuously unless the knob declares a _snap_step "
                f"(a real likith campaign logged 15,484 distinct clock values "
                f"against a declared step of 0.5). Ranges match; grids do not.")

        # ── the clock: a platform range, NOT a knob override ──────────────────
        if key == _CLOCK_KEY:
            lo_ns, hi_ns = lo / unit, hi / unit
            out["clock_range_ns"] = [lo_ns, hi_ns]
            out["default_clock_ns"] = round((lo_ns + hi_ns) / 2.0, 6)
            if unit != 1.0:
                notes.append(
                    f"UNIT {key}: [{lo:g}, {hi:g}] in {platform} native units "
                    f"(ps) -> [{lo_ns:g}, {hi_ns:g}] ns. eda-rl stores ns "
                    f"everywhere; PLATFORM_TIME_UNIT converts back downstream.")
            continue

        name = _PSEUDO_MAP.get(key, key)
        if key.startswith("_FR_LAYER_ADJUST_"):
            notes.append(f"SKIPPED {key}: eda-rl's ROUTING_LAYER_ADJUSTMENT is "
                         f"global; per-layer adjustment has no equivalent")
            continue
        if key.startswith("_") and key not in _PSEUDO_MAP:
            notes.append(f"SKIPPED {key}: unrecognised AutoTuner pseudo-parameter")
            continue

        if known_knobs is not None and name not in known_knobs:
            notes.append(f"SKIPPED {key}: {name!r} is not in eda-rl's KnobRegistry, "
                         f"so it would never be sampled")
            continue

        # Time-valued SDC parameters share the clock's unit and need the same
        # conversion.  Everything else (seeds, layer adjustments, cell padding)
        # is unitless and passes through untouched.
        if key in _TIME_VALUED and unit != 1.0:
            lo_ns, hi_ns = lo / unit, hi / unit
            notes.append(
                f"UNIT {key}: [{lo:g}, {hi:g}] in {platform} native units (ps) "
                f"-> [{lo_ns:g}, {hi_ns:g}] ns for {name}.")
            lo, hi = lo_ns, hi_ns

        out["override"][name] = {"range": [lo, hi]}
        if name in _OPT_IN:
            out["enable"].append(name)
            notes.append(
                f"OPT-IN {name}: rewrites the SDC/routing environment, so eda-rl "
                f"only samples it when the design names it. Added to knobs.enable "
                f"— be aware this re-opens the reward-gaming surface audit F1 "
                f"closed, which is why it is not on by default.")

    return out, notes


def render_yaml(result: dict, platform: str) -> str:
    """Render the YAML fragments to paste into a design spec."""
    lines: list[str] = []
    if result["clock_range_ns"]:
        lo, hi = result["clock_range_ns"]
        lines.append("# --- merge into the design's platforms: block ---")
        lines.append("platforms:")
        lines.append(f"  {platform}:")
        lines.append(f"    clock_range_ns: [{lo:g}, {hi:g}]   # ALWAYS ns")
        lines.append(f"    default_clock_ns: {result['default_clock_ns']:g}")
        lines.append("")

    if result["override"] or result["enable"]:
        lines.append("# --- merge into the design's knobs: block ---")
        lines.append("knobs:")
        if result["enable"]:
            lines.append("  enable:")
            for n in sorted(result["enable"]):
                lines.append(f"    - {n}")
        if result["override"]:
            lines.append("  override:")
            for n in sorted(result["override"]):
                r = result["override"][n]["range"]
                lines.append(f"    {n}:")
                lines.append(f"      range: [{r[0]:g}, {r[1]:g}]")
    if not lines:
        lines.append("# (nothing convertible found in this AutoTuner config)")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert an ORFS AutoTuner autotuner.json into eda-rl design "
                    "YAML fragments (clock range + knobs block).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to an AutoTuner autotuner.json")
    p.add_argument("--platform", required=True,
                   help="Target platform — decides ps-vs-ns conversion for the "
                        "clock range (asap7 is ps; nangate45/sky130hd are ns)")
    p.add_argument("--diff", default=None, metavar="DESIGN",
                   help="Also load this eda-rl design and report where its "
                        "current ranges differ from the AutoTuner config")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: no such file: {cfg_path}")
        sys.exit(1)
    try:
        config = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: {cfg_path} is not valid JSON: {exc}")
        sys.exit(1)

    known: set[str] | None = None
    try:
        from eda_rl.common.knobs import KnobRegistry
        known = {k.name for k in KnobRegistry.load().all_knobs()}
    except Exception as exc:   # noqa: BLE001
        print(f"  [WARNING] could not load KnobRegistry ({exc}); "
              f"skipping the 'is this a real knob' check.")

    result, notes = convert(config, args.platform, known_knobs=known)

    print(f"\n=== {cfg_path} -> eda-rl ({args.platform}) ===\n")
    print(render_yaml(result, args.platform))
    if notes:
        print("\n--- notes ---")
        for n in notes:
            print(f"  {n}")

    if args.diff:
        _print_diff(args.diff, args.platform, result)
    print()


def _print_diff(design_name: str, platform: str, result: dict) -> None:
    """Report where the design's live space differs from the AutoTuner config."""
    try:
        from eda_rl.common.knobs import KnobRegistry
        reg = KnobRegistry.load()
        space = reg.space(max_tier=4, design=design_name, platform=platform)
    except Exception as exc:   # noqa: BLE001
        print(f"\n--- diff vs {design_name} ---\n  could not load: {exc}")
        return

    print(f"\n--- diff vs eda-rl design {design_name!r} (max_tier=4) ---")
    rows = []
    if result["clock_range_ns"]:
        live = space.get("clock_period_ns", {}).get("range")
        want = result["clock_range_ns"]
        rows.append(("clock_period_ns", live, want))
    for name, spec in sorted(result["override"].items()):
        live = space.get(name, {}).get("range")
        rows.append((name, live, spec["range"]))

    same = 0
    for name, live, want in rows:
        if live is None:
            print(f"  {name:32s} NOT IN SPACE   (AutoTuner wants {want}) "
                  f"— raise --max-tier or add it to knobs.enable")
        elif [round(float(x), 9) for x in live] == [round(float(x), 9) for x in want]:
            same += 1
        else:
            print(f"  {name:32s} eda-rl {live}  vs  AutoTuner {want}")
    print(f"  ({same}/{len(rows)} axes already match)")


if __name__ == "__main__":
    main()
