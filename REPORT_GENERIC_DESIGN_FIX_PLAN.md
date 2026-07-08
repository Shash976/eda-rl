# Plan: make `eda_rl/viz/report.py`'s Supervisor Overview design-agnostic

## Problem

`eda_rl/viz/report.py`'s "Supervisor Overview" section — the part that
renders the optimization-process report's before/after comparison — silently
assumes every campaign is TinyMAC/TinyVAD. Running `eda-rl report` on a
non-TinyMAC campaign (e.g. `likith`/`sagar`) today produces a **misleading**
report, not just an incomplete one:

- `_ASAP7_BASELINE` (module-level constant, ~line 45) is a **hand-picked
  TinyMAC data point** (`area_um2=1433`, `fmax_mhz=509`, `config={mac_lanes:
  4, accumulator_width: 24, ...}`).
- `build_comparison_table()` (~line 401) unconditionally uses it as the
  "before" row of the before/after table, and unconditionally emits
  `Lanes`/`Acc_W` columns from `cfg.get("mac_lanes")` /
  `cfg.get("accumulator_width")` — both `None` for a design like likith.
- `build_summary_banner()` (~line 710) unconditionally computes "+X% vs
  hand-picked baseline" from the same constant, **and** a "Peak inference
  speedup vs software baseline (112 ms @ 100 MHz)" KPI using TinyMAC's cycle
  model (`r["config"].get("mac_lanes", 4)` — silently defaults to 4 even when
  the key doesn't exist for the design at all).
- `build_pareto_figure()` (~line 232) always plots the `_ASAP7_BASELINE` star
  on the Area-vs-Fmax scatter and labels the legend "mac_lanes".

For likith (area ≈ 2 µm², fmax ≈ 17,000 MHz, no `mac_lanes` axis at all) this
means: a fabricated "speedup vs software baseline" number that means nothing
for a combinational decoder, a comparison-table row showing TinyMAC's 1433
µm²/509 MHz as "baseline", and a Pareto plot whose axes get stretched by an
unrelated data point three orders of magnitude off scale.

`build_speedup_figure()` a few lines below **already** guards itself
correctly:
```python
if any("mac_lanes" in (r.get("config") or {}) for r in data.rows):
    figs.append(build_speedup_figure(data.rows))
```
The three functions above never got the same treatment. This is the same
class of bug the repo's audit history (`AGENTS.md`) has repeatedly fixed
elsewhere (design-aware reward, surrogate schema guard, per-design state
normalization, etc.) — fix it the same way: make the report design-aware
instead of TinyMAC-only, don't fabricate numbers for designs that don't have
the reference point they're computed from.

## Fix

In `eda_rl/viz/report.py`:

1. Add one helper near `_f3_ok_rows`/`_pareto_front`:
   ```python
   def _is_tinymac_campaign(rows: list[dict]) -> bool:
       """True if any row's config declares mac_lanes — the axis the hand-picked
       _ASAP7_BASELINE and SW-speedup KPI are calibrated against."""
       return any("mac_lanes" in (r.get("config") or {}) for r in rows)

   def _earliest_f3(rows: list[dict]) -> dict | None:
       """Earliest (by ts) successful F3 build — the generic 'before optimization'
       reference point when there's no hand-picked baseline for this design."""
       f3 = _f3_ok_rows(rows)
       return min(f3, key=lambda r: r.get("ts", 0)) if f3 else None
   ```
   Replace the inline `any("mac_lanes" in (r.get("config") or {}) for r in
   data.rows)` check in `main()` (the existing `build_speedup_figure` guard)
   with a call to `_is_tinymac_campaign` — pure dedup, no behavior change.

2. `build_pareto_figure`: only add the `_ASAP7_BASELINE` star trace, and only
   set `legend=dict(title="mac_lanes")`, when `_is_tinymac_campaign(rows)` is
   true. Otherwise skip the star and leave the legend untitled.

3. `build_comparison_table`: keep the existing TinyMAC path byte-for-byte when
   `_is_tinymac_campaign(rows)`. Add a generic branch otherwise:
   - baseline row = `_earliest_f3(rows)`, labeled `"First F3 build (campaign
     start)"` — a real logged data point, not fabricated.
   - drop the `Lanes`/`Acc_W` columns (or render `"—"`) since generic configs
     don't have them.
   - `ΔFmax`/`ΔArea` computed against that real baseline row's own
     `area_um2`/`fmax_mhz`, reusing the `_delta()` helper already in the
     function.

4. `build_summary_banner`: keep the TinyMAC path unchanged. Add a generic
   branch that drops the "vs hand-picked baseline" and "vs software baseline"
   KPIs entirely (no substitute fabrication) and instead shows: best Fmax
   found, min area found, ΔFmax/ΔArea vs `_earliest_f3(rows)`, and the
   existing build-count/hours/variables-searched KPI unchanged.

Scope note: plot titles that say "(asap7)" are left as-is — fixing them for
other platforms (e.g. sky130hd/`sagar`) is a separate, unrelated cleanup, not
part of this fix.

## Verification

Run the report generator against the real likith log and confirm it no longer
references TinyMAC:
```bash
eda-rl report --log eda_rl/campaigns/likith/asap7/results_funnel_campaigns.jsonl --open
```
Check: the KPI banner shows likith-scale numbers (µm² / tens-of-GHz) with no
"vs software baseline" line, the comparison table has no Lanes/Acc_W columns
and its baseline row is a real first-F3 build, and the Pareto scatter has no
red star / no "mac_lanes" legend title. Also spot-check a TinyMAC/
`tinymac_accel` log (if one is handy) to confirm that path is unaffected,
since it must stay byte-for-byte identical.

*(Drafted while investigating how to report/compare a live `likith` campaign;
not yet implemented — do this on a follow-up branch.)*
