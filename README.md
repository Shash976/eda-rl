# eda-rl

**A multi-fidelity RL/DSE optimizer for RTL→GDS chip design-space exploration.**

Drop in a design (RTL + a ~10-line YAML), point it at an [OpenROAD-flow-scripts][orfs]
(ORFS) install, and `eda-rl` searches the flow for configurations that trade off **area /
Fmax / power** — promoting promising candidates through cheap proxies up to the full
RTL→GDS flow, and learning where to spend the synthesis budget.

It is **design-agnostic**: any design becomes an input via a `DesignSpec` YAML. The
TinyMAC accelerator and the ORFS reference `gcd` are worked examples.

```
                 cheap ───────────────────────────────► expensive
   F0 legality ─►  F1 functional ─►  F2 synth+STA proxy ─►  F3 full RTL→GDS
   (validate)      (optional sim)    (seconds)              (minutes, real area/Fmax)
        └────────── a promotion policy decides what advances ──────────┘
```

---

## Install

```bash
git clone <your-fork>/eda-rl && cd eda-rl
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Real RTL→GDS runs need OpenROAD-flow-scripts. Point at your install:
export ORFS_DIR=/opt/OpenROAD-flow-scripts
```

No ORFS? Set `PHYSICAL_MOCK=1` to drive the whole pipeline with synthetic-but-plausible
metrics — useful for trying the tool and for CI.

## The walkthrough: design in → optimized GDS out

Four steps. Try the whole thing with no tools installed by prefixing `PHYSICAL_MOCK=1`.

### 1. Run a campaign (the optimizer)

```bash
eda-rl optimize --design gcd --platform nangate45 --budget-hours 4
```

Toggle the search however you like:

| Flag | Choices / default | What it does |
|------|-------------------|--------------|
| `--design` | name or path to a YAML | the design to optimize |
| `--platform` | `nangate45`, `asap7`, … | target PDK |
| `--budget-hours` | float, `4` | wall-clock budget; the optimizer stops when spent |
| `--max-tier` | `1`–`4`, `1` | how many ORFS knob tiers to search — tiers are cumulative (see [ORFS knob tiers](#orfs-knob-tiers) below). Tier-2+ knobs flow all the way to the F3 full build, not just the F2 proxy |
| `--sampler` | `tpe` \| `surrogate_ucb` \| `random` | how candidates are proposed |
| `--promotion` | `fixed` \| `linucb` \| `random` | the policy that decides what advances F0→F3 |
| `--seed` | int, `0` | reproducibility |
| `--out` | path | where the campaign log goes |

### ORFS knob tiers

`--max-tier N` includes every knob with `tier ≤ N` — tiers are cumulative, so
`--max-tier 3` gets tiers 1+2+3. Regardless of `--max-tier`, three axes are
always in the search space: your design's `params:` RTL axes (if any),
`clock_period_ns` (from the platform's `clock_range_ns`), and `abc_recipe`
(`orfs_speed` / `orfs_area` / `plain`). Tier 4 is additionally gated on
`has_macros: true` in the design YAML — otherwise those 5 knobs are
suppressed even at `--max-tier 4`.

To pin, drop, or retune any knob for one design (instead of editing
`search_space_funnel.yaml`), use the design's `knobs:` block — see
[Bring your own design](#bring-your-own-design) below.

**Tier 1 — dominant axes (always on)**

| Axis | Type | Default | Range / choices | Controls |
|---|---|---|---|---|
| `clock_period_ns` | float | platform's `default_clock_ns` | platform's `clock_range_ns` | target clock period (written to the SDC) |
| `abc_recipe` | categorical | `orfs_speed` | `orfs_speed`, `orfs_area`, `plain` | synthesis ABC script — speed- or area-optimized, or bare (`plain` is proxy-only) |
| `CORE_UTILIZATION` | float | 40.0 | 20.0 – 60.0 | target core utilization % at floorplan |
| *(design `params:` axes)* | varies | design-defined | design-defined | RTL chparams, e.g. TinyMAC's `mac_lanes` / `accumulator_width` |

**Tier 2 — moderate / design-dependent (adds to tier 1)**

| Knob | Type | Default | Range | Controls |
|---|---|---|---|---|
| `CORE_ASPECT_RATIO` | float | 1.0 | 0.5 – 2.0 | die width/height aspect ratio |
| `CORE_MARGIN` | float | 1.0 | 1.0 – 3.0 | spacing (µm) between core and die boundary |
| `PLACE_DENSITY` | float | 0.60 | 0.40 – 0.80 | target global-placement density |
| `PLACE_DENSITY_LB_ADDON` | float | 0.0 | 0.0 – 0.20 | delta on top of the platform's computed minimum density (preferred over `PLACE_DENSITY` — adapts per platform) |
| `CELL_PAD_IN_SITES_GLOBAL_PLACEMENT` | int | 0 | 0 – 3 | cell padding (sites) during global placement, eases routability |
| `CELL_PAD_IN_SITES_DETAIL_PLACEMENT` | int | 0 | 0 – 3 | cell padding during detail placement / CTS / GRT legalization |

**Tier 3 — fine-tuning: CTS / timing-repair / routing (adds to tiers 1–2)**

| Knob | Type | Default | Range | Controls |
|---|---|---|---|---|
| `CTS_CLUSTER_SIZE` | int | 20 | 10 – 200 | max sinks per clock-tree cluster (smaller → deeper tree, better skew) |
| `CTS_CLUSTER_DIAMETER` | float | 100.0 | 20.0 – 400.0 | max spatial diameter (µm) of a CTS sink cluster |
| `TNS_END_PERCENT` | float | 100.0 | 5.0 – 100.0 | % of violating timing endpoints to repair (100 = full closure; lower = faster exploratory builds) |
| `SETUP_SLACK_MARGIN` | float | 0.0 | -0.5 – 0.5 | extra setup-slack margin during repair (negative caps runaway repair) |
| `ROUTING_LAYER_ADJUSTMENT` | float | 0.5 | 0.1 – 0.7 | global-route congestion adjustment |
| `RECOVER_POWER` | float | 0.0 | 0.0 – 30.0 | % of positive-slack paths eligible for power-saving downsizing |
| `DETAILED_ROUTE_END_ITERATION` | int | 64 | 32 – 64 | max detailed-route iterations (lower cuts runtime, risks DRC on congested designs) |
| `MIN_PLACE_STEP_COEF` | float | 0.95 | 0.95 – 1.00 | lower bound on the global-placement Nesterov step size |
| `MAX_PLACE_STEP_COEF` | float | 1.05 | 1.00 – 1.15 | upper bound on the global-placement Nesterov step size |

**Tier 4 — macro-only (adds to tiers 1–3; requires `has_macros: true`)**

| Knob | Type | Default | Range | Controls |
|---|---|---|---|---|
| `MACRO_PLACE_HALO` | float | 5.0 | 1.0 – 20.0 | exclusion halo (µm, x=y) around each macro for std-cell placement |
| `MACRO_BLOCKAGE_HALO` | float | 2.0 | 1.0 – 15.0 | explicit routing/placement blockage halo around macros |
| `RTLMP_MAX_LEVEL` | int | 2 | 1 – 4 | max depth of the physical hierarchy tree for the RTL macro placer |
| `RTLMP_WIRELENGTH_WT` | float | 100.0 | 10.0 – 200.0 | wirelength weight in the macro-placement cost function |
| `RTLMP_BOUNDARY_WT` | float | 50.0 | 10.0 – 100.0 | weight for pulling macro clusters toward die boundaries |

Each evaluated config is streamed to
`campaigns/<design>/<platform>/results_funnel_campaigns.jsonl` — one JSON line carrying the
config, the fidelity it reached, the reward, and (for full builds) the real
`area_um2 / fmax_mhz / power_mw / timing_met` and the **path to its `6_final.gds`**.

### 2. Get the graphical dashboard (report)

```bash
eda-rl report --campaign latest --open
```

Produces one self-contained HTML file (no server) with: a supervisor overview
(summary banner, comparison table, **area-vs-Fmax Pareto**), optimization history vs
episode and wall-clock, per-parameter reward analysis, the fidelity funnel (how many
configs died at each gate), and Optuna parameter-importance / slice / contour plots.

### 3. Harvest the best configs + their GDS (collect)

```bash
eda-rl collect --campaign latest --render --open
```

Picks the standout F3 builds — best overall score, max Fmax, min area, min power, and
the top-N by score — and writes to `best_configs/<design>_<platform>/`:

```
best_configs/gcd_nangate45/
├── best_configs.html          # before/after comparison page (layout thumbnails with --render)
├── best_configs.json          # manifest: config + metrics + GDS paths
├── BEST_OVERALL__<variant>/
│   ├── 6_final.gds            # the optimized layout, ready to hand off
│   └── 6_finish.rpt
├── MAX_FMAX__<variant>/ …
└── MIN_AREA__<variant>/ …
```

`--render` rasterizes each layout to a thumbnail with KLayout (skipped automatically if
klayout isn't installed). The GDS files are copied from the work dir; if you've cleared
`EDA_RL_WORK`, re-run those configs to regenerate them.

### 4. (optional) Live/interactive dashboard

```bash
pip install -e '.[dashboard]'
eda-rl dashboard --campaign latest
```

---

## Quick smoke test (no ORFS)

```bash
PHYSICAL_MOCK=1 eda-rl optimize --design gcd --platform nangate45 \
    --budget-hours 0.02 --sampler random
PHYSICAL_MOCK=1 eda-rl report  --campaign latest --out /tmp/r.html
PHYSICAL_MOCK=1 eda-rl collect --campaign latest --out /tmp/best
```

## Bring your own design

Create a `DesignSpec` YAML. Relative `rtl_files` resolve against the YAML's own directory,
so a design folder is self-contained and portable:

```yaml
name: my_block
top:  my_block
clock_port: clk

rtl_files:
  - rtl/my_block.v          # relative to this YAML's directory
  - rtl/submodule.v

# Optional: RTL parameters the optimizer may sweep (chparam axes).
params:
  lanes:
    choices: [1, 2, 4, 8]
    default: 4
    rtl_param_name: LANES   # the actual Verilog parameter name

platforms:
  nangate45:
    clock_range_ns: [3.0, 8.0]
    default_clock_ns: 5.0

has_macros: false           # omit to auto-detect at first synth
functional_eval:
  kind: none                # or 'tinyvad_sim' for a design with a behavioral hook

# Optional: reward: (generic-PPA weights/anchors) and knobs: (per-design ORFS
# search-space control) blocks — see the dedicated sections below.
```

### Field reference

Only `name`, `top`, `rtl_files`, `clock_port` are required; everything else is
optional and defaults as shown.

| Field | Required | Type | Default | Notes |
|---|---|---|---|---|
| `name` | yes | str | — | registry key / results-dir nickname (`DESIGN_NICKNAME`). Must match `[A-Za-z_][A-Za-z0-9_]*` — it's embedded in filesystem paths and shell commands, so slashes/quotes/shell metacharacters are rejected. |
| `top` | yes | str | — | Verilog top module (`DESIGN_NAME`, yosys). Same identifier rule as `name`. Can differ from `name` — e.g. `aes.yaml` has `name: aes`, `top: aes_cipher_top`. |
| `rtl_files` | yes | list of paths | — | Relative paths resolve against the YAML's own directory (or `EDA_RL_DESIGN_ROOT` if set — see below); absolute paths are used as-is. |
| `clock_port` | yes | str | — | Clock port name used in the generated SDC. |
| `params` | no | dict | `{}` | RTL chparam sweep axes. Each entry: `choices: [...]` (categorical) or `range: [lo, hi]` (int), optional `default`, optional `rtl_param_name` if the Verilog parameter name differs from the YAML key. Omit or leave `{}` for designs with no configurable RTL parameters (`gcd`, `aes`). |
| `platforms` | no | dict | one `nangate45` entry, `clock_range_ns: [3.0, 8.0]`, `default_clock_ns: 5.0` | Keyed by platform name; in practice only `nangate45` and `asap7` are wired up end-to-end elsewhere in the flow. Each entry: `clock_range_ns: [lo, hi]`, `default_clock_ns: <float>`. |
| `has_macros` | no | bool | `None` (auto-detect at first F2 synth) | Gates the tier-4 macro-placement knobs (see [ORFS knob tiers](#orfs-knob-tiers)) — set `true` only once the design actually instantiates SRAM/macros. |
| `functional_eval.kind` | no | str | `None` (≈ `none`) | Names the **functional-model plugin** the design opts into (`common/functional_models/`). `tinyvad_sim` selects the TinyVAD plugin — its behavioral-sim fidelity (F1) and speedup/accuracy composite reward; any other value — including omitting `functional_eval` — runs the generic PPA reward and skips F1. The core dispatches through `DesignSpec.functional_model()`; add a plugin to introduce a new family. |
| `reward` | no | dict | `None` (auto-anchored) | **Generic designs only** — ignored when `functional_eval.kind: tinyvad_sim`. See below. |
| `knobs` | no | dict | `None` (nothing pinned/dropped) | Per-design control of the ORFS knob search space. See below. |

### `reward:` — tuning the generic PPA objective

Only read for designs that are *not* `functional_eval.kind: tinyvad_sim`
(TinyVAD designs use a separate speedup/accuracy reward, configured in
`search_space_funnel.yaml`, not here). All fields are optional; shown values
are the defaults:

```yaml
reward:
  w_fmax: 1.0                 # weight on normalized Fmax (higher is better)
  w_area: -1.0                 # weight on normalized area (negative = penalize)
  w_power: -0.4                # weight on normalized power (negative = penalize)
  w_timing_violation: -3.0     # penalty applied when timing isn't met
  area_ref_um2: 15000.0        # optional normalization anchors
  power_ref_mw: 2.0
  fmax_ref_mhz: 300.0
```

If you omit `area_ref_um2` / `power_ref_mw` / `fmax_ref_mhz`, the optimizer
auto-anchors each from the design's first successful F3 build (so that build
scores ≈0 and later builds are scored relative to it) — this is what
`gcd.yaml` and `aes.yaml` do today, by omitting `reward:` entirely. Declare
the anchors explicitly once you know representative PPA numbers for the
design (e.g. from a baseline build), so scores are comparable across
separate campaigns.

### `knobs:` — per-design control of the ORFS search space

Optional; lets a design be fully described by its own YAML with no need to
edit `search_space_funnel.yaml`. Omit entirely → every knob up to
`--max-tier` is optimized, nothing pinned. See
[ORFS knob tiers](#orfs-knob-tiers) above for the full knob catalog.

```yaml
knobs:
  fix:                      # pin to a constant + drop from the sampled space
    CORE_UTILIZATION: 40
    PLACE_DENSITY: 0.60
  exclude:                  # drop from the space, use the tool default
    - CORE_MARGIN
  override:                 # retune a knob's range / choices / default
    CORE_ASPECT_RATIO:
      range: [0.8, 1.2]
```

### Worked examples

`eda_rl/designs/` ships three reference shapes:

- **`gcd.yaml`** — self-contained: RTL vendored in-repo, no `params`, no
  functional hook. The minimal shape to copy for a new pure-std-cell design.
- **`tinymac_accel.yaml`** — external repo: RTL lives in another checkout,
  resolved via `EDA_RL_DESIGN_ROOT`; uses `params` (chparam sweep),
  `has_macros: false`, `knobs.fix`, and `functional_eval.kind: tinyvad_sim`.
- **`aes.yaml`** — `top` differs from `name` (`aes_cipher_top` vs `aes`), and
  `rtl_files` are absolute paths into an ORFS install rather than relative
  ones.

Then:

```bash
eda-rl optimize --design path/to/my_block.yaml --platform nangate45 --budget-hours 4
```

**Optimizing a design that lives in another repo?** Keep `rtl_files` relative to that
repo's root and set `EDA_RL_DESIGN_ROOT`:

```bash
EDA_RL_DESIGN_ROOT=/path/to/voiceAI \
  eda-rl optimize --design eda_rl/designs/tinymac_accel.yaml --platform nangate45
```

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `ORFS_DIR` | `/opt/OpenROAD-flow-scripts` | OpenROAD-flow-scripts install |
| `EDA_RL_WORK` | `./eda_rl_runs` | scratch / ORFS WORK_HOME for all per-variant staging + build output |
| `EDA_RL_DESIGN_ROOT` | *(YAML dir)* | base for resolving relative `rtl_files` (e.g. an external repo root) |
| `PHYSICAL_MOCK` | *(unset)* | `1` → skip OpenROAD, return synthetic metrics |
| `ORFS_TIMEOUT` | `2400` | per-build timeout (seconds) |

## Commands

| Command | Description |
|---------|-------------|
| `eda-rl optimize` | run an optimization campaign on a design (the main pipeline) |
| `eda-rl report` | render the graphical HTML analysis dashboard from a campaign |
| `eda-rl collect` | harvest the best configs: copy their GDS + reports + a comparison page |
| `eda-rl dashboard` | launch the live/interactive Optuna dashboard (`[dashboard]` extra) |
| `eda-rl build-table` | pre-build an offline F0–F2 evaluation table (resumable) |
| `eda-rl benchmark` | compare promotion / candidate strategies on the table simulator |

Run `eda-rl <command> --help` for per-command options.

## How it works

See [`docs/08_funnel_optimizer.md`](docs/08_funnel_optimizer.md) (operator guide),
[`docs/07_rl_pipeline_design.md`](docs/07_rl_pipeline_design.md) (rationale + audit), and
[`docs/04_optimizer.md`](docs/04_optimizer.md). In short: a gym-style `FunnelEnv` runs each
candidate through fidelity gates F0→F3; a candidate generator (Optuna TPE / surrogate-UCB /
random) proposes configs; a promotion policy (fixed gates / LinUCB / random) decides what
advances; a quantile-GBT surrogate predicts area/period to rank candidates. Reward pays only
on a terminal F3 build; the failure ladder is monotone so partial progress is scored honestly.

[orfs]: https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
