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
| `--max-tier` | `1`–`4`, `1` | how many ORFS knob tiers to search (1 = core axes; 4 = +macro knobs). Tier-2+ knobs now flow all the way to the F3 full build, not just the F2 proxy |
| `--sampler` | `tpe` \| `surrogate_ucb` \| `random` | how candidates are proposed |
| `--promotion` | `fixed` \| `linucb` \| `random` | the policy that decides what advances F0→F3 |
| `--seed` | int, `0` | reproducibility |
| `--out` | path | where the campaign log goes |

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

# Optional: per-design control of the ORFS knob search space, so a design is
# fully described by its own YAML (no need to edit search_space_funnel.yaml).
# Omit entirely → every knob up to --max-tier is optimized, nothing pinned.
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
