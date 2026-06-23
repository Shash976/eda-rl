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

## Quick start

```bash
# Smoke test on the bundled, self-contained gcd example (no ORFS needed):
PHYSICAL_MOCK=1 eda-rl optimize --design gcd --platform nangate45 \
    --budget-hours 0.02 --sampler random

# A real campaign once ORFS_DIR is set:
eda-rl optimize --design gcd --platform nangate45 --budget-hours 4
```

Results stream to `campaigns/<design>/<platform>/results_funnel_campaigns.jsonl` (one
JSON line per evaluated config, with the terminal area/Fmax/power), and a running
incumbent is printed. Override the output path with `--out`. Render an HTML report:

```bash
eda-rl report --campaign all --open
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
| `eda-rl build-table` | pre-build an offline F0–F2 evaluation table (resumable) |
| `eda-rl benchmark` | compare promotion / candidate strategies on the table simulator |
| `eda-rl report` | render an HTML report from a campaign log |

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
