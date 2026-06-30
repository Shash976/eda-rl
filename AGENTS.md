# AGENTS.md — orientation for AI agents working on `eda-rl`

Read this first. It captures where the project is, how it's laid out, and the
invariants you must not silently re-break. For depth, see `docs/07`, `docs/08`,
`docs/04` and `README.md` — don't re-read every source file to get oriented.

## What this is

A **multi-fidelity RL/DSE optimizer for RTL→GDS chip design-space exploration**.
You drop in a design (RTL + a ~10-line `DesignSpec` YAML), point it at an
OpenROAD-flow-scripts (ORFS) install, and it searches the flow for configs that
trade off **area / Fmax / power**, promoting promising candidates through cheap
proxies (legality → synth → full place-and-route) and learning where to spend
the synthesis budget. It is **design-agnostic**: TinyMAC (a TinyVAD accelerator)
and ORFS's `gcd`/`aes` are worked examples.

## Current status (post audit, PR #1 merged into `main`)

The **gen2 funnel** is the active system. A logic audit (see git history
`fix(gen2): design-aware reward …`) landed these fixes — treat them as
invariants, not optional:

- **Design-aware reward.** TinyVAD designs use the speedup/accuracy composite;
  generic designs use a pure PPA reward. (Was hardwired to TinyMAC.)
- **Surrogate conditions on F2 obs** (live area/WNS now reach it).
- **Runner records the pure terminal PPA reward**, not the shaped accumulator.
- **FixedGate kill-gate** no longer lets catastrophic-timing configs escape.
- **One canonical 22-dim state spec** (`gen2/state_spec.py`).
- **Cell counts** surfaced at F2 (synth) and F3 (post-PnR) in logs + report + collect.

**What remains** (from `docs/08`, "What remains"): build real F3 rows into the
offline table → run the real LinUCB-vs-fixed-gates benchmark; the full
594-config F2 table; asap7 transfer test; RTL requantize-pipelining (the ~3.7 ns
Fmax wall); PPO upgrade of the promotion policy *only if* the bandit measurably
loses to lookahead. Honest result so far: **cold-start LinUCB does not beat fixed
gates** on the synthetic table — learning must earn its keep.

## Repo layout

```
eda_rl/
  cli.py            # `eda-rl` entry point → dispatches to subcommand main()s
  gen1/             # 1st gen: single-step black-box DSE (env, cascade, agents/)
  gen2/             # THE FUNNEL (active system) — see below
  common/           # shared plumbing (runner, rewards, designs, knobs, constants)
  viz/              # report.py (static HTML), dashboard.py (live Optuna), campaign_data.py
  designs/          # per-design DesignSpec YAMLs (tinymac_accel, gcd, aes) + gcd/gcd.v
  campaigns/        # committed example campaign logs (results_funnel_campaigns.jsonl)
  results/          # offline tables (gen1/gen2 jsonl)
docs/               # 04 (gen1 optimizer), 07 (RL rationale + audit), 08 (funnel operator guide)
```

### gen2 — the funnel (the part you'll mostly touch)

| File | Role |
|---|---|
| `funnel.py` | `FunnelEnv` — gym-style env over fidelity gates **F0 validate+cycle model → F1 behavioral sim → F2 synth+STA proxy → F3 full ORFS flow**. `reset(config)` runs F0; `step(action)` with `{kill, re-proxy, promote, commit}`; terminal on kill/after F3. Live (real tools) or table mode (replays logged rows). Logs every `(config, fidelity, obs)` row. |
| `state_spec.py` | **Single source of truth** for the 22-dim state vector (`IDX_*`, normalization, `unrun = 0.0`). `funnel`, `promotion_agent`, `benchmark_funnel` all import from here. |
| `candidates.py` | `CandidateGenerator` — Optuna TPE / surrogate-UCB / random. **F3-only tell rule**: only terminal F3 rewards feed the study. |
| `surrogate.py` | Per-metric quantile-GBT surrogate (area/period/power). Conditions on F2 observables. `predict_reward_stats(reward_kind=…)` matches the design-aware reward. |
| `promotion_agent.py` | `PromotionAgent` (LinUCB), `FixedGateAgent` (the baseline to beat), `RandomPromotionAgent`. |
| `run_funnel_optimizer.py` | Live campaign driver (`eda-rl optimize`). |
| `build_table.py` | Resumable offline F0–F2 table builder. |
| `benchmark_funnel.py` | Table-simulator benchmark: random vs fixed-gate vs LinUCB. |
| `collect_best.py` | `eda-rl collect` — harvest best F3 builds (GDS + comparison page). |
| `fit_surrogate.py` | Mine campaign logs / report tree, fit + CV-validate the surrogate. |

### common — shared

`physical_runner.py` (drives ORFS / `run_synth_sta` proxy / `_mock_metrics`,
parses reports incl. `6_report.json` cell counts), `physical_reward.py`
(`compute_physical_reward` = TinyVAD, `compute_generic_reward` = PPA),
`cascade_reward.py` (monotone failure ladder), `designs.py` (`DesignSpec.load`),
`knobs.py` (`KnobRegistry`, 24 ORFS knobs in 4 tiers), `constants.py` (measured
cycle model, `MAX_SPEEDUP_*`), `recipe.py` (ABC recipe axis).

## Commands

```bash
pip install -e .                       # editable install (entry point: eda-rl)
export ORFS_DIR=/opt/OpenROAD-flow-scripts     # real runs; or set PHYSICAL_MOCK=1

eda-rl optimize --design gcd --platform nangate45 --budget-hours 4 \
       --sampler tpe|surrogate_ucb|random --promotion fixed|linucb|random
eda-rl report  --campaign latest --open        # static HTML (Pareto, funnel, importances…)
eda-rl collect --campaign latest --render      # best GDS + before/after page
eda-rl build-table --design gcd --max-tier 2   # offline F0–F2 table (resumable)
eda-rl benchmark --seeds 20                     # promotion-policy table benchmark

# No ORFS? prefix any command with PHYSICAL_MOCK=1 (synthetic metrics).
PHYSICAL_MOCK=1 eda-rl optimize --design gcd --budget-hours 0.02 --sampler random
```

**Self-tests** (no real tools needed) — run these after touching gen2:

```bash
PHYSICAL_MOCK=1 python -m eda_rl.gen2.funnel
python -m eda_rl.gen2.promotion_agent
python -m eda_rl.gen2.candidates
python -m eda_rl.gen2.benchmark_funnel --selftest
PHYSICAL_MOCK=1 python -m eda_rl.gen2.build_table --subset strategic --limit 5
```

## Invariants & gotchas (don't re-break these)

- **Design-aware reward.** `FunnelEnv._terminal_reward` branches on
  `design.is_tinyvad()`: TinyVAD → `compute_physical_reward` (speedup/accuracy,
  TinyMAC anchors); generic → `compute_generic_reward` (PPA, refs auto-anchored
  from the design's first F3 build or its YAML `reward:` block). Never hardcode
  TinyMAC constants for all designs.
- **State vector is owned by `gen2/state_spec.py`.** 22 dims, `unrun = 0.0`,
  `[3]=recipe_idx/2`, `[4]=platform flag`. Don't fork the layout.
- **Design-authoritative knob control.** A design's optional YAML `knobs:` block
  (`fix` / `exclude` / `override`) is the single place knobs are pinned/dropped/
  retuned, applied centrally in `KnobRegistry.space()` so the live optimizer and
  `build_table` agree. Don't reintroduce knob-fixing in
  `search_space_funnel.yaml` (its `fixed:` block is dead). TinyMAC's
  `CORE_UTILIZATION=40`/`PLACE_DENSITY=0.60` live in `tinymac_accel.yaml`.
- **Tier-2+ knobs reach F3.** `FunnelEnv._effective_orfs_knobs()` merges
  design-fixed constants with the sampled config (minus the four non-ORFS axes:
  `clock_period_ns`, `abc_recipe`, `mac_lanes`, `accumulator_width`) and passes
  them to `run_physical`. F2 stays untouched (proxy = synth+STA, no
  floorplan/place/route). Don't re-hardcode util/density at F3.
- **`DESIGN_NAME` vs `DESIGN_NICKNAME`.** The runner emits
  `DESIGN_NAME = design.top` (yosys top module) and `DESIGN_NICKNAME = design.name`
  (results/logs dir). Don't collapse them — designs whose top ≠ registry name
  (e.g. aes, top `aes_cipher_top`) break if `DESIGN_NAME` is overloaded.
- **Variant `L/A` tokens are conditional.** `variant_name()` only embeds
  `L{lanes}_A{acc_w}` when `design.params` declares `mac_lanes`/`accumulator_width`
  (the legacy tinymac/`design is None` path keeps the old prefix for cache reach).
- **F3-only TPE tell.** Only terminal F3 rewards feed the Optuna study; kills/
  proxy results go to a skip-memo (`candidates.py`).
- **Reward bookkeeping.** The runner uses `info["terminal_reward"]` (pure PPA),
  not the shaped per-step accumulator, for TPE/best/log.
- **Surrogate obs aliasing.** Live obs keys `area_um2`/`wns_ns` map to the
  surrogate's `proxy_area_um2`/`proxy_wns_ns` columns (`surrogate._OBS_ALIASES`).
- **`PHYSICAL_MOCK` metrics are TinyMAC-shaped** — they depend only on
  lanes/acc_w/clk and ignore the design. So a gcd mock campaign yields constant
  area/Fmax and always-violated timing; **don't read real PPA behaviour from gcd
  in mock.** Validate generic-reward *ordering* with unit tests, not mock runs.
- **Cache invalidation.** Variant names embed an 8-hex RTL content hash, so any
  RTL edit invalidates cached builds automatically.
- **Cells.** F2 reports a single total synth cell count (so `ff_count == cell_count`
  there); F3 carries post-PnR total **and** a real FF count from `6_report.json`.
- **Units.** asap7 SDC/reports are picoseconds — `PLATFORM_TIME_UNIT` converts;
  all stored `*_ns` keys are nanoseconds.

## Env vars

`ORFS_DIR` (ORFS install), `EDA_RL_WORK` (scratch/WORK_HOME, default
`./eda_rl_runs`), `EDA_RL_DESIGN_ROOT` (base for relative `rtl_files`),
`PHYSICAL_MOCK=1` (synthetic metrics), `ORFS_TIMEOUT` (per-build seconds).

## Conventions

- **Branch before committing on `main`.** End commit messages with the
  `Co-Authored-By` / `Claude-Session` trailers per the harness.
- **Don't commit run artifacts.** Per-fidelity traces (`funnel_*.jsonl`) are
  gitignored; only the small `results_funnel_campaigns.jsonl` example logs under
  `campaigns/` are committed (kept for demos).
- A non-TinyMAC design needs RTL files resolvable on the machine; `aes.yaml`'s
  example campaign log is committed but its RTL is not vendored (gcd's is).
