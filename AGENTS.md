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
and ORFS's `gcd`/`aes` are worked examples, joined by `likith` (a combinational
decoder on asap7) and `sagar` (a combinational ALU on sky130hd) — two tiny
combinational blocks that exercise the generic-reward and new-platform paths.
**PDN-0185 lesson:** on tiny floorplans the tool-default `CORE_UTILIZATION`
ranges make ORFS's PDN step fail; use `knobs.override` to pin utilization low
(likith util≈5, sagar util≈15) and `eda-rl doctor --probe-f3` to find the floor.

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

Two framing updates from the third audit (see `RECOMMENDATIONS.md`): (R1) the
`likith`/`sagar` campaign corpora predate the measurement fixes (F1–F4) and must
be **re-run before any learning conclusion is drawn** — they were graded on a
gameable objective and F2 zeros/echoes. (R2) the LinUCB-vs-fixed-gates question
now has a recommended alternative on record: a surrogate-driven expected-
improvement-per-cost decision rule (`E[max(0, reward−incumbent) | obs] / cost >
τ(budget)`), which degrades to the fixed-gate funnel at cold start; keep LinUCB
as a comparison arm, not the protagonist.

### Second audit round (2026-07-01, branch `fix/gen2-audit-findings`)

A follow-up audit (scope: `eda_rl/` excluding `gen1/`) found a real security
hole and three silent-corruption bugs that lived in code paths the first
audit didn't reach (`run_funnel_optimizer.py`, `build_table.py`,
`candidates.py`). All 18 findings were fixed as 18 atomic commits on
`fix/gen2-audit-findings` (not yet merged — see that branch for review/PR).
Treat these as invariants once merged:

- **Design name/top are validated.** `DesignSpec.load()` now rejects any
  `name`/`top` that isn't `[A-Za-z_][A-Za-z0-9_]*`. Before this, a malicious
  design YAML could path-traverse (`Path.__truediv__` on an absolute-path
  name discards the base dir) or break out of the single-quoted `bash -c`
  strings in `physical_runner.py`, since both sinks build paths/shell
  commands straight from the YAML-supplied name.
- **`build_table.py` threads `design` through F0/F1/F2.** Previously
  `_eval_f0`/`_eval_f1`/`_eval_f2` took no `design` param, so
  `build-table --design gcd` silently synthesized TinyMAC RTL for every row
  (`run_synth_sta(design=None)` defaults to tinymac_accel).
  `build_table --subset strategic --limit 5` and building the design-authoritative
  YAML `knobs:` block both depend on `design` reaching these evaluators now.
- **`run_funnel_optimizer.py` routes table-miss episodes to the skip-memo,
  not `_study.tell()`.** Previously a table-miss episode kept
  `fidelity_reached == "F3"` (deepest fidelity *attempted*, not real), so
  `CandidateGenerator.update()` told Optuna a phantom near-zero reward as if
  it were a genuine F3 result. Mirrors the pattern `benchmark_funnel.py`
  already had right (`fidelity="table_miss"`).
- **`CandidateGenerator`'s `surrogate_ucb` sampler is reward-kind aware.**
  `_rebuild_ucb_pool` now passes `reward_kind`/`refs` (derived the same way
  `FunnelEnv._surrogate_reward_kind()` does) into
  `Surrogate.predict_reward_stats()`. Before, every design — TinyVAD or not —
  got scored with the TinyVAD reward formula and TinyMAC reference constants.
- **Doomed placer configs are rejected before the ORFS timeout, not after.**
  `run_physical()` now runs `validate_config()`'s ABORT-RISK/ERROR checks
  before invoking `make`, returning `status: "config_abort"` immediately
  instead of burning up to `ORFS_TIMEOUT` seconds on a build that was always
  going to fail. F2 (`run_synth_sta`) is untouched — it never floorplans/places,
  so placer-abort risk doesn't apply there.
- **`compute_generic_reward` warns when called with empty `refs`.** The
  self-normalizing bootstrap behavior (`FunnelEnv`'s "first F3 build anchors
  the refs") is unchanged and does not warn; only *other* callers that pass
  no refs at all now get a `UserWarning` instead of a silent constant reward.
- **`PromotionAgent.save()`/`load()` round-trips a custom `actions` tuple.**
  Was previously dropped on load and reconstructed from `_DEFAULT_ACTIONS`
  regardless of what the saved agent actually used.
- **`compute_physical_reward`'s `max_speedup` default matches
  `constants.MAX_SPEEDUP_FULL` (1024), not a stale literal `576.0`.** They
  only agreed before because `search_space_funnel.yaml` always supplies
  `reward.max_speedup: 1024.0`; a custom space YAML without that key would
  silently reintroduce a previously-fixed miscalibration.
- **`funnel.py`'s constraint-skip logic is word-boundary, not substring.**
  Guards against a future axis name that's a substring of another
  constraint's variable name.
- **Process-group cleanup on any `communicate()` failure**, not just
  `TimeoutExpired` — `physical_runner.run_physical` now kills the ORFS
  process group on any exception from `proc.communicate()`, not just a
  timeout.
- **Dead code removed:** `funnel.py`'s unused `compute_cascade_reward`
  import and unenforced `self._gates` assignment (gates are loaded from YAML
  but not applied — see the file's comment); `viz/comparison.py` (unreachable
  via `cli.py`, TinyMAC-hardcoded, stale paths — deleted outright).
- **`pyproject.toml`'s `[dashboard]` extra installs `optuna-dashboard`**, not
  `streamlit` (the latter is gen1-only; `eda-rl dashboard` needs the former).
- **`viz/report.py` HTML-escapes campaign-derived strings** (`abc_recipe`,
  titles, section headings, etc.) before interpolating into HTML/Plotly
  hovertext.
- `docs/08_funnel_optimizer.md` now points at the real `eda_rl/...` module
  paths and CLI subcommands instead of a fictional `optimizer/` package.
- `.gitignore` covers `*_trial.jsonl` and `best_configs/` (ad hoc trial logs
  / `eda-rl collect` output).
- `validate.py`'s `eval()`-based constraint sandbox (`__builtins__: {}`) is
  documented in-code as a known-insufficient sandbox — fine under this
  repo's threat model (constraint expressions are author-controlled YAML,
  not user input); revisit only if that assumption changes.

### Third audit round (2026-07-06, branch `fix/audit-2026-07-06`)

A third audit (scope: `gen2/`, `common/`, the working-tree diff that added the
`likith`/`sagar` designs + asap7 F2 proxy + SDC/fastroute knobs, and the two
real overnight campaigns) found that the *measurement layer* was the weak
point: the objective could be gamed through the SDC and the cheap fidelities
were feeding zeros and constraint echoes to everything that learns. 17 findings
(F1–F17, `AUDIT_FINDINGS.md`) landed as atomic commits; 9 recommendations
(R1–R9, `RECOMMENDATIONS.md`) frame the follow-up. Treat these as invariants:

- **Reward is scored on a fixed-ruler reference SDC, never the sampled
  constraints (F1).** After a successful F3 build the final netlist is re-timed
  under the design's *default* constraints (io = 0.2·clock, no uncertainty) and
  the reward reads `wns_ref_ns`/`fmax_ref_mhz`/`period_ref_ns`; the sampled-SDC
  metrics stay in the obs for flow visibility only. Without this the optimizer
  earns its best reward by relaxing its own timing budget (real sagar campaign:
  corr(reward, IO_DELAY) = −0.83). Legacy tinymac already uses the reference
  ruler, so its ref keys mirror the sampled ones (no extra STA).
- **Constraint/environment knobs are opt-in per design (R6/F7).** Each knob
  carries an `affects` tag (`netlist | layout | constraints | environment`);
  `constraints`/`environment` knobs (CLOCK_UNCERTAINTY, IO_DELAY, GR_SEED) enter
  a design's space only when its YAML names them under `knobs.override`/
  `knobs.enable`. **CLOCK_PERIOD is the one always-on exception** — it is the
  performance target, not a measurement-relaxing knob. Don't let SDC-ish knobs
  auto-enter every tier-2+ space again.
- **GR_SEED keeps ROUTING_LAYER_ADJUSTMENT env-var-live (F2).** The generated
  `fastroute.tcl` reproduces the platform file but substitutes
  `$::env(ROUTING_LAYER_ADJUSTMENT)` for the hardcoded literal before appending
  the seed. Copying the platform file verbatim (the old behavior) made ORFS
  source it *instead of* its env-var branch, so the sampled adjustment silently
  never reached the router.
- **F2 cell counts parse the installed yosys tabular stat (F3).** The regex
  matches both `Number of cells: N` and the new `N <area> cells` total line
  (last occurrence). The old parser matched nothing on current yosys, so state
  dims [11]/[12] were 0.0 in every live episode.
- **Combinational F2 fmax is `None` + a `combinational` marker, never a
  1000/clk echo (F4).** `report_clock_min_period` prints `fmax = inf` for a
  design with no reg-to-reg path; that is recorded honestly, not fabricated from
  the clock. An inferred (slack-fallback) fmax carries `fmax_inferred = True`.
  The honest combinational speed number comes from F1's reference STA
  (`report_checks -path_delay max`).
- **F2 receives the sampled SDC knobs so both fidelities time under one ruler
  (F8).** `_run_f2` forwards the SDC-owned subset (CLOCK_UNCERTAINTY, IO_DELAY)
  to `run_synth_sta`; placement/routing knobs still stay out (proxy has no
  floorplan/place/route). Preserves the F2→F3 timing correlation the kill
  decisions depend on.
- **CORE_UTILIZATION is an int end-to-end (F9).** Declared `type: int` so the
  sampler, log, variant hash, and ORFS emission agree; the funnel emits
  `int(round(...))`. Per-design float override ranges still work via
  `suggest_int`.
- **Per-design clock/WNS state normalization + 3-level platform ordinal (F10).**
  Generic designs normalize clock (state[2]) as `(clk−lo)/(hi−lo)` from their
  own `clock_range_ns` and WNS (state[10]) by the actual clock period; state[4]
  is a 3-level ordinal (0.0=nangate45, 0.5=sky130hd, 1.0=asap7). The
  tinymac/no-design legacy path keeps the old fixed rulers and is **bit-
  compatible** (saved LinUCB agents / benchmark tables unaffected). STATE_DIM
  stays 22 — the deliberate bump + knob-summary block is deferred to R2.
- **FixedGate F2 kill is clock-relative (F11).** Kills when
  `wns < −0.5·clock_period` (== normalized state[10] < −0.5), computable from
  the F10-normalized state. The old absolute −2.5 ns threshold was inert on
  sub-ns platforms, degenerating the "fixed gates" baseline to always-promote.
- **Proxy/elaborate subprocesses are process-group-killed (F12).**
  `run_synth_sta`/`run_elaborate` use the same `start_new_session` +
  `communicate(timeout)` + `os.killpg` pattern as `run_physical` (factored into
  `_run_capture`/`_killpg`), so a hung yosys/openroad grandchild no longer
  survives the timeout.
- **RTL staging is content-addressed + variant-locked (F13).** RTL stages into
  `src/<design>_<rtlhash8>/` (the digest the variant name already uses) so two
  campaigns sharing an `EDA_RL_WORK` can't cross-contaminate; an exclusive
  `flock` per variant dir serializes same-variant races (poll-for-peer-GDS).
- **Campaign logs are self-describing (F15).** Each episode row carries
  design/platform/sampler/promotion/max_tier/seed, and the run ends with a
  trailing `{"campaign_summary": …}` row (no `config` key, so
  `campaign_data.load_campaign_rows` skips it — old logs stay loadable). Also
  fixes the budget double-count and adds a reset-failure spin-guard.
- **gen2/common no longer import gen1 (F16).** `SW_BASELINE_CLOCK_NS`/
  `SW_BASELINE_LATENCY_NS`/`acc_overflows` moved to `common/constants.py` and
  the mock-aware `_run_sim` wrapper to `common/sim.py`. The reward and surrogate
  are gen1-free; only the real Verilator harness (TinyVAD-only) still lives in
  gen1, reached lazily through `common/sim.py`.
- **Surrogate featurizes the discovered axis schema and refuses cross-design
  predictions (F5).** `fit()` learns the config-axis schema from the corpus
  (every numeric axis sorted by name, small categoricals one-hot; `util ←
  CORE_UTILIZATION`, `density ← PLACE_DENSITY`) and stores it in the joblib;
  `predict()` refuses (ValueError) any config not covering the stored axes —
  which also neutralizes the `surrogate_n45.joblib` cross-design auto-load.
  Pre-schema fitted payloads are refused with a refit instruction.
- **Snap only applies to axes declaring `_snap_step` (F6).** Absence means
  "don't snap"; the old 0.5 ns default pinned every sub-range float axis to its
  lower bound under `grid_snap`. `build_table`'s clock grid step is now
  range-derived and asserted in-range.
- **`clock_port` and YAML knob/param values are injection-validated (F14).**
  `_SAFE_IDENT_RE` now also guards `clock_port` (interpolated into SDC TCL) and
  `knobs.fix`/`knobs.override`/`params` values (which flow into config.mk
  `export`), closing the sibling holes to the earlier name/top fix.

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
| `surrogate.py` | Per-metric quantile-GBT surrogate (area/period/power). Conditions on F2 observables. `predict_reward_stats(reward_kind=…)` matches the design-aware reward. `fit()` learns the config-axis schema from the corpus and stores it; `predict()` refuses configs that don't cover it (no cross-design predictions). |
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
`knobs.py` (`KnobRegistry`, 27 ORFS knobs in 4 tiers — the original 24 plus the
three opt-in SDC/route knobs `CLOCK_UNCERTAINTY`, `IO_DELAY`, `GR_SEED`; each
knob carries an `affects` tag, see the third-audit invariant), `constants.py`
(measured cycle model, `MAX_SPEEDUP_*`, the TinyVAD SW-baseline constants +
`acc_overflows` formerly in gen1), `sim.py` (mock-aware `_run_sim` wrapper),
`recipe.py` (ABC recipe axis).

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
eda-rl doctor --design likith --platform asap7 # per-design preflight (F2 proxy sanity;
                                               #   --probe-f3 bisects util for the PDN floor)

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
python3 tests/test_parsers.py                  # golden-log parser tests (real tool output)
PHYSICAL_MOCK=1 eda-rl doctor --design gcd --platform nangate45   # preflight smoke
```

(`eda-rl doctor` and `tests/test_parsers.py` land in this same branch — a
concurrent change. The parser fixtures catch the F3/F4-class silent parser
deaths that mock-based self-tests structurally can't — mock fabricates exactly
the fields the parsers should produce.)

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
- **Surrogate schema guard.** The fitted joblib stores the config-axis schema it
  was trained on (sorted numeric axes + one-hot categoricals; `util ←
  CORE_UTILIZATION`, `density ← PLACE_DENSITY`); `predict()` refuses any config
  that doesn't cover it, so a surrogate can't silently score another design's
  space. Refit if you change the axis set; pre-schema payloads are rejected.
- **`PHYSICAL_MOCK` metrics are TinyMAC-shaped** — they depend only on
  lanes/acc_w/clk and ignore the design. So a gcd mock campaign yields constant
  area/Fmax and always-violated timing; **don't read real PPA behaviour from gcd
  in mock.** Validate generic-reward *ordering* with unit tests, not mock runs.
- **Cache invalidation.** Variant names embed an 8-hex RTL content hash, so any
  RTL edit invalidates cached builds automatically.
- **Cells.** F2 reports a single total synth cell count (so `ff_count == cell_count`
  there), parsed from the installed yosys tabular `stat` (F3 fix); F3 carries the
  post-PnR total **and** a real FF count from `6_report.json` *when present* —
  `ff_count` legitimately stays `None` for a purely combinational design (no
  sequential-cell key), so the "real FF count" invariant holds "when present".
- **Units.** asap7 SDC/reports are picoseconds — `PLATFORM_TIME_UNIT` converts;
  all stored `*_ns` keys are nanoseconds.

## Env vars

`ORFS_DIR` (ORFS install), `EDA_RL_WORK` (scratch/WORK_HOME, default
`./eda_rl_runs`), `EDA_RL_DESIGN_ROOT` (base for relative `rtl_files`),
`PHYSICAL_MOCK=1` (synthetic metrics), `ORFS_TIMEOUT` (per-build seconds).

## Conventions

- **Branch before committing on `main`.** Instead of commiting with co-author, just add a note at the bottom that says "Done by an AI agent"
- **Don't commit run artifacts.** Per-fidelity traces (`funnel_*.jsonl`) are
  gitignored; only the small `results_funnel_campaigns.jsonl` example logs under
  `campaigns/` are committed (kept for demos).
- A non-TinyMAC design needs RTL files resolvable on the machine; `aes.yaml`'s
  example campaign log is committed but its RTL is not vendored (gcd's is).
