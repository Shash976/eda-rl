# AGENTS.md — the manual for agents (and humans) working on `eda-rl`

Read this first, top to bottom. It is four things: (1) what this project is,
(2) how to operate it, (3) the invariants you must not silently re-break, and
(4) the history of how it broke before — four audit rounds' worth — so you
recognize the failure patterns before repeating them. For depth:
`docs/rl_system.md` (the RL machinery + role of every file — the one current
doc), `legacy/audits/` (the third audit's findings/recommendations, in full),
and `git log` (every fix is an atomic commit whose body explains what and why).

**The single most important lesson from this repo's history:** its self-tests
check structure, not behavior under real tools. Three separate silent-
corruption bugs (a knob pinned to a constant every episode, a dead cell-count
parser, a fake fmax) passed every self-test and were only caught by running a
real campaign and looking at whether the numbers varied and made physical
sense. When you change anything in the measurement or sampling path, **verify
with a real ORFS run**, not just the mock suite.

## What this is

A **multi-fidelity RL/DSE optimizer for RTL→GDS chip design-space
exploration**. You drop in a design (RTL + a small `DesignSpec` YAML), point
it at an OpenROAD-flow-scripts (ORFS) install, and it searches flow
configurations trading off **area / Fmax / power**, promoting promising
candidates through cheap fidelity gates (legality → behavioral sim → synth+STA
proxy → full place-and-route) and learning where to spend the synthesis
budget. The open research question (docs/rl_system.md §4, audit R2): does a
*learned* promotion policy actually beat fixed gates at spending that budget?
Honest status: not yet demonstrated — and the corpora that could answer it
must be regenerated post-measurement-fixes before any conclusion is drawn.

Worked example designs (`eda_rl/designs/`):

| design | what | platform | why it's here |
|---|---|---|---|
| `tinymac_accel` | TinyVAD MAC accelerator (RTL params: lanes, acc width) | nangate45/asap7 | the original design; TinyVAD composite reward; RTL not vendored here |
| `gcd` | ORFS reference GCD (~200 cells) | nangate45 | smoke test; RTL vendored, fully self-contained |
| `aes` | ORFS reference AES-128 | asap7 | bigger block; top ≠ name exercises DESIGN_NAME/NICKNAME split; RTL from ORFS install |
| `likith` (`id`) | tiny combinational decoder | asap7 | generic reward + asap7 F2 proxy + opt-in constraint knobs; PDN-floor lesson |
| `sagar` (`alu4b`) | tiny combinational 4-bit ALU | sky130hd | generic reward + sky130hd path; PDN-floor lesson |

The superseded first-generation system lives in **repo-root `legacy/`** —
outside the installed package, imported by nothing, frozen verbatim (see
`legacy/README.md`). Its one live piece, the TinyVAD Verilator driver, was
rescued to `common/verilator_sim.py`. Do not spend time under `legacy/`.

## Operating it

```bash
pip install -e .                                # entry point: eda-rl
export ORFS_DIR=/opt/OpenROAD-flow-scripts      # real runs; or PHYSICAL_MOCK=1

# ALWAYS preflight a design before burning campaign budget on it:
eda-rl doctor --design likith --platform asap7            # seconds
eda-rl doctor --design likith --platform asap7 --probe-f3 # + real build, finds the PDN util floor

eda-rl optimize --design gcd --platform nangate45 --budget-hours 4 \
       --sampler tpe|surrogate_ucb|random --promotion fixed|linucb|random \
       --max-tier N       # N must cover the design's declared knobs — doctor prints the minimum
eda-rl report  --campaign latest --open        # static HTML (Pareto, funnel, importances…)
eda-rl collect --campaign latest --render      # best GDS + before/after page
eda-rl build-table --design gcd --max-tier 2   # offline F0–F2 table (resumable)
eda-rl benchmark --seeds 20                    # promotion-policy table benchmark
eda-rl dashboard --log <campaign jsonl> --port 8080   # live Optuna view; pass --log
                                               # explicitly when >1 campaign runs (the
                                               # default picks the most recently written log)
eda-rl fit-surrogate                           # mine campaign logs, fit + CV the surrogate

# No ORFS? prefix any command with PHYSICAL_MOCK=1 (synthetic metrics).
```

Campaign logs land in `eda_rl/campaigns/<design>/<platform>/
results_funnel_campaigns.jsonl` — one design+platform per file, so concurrent
campaigns on different designs are safe (and same-design concurrency is safe
too: RTL staging is content-addressed, variants are flock-serialized).

**Self-tests — run after touching funnel/common (all must pass):**

```bash
PHYSICAL_MOCK=1 python -m eda_rl.funnel.env
python -m eda_rl.funnel.promotion_agent
python -m eda_rl.funnel.candidates
python -m eda_rl.funnel.benchmark_funnel --selftest
python -m eda_rl.common.knobs
python3 tests/test_parsers.py                  # golden-log parser tests (REAL tool output)
PHYSICAL_MOCK=1 eda-rl doctor --design gcd --platform nangate45
PHYSICAL_MOCK=1 python -m eda_rl.funnel.build_table --subset strategic --limit 5  # auto-writes to a temp path under mock
```

These are necessary, not sufficient. `PHYSICAL_MOCK` metrics are TinyMAC-
shaped (they ignore the design), and mock mode fabricates exactly the fields
the real parsers produce — so mock tests **cannot** catch parser/measurement
regressions. That's what `tests/test_parsers.py` (real captured fixtures) and
`eda-rl doctor` (real F2 run) exist for; after touching the measurement path,
also run one real campaign episode and eyeball the numbers.

## Repo layout

```
eda_rl/
  cli.py            # `eda-rl` entry point → dispatches to subcommand main()s
  funnel/           # THE FUNNEL (active system) — see below
  common/           # shared plumbing (runner, rewards, designs, knobs, sim, constants)
  viz/              # report.py (static HTML), dashboard.py (live Optuna), campaign_data.py
  designs/          # per-design DesignSpec YAMLs + vendored RTL (gcd/, likith/, sagar/)
  campaigns/        # committed example campaign logs
  results/          # offline tables (jsonl)
tests/              # golden-log parser tests + real-output fixtures
docs/rl_system.md   # THE doc: how the RL works + role of every file
legacy/             # frozen history, outside the package: gen1/, dead modules,
                    # superseded docs 04/07/08, audits/ (third-audit records)
```

### funnel/ — the active system

| File | Role |
|---|---|
| `env.py` | `FunnelEnv` — gym-style env over fidelity gates **F0 validate+cycle model → F1 behavioral sim (TinyVAD only) → F2 synth+STA proxy → F3 full ORFS flow**. `reset(config)` runs F0; `step(action)` with `{kill, re-proxy, promote, commit}`; terminal on kill/after F3. Live or table mode. Logs every `(config, fidelity, obs)` row. |
| `state_spec.py` | **Single source of truth** for the 22-dim state vector (`IDX_*`, normalization, `unrun = 0.0`). Everything imports from here. |
| `candidates.py` | `CandidateGenerator` — Optuna TPE / surrogate-UCB / random. F3-only tell rule; kill-memo for non-F3 outcomes. |
| `surrogate.py` | Per-metric quantile-GBT surrogate (area/period/power), conditioned on F2 obs. Learns the config-axis schema from its corpus; refuses cross-design predictions. |
| `promotion_agent.py` | `PromotionAgent` (LinUCB), `FixedGateAgent` (baseline, clock-relative kill), `RandomPromotionAgent`. |
| `run_funnel_optimizer.py` | Live campaign driver (`eda-rl optimize`). Probes any auto-loaded surrogate against the campaign design's space; drops it loudly on schema mismatch. |
| `build_table.py` | Resumable offline F0–F2 table builder. |
| `benchmark_funnel.py` | Table-simulator benchmark: random vs fixed vs LinUCB. Scores the **pure terminal reward** (`info["terminal_reward"]`), never the shaped accumulator. |
| `collect_best.py` | `eda-rl collect` — harvest best F3 builds. |
| `fit_surrogate.py` | `eda-rl fit-surrogate` — mine campaign logs, fit + CV-validate the surrogate. |
| `doctor.py` | `eda-rl doctor` — per-design preflight (parsers, knob-range coherence, PDN util floor). |

### common — shared

`physical_runner.py` (drives ORFS `make` at F3, the yosys+OpenSTA proxy at F2,
mock metrics, all report parsing — the extracted `_parse_synth_stat` /
`_parse_sta_timing` / `_parse_metrics` are what `tests/test_parsers.py`
covers, plus the F1 reference-STA `_reference_sta`), `physical_reward.py`
(`compute_physical_reward` = TinyVAD composite, `compute_generic_reward` =
pure PPA), `designs.py`
(`DesignSpec.load`, SDC generation, injection validation), `knobs.py`
(`KnobRegistry`: **27 ORFS knobs in 4 tiers** — the original 24 plus opt-in
`CLOCK_UNCERTAINTY`/`IO_DELAY`/`GR_SEED`; every knob carries an `affects` tag:
`netlist | layout | constraints | environment`), `constants.py` (measured
cycle model, TinyVAD SW-baseline constants, `acc_overflows`), `sim.py`
(mock-aware behavioral-sim wrapper) + `verilator_sim.py` (the real TinyVAD
Verilator driver, rescued from gen1), `recipe.py` (ABC recipe axis).

## How a new design gets optimized (the part people get wrong)

1. **Write the YAML** (`eda_rl/designs/<name>.yaml`): `name`/`top` (safe
   identifiers — validated), `rtl_files` (relative to the YAML's dir or
   absolute), `clock_port` (validated identifier; for a **combinational
   design with no clock pin, any placeholder name works** — OpenSTA turns
   `get_ports` on a missing port + `create_clock` into a proper virtual
   clock; verified on real designs, no code change needed), per-platform
   `clock_range_ns` (**always ns** — `PLATFORM_TIME_UNIT` converts to asap7's
   ps downstream; forgetting this conversion is exactly what
   `eda-rl doctor` FAILs on), `params` (RTL chparam axes, may be `{}`),
   `has_macros`, `functional_eval` (`tinyvad_sim` or `none`).
2. **Knob control lives in the YAML** (`knobs: fix / exclude / override /
   enable`), applied centrally in `KnobRegistry.space()` so live campaigns
   and `build_table` agree. Overrides may deliberately exceed registry
   ranges (doctor WARNs). `constraints`/`environment`-tagged knobs
   (CLOCK_UNCERTAINTY, IO_DELAY, GR_SEED) are **opt-in**: they enter the
   space only if the YAML names them.
3. **Tiny design? Expect PDN-0185.** ORFS's default power-grid needs more
   floorplan than a few dozen cells give it at default utilization — the
   build fails deep in P&R. Override `CORE_UTILIZATION` low (likith ≈5 on
   asap7, sagar ≈15 on sky130hd) or let `eda-rl doctor --probe-f3` bisect
   the floor and print the override block for you.
4. **Mind `--max-tier`.** A design's declared knobs are only sampled if
   their tier is ≤ the campaign's `--max-tier` (doctor prints the minimum
   needed). YAML `override` cannot promote a knob past `max_tier`.
5. **Preflight**: `eda-rl doctor --design X --platform Y` before any real
   campaign. It catches dead parsers, ps-vs-ns range mistakes, missing RTL,
   and tier mismatches in seconds.

## Invariants (do not silently re-break)

### Measurement integrity — the reward must measure the chip, not the ruler
- **Reward is scored on a fixed-ruler reference SDC, never the sampled
  constraints.** After a successful F3, the final netlist is re-timed under
  the design's default constraints (io = 0.2·clock, no uncertainty);
  `wns_ref_ns`/`fmax_ref_mhz`/`period_ref_ns`/`comb_delay_ns` feed the
  reward; sampled-SDC metrics stay in the obs for flow visibility only.
  (Without this, the optimizer's best reward came from loosening its own
  timing budget: corr(reward, IO_DELAY) = −0.83 in a real campaign.)
- **Combinational F2 fmax is `None` + a `combinational` marker, never a
  1000/clk echo.** An inferred (slack-fallback) fmax carries
  `fmax_inferred=True`. The honest combinational speed number is the
  reference STA's measured path delay.
- **F2 cell counts parse the installed yosys tabular `stat`** (and the
  legacy `Number of cells:` form). If a toolchain upgrade changes an output
  format, `tests/test_parsers.py` is where it must fail loudly.
- **F2 times under F3's ruler**: the SDC-owned sampled knobs
  (CLOCK_UNCERTAINTY, IO_DELAY) are forwarded to the F2 proxy; place/route
  knobs are not (the proxy has no floorplan/place/route).
- **Design-aware reward.** TinyVAD designs → `compute_physical_reward`
  (speedup/accuracy composite); generic designs → `compute_generic_reward`
  (PPA; refs auto-anchor from the first F3 build or the YAML `reward:`
  block). Never hardcode TinyMAC constants for all designs.
- **Reward bookkeeping.** Everything outside the promotion bandit uses
  `info["terminal_reward"]` (pure terminal PPA), not the shaped per-step
  accumulator: the driver for TPE/best/log, and the benchmark for scoring
  and its TPE tell. (The shaped sum is path-dependent — promote-through
  policies pay F1+F2+F3 cost, commit pays only F3 — so scoring it against a
  pure-terminal optimum penalized fixed/LinUCB by ~5% at small budgets; this
  exact bug shipped once.)
- **Surrogate-Δ shaping prior is captured pre-stage.** In `env.py
  _run_stage`, `prior_mu` must be read *before* the stage runs — the stage
  mutates `_f2_obs`, and a prior taken afterwards equals the posterior,
  silently zeroing the shaping term for every step (this exact bug shipped
  once, too).
- **The report is design-aware.** `viz/report.py` renders TinyMAC's
  hand-picked baseline, SW-speedup KPI, Lanes/Acc_W columns, and knob-tier
  table only when `_is_tinymac_campaign()`; generic designs anchor
  before/after deltas to their own earliest F3 build. Never reintroduce an
  unconditional TinyMAC reference.

### Knobs & search space
- **Knob ontology is load-bearing.** `affects: constraints|environment`
  knobs are opt-in per design (CLOCK_PERIOD is the one always-on exception —
  it is the performance *target*). Any future SDC-rewriting or
  noise-injecting knob must be tagged and opt-in, or it re-opens the
  reward-gaming hole.
- **Pseudo-typed knobs need a sampling type.** `Knob.type` values like
  `pseudo_sdc`/`pseudo_fastroute` route *emission* (SDC / fastroute.tcl
  instead of config.mk); `space()` maps them to real sampling types
  (`float`/`int`) for Optuna. Skip that mapping and the axis silently pins
  to its default every episode — this exact bug shipped once.
- **GR_SEED keeps ROUTING_LAYER_ADJUSTMENT env-var-live.** The generated
  per-variant `fastroute.tcl` substitutes `$::env(ROUTING_LAYER_ADJUSTMENT)`
  for the platform file's hardcoded literal before appending the seed —
  ORFS sources `FASTROUTE_TCL` *instead of* its env-var branch, so a
  verbatim copy silently disconnects the adjustment axis.
- **CORE_UTILIZATION is an int end-to-end** (sampler, log, variant hash,
  emission). **Tier-2+ knobs reach F3** via
  `FunnelEnv._effective_orfs_knobs()`; don't re-hardcode util/density.
  **Snap only applies to axes declaring `_snap_step`** — a default snap step
  pinned every sub-range float axis to its lower bound once.
- **Design-authoritative knob control**: the YAML `knobs:` block is the only
  place knobs are pinned/dropped/retuned (don't add knob-fixing back into
  `search_space_funnel.yaml`). **F3-only TPE tell**: only terminal F3
  rewards feed Optuna; kills/proxies/table-misses go to the skip-memo.

### State & learning
- **State vector is owned by `funnel/state_spec.py`.** 22 dims, `unrun=0.0`.
  Generic designs: clock (slot [2]) normalizes per-design
  `(clk−lo)/(hi−lo)`; WNS (slot [10]) normalizes by the actual clock
  period; slot [4] is a 3-level platform ordinal (0.0 nangate45 /
  0.5 sky130hd / 1.0 asap7). The tinymac/no-design legacy path keeps the
  old fixed rulers **bit-compatible** (saved agents/tables depend on it).
- **FixedGate's F2 kill is clock-relative** (`wns < −0.5·clock_period` ==
  normalized state[10] < −0.5). An absolute-ns threshold is inert on
  sub-ns platforms and turns the baseline into always-promote.
- **Surrogate schema guard.** The fitted joblib stores the config-axis
  schema (sorted numeric axes + one-hot categoricals; `util ←
  CORE_UTILIZATION`, `density ← PLACE_DENSITY`; recipe flag matches any
  `*area*`); `predict()` refuses configs that don't cover it — no
  cross-design predictions, and pre-schema payloads are rejected with a
  refit instruction. **Obs aliasing**: live keys `area_um2`/`wns_ns` map to
  `proxy_area_um2`/`proxy_wns_ns` via `surrogate._OBS_ALIASES`.

### Security (author-controlled YAML is still validated)
- `name`/`top`/`clock_port` must match `[A-Za-z_][A-Za-z0-9_]*`; YAML
  knob/param values are validated numeric-or-safe-string at load. These flow
  into filesystem paths, `bash -c` strings, SDC TCL, and config.mk `export`
  lines — all real injection sinks that were each exploitable once.
- `validate.py`'s `eval()` constraint sandbox is documented-insufficient but
  acceptable under the threat model (constraint expressions are
  author-controlled YAML); revisit if that changes.

### Operational
- **All tool subprocesses are process-group-killed on timeout/failure**
  (`_run_capture`/`_killpg` for the proxy/elaborate/reference-STA paths, the
  same pattern `run_physical` uses). No detached yosys/openroad survivors.
- **RTL staging is content-addressed** (`src/<design>_<rtlhash8>/`) and
  **variant builds are flock-serialized** — concurrent campaigns are safe,
  including same-design. Variant names embed the RTL hash, so any RTL edit
  invalidates cached builds automatically.
- **Campaign logs are self-describing**: every episode row carries
  design/platform/sampler/promotion/max_tier/seed; the run ends with a
  `{"campaign_summary": …}` row (no `config` key — old readers skip it).
- **Doomed configs are rejected pre-make** (`validate_config` ABORT-RISK →
  `config_abort` immediately, no ORFS_TIMEOUT burn); repeated knob warnings
  are printed once with a suppressed-count summary at exit.
- **`DESIGN_NAME` (= design.top, the yosys top) vs `DESIGN_NICKNAME`
  (= design.name, the results dir) stay split** — aes-style top≠name designs
  break otherwise. **Units**: all stored `*_ns` keys are nanoseconds;
  `PLATFORM_TIME_UNIT` owns the asap7 ps conversion at every boundary.
- **Cells**: F2 has one total synth cell count; F3 carries the post-PnR
  total and an FF count from `6_report.json` *when present* (`ff_count` is
  legitimately `None` for combinational designs).

## History: how this repo broke, and what fixed it

Four audit rounds, each an atomic-commit series (see `git log`); the pattern
each time was *silent* corruption — nothing crashed, numbers just stopped
meaning what everyone thought they meant.

**Round 1** (`fix(gen2)` series, merged): the funnel was quietly
TinyMAC-hardcoded — reward, surrogate conditioning, terminal-reward
bookkeeping, kill gates, and a forked state layout. Established the
design-aware reward, the single 22-dim state spec, and F2/F3 cell surfacing.

**Round 2** (2026-07-01, 18 commits): a real path-traversal/shell-injection
hole via YAML `name`/`top`; `build_table` silently synthesizing TinyMAC RTL
for every design; table-miss episodes feeding phantom rewards to Optuna; the
surrogate scoring every design with TinyVAD math. Established input
validation, design threading, the table-miss skip-memo, reward-kind-aware
UCB, pre-make config rejection, and process-group cleanup in `run_physical`.

**Round 3** (2026-07-06, `legacy/audits/AUDIT_FINDINGS.md`, 17 findings): the
measurement layer itself. The optimizer was *gaming its own reward* through
sampled SDC constraints (F1); a knob fix had silently disconnected another
knob (F2); the F2 cell parser had been dead since a yosys upgrade (F3); F2
fmax was an echo of the requested clock (F4); the surrogate was blind to
almost every axis campaigns actually vary (F5); the "fixed gates" baseline
was inert on fast platforms (F11). Fixed measurement (reference-ruler reward,
honest parsers, knob ontology + opt-in), state normalization, operational
hardening, and added `eda-rl doctor` + golden-log tests. Two framing
consequences stand (see `legacy/audits/RECOMMENDATIONS.md`): the pre-fix
likith/sagar campaign corpora are **unusable for learning conclusions** and
must be re-run (R1), and the LinUCB-vs-fixed question has a recommended
reframing as a surrogate-driven expected-improvement-per-cost rule (R2).

**Round 4** (2026-07-08, the restructure audit + this branch): the learning
signal itself, plus the repo shape. gen1 was retired wholesale to repo-root
`legacy/` (the Verilator driver rescued to `common/verilator_sim.py`; four
dead modules and the stale docs went with it), `gen2/` became `funnel/`
(`funnel.py` → `env.py`), and `fit_surrogate` was wired into the CLI. Two
confirmed RL-signal bugs were fixed: the surrogate-Δ shaping term was
identically zero (prior captured after the stage had already mutated the obs
it conditions on) and the benchmark scored the path-dependent shaped
accumulator against a pure-terminal optimum (penalizing promote-through
policies ~5% at small budgets). `viz/report.py` stopped fabricating TinyMAC
baselines/speedups for generic-design campaigns. Known-but-unchanged
learning-signal weaknesses (budget cost ~100× under-scaled, kill pinned at 0,
no terminal credit to earlier promotes) are documented in
`docs/rl_system.md` §5 — do not "fix" them casually; they change every
historical number and the corpora need the R1 re-run first.

**Discoveries from the likith/sagar onboarding session** (verified on real
tools; the reason several invariants above exist):
- *The clockless-design scare was a false alarm*: OpenSTA silently makes a
  virtual clock from a missing port — static analysis predicted a crash;
  a real run disproved it. Verify hypotheses with runs, in both directions.
- *PDN-0185 is the tiny-design wall*: tool-default CORE_UTILIZATION ranges
  physically cannot fit ORFS's power grid on a dozens-of-cells floorplan.
  Now automated in `eda-rl doctor --probe-f3`.
- *The pseudo-knob sampling pin*: three new knobs passed every structural
  test while being constant in every live episode (`Knob.type` leaked into
  the Optuna sampling layer, which didn't know the pseudo types). Caught
  only by eyeballing sampled values in a real campaign log.
- *The surrogate schema round-trip gap*: the F5 featurization was correct
  in-process but didn't persist its schema through save/load — a loaded
  model would have silently lost both its features and its cross-design
  guard. Caught in review of a half-landed change; the save/load round-trip
  is now asserted in verification.
- *asap7 has no merged liberty*: its std cells ship as 5 split NLDM libs
  (4 gzipped), which is why the F2 proxy passes repeated `-liberty` flags
  (dfflibmap gets the single SEQ lib) instead of one merged file.

The raw ORFS-AutoTuner bundles the likith/sagar designs originally arrived
as (config.mk / fastroute.tcl / constraint.sdc) were deleted in the cleanup —
eda-rl never read them, and likith's config.mk referenced a file that never
existed. Each design keeps its `autotuner.json` purely as the provenance
record its YAML knob ranges mirror.

## Env vars

`ORFS_DIR` (ORFS install), `EDA_RL_WORK` (scratch/WORK_HOME, default
`./eda_rl_runs` — no GC yet, watch disk on long campaigns),
`EDA_RL_DESIGN_ROOT` (base for relative `rtl_files`), `PHYSICAL_MOCK=1`
(synthetic metrics), `ORFS_TIMEOUT` / `PROXY_TIMEOUT` (per-build / per-proxy
seconds).

## Conventions

- **Branch before committing on `main`.** Commit messages end with a
  `Done by an AI agent` line instead of a co-author trailer.
- **Don't commit run artifacts.** Per-fidelity traces (`funnel_*.jsonl`) are
  gitignored; only small example campaign logs under `campaigns/` are
  committed. `eda_rl/results/funnel/*.joblib` is gitignored.
- A non-TinyMAC design needs RTL resolvable on the machine (gcd/likith/sagar
  RTL is vendored; aes/tinymac RTL is not).
- Keep this file true. Every audit round found stale claims here being
  re-trusted by the next reader — when you land a behavior change, update
  the invariant in the same commit.
