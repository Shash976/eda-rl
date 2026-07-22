# eda-rl vs. ORFS AutoTuner — architectural & theoretical differences

*Scope: this first half is a **conceptual comparison** of the two systems as
they exist today, grounded in their source. A second half (appended after the
head-to-head runs) reports **empirical results** on common designs. Evidence is
cited as `path:line`.*

Systems compared:
- **eda-rl** — the funnel in this repo (`eda_rl/funnel/`), a multi-fidelity
  RL/DSE optimizer. See `docs/rl_system.md`.
- **ORFS AutoTuner** — `OpenROAD-flow-scripts/tools/AutoTuner/`
  (`src/autotuner/distributed.py`, `utils.py`), the upstream Ray-Tune
  hyperparameter-tuning framework for ORFS.

---

## 0. TL;DR

Both search ORFS flow configurations for better PPA (area / speed / power), but
they answer **different questions**:

- **AutoTuner** asks *"given many full-flow evaluations, which knob vector is
  best?"* — it is a **single-fidelity, distributed hyperparameter optimizer**.
  Every trial runs the **entire** RTL→GDS flow to completion, then scores it.
  Its leverage is **parallelism** (Ray across many CPUs/nodes) plus a choice of
  off-the-shelf search algorithms.

- **eda-rl** asks *"given a limited synthesis budget, where should I spend it?"*
  — it is a **multi-fidelity optimizer with a learned promotion policy**. Cheap
  gates (validate → sim → synth+STA proxy) triage candidates before any full
  place-and-route, and an RL agent (LinUCB) learns which candidates deserve the
  expensive F3 build. Its leverage is **not wasting P&R runs** on candidates a
  cheap proxy already predicts are bad.

In one line: **AutoTuner parallelizes the brute force; eda-rl tries to avoid the
brute force.**

---

## 1. Side-by-side

| Axis | **eda-rl** (funnel) | **ORFS AutoTuner** |
|---|---|---|
| Category | Multi-fidelity RL/DSE with learned budget allocation | Single-fidelity distributed hyperparameter optimization |
| Evaluation cost model | 4 fidelity gates **F0** validate+cycle-model → **F1** behavioral sim (only when the design's functional model enables it) → **F2** yosys+OpenSTA synth/STA proxy → **F3** full ORFS P&R (`eda_rl/funnel/env.py`) | One fidelity: **every** trial runs `make` to `stop_stage` (default `finish` = full detailed route) (`utils.py:openroad()`, `distributed.py:369`) |
| Who decides to spend a full build | **Learned promotion policy** — LinUCB contextual bandit over a 22-dim state (`funnel/promotion_agent.py`); baselines FixedGate / Random | Nobody — **all** sampled trials get a full build |
| Search / sampler | Optuna **TPE** / **surrogate-UCB** / random (`funnel/candidates.py`) | Ray Tune: **hyperopt (TPE, default)** / ax / optuna / pbt / random (`distributed.py:327`) |
| Surrogate model | **Quantile-GBT** per-metric (area/period/power), conditioned on F2 obs, schema-guarded (`funnel/surrogate.py`) | **None** |
| Objective | **Balanced PPA** on a **fixed-ruler reference SDC**: `+1·fmax/ref − 1·area/ref − 0.4·power/ref` + timing penalty (`common/physical_reward.py:compute_generic_reward`) | Default: **minimize `effective_clk_period`** (`clk_period − worst_slack`) only (`distributed.py` `AutoTunerBase.evaluate`); optional `ppa-improv`: perf/power/area weighted **10000/100/100** vs a reference (`PPAImprov.get_ppa`) |
| Measurement integrity | Reward from a **re-timed reference SDC** (io=0.2·clk, no uncertainty), *not* the sampled constraints — closes the reward-gaming hole (AGENTS.md "Measurement integrity") | Scores the **sampled** run's metrics directly; `effective_clk_period` subtracts slack so loosening the clock does not trivially help, but there is no separate fixed ruler |
| Search space | 27 ORFS knobs in **4 tiers** + RTL chparams; constraint/env knobs **opt-in** & ontology-tagged (`common/knobs.py`) | Flat **JSON** of ORFS make-vars + pseudo-vars `_SDC_*`, `_FR_*` (`designs/<plat>/<design>/autotuner.json`) |
| Design-awareness | Reward dispatches on `design.functional_model()` (a plugin's composite reward, e.g. TinyVAD's, vs `compute_generic_reward`); refs auto-anchor per design | Design-agnostic; user picks coeffs / reference file |
| State / learning signal | Explicit **22-dim state vector** (`funnel/state_spec.py`); RL credit assignment | No RL state; each trial is i.i.d. to the searcher |
| Parallelism | **Single process**, variant builds **flock-serialized** (safe concurrency, but one build at a time) | **Ray**-distributed, `--jobs` concurrent trials across cores/nodes/cloud |
| Reproducibility unit | Content-addressed RTL staging; per-episode JSONL log with full provenance | Ray experiment dir + `params.json` / `metrics.json` per trial |
| Provenance | eda-rl's likith/sagar knob ranges were **mirrored from** AutoTuner `autotuner.json` bundles | The upstream format & flow eda-rl borrows from |

---

## 2. Core philosophy: multi-fidelity RL vs. single-fidelity HPO

**AutoTuner** is a classic **black-box hyperparameter optimizer**. It treats one
full RTL→GDS run as the objective function `f(config) → score` and hands that to
Ray Tune. The intelligence lives entirely in (a) the search algorithm choosing
the next config and (b) Ray's `AsyncHyperBandScheduler` early-stopping trials
that run multiple `--iterations` (for ORFS a trial is normally a single
`training_iteration`, so ASHA rarely bites). Cost per evaluation is **fixed and
high** — a full P&R — and the framework's job is to spread those evaluations
across hardware.

**eda-rl** rejects the premise that every candidate deserves a full build. It
models the flow as a **funnel of increasing fidelity** (`env.py`):

- **F0** — legality / `validate_config` + analytic cycle model. ~free.
- **F1** — behavioral sim (only for TinyVAD designs; skipped for generic).
- **F2** — a **yosys + OpenSTA synth/STA proxy**: real synthesis and static
  timing, *no* floorplan/place/route. Seconds–minutes, not minutes–tens.
- **F3** — the **full ORFS flow** (what AutoTuner runs every time).

A **promotion policy** decides, per candidate, whether to keep spending. The
research bet (`docs/rl_system.md §4`) is that a *learned* policy (LinUCB over the
state vector) can allocate the F3 budget better than fixed gates — i.e. get more
PPA-per-P&R-hour. **This is the fundamental difference**: AutoTuner has no notion
of cheap gates or of *learning where to spend*; eda-rl is built entirely around
it.

> Honest status (from `AGENTS.md`): whether the learned policy actually beats
> fixed gates is **not yet demonstrated** — the point of runs like this one is to
> generate the evidence, and eda-rl's own docs flag that its budget-cost model is
> still under-tuned. This document does not claim eda-rl "wins."

---

## 3. Search space & knob representation

**AutoTuner** — the search space is a **flat JSON** per design
(`designs/nangate45/gcd/autotuner.json`). Each entry is an ORFS make-variable or
a pseudo-variable with `{type, minmax, step}`:

```json
"_SDC_CLK_PERIOD":  {"type":"float","minmax":[0.3,1.0],"step":0},
"CORE_MARGIN":      {"type":"int","minmax":[1,3],"step":1},
"CTS_CLUSTER_SIZE": {"type":"int","minmax":[10,200],"step":1},
"_FR_LAYER_ADJUST": {"type":"float","minmax":[0.1,0.3],"step":0}
```

Pseudo-vars route to SDC / fastroute.tcl edits: `_SDC_CLK_PERIOD`,
`_SDC_UNCERTAINTY`, `_SDC_IO_DELAY`, `_FR_LAYER_ADJUST`, `_FR_GR_SEED`
(`AutoTuner/README.md`). Anything settable on the ORFS `make` command line can be
swept. There is **no tiering and no opt-in gating** — the JSON is the whole
contract.

**eda-rl** — a **tiered registry** (`common/knobs.py`, 27 knobs in 4 tiers). Each
knob carries an `affects` tag (`netlist | layout | constraints | environment`).
Two design-level ideas AutoTuner has no equivalent of:

1. **Tiers** — a campaign's `--max-tier N` bounds which knobs are in scope, so
   you can search a small core space cheaply or open it up.
2. **Opt-in constraint/environment knobs** — knobs that *rewrite the SDC or
   inject noise* (`CLOCK_UNCERTAINTY`, `IO_DELAY`, `GR_SEED`) enter the space
   **only if the design's YAML names them** (`knobs.space()` opt-in gate). This
   exists specifically because letting the optimizer freely loosen its own
   timing constraints was found to be a **reward-gaming hole** (see §5).

Knob control lives in the design YAML (`knobs: fix/exclude/override/enable`),
applied centrally so live campaigns and offline table-builds agree.

**Consequence for a fair comparison:** the two spaces are configured
independently and by default differ (eda-rl tier-3 gcd = 18 axes incl. an ABC
synthesis-recipe axis; AutoTuner gcd = 8 axes). For a true head-to-head one must
**pin them to the same knobs and ranges** in each system's native format — which
is exactly what the empirical section does.

---

## 4. Objective / reward function

This is where the two systems' *intent* diverges most.

**AutoTuner (default, `AutoTunerBase.evaluate`)** minimizes
`effective_clk_period = clk_period − worst_slack`. That is a **pure speed
objective** — area and power do not enter the score at all. A config that is 2×
larger but 1 ps faster scores better.

**AutoTuner (`--eval ppa-improv`, `PPAImprov.get_ppa`)** compares against a
**reference** run and forms a weighted % improvement:

```
coeff_perform, coeff_power, coeff_area = 10000, 100, 100
performance = %Δ effective_clk_period vs reference
power       = %Δ total_power           vs reference
area        = %Δ (100 − final_util)    vs reference
ppa = 10000·perf + 100·power + 100·area   (then bounded & negated to minimize)
```

Even here performance is weighted **100× more** than power or area — AutoTuner is
speed-first by construction, and `ppa-improv` additionally needs a user-supplied
reference metrics file.

**eda-rl (`compute_generic_reward`)** is a **balanced PPA** objective normalized
per-design (`physical_reward.py`):

```
reward = +1.0·(fmax/fmax_ref) − 1.0·(area/area_ref) − 0.4·(power/power_ref)
         − 3.0·[timing violated]
```

Area is weighted **equal** to speed, and power at 0.4 — eda-rl actively trades
speed against area/power, whereas AutoTuner (both modes) is dominated by speed.
For a design with a functional model the reward is a different, functional
composite (the plugin's `terminal_reward`, e.g. TinyVAD's speedup×accuracy) —
**design-aware**, which AutoTuner is not.

---

## 5. Measurement integrity — "measure the chip, not the ruler"

eda-rl carries an invariant AutoTuner has no analogue for. After a successful
F3, the final netlist is **re-timed under a fixed reference SDC**
(io = 0.2·clock, no uncertainty), and the reward reads *those* numbers
(`wns_ref_ns`, `fmax_ref_mhz`, …) — never the sampled constraints
(AGENTS.md "Measurement integrity"). The reason is empirical: without it, a real
campaign found the optimizer's best "reward" came from **loosening its own timing
budget** via SDC knobs (`corr(reward, IO_DELAY) = −0.83`). The opt-in gating of
constraint/environment knobs (§3) is the second half of the same fix.

AutoTuner is **less exposed** to this because its default objective already
subtracts slack (`effective_clk_period`), so simply loosening the clock does not
inflate the score. But it also has **no separate fixed ruler**: if a user adds
`_SDC_IO_DELAY` / `_SDC_UNCERTAINTY` to the JSON and switches to `ppa-improv`,
the comparison is against a reference measured under *different* constraints, and
the same class of gaming can reappear. eda-rl treats the ruler as a first-class,
protected invariant; AutoTuner leaves it to the user's JSON.

Related eda-rl-only integrity rules (all from prior silent-corruption bugs, see
`AGENTS.md` history): honest combinational Fmax (`None`, not a `1000/clk` echo);
parser tests against **real** tool output (`tests/test_parsers.py`); terminal-
reward bookkeeping separated from the shaped per-step accumulator; the
surrogate-Δ shaping prior captured pre-stage.

---

## 6. Search algorithm & learning signal

**AutoTuner** delegates to **Ray Tune** search algorithms
(`distributed.py:set_algorithm`): hyperopt (TPE, default), Ax (Bayesian+bandit),
Optuna (TPE+CMA-ES), PBT (evolutionary), or random. These are **stateless w.r.t.
the flow** — each config is an opaque point; there is no per-stage observation
feeding back except the final score. Multiple `--iterations` + ASHA enable
early-stopping, but for a one-shot ORFS build that path is mostly inert.

**eda-rl** has **two** learners:
1. The **candidate generator** (Optuna TPE / surrogate-UCB) proposes configs —
   but crucially, **only terminal F3 rewards are told to Optuna**
   (`candidates.py` F3-only tell rule); kills/proxies go to a skip-memo so
   phantom rewards never poison the model.
2. The **promotion agent** (LinUCB) consumes the **22-dim state**
   (`state_spec.py`) — which fidelity has run, F2 area/WNS/cells, clock,
   platform ordinal, surrogate-Δ, etc. — and learns a promote/kill/commit policy.
   This is the reinforcement-learning core AutoTuner simply does not have.

---

## 7. Parallelism & infrastructure

- **AutoTuner** is **built for scale-out**: Ray schedules `--jobs` concurrent
  trials, optionally against a remote Ray cluster (`--server`), with
  `resources_per_trial = cpu_count/jobs`. Throughput scales with hardware. It
  writes a full Ray experiment tree (`experiment_state-*.json`, per-trial
  `params.json`/`metrics.json`) into `flow/logs/<plat>/<design>/`.

- **eda-rl** is **single-process** and deliberately **serializes** variant
  builds with an flock (`AGENTS.md` Operational), so same-design concurrent
  campaigns are *safe* but the optimizer itself does one F3 at a time. Its
  parallelism story is "spend fewer builds," not "run more builds at once." It
  logs one JSONL row per `(config, fidelity, obs)` with full campaign provenance.

Practical corollary: on a big cluster AutoTuner will simply *do more full builds
per hour*. eda-rl's thesis only pays off if avoiding builds beats doing more of
them — which is why the interesting regime is **limited compute**, not a cluster.

---

## 8. When each makes sense

- **Use AutoTuner** when you have **lots of compute** (a Ray cluster), want a
  **battle-tested** framework, care mostly about **speed/Fmax** (or will hand-tune
  the PPA coeffs + reference), and are fine paying a full P&R per trial.
- **Use eda-rl** when compute is **scarce**, you want a **balanced PPA** objective
  with **anti-gaming** measurement guarantees, you value **cheap triage** before
  P&R, or you want the **design-aware** (TinyVAD) reward. It is also a research
  vehicle for *learned budget allocation*, which AutoTuner does not attempt.

---

## 9. Evidence index (for the reader who wants to verify)

- Multi-fidelity gates: `eda_rl/funnel/env.py` (`FunnelEnv`, F0–F3).
- Promotion policy: `eda_rl/funnel/promotion_agent.py`.
- State vector: `eda_rl/funnel/state_spec.py`.
- eda-rl reward: `eda_rl/common/physical_reward.py` (`compute_generic_reward`).
- eda-rl knobs/tiers/opt-in: `eda_rl/common/knobs.py` (`space()`).
- AutoTuner objective: `tools/AutoTuner/src/autotuner/distributed.py`
  (`AutoTunerBase.evaluate`, `PPAImprov.get_ppa`).
- AutoTuner full-flow-per-trial: `tools/AutoTuner/src/autotuner/utils.py`
  (`openroad()`, `read_metrics()`), `distributed.py:369`.
- AutoTuner search algos: `distributed.py:set_algorithm`, `--algorithm` arg.
- AutoTuner space format: `designs/<plat>/<design>/autotuner.json`,
  `tools/AutoTuner/README.md`.

---

# Empirical head-to-head (gcd/nangate45 + aes/asap7)

Real runs with **real tools** (OpenROAD + yosys from `/opt/OpenROAD-flow-scripts`),
no `PHYSICAL_MOCK`. Both systems were pinned to the **same knobs over the same
ranges**, each in its native format (see "Knob parity" below).

## Setup & knob parity

The eda-rl design YAMLs were edited (uncommitted) so eda-rl's search space equals
each ORFS `autotuner.json` exactly:

| design / platform | shared knobs (identical ranges both sides) |
|---|---|
| gcd / nangate45 (8) | clock `[0.3,1.0]`, `CORE_MARGIN [1,3]`, `CELL_PAD` global/detail `[0,3]`, `ROUTING_LAYER_ADJUSTMENT`=`_FR_LAYER_ADJUST [0.1,0.3]`, `PLACE_DENSITY_LB_ADDON [0,0.2]`, `CTS_CLUSTER_SIZE [10,200]`, `CTS_CLUSTER_DIAMETER [20,400]` |
| aes / asap7 (9) | clock `[0.3,0.6] ns` = `[300,600] ps`, `CORE_UTILIZATION [1,5]`, `CORE_ASPECT_RATIO [0.9,1.1]`, `CELL_PAD` global/detail `[0,3]`, `_FR_LAYER_ADJUST [0.0,0.1]`, `PLACE_DENSITY_LB_ADDON [0,0.2]`, `CTS_CLUSTER_SIZE`, `CTS_CLUSTER_DIAMETER` |

eda-rl ran `--max-tier 3` with all non-parity tier-3 axes (incl. `abc_recipe`)
excluded via the YAML `knobs:` block; AutoTuner used the matching `autotuner.json`.
Machine: 8-core AMD EPYC, 30 GB RAM, **shared with other tenants** (~15 GB used
by others throughout).

### Commands (exact)

eda-rl (both designs, concurrent, 2 h budget each):
```bash
ORFS_DIR=/opt/OpenROAD-flow-scripts python3 -m eda_rl.cli optimize \
  --design gcd --platform nangate45 --budget-hours 2 \
  --max-tier 3 --sampler tpe --promotion linucb --seed 0
ORFS_DIR=/opt/OpenROAD-flow-scripts python3 -m eda_rl.cli optimize \
  --design aes --platform asap7 --budget-hours 2 \
  --max-tier 3 --sampler tpe --promotion linucb --seed 0
```
AutoTuner (py3.11 env with pinned deps; run from a writable ORFS mirror because
`/opt` is read-only; Ray node-memory OOM monitor disabled — see finding #3):
```bash
RAY_memory_monitor_refresh_ms=0 PYTHONPATH=$MIRROR/tools/AutoTuner/src \
python -u -m autotuner.distributed --design gcd --platform nangate45 \
  --config $MIRROR/flow/designs/nangate45/gcd/autotuner.json \
  --experiment gcd_clean2 --jobs 4 --openroad_threads 4 \
  tune --algorithm hyperopt --eval default --samples 80 --seed 42
RAY_memory_monitor_refresh_ms=0 PYTHONPATH=$MIRROR/tools/AutoTuner/src \
python -u -m autotuner.distributed --design aes --platform asap7 \
  --config $CFG/aes_asap7_parity.json \
  --experiment aes_clean2 --jobs 2 --openroad_threads 4 \
  tune --algorithm hyperopt --eval default --samples 10 --seed 42
```

## Results — gcd / nangate45

Each system's best design **by its own objective** (eda-rl: balanced-PPA reward;
AutoTuner: min effective clock period), plus eda-rl's fastest timing-clean build
for a speed-to-speed view:

| | achievable clk period | instance area (µm²) | power (mW) | DRC | selected by |
|---|---|---|---|---|---|
| **eda-rl** best (balanced PPA) | 0.75 ns¹ | **676** | **1.58** | 0 | its reward |
| eda-rl fastest timing-clean | 0.67 ns¹ | 782 | 2.59 | 0 | — |
| **AutoTuner** best | **0.472 ns**² | 1012 | 4.31 | 0 | eff. clk period |

**Search effort:** eda-rl evaluated **1,524 candidates → 57 full F3 builds**
(F0=504, F2=963, F3=57; 47 F3 valid, 10 failed) in ~1.9 h. AutoTuner ran **80
full-build samples → 43 valid** (22 rejected for invalid padding `detail>global`,
15 build-failed) in ~29 min.

**Reading:** the objective difference shows up exactly as theory predicts.
AutoTuner (pure speed) found a **faster** design (0.472 ns) but **1.5× larger and
~2.7× more power** than eda-rl's balanced pick (676 µm² / 1.58 mW). eda-rl's own
fastest timing-clean build (0.67 ns) is still slower than AutoTuner's — AutoTuner
is genuinely better at the single axis it optimizes. Neither "wins": they
optimize different things. eda-rl considered **~19× more candidates per full
build** thanks to F0/F2 triage.

## Results — aes / asap7  *(low confidence — few valid builds each)*

| | achievable clk period | instance area (µm²) | power (mW) | DRC | builds |
|---|---|---|---|---|---|
| **eda-rl** best (balanced PPA) | 0.361 ns¹ | 1871 | 126 | 0 | 7 F3 (6 valid) / 136 cand |
| eda-rl fastest timing-clean | 0.354 ns¹ | 1912 | 131 | 0 | — |
| **AutoTuner** best | 0.474 ns² | 1902 | 112 | 0 | 4 valid / 10 samples |

**Caveat:** both explored very few valid builds on this slow (~30 min/build)
asap7 block, so this is **not statistically meaningful**. Directionally eda-rl's
best is faster with similar area and slightly higher power, but with 4–6 builds
per side, seed noise dominates. Report it as "both produced working GDS in the
same parity space," not as a winner.

¹ eda-rl "achievable clk period" = `period_ref_ns`, re-timed on eda-rl's **fixed
reference SDC** (io = 0.2·clk, no uncertainty).
² AutoTuner "achievable clk period" = `effective_clk_period = clk_period −
worst_slack` on the **sampled SDC**. The two rulers are close but not identical;
do not over-read sub-0.1 ns gaps between the systems.

## Cross-cutting findings (the interesting part)

1. **Objective drives the design, as designed.** gcd is the clean demonstration:
   speed-only AutoTuner → fast, dense (util 0.87), power-hungry; balanced-PPA
   eda-rl → small, cool, moderate speed. This is the single most important
   practical difference between the tools — *choose the optimizer whose objective
   matches your goal.*
2. **Multi-fidelity really does amortize P&R.** eda-rl screened 1,524 (gcd) /
   136 (aes) candidates while paying for only 57 / 7 full builds. AutoTuner pays
   one full build per sample. In a build-bound regime this is eda-rl's core
   advantage; on a big cluster AutoTuner's parallelism narrows it.
3. **Memory robustness under contention (an unplanned but real result).** In the
   first, fully-concurrent run (all 4 processes at once, per the experiment
   design), **AutoTuner's parallel `--jobs` were OOM-killed by Ray** — the aes
   run produced **0 valid builds**, gcd only ~10. eda-rl's **serialized single
   build** sailed through (57 / 7 builds) on the same box at the same time. Root
   cause: Ray's OOM monitor measures *whole-node* memory, which on this shared
   box includes other tenants' ~15 GB, so it false-triggers; the clean AutoTuner
   numbers above required `RAY_memory_monitor_refresh_ms=0` **and** running the
   two designs sequentially. eda-rl's one-build-at-a-time design is simply more
   robust when memory is contended.
4. **Wasted samples.** AutoTuner spent 27 % (gcd, 22/80) and 40 % (aes, 4/10) of
   its samples on **invalid padding** combos (`detail>global`) it rejects after
   sampling, plus outright build failures (15/80 gcd). eda-rl's F0 `validate_config`
   gate rejects illegal configs *before* any build, so its "wasted" candidates
   cost ~nothing.
5. **Measurement ruler.** eda-rl reports speed on a fixed reference SDC;
   AutoTuner on the sampled SDC. eda-rl's ruler is the more gaming-resistant
   choice (§5), but it also means cross-tool speed numbers carry a small
   systematic offset — hence the ¹/² caveats.

## Honesty box

- **Two designs, one seed each, a shared/contended box** — this is an existence
  demonstration of the architectural differences, **not** a benchmark. No claim
  that either tool universally wins.
- The aes/asap7 comparison is **underpowered** (≤7 builds/side).
- eda-rl's `docs/rl_system.md §5` flags that its budget-cost model is still
  under-tuned; this run does **not** attempt to show the *learned* promotion
  policy beats fixed gates (a separate question — it only shows multi-fidelity
  triage vs full-build-per-trial).
- AutoTuner was run in its **default speed objective** (`--eval default`); with
  `--eval ppa-improv` + a reference it would weigh area/power too (still
  speed-dominant at 10000:100:100).

## Reproduce

- eda-rl campaign logs: `eda_rl/campaigns/{gcd/nangate45,aes/asap7}/results_funnel_campaigns.jsonl`
  (per-episode `(config, fidelity, obs)` rows + a `campaign_summary`).
- AutoTuner Ray results: `$MIRROR/flow/logs/<plat>/<design>/<experiment>-tune/`
  (`variant-*-ray/result.json` per trial; best trial's `.../metrics.json` = full
  METRICS2.1 for the winning build).
- Parity YAML edits: `eda_rl/designs/{gcd,aes}.yaml` `knobs:` blocks (uncommitted).
