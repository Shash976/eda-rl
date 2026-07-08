# The eda-rl RL system — how it works, file by file

*The single authoritative doc for the active system (`eda_rl/`). It replaces
`legacy/docs/04` (gen1 optimizer), `07` (gen1-era design rationale), and `08`
(early funnel guide — predates the third audit). Operating instructions live in
`AGENTS.md`; this doc explains the learning machinery and each file's role.*

---

## 1. System overview

`eda-rl` searches OpenROAD-flow-scripts (ORFS) flow configurations for a given
RTL design, trading off **area / Fmax / power**. Full physical builds are
expensive (~7 min for a tiny block, hours for a real one), so candidates are
evaluated through a **multi-fidelity funnel** and only promising ones earn the
full build:

```
candidate config
      │
      F0  legality + analytic cycle model      ~0 s    (always runs)
      │        kill? ────────────────────────────────► episode over
      F1  behavioral Verilator sim             ~5 s    (TinyVAD designs only)
      │        kill?
      F2  yosys synth + OpenSTA proxy          ~45 s
      │        kill?
      F3  full ORFS RTL→GDS flow               ~420 s  (terminal, real reward)
```

Two learning loops run at once, at different levels:

- **Across episodes** — a `CandidateGenerator` (Optuna TPE, surrogate-UCB, or
  random) proposes *which config to try next*, learning from terminal F3
  rewards only.
- **Within an episode** — a promotion policy (LinUCB bandit, fixed gates, or
  random) decides *how far up the funnel this candidate deserves to go*:
  `{kill, re-proxy, promote, commit}` after each fidelity result.

Everything can run **live** (real tools) or in **table mode** (replaying a
pre-built F0–F2/F3 table from `eda-rl build-table`, used by `eda-rl benchmark`
to compare promotion policies at zero tool cost). `PHYSICAL_MOCK=1` substitutes
synthetic metrics for every tool call (note: mock metrics are TinyMAC-shaped
and cannot catch parser/measurement regressions — that is `tests/test_parsers.py`
and `eda-rl doctor`'s job).

## 2. The RL formulation, precisely

One **episode = one candidate config** walked through the funnel by
`FunnelEnv` (`eda_rl/funnel/env.py`).

### State — 22-dim float32 vector

Owned by `eda_rl/funnel/state_spec.py` (the single source of truth — do not
fork it). Unrun slots are `0.0`, never −1.

| idx | content | normalization |
|-----|---------|---------------|
| 0 | mac_lanes | log2(lanes)/5; sentinel 1.0 for designs without the axis |
| 1 | accumulator_width | (acc_w−16)/16 |
| 2 | clock period | generic: (clk−lo)/(hi−lo) from the design's own `clock_range_ns`; tinymac/legacy: fixed platform rulers (bit-compat) |
| 3 | ABC recipe | recipe_idx/2 (orfs_speed=0, orfs_area=0.5, plain=1.0) |
| 4 | platform | ordinal 0.0 nangate45 / 0.5 sky130hd / 1.0 asap7 |
| 5–6 | F0 cycle-model + accuracy | log2(SW_BASELINE/cycles)/10; accuracy 0..1 (0 for generic designs) |
| 7–8 | F1 sim cycles + accuracy | as above; 0 if unrun |
| 9 | F2 proxy area | area/20000, clip [0,3] |
| 10 | F2 WNS | generic: wns/clock_period; tinymac/legacy: wns/5 ns; clip [−2,2] |
| 11–13 | F2 ff/cells/levels | /1000, /10000, /50 (note: the F2 proxy has no separate FF count, so [11]==[12]) |
| 14–15 | surrogate μ/4.5, σ | 0 if no surrogate |
| 16 | incumbent best reward/4.5 | 0 if none |
| 17 | remaining budget fraction | 1 → 0 |
| 18–21 | depth one-hot | highest fidelity already run (F0..F3) |

### Actions

`kill` (terminate, reward 0), `re-proxy` (re-run F2), `promote` (run the next
unrun fidelity), `commit` (jump straight to F3, terminal). Episodes always end
on kill or after F3.

### Reward

Each `step()` returns a shaped scalar:

```
r_step = −λ·cost_s/budget_s                    (cost of the stage just run)
       + [surrogate Δ:  μ_after_obs − μ_before_obs]   (0 without a surrogate;
                                                        nonzero mainly at F2)
       + [terminal reward, F3 only]
```

- The **surrogate-Δ prior is captured *before* the stage runs** — the stage
  mutates the F2 observation the surrogate conditions on. (A regression here
  once made the term identically zero; see §5.)
- The **terminal reward** is design-aware (`common/physical_reward.py`):
  - TinyVAD designs → `compute_physical_reward`, a speedup/accuracy/area/power
    composite against the measured software baseline (peaks ≈ +4; correctness
    failure −50).
  - Generic designs → `compute_generic_reward` = `+1.0·(fmax/ref) −
    1.0·(area/ref) − 0.4·(power/ref)` + timing-violation penalty, with refs
    auto-anchored from the design YAML `reward:` block or the first successful
    F3 build.
  - Genuine F3 failure → monotone ladder penalty (−20 family) from
    `FunnelEnv._terminal_reward`; a table-miss scores nothing.
- **Measurement integrity**: the terminal reward reads the fixed-ruler
  reference-SDC metrics (`*_ref_*`, re-timed under default constraints after
  F3), never the sampled constraints — otherwise the optimizer games its own
  ruler (this exact failure shipped once: corr(reward, IO_DELAY) = −0.83).

`info["terminal_reward"]` always carries the *pure* terminal value (no
shaping). Everything outside the promotion bandit — the Optuna tell, the
best-config tracker, the benchmark score — must use it, never the shaped
accumulator.

### The promotion policy — LinUCB

`PromotionAgent` (`eda_rl/funnel/promotion_agent.py`) is a **disjoint linear
contextual bandit**: one ridge model per action over the 22-dim state.

- Init: `A_a = λI` (λ=1), `b_a = 0` for each action `a`.
- Act: `argmax_a  θ_aᵀs + α·√(sᵀA_a⁻¹s)` with `θ_a = A_a⁻¹b_a`, α=1.
- Update after every step, with the **pre-action** state and that step's
  shaped reward: `A_a += ssᵀ`, `b_a += r·s`.

Baselines: `FixedGateAgent` — deterministic kill gates (F0/F1 accuracy < 0.95;
F2 `state[10] < −0.5`, i.e. **WNS worse than −0.5·clock_period** — the
threshold is clock-relative so it still fires on sub-ns platforms) — and
`RandomPromotionAgent`.

### A traced episode (live, LinUCB)

1. Driver asks `CandidateGenerator.suggest()` → config C; `env.reset(C)`
   validates, runs F0, returns s₀ (depth one-hot = F0).
2. `agent.act(s₀)` → `promote` → F1 runs; r₁ = −λ·5/budget; `agent.update(s₀,
   promote, r₁)`.
3. `act(s₁)` → `promote` → F2; r₂ = −λ·45/budget + surrogate-Δ.
4. `act(s₂)` → `promote` → F3; r₃ = −λ·420/budget + terminal. Episode done.
5. Driver reads `info["terminal_reward"]`, tells Optuna (F3 only), updates
   the incumbent, logs the episode row.

## 3. Candidate generation

`CandidateGenerator` (`eda_rl/funnel/candidates.py`):

- **tpe** — Optuna TPE over the design's knob space (built by
  `KnobRegistry.space()` from `common/knobs.py`, honoring the design YAML's
  `knobs: fix/exclude/override/enable` block and `--max-tier`).
- **surrogate_ucb** — ranks a random pool by `μ + κ·σ` (κ=1) from the fitted
  surrogate; falls back to random ordering when no surrogate covers the design.
- **random** — uniform sampling.

**F3-only tell rule**: only terminal F3 rewards are told to Optuna. Kills,
proxies, and table-misses go to the **kill-memo** (skip list) and close the
pending trial as FAIL — phantom rewards for unmeasured configs poisoned TPE
once.

The **surrogate** (`eda_rl/funnel/surrogate.py`) is a per-metric quantile-GBT
model (area/period/power at q16/q50/q84), conditioned on the config axes plus
optional F2 observations. It learns its config-axis schema from its corpus and
**refuses cross-design predictions** (schema guard, persisted through
save/load). It is fitted offline from campaign logs by `eda-rl fit-surrogate`.
The live driver probes any auto-loaded surrogate against the campaign design's
space and drops it loudly on mismatch rather than letting UCB silently return
0.0 for every candidate.

## 4. What the benchmark can (and cannot) conclude

`eda-rl benchmark` (`eda_rl/funnel/benchmark_funnel.py`) replays a fixed table
through random / fixed-gate / LinUCB promotion policies: same table, same
candidate order per seed, same budget accounting, fresh agent per seed;
scoring uses the pure terminal reward (`info["terminal_reward"]`). Metrics:
time-to-95%-of-table-optimum and best-found.

Honest caveats:

- LinUCB learning *during* the measured run is the intended online-bandit
  protocol, not leakage — but `--pretrain-campaigns > 0` warms the bandit on
  the same table it is then measured on, which is optimistic. Leave it 0 for
  claims.
- The open research question — *does a learned promotion policy beat fixed
  gates at spending synthesis budget?* — is **not demonstrated**. On small
  deterministic tables LinUCB does not reliably beat fixed gates, and the
  pre-measurement-fix likith/sagar corpora are unusable for learning
  conclusions until re-run (audit R1). The recommended reframing (audit R2,
  still open) is a surrogate-driven expected-improvement-per-cost promotion
  rule instead of a bigger bandit.

## 5. Known limitations of the learning signal (deliberate, documented)

Found in the 2026-07 restructure audit; semantics intentionally left unchanged
pending the R1 corpus re-run — fixing them mid-stream would invalidate
comparability. Ranked by importance:

1. **Budget cost is ~100× smaller than terminal magnitudes.** A full F3 costs
   `−λ·420/14400 ≈ −0.03` reward vs terminal magnitudes of ±4…50, so the
   bandit has almost no incentive to kill early — the funnel's core premise is
   nearly invisible in its reward. Scaling λ (or crediting kills with the
   avoided F3 cost) is the obvious knob, but it changes every historical
   number.
2. **`kill` pays exactly 0 and is never charged or credited**, so `θ_kill ≡ 0`
   forever; the agent kills only when every other arm's UCB predicts < 0.
3. **Terminal credit reaches only the final-step arm.** A per-step bandit has
   no return propagation: the F0→F1 and F1→F2 promotes that *enabled* a great
   chip train only on the tiny cost shaping. The depth one-hot lets θ separate
   regimes, but this is a compromise, not credit assignment.
4. **Two failure scales coexist**: the funnel's −20 ladder vs
   `physical_reward`'s −100 (reachable only through driver reset-failure
   paths). Harmless today; mind it when retuning.

Two *fixed* regressions to not re-break (regression details in `git log`):
the surrogate-Δ prior must be captured **before** the stage runs (else the
term is identically 0), and the benchmark must score **pure terminal** reward
(the shaped accumulator is path-dependent and penalized promote-through
policies by ~5% at small budgets).

## 6. File-by-file roles

### CLI

| subcommand | module | role |
|---|---|---|
| `optimize` | `funnel/run_funnel_optimizer.py` | live campaign driver (the main pipeline) |
| `report` | `viz/report.py` | static HTML report (design-aware; no TinyMAC baselines for generic designs) |
| `collect` | `funnel/collect_best.py` | harvest best F3 GDS + before/after page |
| `dashboard` | `viz/dashboard.py` | live Optuna dashboard |
| `build-table` | `funnel/build_table.py` | resumable offline F0–F2 table builder |
| `benchmark` | `funnel/benchmark_funnel.py` | promotion-policy table benchmark |
| `doctor` | `funnel/doctor.py` | per-design preflight: dead parsers, ps-vs-ns ranges, PDN util floor, minimum `--max-tier` |
| `fit-surrogate` | `funnel/fit_surrogate.py` | mine campaign logs, fit + CV-validate the surrogate |

### `eda_rl/funnel/` — the learning system

| file | role |
|---|---|
| `env.py` | `FunnelEnv` — the episode environment (fidelity gates, shaped reward, logging, live/table modes) |
| `state_spec.py` | the 22-dim state layout (single source of truth) |
| `promotion_agent.py` | LinUCB / FixedGate / Random promotion policies |
| `candidates.py` | TPE / surrogate-UCB / random candidate generation, F3-only tell, kill-memo |
| `surrogate.py` | quantile-GBT PPA surrogate with persisted schema guard |
| `run_funnel_optimizer.py` | live driver wiring generator + env + policy + logs |
| `build_table.py`, `benchmark_funnel.py`, `collect_best.py`, `doctor.py`, `fit_surrogate.py` | as in the CLI table |
| `search_space_funnel.yaml` | the reduced TinyMAC table-mode space (build_table default) |

### `eda_rl/common/` — measurement & plumbing

| file | role |
|---|---|
| `physical_runner.py` | drives ORFS make (F3), the yosys+OpenSTA proxy (F2), the reference-SDC re-time, mock metrics, and all report parsing (`tests/test_parsers.py` covers the extracted parsers) |
| `physical_reward.py` | `compute_physical_reward` (TinyVAD composite) + `compute_generic_reward` (pure PPA) |
| `designs.py` | `DesignSpec.load`, SDC generation, injection-safe YAML validation |
| `knobs.py` | `KnobRegistry`: 27 ORFS knobs in 4 tiers, `affects` ontology (constraints/environment knobs are opt-in per design), pseudo-type → sampling-type mapping |
| `constants.py` | measured TinyVAD cycle model + software-baseline constants |
| `recipe.py` | ABC synthesis recipe axis |
| `sim.py` / `verilator_sim.py` | mock-aware F1 wrapper / the real TinyVAD Verilator driver (rescued from gen1) |

### `eda_rl/viz/` — reporting

`report.py` (static HTML), `dashboard.py` (live Optuna), `campaign_data.py`
(campaign JSONL loader shared by both).

### What lives in `legacy/` and why

The retired first-generation optimizer (`legacy/gen1/`), its results, dead
modules whose only importers were gen1 (`cascade_reward.py`, `validate.py`),
the orphaned `measure_real.py`, the superseded docs 04/07/08, and the third
audit's records (`legacy/audits/`). Nothing under `legacy/` is imported,
packaged, or executed — see `legacy/README.md`.

## 7. Still-open items (from the third audit, tracked here so they aren't lost)

- **R1** — re-run the likith/sagar corpora on the fixed measurement stack
  before drawing any learning conclusions.
- **R2** — replace/beat LinUCB with a surrogate-driven
  expected-improvement-per-cost promotion rule; **R7** — the four-way
  benchmark (random/fixed/LinUCB/R2-rule) on real design tables.
- **R8 residue** — an `eda-rl gc` for `eda_rl_runs/` (no GC yet; watch disk).
- **F10 residue** — the state has no ORFS-knob summary block; tier-2+ knobs
  are invisible to the promotion policy.
