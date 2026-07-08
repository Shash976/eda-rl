# AUDIT_FINDINGS.md — independent technical audit, 2026-07-06

Scope: `eda_rl/gen2/`, `eda_rl/common/`, `eda_rl/viz/` (touched lightly), the
uncommitted working-tree diff (asap7 F2 proxy support, CLOCK_UNCERTAINTY /
IO_DELAY / GR_SEED knobs, likith/sagar designs), and the real campaign
artifacts under `eda_rl/campaigns/{likith,sagar}` and `./eda_rl_runs/`.
`eda_rl/gen1/` out of scope except where gen2/common imports it.

Method: full read of the core modules, all canonical self-tests executed
(all pass), real (non-mock) `run_synth_sta` runs on likith (asap7) and sagar
(sky130hd) with `EDA_RL_WORK` pointed at a scratch dir, forensic analysis of
the two real overnight campaigns (1,134 + 1,453 episodes, 206 + 224 real F3
builds) and their on-disk ORFS build artifacts, plus direct inspection of the
ORFS install at `/opt/OpenROAD-flow-scripts`. No tracked file was modified;
no campaign rows were added.

Verified-good (for the record): the second-audit fixes are all present and
working as described (design name/top validation, F3-only TPE tell with
table-miss → skip-memo, design threading in build_table, process-group kill in
`run_physical`, DESIGN_NAME/NICKNAME split, PLATFORM_TIME_UNIT applied
correctly at every call site I checked — SDC write ×unit, report parse ÷unit,
including the new CLOCK_UNCERTAINTY/IO_DELAY conversions). The asap7 F2 proxy
lib/corner claims check out against the ORFS install (`CORNER ?= BC` at
`platforms/asap7/config.mk:198`; `set_global_routing_random` exists in the
installed OpenROAD). The knobs.py `_SPACE_TYPE` fix works: all 21 axes varied
in both live campaigns.

---

## F1. The optimizer's best rewards come from relaxing its own timing constraints (reward gaming via IO_DELAY / CLOCK_UNCERTAINTY)

**Severity: critical** (for the project's stated purpose — the campaign optimum is not a better chip)

**Evidence**
- `eda_rl/common/knobs.py:507-530` (IO_DELAY) and `:482-505`
  (CLOCK_UNCERTAINTY) are sampled axes that rewrite the SDC
  (`designs.py:_sdc_text`, `physical_runner.py:605-621`). The SDC defines the
  constraints under which `wns_ns`, `timing_met`, and (for these combinational
  designs) `fmax_mhz` are *measured*.
- `compute_generic_reward` (`eda_rl/common/physical_reward.py:66-143`) rewards
  `+w_fmax·(fmax/ref)` and penalizes `timing_violation` — both functions of the
  sampled constraints, not only of the netlist.
- Real sagar campaign (`eda_rl/campaigns/sagar/sky130hd/results_funnel_campaigns.jsonl`,
  223 ok F3 builds): **corr(f3_reward, IO_DELAY) = −0.833**, the strongest
  correlation of any axis. Concretely:
  - top rewards: `r=+0.34, fmax=507 MHz, area=490 µm², IO_DELAY=0.020` (range min)
  - bottom rewards: `r=−0.55, fmax=256 MHz, area=489 µm², IO_DELAY=0.97` (range max)
  Same design, same area (±1 µm²), timing met in both — the "fmax" doubled purely
  because the I/O timing budget got looser. For a combinational block the
  critical path is in→out, so min-period ≈ io_delay_in + comb + io_delay_out:
  the metric largely *is* the knob.
- Real likith campaign: all top-reward configs sit at `IO_DELAY=0.0005`
  (range min), `CLOCK_UNCERTAINTY=0.0` (range min), `clock_period_ns≈0.070`
  (range max); corr(f3_reward, CLOCK_UNCERTAINTY) = −0.34.

**Failure scenario**
Given any design whose reward uses fmax/timing (all generic designs), and
tier ≥ 2 (which auto-activates these knobs — see F7), the sampler converges on
"loosest constraints", the campaign reports a false best config, `eda-rl
collect` harvests a GDS that is *physically identical* to a mid-pack one, and
any learned promotion policy / surrogate is trained on a corrupted objective.

**Fix plan**
Pick one (they're compatible):
1. **Measure with a fixed ruler.** Compute reward metrics under a *reference*
   SDC (design-declared io fraction/uncertainty), independent of the sampled
   SDC. Cheapest: after F3, re-run the 10-second pre-layout STA (or
   `report_checks -path_delay max` on the routed design) with the reference SDC
   and use that fmax/wns in the reward; keep the sampled SDC only as a flow
   input.
2. **Make constraint knobs opt-in** (see F7) and document in the knob notes
   that any design enabling them must declare `reward:` anchors that neutralize
   the fmax term, or exclude timing from its reward weights.
3. Minimum: add a `validate_config` ERROR when IO_DELAY/CLOCK_UNCERTAINTY are
   in the sampled space of a design whose reward weights include `w_fmax`/
   `w_timing_violation` and no fixed reference SDC is configured.

---

## F2. GR_SEED's generated fastroute.tcl silently disables the sampled ROUTING_LAYER_ADJUSTMENT (phantom axis — same class as Known Trap #3, introduced by the fix for Known Trap #3)

**Severity: high**

**Evidence**
- `eda_rl/common/physical_runner.py:626-635` (uncommitted diff): when GR_SEED
  is sampled, the platform's default `fastroute.tcl` is copied verbatim and a
  seed line appended, and `FASTROUTE_TCL` is exported (`:511-514`).
- ORFS `flow/scripts/floorplan.tcl:104-111`:
  ```tcl
  if { [env_var_exists_and_non_empty FASTROUTE_TCL] } {
    log_cmd source $::env(FASTROUTE_TCL)
  } else {
    set_global_routing_layer_adjustment ... $::env(ROUTING_LAYER_ADJUSTMENT)
    ...
  }
  ```
  i.e. FASTROUTE_TCL *replaces* the env-var path entirely.
- The asap7 platform `fastroute.tcl` hardcodes
  `set_global_routing_layer_adjustment $MIN-$MAX 0.25`.
- Real campaign artifact: `eda_rl_runs/asap7/id/config_id_c0p04532_u6_d0p72_area_k2013a3_re1ca1bf0.mk`
  exports `ROUTING_LAYER_ADJUSTMENT = 0.7415…` **and**
  `FASTROUTE_TCL = …/fastroute_id_….tcl` (which contains the hardcoded 0.25 +
  the seed). The build log
  (`eda_rl_runs/logs/asap7/id/id_c0p04532_…/2_1_floorplan.log:30`) shows the
  fastroute file being sourced; the sampled 0.74 never appears in any log.

**Failure scenario**
At `--max-tier 3` (which likith.yaml explicitly instructs), GR_SEED is sampled
*every* episode, so ROUTING_LAYER_ADJUSTMENT never reached the router in the
entire 1,120-episode likith campaign. Optuna's importance analysis, the report,
and any surrogate treat it as a real axis with (noise-only) observed effects.
likith.yaml's own header ("Campaigns need --max-tier 3 to actually exercise
ROUTING_LAYER_ADJUSTMENT and GR_SEED") is self-defeating as written.

**Fix plan**
Generate the fastroute.tcl from a template that preserves env-var
configurability, e.g. write:
```tcl
set_global_routing_layer_adjustment $::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER) $::env(ROUTING_LAYER_ADJUSTMENT)
set_routing_layers -clock ... -signal ...
set_global_routing_random -seed <N>
```
(i.e. reproduce ORFS's *else* branch plus the seed line) instead of copying the
platform file. Add a check to the knobs self-test that a config sampling both
knobs produces a build log containing the sampled adjustment value.

---

## F3. F2 cell-count parsing is dead against the installed yosys (state dims [11]/[12] silently zero everywhere; AGENTS.md "Cells" invariant broken)

**Severity: high**

**Evidence**
- Parser: `eda_rl/common/physical_runner.py:1062`
  `re.findall(r"Number of cells:\s+(\d+)", p1.stdout)`.
- The installed yosys `stat` prints a new tabular format (real synth log,
  scratch run `proxy/alu4b_c6p5_re059b768/synth.log:739`):
  ```
         36  231.472 cells
  ```
  `grep -c "Number of cells"` over both new-design proxy logs: **0**.
- Corpus-wide impact: F2 rows with `cell_count != None`:
  likith campaign **0/1383**, sagar campaign **0/1821**, and the *committed*
  table `eda_rl/results/gen2/results_funnel.jsonl` **0/84**.
- Consequence: `FunnelEnv._run_f2` sets `cell_count`/`ff_count` from
  `result["cells"]` (None), so state dims [11] and [12] are 0.0 in every live
  episode; the surrogate's `cell_count`/`ff_count` conditioning columns are
  always "missing". Nothing crashed and no test noticed (mock mode fabricates
  `cells`, so `PHYSICAL_MOCK=1` self-tests pass).

**Failure scenario**
Any consumer of F2 netlist-size signals — the promotion policy, the surrogate,
`fit_surrogate`, the report — has been operating on zeros since the yosys
version bump. AGENTS.md's invariant "Cell counts surfaced at F2" is
documentation-vs-reality drift.

**Fix plan**
Make the regex accept both formats, e.g. also match
`^\s*(\d+)\s+[\d.]+\s+cells\s*$` (take the last occurrence), or parse the
`=== <top> ===` stats block structurally. Add a **real-tool** regression test:
a tiny fixture .v synthesized once in CI (or a golden captured log snippet)
asserted to yield a non-None cell count.

---

## F4. F2 fmax/period for combinational designs is a constraint echo: `report_clock_min_period` prints `fmax = inf`, the regex fails, and the fallback silently substitutes 1000/(clk − wns)

**Severity: high**

**Evidence**
- Real STA logs (scratch runs): `proxy/id_c0p05_…/sta.log:19` and
  `proxy/alu4b_c6p5_…/sta.log:9`: `core_clock period_min = 0.00 fmax = inf`.
- Parser `physical_runner.py:1099` requires both groups to match `[\d.]+`;
  `inf` doesn't match, so the whole match fails and `:1107-1110` computes
  `fmax = 1000/(clk_ns − wns)`.
- Corpus proof: every F2 `fmax_mhz` in the sagar campaign is ≈ 1000/clk
  (125.0–181.8 for clk 5.5–8.0); likith similarly 14285–22222 = 1000/clk.
  Distinct-value analysis over 1,821 sagar F2 rows shows fmax is a pure
  function of the sampled clock.

**Failure scenario**
For any design with no reg-to-reg path (both new designs; also any
I/O-dominated block), the F2 "speed" observable carries zero design
information but *looks* plausible. The surrogate's `period_ns` target and the
docs' claim that proxy Fmax is "target-clock-independent, a fair cross-config
speed metric" (`physical_runner.py:941-947`) are both false in this regime.

**Fix plan**
1. Parse `inf`/`0.00` explicitly (`fmax\s*=\s*([\d.]+|inf)`); when
   period_min == 0 / fmax == inf, record `fmax_mhz = None` (or a
   `combinational: true` marker) instead of the fallback.
2. For combinational designs, derive an honest speed proxy from the max
   in→out path under a *fixed reference* io_delay (ties into F1's fix):
   `report_checks -path_delay max` gives the comb delay directly.
3. The fallback `fmax = 1000/(clk − wns)` should only ever fire when wns is a
   real violation figure, and should set a flag in the obs so downstream
   consumers can distinguish measured from inferred fmax.

---

## F5. Surrogate feature encoding cannot see what the campaigns actually vary (recipe flag dead on gen2 corpora; util/density read the wrong keys; all tier-2/3 knobs unfeaturized)

**Severity: high** (for the surrogate/surrogate_ucb path; currently latent in live runs because no surrogate file exists)

**Evidence**
- `eda_rl/gen2/surrogate.py:555-556`: `abc_flag = 1.0 if str(abc_raw).lower() == "area"`.
  Gen2 configs carry `abc_recipe="orfs_area"`. Verified live:
  `_build_feature_row` for `orfs_area` and `orfs_speed` configs returns
  **identical** vectors. (`fit_surrogate.py:264` feeds `orfs_area` through
  unchanged; only the *legacy* report-miner (`fit_surrogate.py:100`) emits
  `'area'`.) The 43%-area-spread recipe axis is invisible on exactly the
  corpus gen2 produces.
- `surrogate.py:565-566` reads `flat.get("util", 40)` / `flat.get("density", 0.60)`;
  funnel configs use `CORE_UTILIZATION` / `PLACE_DENSITY`, so these features
  are constants on gen2 corpora.
- No knob axis (IO_DELAY, PLACE_DENSITY_LB_ADDON, CTS_*, …) is featurized at
  all: verified that two configs differing in CORE_UTILIZATION 21→55,
  PLACE_DENSITY 0.41→0.75, IO_DELAY 0.02→0.9 produce identical feature rows.
- Also: `run_funnel_optimizer.py:200-207` auto-loads
  `results/gen2/surrogate_n45.joblib` for **any** design/platform with no
  compatibility check (file currently absent, so latent).

**Failure scenario**
`--sampler surrogate_ucb` or a fitted surrogate feeding state dims [14]/[15]
on a likith/sagar-style campaign would rank candidates on lanes/acc_w/clk
alone — for these designs, essentially on clock only — while confidently
reporting µ/σ. The obs-join key in `fit` (`(lanes, acc_w, clk)`,
`surrogate.py:281-287`) additionally collides all knob-differing configs.

**Fix plan**
- Normalize recipe: `abc_flag = 1.0 if "area" in str(abc_raw).lower()`.
- Alias `util ← CORE_UTILIZATION`, `density ← PLACE_DENSITY` in `_flatten_row`.
- Generalize the feature vector: featurize every numeric axis of the active
  space (ordered by sorted axis name, stored in the joblib metadata so
  fit/predict agree), one-hot small categoricals. Refuse to `load()` a
  surrogate whose stored axis list doesn't match the campaign's space
  (also fixes the cross-design auto-load).

---

## F6. `_snap_config` applies a 0.5 ns default snap step to *every* float axis — pinning sub-range axes to their lower bound whenever grid_snap is on

**Severity: medium-high** (off in live mode; on in table-mode `eda-rl optimize --table`, in `benchmark_funnel`, and by default for any direct `CandidateGenerator` user)

**Evidence**
- `eda_rl/gen2/candidates.py:115-133`: `step = spec.get("_snap_step", _CLOCK_SNAP_STEP)`
  (0.5) for **all** `type: float` axes; only `_fallback_space()`'s
  clock axis actually declares `_snap_step`. KnobRegistry spaces declare none.
- Demonstrated:
  `_snap_config({"clock_period_ns":0.062,"IO_DELAY":0.0018,"PLACE_DENSITY_LB_ADDON":0.15}, likith-style space)`
  → `{'clock_period_ns': 0.045, 'IO_DELAY': 0.0005, 'PLACE_DENSITY_LB_ADDON': 0.0}` —
  all three pinned to range-lo.
- Activation sites: `run_funnel_optimizer.py:254` `grid_snap=(table is not None)`;
  `benchmark_funnel.py:328` `grid_snap=True`; `CandidateGenerator.__init__`
  default `grid_snap=True`.
- Same hardcoded 0.5 appears independently in `build_table.py:536-540`:
  for likith's clock range [0.045, 0.070] it computes
  `clks = [0.045, 0.545]` — the second grid point **outside the design's own
  declared range**, and only 2 points where the live campaign explored 796.

**Failure scenario**
`eda-rl optimize --design likith --table <table>` silently searches a
one-point clock space (every candidate snapped to 0.045) and constant
IO_DELAY/LB_ADDON — the exact "silently pinned to a constant" failure mode
this repo has already been burned by twice, now via a third mechanism.

**Fix plan**
Snap only axes that declare `_snap_step` (make absence mean "don't snap"), and
derive `build_table`'s clock grid step from the range (e.g. `(hi−lo)/10`) or a
per-design `table_grid` field. Add an assert that generated grid points lie
within the axis range.

---

## F7. CLOCK_UNCERTAINTY / IO_DELAY silently enter *every* design's tier-2+ space with platform-inappropriate absolute-ns ranges (opt-out, though the knob notes read as opt-in)

**Severity: medium-high**

**Evidence**
- `knobs.py:258-279`: `space()` adds every `active(max_tier)` knob; the new
  SDC knobs are tier 2, so any `--max-tier 2+` campaign on any design samples
  them. Their notes claim "Not sampled unless a design's active knob space
  includes it (tier 2+)" — which is technically true but reads as opt-in; in
  practice it is automatic.
- sagar.yaml declares no override for them, and its real campaign sampled
  `IO_DELAY ∈ [0.020, 0.9999]` and `CLOCK_UNCERTAINTY ∈ [0.0, 0.0964]` (ns) —
  the registry defaults. On asap7 those same defaults would exceed the entire
  clock period (0.045–0.070 ns); likith only avoided this because its YAML
  hand-overrides the ranges.
- Combined with F1, this means every existing tier-2+ design campaign changed
  objective semantics the moment this diff landed, with no per-design action.

**Failure scenario**
`eda-rl optimize --design gcd --platform nangate45 --max-tier 2` (a documented
command) now spends budget on two nuisance axes whose reward effect is
constraint-relaxation (F1), and produces rewards incomparable with any pre-diff
gcd campaign.

**Fix plan**
Make SDC-rewriting knobs opt-in: skip them in `space()` unless the design YAML
lists them under `knobs.override`/an explicit `enable:` list. Alternatively
express them as *fractions of the clock period* (platform-independent), with
conservative defaults equal to the current SDC behavior (io_frac=0.2,
uncertainty_frac=0).

---

## F8. F2 proxy no longer sees the same SDC as F3 when SDC knobs are sampled (`_run_f2` doesn't forward knob_values; its "knobs can't affect the proxy" rationale is now stale)

**Severity: medium**

**Evidence**
- `eda_rl/gen2/funnel.py:870-877` (comment: "tier-2/3 ORFS knobs do not affect
  its output and are intentionally not forwarded") and `:902`
  (`run_synth_sta(lanes, acc_w, clk, platform, **kwargs)` — no `knob_values`).
- `run_synth_sta` *does* accept `knob_values` and `_stage_inputs` would write
  CLOCK_UNCERTAINTY/IO_DELAY into the SDC that the F2 STA reads
  (`physical_runner.py:605-621`, `:1027-1029`).
- Consequence in the real campaigns: every F2 wns/timing evaluation used the
  clk·0.2 io fraction while the F3 build used the sampled IO_DELAY — for
  combinational designs the two stages were timed under different constraint
  sets, degrading exactly the F2→F3 correlation the funnel depends on.

**Failure scenario**
A config with tiny IO_DELAY looks *worse* at F2 (io=0.2·clk is stricter) than
at F3 and vice versa; kill decisions at F2 are made against the wrong ruler.

**Fix plan**
Forward the SDC-owned subset of `_effective_orfs_knobs()` (CLOCK_UNCERTAINTY,
IO_DELAY) into `run_synth_sta(knob_values=…)`, and update the comment to say
"placement/routing knobs are not forwarded; SDC knobs are". Note the F2 cache
key already includes knob_values, so caching stays correct.

---

## F9. Sampled CORE_UTILIZATION is silently floor-truncated to int at F3

**Severity: medium**

**Evidence**
- `funnel.py:932`: `util = int(eff.pop("CORE_UTILIZATION", 40))`.
- Knob is `type: float` (`knobs.py:560-561`); likith overrides range to
  [3.0, 7.0]. Real artifact: sampled 6.x emitted as
  `export CORE_UTILIZATION = 6` (config_id_c0p04532_u6_… above).
- The campaign log stores the float (`CORE_UTILIZATION: 1125 distinct values`),
  so the Optuna study and any analysis see resolution the build never had:
  on likith the effective axis has ~5 levels (3..7), not 1125.

**Failure scenario**
TPE fits structure to sub-integer variation that is pure noise; two "different"
configs (6.01 vs 6.99) are the same build but can't share the cache (variant
hash differs), wasting budget on duplicate physical builds.

**Fix plan**
`util = int(round(float(...)))` at minimum; better, declare CORE_UTILIZATION as
`type: int` in the registry (ORFS accepts integers) or stop truncating (ORFS
accepts float utilization) — pick one representation and make sampler, log,
variant name, and emission agree. Record the *effective* (emitted) value in the
episode obs.

---

## F10. The 22-dim state under-represents exactly what drives reward on the new designs (clock normalization, sky130hd, and all knob axes)

**Severity: medium** (it's the load-bearing input of the research question)

**Evidence**
- `funnel.py:138-141` `_CLK_NORM`: asap7 `(clk−0.3)/1.2` maps likith's entire
  clock range [0.045, 0.070] to [−0.2125, −0.1917] — ~2% of the nominal scale —
  while clock is the *strongest* reward correlate in that campaign (+0.77).
  sky130hd has no entry at all (falls back to nangate45's (3,5); workable for
  sagar by coincidence).
- `state_spec.py` [4] is a binary nangate45/asap7 flag: sky130hd encodes as
  nangate45 (`funnel.py:337`).
- None of the 17 sampled ORFS knob axes appear in the state; with F3
  (cell/ff, F3) and F2 signals degraded (F3-slot obs fine, but see F3/F4
  above), LinUCB's context for a likith episode is effectively
  {~constant clk term, recipe, budget, depth one-hot}.

**Failure scenario**
The central research question ("can a learned promotion policy beat fixed
gates?") is being tested with a state that cannot express the factors that
determine the outcome, on top of observables that are zero (F3) or constraint
echoes (F4). A negative result about LinUCB is uninterpretable in this setup.

**Fix plan**
Per-design normalization: compute clk norm from the design's own
`clock_range_ns` (`(clk−lo)/(hi−lo)`), make [4] a small categorical/embedding
per platform, and add a compact knob summary (e.g. each active knob min-max
scaled into a fixed-order block, or at least util/density/io_delay). This
changes the state contract — bump STATE_DIM deliberately in `state_spec.py`
(one place, per the invariant) with a migration note.

---

## F11. FixedGateAgent's F2 kill threshold is an absolute −2.5 ns: inert on sub-ns platforms, so the "fixed gates" baseline degenerates to always-promote

**Severity: medium**

**Evidence**
- `promotion_agent.py:282-284`: `_RAW_WNS_KILL_NS = −2.5` (normalized −0.5),
  calibrated on tinymac/nangate45 (3–8 ns clocks).
- likith F2 WNS values are O(±0.003 ns) (real proxy run: wns 0.0; F3 wns
  −0.0014…−0.0035); sagar timing always met. The gate can never fire; at F0
  generic designs always promote (accuracy sentinel).

**Failure scenario**
On fast platforms the LinUCB-vs-fixed benchmark compares LinUCB against
"promote everything", not against a meaningful gate — a too-easy baseline that
also burns budget (every candidate reaches F3 unless the bandit kills it).

**Fix plan**
Express the threshold in clock-relative units (e.g. kill when
`wns < −k·clock_period`, k≈0.5) or normalize state[10] by the design's clock
instead of the fixed /5 ns. Note this interacts with F10's normalization work.

---

## F12. F2/elaborate subprocesses leak tool processes on timeout (no process-group kill, unlike run_physical)

**Severity: medium**

**Evidence**
- `physical_runner.py:1051-1054` and `:1089-1092` (`run_synth_sta`) and
  `:852-855` (`run_elaborate`) use `subprocess.run(["bash","-c", …],
  timeout=PROXY_TIMEOUT)` without `start_new_session` / `killpg`. On
  `TimeoutExpired`, Python kills the direct child (bash); the `yosys` /
  `openroad` grandchild keeps running detached.
- Contrast with `run_physical` (`:749-782`), which was explicitly hardened for
  this in the second audit.

**Failure scenario**
A hung yosys on a pathological netlist survives the timeout; over a long
unattended campaign these accumulate and starve the machine. FunnelEnv catches
the exception and marks F2 FAIL, so nothing surfaces in the logs.

**Fix plan**
Use the same `Popen(start_new_session=True)` + `communicate(timeout)` +
`os.killpg` pattern as `run_physical` (factor it into a helper —
three call sites want it).

---

## F13. Concurrent campaigns on the same design can cross-contaminate staged RTL (and same-variant runs race)

**Severity: medium** (probability moderate, effect is silent wrong data)

**Evidence**
- `physical_runner.py:592-599` (`_stage_inputs`): RTL is copied to the shared
  `MAKE_DIR/src/<design_name>/<fname>` — *not* variant-scoped. The variant name
  embeds the RTL hash (`variant_name`, `:239-246`), but the staged source is
  whatever the most recent campaign copied.
- Two campaigns on the same design name with different RTL content (e.g. one
  on an edited working tree, one on an older checkout, same `EDA_RL_WORK`):
  campaign B overwrites `src/id/id.v`; campaign A's next build synthesizes B's
  RTL into a variant directory whose name claims A's hash → permanently
  poisoned cache (`6_final.gds` exists → skipped forever).
- Same-variant concurrency: `config_<variant>.mk` / `constraint_<variant>.sdc`
  writes and the ORFS make invocation share paths; two identical configs from
  two seeds running concurrently collide in `results/<plat>/<design>/<variant>/`.

**Fix plan**
Stage RTL under a content-addressed dir (`src/<design>_<rtlhash>/`) and point
`VERILOG_FILES` there — one-line change in `_config_mk`/`_stage_inputs` that
makes the staging immutable per hash. For same-variant races, take an exclusive
`flock` on the variant dir around the make invocation (skip-if-locked +
poll-for-result is enough).

---

## F14. `clock_port` (and design-YAML knob override values) are unvalidated injection sinks into generated TCL/config.mk — same class as the fixed name/top hole

**Severity: low** (author-controlled YAML per the documented threat model, but the prior audit chose to harden this class)

**Evidence**
- `designs.py:231` accepts any string for `clock_port`; `_sdc_text` interpolates
  it into `set clk_port_name {clock_port}` — a `clock_port: "clk]; exec touch /tmp/pwned;#"`
  executes inside OpenSTA/OpenROAD (TCL), the same interpreters the name/top
  fix (`_SAFE_IDENT_RE`, `designs.py:62,189-198`) was protecting.
- Similarly `knobs.override.choices` values and `params` values flow into
  config.mk `export NAME = value` lines (`knobs.py:emit_lines`,
  `physical_runner._config_mk:500-506`), where `$(shell …)` would be expanded
  by make.

**Fix plan**
Apply `_SAFE_IDENT_RE` to `clock_port` in `DesignSpec.load` (it is an
identifier by construction), and validate knob/param values as numeric or
`[A-Za-z0-9_ .+-]`-safe at load time. Cheap, closes the sibling holes to the
one already fixed.

---

## F15. Campaign driver bookkeeping nits: budget double-count, invalid-config spin, non-self-describing logs

**Severity: low**

**Evidence**
- `run_funnel_optimizer.py:310`: `if env.spent_s + env._episode_spent_s >= budget_s`
  — but `FunnelEnv._charge` (`funnel.py:1112-1114`) adds each cost to *both*
  counters, so the episode's spend is counted twice; campaigns stop early
  (conservative, but wrong).
- `:290-297`: a `ValueError` from `env.reset` (invalid config) costs zero
  budget and loops silently; if the generator's space systematically mismatches
  the table/constraints (easy in table mode, see F6), the driver busy-loops
  indefinitely with no output.
- Per-episode log rows (`:381-399`) carry no `design`, `sampler`, `promotion`,
  `max_tier`, or `seed`; the summary dict that has them is printed but never
  persisted. The two real campaign logs cannot be attributed to a promotion
  policy after the fact.

**Fix plan**
Use `env.spent_s` alone in the mid-episode check; count consecutive
reset-failures and abort with a clear message after N; add the campaign
metadata to each log row (or write one `campaign_meta` header row + persist the
summary line at exit).

---

## F16. gen2/common still import gen1 (live coupling, not historical leftover)

**Severity: low (flagged as required)**

**Evidence**
- `eda_rl/gen2/funnel.py:101` `from eda_rl.gen1.cascade import _run_sim`
- `eda_rl/gen2/build_table.py:315` `from eda_rl.gen1.runner import run_sim`
- `eda_rl/gen2/surrogate.py:471` `from eda_rl.gen1.reward import SW_BASELINE_CLOCK_NS`
- `eda_rl/common/physical_reward.py:26` `from eda_rl.gen1.reward import SW_BASELINE_LATENCY_NS, acc_overflows`

**Failure scenario**
Deleting/refactoring gen1 (its stated status is "kept only for history")
breaks F1, the TinyVAD reward, and the surrogate's tinyvad branch.

**Fix plan**
Move `SW_BASELINE_LATENCY_NS` / `SW_BASELINE_CLOCK_NS` / `acc_overflows` into
`common/constants.py` (they are measured constants + a 5-line predicate), and
move the Verilator `_run_sim` wrapper into `common/`.

---

## F17. Assorted small drift / hygiene

**Severity: nitpick**

- AGENTS.md says "24 ORFS knobs in 4 tiers"; the working tree has 27
  (`knobs.py:1065`). Update on commit.
- `surrogate.py:125` docstring: "6-element config feature vector" — it is 8.
- likith F3 rows have `ff_count=None` (206/206) — `6_report.json` has no
  sequential-cell key for a combinational design; harmless but the
  "F3 carries a real FF count" invariant should say "when present".
- `_parse_metrics` area regex resolution is 1 µm² — on likith (2–4 µm² total)
  the generic reward's `norm_area` jumps 2× from quantization alone.
- `validate_config` exclusivity warning (PLACE_DENSITY>0.6 + LB_ADDON>0)
  printed to stderr on ~every sagar episode (~1,400×) — pre-aggregate or log once.
- `PHYSICAL_MOCK=1 build_table` (a canonical AGENTS.md self-test) appends
  `mock-proxy` rows to the *tracked* `eda_rl/results/gen2/results_funnel.jsonl`
  when its configs aren't already present (currently a no-op only because the
  strategic subset is already covered by real rows). Default the self-test
  `--out` to a temp path.
- `FunnelEnv` default `results_path` points inside the installed package tree
  (`funnel.py:479`) — fine for editable installs, writes into site-packages
  otherwise.
- `eda_rl_runs/` is 17 GB after two overnight campaigns; there is no GC or
  size cap (operational, see RECOMMENDATIONS).
