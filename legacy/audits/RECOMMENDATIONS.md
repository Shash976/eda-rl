# RECOMMENDATIONS.md — forward-looking, prioritized (audit 2026-07-06)

Companion to AUDIT_FINDINGS.md (bug list). This file is about making the
project achieve its stated goal: *a multi-fidelity optimizer whose learned
promotion policy demonstrably beats simple baselines at spending a synthesis
budget well*. Ordered by expected impact per unit effort.

---

## R1. Fix the measurement before doing any more learning

**What:** Land fixes for findings F1–F4 (constraint-independent reward
metrics, GR_SEED/ROUTING_LAYER_ADJUSTMENT, F2 cell-count parse, F2 fmax echo)
*before* running another campaign or fitting another surrogate — and then
re-run the likith and sagar campaigns to regenerate honest corpora.

**Why it moves the needle:** Every downstream artifact — TPE study, surrogate,
promotion-policy state, report, docs conclusions — is currently trained/graded
on (a) an objective that pays for relaxing constraints, (b) an F2 observable
set that is partially zeros and partially `1000/clk`. Any conclusion drawn
from the existing likith/sagar campaign data about "which configs are good" or
"whether learning helps" is unusable. The two overnight campaigns are ~450
real F3 builds of sunk cost; budget one more night after the fixes.

**Cost/risk:** Days of work; the reference-SDC re-timing (F1 fix) adds ~10 s
per F3 build. Negligible risk vs. status quo.

---

## R2. Answer the LinUCB question with a decision rule, not a bigger bandit — my read: the bandit formulation is the wrong tool here

**What:** Replace (or at least benchmark against) the LinUCB promotion policy
with an explicit value-of-information rule driven by the surrogate you already
have:

> promote a candidate from Fk to Fk+1 iff
> `E[max(0, reward − incumbent) | config, Fk obs] / cost(Fk+1) > τ(budget)`
> using the surrogate's (q16, q50, q84) — which exists precisely to support
> this — with τ annealed as budget depletes; kill otherwise.

**Why (the opinion the audit brief asked for):** docs/08 already concedes
cold-start LinUCB doesn't beat fixed gates. Having read the code and the real
campaign data, I don't think that's (only) a data or tuning problem — the
formulation is mismatched three ways:

1. **The decision is not a contextual-bandit decision.** Promotion is a
   sequential value-of-information problem: "is the expected improvement from
   knowing F3 worth 420 s?" LinUCB per-action linear models can only encode
   this indirectly, and the `promote` arm must average over wildly different
   regimes (F0→F1 shaping micro-costs of ~−0.001 and F2→F3 payoffs of −20…+4,
   noted in the code itself at `promotion_agent.py` "audit M6").
2. **The state can't carry the signal** (F10): on the new designs the context
   collapses to ≈{clock-term that spans 2% of its scale, recipe, budget,
   depth}, and the F2 observables it would need are broken (F3/F4). A linear
   model over a broken 22-dim context has no chance against a threshold that
   encodes real domain knowledge — that's not a fair fight, and fixing the
   inputs (R1, F10) is a precondition for *any* learner.
3. **The baseline is mis-specified on fast platforms** (F11): "fixed gates"
   currently means "always promote" for sub-ns designs, so even the benchmark
   you'd use to declare victory is not measuring gate quality.

The surrogate-driven expected-improvement-per-cost rule is (a) the textbook
answer to this exact problem (multi-fidelity BO / freeze-thaw / BOHB family),
(b) already 80% built — `Surrogate.predict_reward_stats` returns exactly the
(µ, σ) it needs, (c) interpretable (you can print *why* a candidate was
killed), and (d) has no cold-start problem: with no data it degrades to cost
ordering, which is the fixed-gate funnel. If it can't beat fixed gates either,
the honest conclusion is that on these small design spaces the funnel's kill
decisions just don't matter much — which is itself a publishable, budget-saving
answer. Keep LinUCB as a comparison arm, not the protagonist; PPO ("only if
the bandit measurably loses to lookahead", docs/08) should stay shelved.

**Cost/risk:** ~1–2 days for the rule + wiring into `run_funnel_optimizer`
and `benchmark_funnel`. Risk: surrogate quality gates everything → do R3 first.

---

## R3. Make the surrogate model the space that campaigns actually search

**What:** F5's fix plan, plus: featurize the full active space generically
(sorted axis names → fixed-order numeric vector, stored in the model file),
train on the (re-run, post-R1) campaign corpora, and report grouped-CV Spearman
ρ per design in the report HTML. Refuse predictions when the stored axis list
mismatches the campaign space.

**Why:** The funnel's entire premise is that cheap observations + a model can
substitute for expensive builds. Today the model is structurally blind to 17
of 21 axes and the recipe flag, so surrogate_ucb and dims [14]/[15] cannot
work regardless of data volume. This is also a precondition for R2.

**Cost/risk:** ~1 day. Risk of overfitting small corpora → keep the
quantile-GBT + GroupKFold machinery, it's sound.

---

## R4. Give every design a "physics sanity" preflight instead of tribal-knowledge YAML overrides

**What:** A `eda-rl doctor --design X --platform Y` step (or automatic at
campaign start) that runs: (1) one F2 proxy at the default config and asserts
parseable area/cells/fmax (catches F3/F4-class regressions and missing libs);
(2) one *real* F3 at the design's default util and, on PDN-0185-class failure,
bisects util downward a couple of steps and reports the working floor; (3)
warns when knob ranges are incoherent with the platform (IO_DELAY > clock
period, CTS_CLUSTER_DIAMETER ≫ die size, etc.).

**Why:** The likith/sagar YAML headers show what onboarding actually costs
today: someone had to discover PDN-0185 via failed real builds, hand-derive
util≈5 and util≈15, hand-convert ps→ns from autotuner.json, and know that
tier flags gate which knobs matter. That knowledge is now frozen in comments
where the next design's author won't see it. The "drop in a 10-line YAML"
story is only true for designs that resemble the worked examples; a preflight
turns each discovered failure mode (this repo's second one already — the
clockless-SDC surprise being the first) into an automated check.

**Cost/risk:** ~2 days including the util bisect (each probe build on these
tiny designs is 1–3 min). No risk — it's advisory.

---

## R5. Treat tool-output formats as a contract: golden-log regression tests

**What:** Capture short real-output fixtures (yosys `stat` block, OpenSTA
`report_clock_min_period` incl. the `fmax = inf` case, `6_finish.rpt`,
`6_report.json`) into `tests/fixtures/`, and unit-test every parser in
`physical_runner.py` against them. Add one *live* smoke test (`eda-rl doctor`,
R4) to the AGENTS.md self-test list so a tool upgrade fails loudly.

**Why:** The audit found two independent silent parser deaths (F3: yosys stat
format; F4: `inf` fmax) that mock-based self-tests are structurally unable to
catch — mock mode fabricates exactly the fields the parsers should be
producing. This repo's known-trap history is "self-tests check structure, not
behavior under real tools"; fixtures are the cheapest way to close that class.

**Cost/risk:** <1 day. None.

---

## R6. Constraint knobs need an explicit ontology: flow knobs vs. environment knobs

**What:** Tag each knob in the registry as `affects: netlist | layout |
constraints`, and make the reward/report machinery aware: `constraints` knobs
(CLOCK_PERIOD is the deliberate exception, it is the performance *target*)
are excluded from reward-visible metrics (R1), shown separately in the report,
and opt-in per design (F7).

**Why:** IO_DELAY/CLOCK_UNCERTAINTY/GR_SEED are qualitatively different from
PLACE_DENSITY: they change the question, not the answer. AutoTuner can afford
to tune them because its objective is usually a fixed post-route metric; this
project's reward is computed *from* the constrained timing report. Encoding
the distinction once in the registry prevents every future SDC-ish knob from
re-opening F1.

**Cost/risk:** Small schema change + docs. None.

---

## R7. Make benchmarks run on the real new-design tables, and fix the baseline before declaring winners

**What:** (a) Fix F6 so `build_table`/table-mode work for sub-ns clock ranges;
(b) build real F0–F2 (+ sampled F3) tables for likith/sagar/gcd from the
(post-R1) campaigns via the existing `load_table` path; (c) re-specify
FixedGateAgent thresholds in clock-relative units (F11) so "fixed gates" is a
real baseline on every platform; (d) then run the
random / fixed / LinUCB / R2-rule four-way benchmark with `--seeds 20` on all
three designs and put the table in docs/08.

**Why:** The headline claim ("cold-start LinUCB does not beat fixed gates")
currently rests on the synthetic tinymac table only, against a baseline that
is inert on two of the three real design/platform combos in the repo. Whatever
R2 concludes, it needs this harness to be credible.

**Cost/risk:** Mostly compute (~1 overnight per design for F3 sampling).

---

## R8. Operational hygiene for long unattended campaigns

**What, in order of value:**
- Work-tree GC: `eda-rl gc --keep-best N --keep-days D` pruning
  `results/<plat>/<design>/<variant>` dirs not referenced by any campaign
  best-list (17 GB after two nights today; a month of campaigns fills a disk).
- Process-group kill for the proxy/elaborate subprocesses (F12).
- Content-addressed RTL staging + variant flock (F13) so two campaigns can
  legally share an `EDA_RL_WORK`.
- Self-describing campaign logs + persisted summary row (F15) — without
  design/sampler/promotion metadata in the JSONL, the two existing overnight
  logs already can't be attributed to a policy.
- Rate-limit repeated `validate_config` warnings; surface counts in the
  end-of-campaign summary instead.

**Why:** All were observed, not hypothesized: the 17 GB tree, the
unattributable logs, and the warning spam are in the current artifacts.

**Cost/risk:** 1–2 days total; each item independent.

---

## R9. Decouple gen2 from gen1 (F16) and finish the docs debt on commit

**What:** Move `SW_BASELINE_LATENCY_NS` / `SW_BASELINE_CLOCK_NS` /
`acc_overflows` / the Verilator `_run_sim` wrapper into `common/`; update
AGENTS.md (27 knobs, the new pseudo-knob types, the F2-cells caveat until F3
is fixed, likith/sagar as worked examples with their PDN-0185 lesson) in the
same commit as the working-tree diff.

**Why:** gen1 is documented as frozen history but is a live dependency of the
reward function; and this audit found AGENTS.md's invariants list is the
single most load-bearing doc in the repo — keeping it true is cheap insurance
given three audits' worth of evidence that stale claims get re-trusted.

**Cost/risk:** Hours. None.

---

## Bottom line

The plumbing generality bet (DesignSpec + KnobRegistry + funnel) is paying
off — two genuinely new designs on two new platform paths ran 2,500+ episodes
unattended, and the second-audit invariants all held. What failed is the
*measurement layer*: the objective can be gamed through the SDC, and the cheap
fidelities are feeding zeros and constraint echoes to everything that learns.
Fix measurement (R1, R5), make the model see the real space (R3), and then
answer the promotion question with expected-improvement-per-cost (R2) rather
than more bandit capacity. If that rule still can't beat fixed gates on honest
data across three designs (R7), the right conclusion is that kill-policy
learning is not where this system's value is — the surrogate-guided *sampler*
is — and the project should say so.
