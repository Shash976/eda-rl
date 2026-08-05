## What changed and why

<!-- Summarize the change and the motivation. Link an issue if there is one. -->

## Self-tests run locally

`ci.yml` re-runs the mock-safe subset automatically, but real-tool
verification (anything touching measurement/reward/knob/parser code) is on
the author — see AGENTS.md.

- [ ] `PHYSICAL_MOCK=1 python -m eda_rl.funnel.env`
- [ ] `python -m eda_rl.funnel.promotion_agent`
- [ ] `python -m eda_rl.funnel.candidates`
- [ ] `python -m eda_rl.funnel.benchmark_funnel --selftest`
- [ ] `python -m eda_rl.common.knobs`
- [ ] `python3 tests/test_parsers.py`
- [ ] `PHYSICAL_MOCK=1 eda-rl doctor --design gcd --platform nangate45`
- [ ] `PHYSICAL_MOCK=1 python -m eda_rl.funnel.build_table --design tinymac_accel --subset strategic --limit 5`
- [ ] Touched measurement/reward/knob/parser code → verified with a **real**
      ORFS run, not just the mock suite (mock metrics are synthetic and
      cannot catch parser/measurement regressions).

## New or changed design?

- [ ] `eda-rl doctor --design <name> --platform <platform>` run (and
      `--probe-f3` if knob ranges or `CORE_UTILIZATION` changed).

## Risk / rollback

<!-- Anything that changes reward semantics, state layout, or campaign log
     schema needs a note here — those are versioned/load-bearing per AGENTS.md. -->
