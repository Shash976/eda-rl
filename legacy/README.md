# legacy/ — frozen history, not part of the installed package

Nothing under this directory is imported, packaged, or executed by the live
system (`eda_rl/`). It is kept for provenance only; expect broken imports and
stale paths inside — files were moved here verbatim, not maintained.

| Path | What it is |
|---|---|
| `gen1/` | The first-generation single-step DSE optimizer (grid space, evo/UCB/bayesian agents, cascade env, its own dashboard and in-package tests). Superseded by the gen2 funnel — now `eda_rl/funnel/`. Its one live piece, the Verilator sim driver (`runner.py`), was rescued to `eda_rl/common/verilator_sim.py`. |
| `measure_real.py` | Standalone TinyMAC baseline re-measurement script (was `eda_rl/common/`, orphaned — imported by nothing, not in the CLI). Re-derives the measured constants in `eda_rl/common/constants.py`. |
| `cascade_reward.py`, `validate.py` | Shared helpers whose only importers were gen1 (the funnel has its own inline failure ladder and uses `knobs.validate_config`). |
| `docs/` | Superseded docs: `04` (gen1 optimizer), `07` (gen1-era RL design rationale), `08` (early funnel operator guide — predates the third audit; says 24 knobs, omits likith/sagar). Replaced by `docs/rl_system.md`. |
| `audits/` | The third audit's records (`AUDIT_FINDINGS.md` F1–F17, `RECOMMENDATIONS.md` R1–R9). Historical snapshots — some evidence line numbers predate the fixes. Still-open items are tracked in `docs/rl_system.md` §Known limitations. |
| `results/gen1/` | gen1 run artifacts. |
| `artifacts/` | Loose campaign JSONLs that used to sit at the repo root (untracked). |
