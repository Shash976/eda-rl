# CLAUDE.md

The full agent orientation guide for this repo lives in **[AGENTS.md](AGENTS.md)** —
project status, layout, commands, and the invariants you must not silently
re-break. Read it first.

@AGENTS.md

## TL;DR

- **`eda-rl`** = multi-fidelity RL/DSE optimizer for RTL→GDS chip design-space
  exploration. The active system is the **funnel** (`eda_rl/funnel/`):
  `FunnelEnv` over fidelity gates F0→F3, Optuna candidate generator, quantile-GBT
  surrogate, and a promotion policy (LinUCB / fixed-gate / random). How the RL
  works + role of every file: `docs/rl_system.md`. Retired gen1 sits frozen in
  repo-root `legacy/` — don't touch it.
- **Design-agnostic via a functional-model plugin registry**
  (`eda_rl/common/functional_models/`). A design opts into a family-specific
  behavioral model + composite reward by naming `functional_eval.kind` in its
  YAML; the core dispatches through `design.functional_model()` and never names
  a design family. TinyVAD's cycle model, composite reward, behavioral sim, and
  report extras all live in `functional_models/tinyvad*.py`. Generic designs
  (`functional_model()` is None) get `compute_generic_reward`, F1 skipped, and
  the default report. **Don't hardcode TinyMAC in core** — add a plugin instead.
  A design is REQUIRED everywhere (env/runner/build_table); there is no silent
  default.
- **One state spec**: `eda_rl/funnel/state_spec.py` (22 dims). Don't fork it.
- **Measure the chip, not the ruler**: reward reads the fixed-ruler reference-SDC
  metrics (`*_ref_*`), constraint/environment knobs are opt-in per design — see
  AGENTS.md "Measurement integrity" before touching reward/SDC/knob code.
- **Try things with no tools installed**: prefix any command with `PHYSICAL_MOCK=1`
  (but note mock metrics are synthetic/design-independent — mock CANNOT catch
  parser/measurement regressions; that's `tests/test_parsers.py` + `eda-rl
  doctor`'s job).
- **New design? `eda-rl doctor --design X --platform Y` first** (it finds dead
  parsers, ps-vs-ns range mistakes, the PDN utilization floor, and the minimum
  `--max-tier`).
- After touching funnel/common, run the self-tests listed in AGENTS.md.
- Learning-signal bookkeeping: score/tell only `info["terminal_reward"]` outside
  the bandit, and the surrogate-Δ prior is captured pre-stage — see
  `docs/rl_system.md` §5 before touching reward code.
