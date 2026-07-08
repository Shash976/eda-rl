# Golden-log parser fixtures (audit R5)

Short real-tool output snippets asserted against the exact parsing rules in
`eda_rl/common/physical_runner.py`. The third audit found two silent parser
deaths (F3: yosys changed its `stat` format; F4: OpenSTA prints `fmax = inf`
for combinational designs) that mock-based self-tests are structurally unable
to catch — mock mode fabricates exactly the fields these parsers produce.
These fixtures make a tool-upgrade format change fail loudly instead.

Provenance (captured 2026-07-07 on this machine's ORFS install,
`ORFS_DIR=/opt/OpenROAD-flow-scripts`, yosys/OpenROAD from
`tools/install`):

| file | source |
|---|---|
| `yosys_stat_tabular.txt` | `run_synth_sta(design=sagar, platform=sky130hd)` synth.log — trimmed to the pre-map `51 cells` stat block and the post-`opt_clean` `36  231.472 cells` block + `Chip area` line |
| `yosys_stat_legacy.txt`  | synthetic — the pre-upgrade `Number of cells:  N` format older yosys printed (kept so the parser stays bi-format) |
| `sta_combinational.txt`  | `run_synth_sta(design=sagar, platform=sky130hd)` sta.log — the `period_min = 0.00 fmax = inf` no-clocked-path case |
| `sta_sequential.txt`     | `run_synth_sta(design=gcd, platform=nangate45)` sta.log — a real clocked `period_min = 0.64 fmax = 1564.07` case |

Run: `python3 tests/test_parsers.py` (plain asserts; also collectable by
pytest if installed).
