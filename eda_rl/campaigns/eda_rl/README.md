# Mispathed campaign logs — real data, wrong directory

These two logs are **genuine campaigns** (15,484 and 7,887 episodes, ~7 h each),
not debris. They landed here because `--design` was given a *path* rather than a
name, and the raw string was pasted straight into the log path:

    eda-rl optimize --design eda_rl/designs/sagar.yaml ...
    -> eda_rl/campaigns/eda_rl/designs/sagar.yaml/sky130hd/...

The cause is fixed: `run_funnel_optimizer.main()` now resolves `--design`
through `DesignSpec.load(...).name` before building the path, so a YAML path and
a bare name produce the same canonical directory.

**They are deliberately left here rather than moved into
`campaigns/{sagar,likith}/`.** Two reasons:

1. Both predate the audit-F18 power-ruler fix, so they are `reward_version: 1`
   corpora. Their rewards are contaminated by the clock-period confound
   (corr(reward, clock_period_ns) = +0.76) and are **not comparable** with v2
   results. Moving them into the canonical directories would mix rulers in the
   set `fit-surrogate` mines — which it would then correctly refuse to fit.
2. `viz/campaign_data` resolves the canonical
   `campaigns/<design>/<platform>/results_funnel_campaigns.jsonl`, so a second
   file there would change which campaign `report`/`collect` pick by default.

Note `fit_surrogate._mine_campaign_rows` derives the design from the first path
component, so these currently mine as design `"eda_rl"` — harmless with a
`--design` filter, wrong without one.

What they are:

| log | campaign | design / platform | episodes | F3 |
|---|---|---|---|---|
| `designs/sagar.yaml/sky130hd/` | `campaign_0_1783716941` | sagar / sky130hd | 15,484 | 58 |
| `designs/likith.yaml/asap7/`   | `campaign_0_1783716891` | likith / asap7   | 7,887  | 2 |

The likith one is the **LinUCB-collapse** exhibit: it fired F3 at episodes 1 and
3, then killed 7,884 consecutive episodes and never promoted again — the failure
`promotion_agent.py`'s docstring describes, reproduced verbatim in a real run.

Both are superseded for learning purposes by the R1 re-run (see AGENTS.md), which
must happen on reward v2.
