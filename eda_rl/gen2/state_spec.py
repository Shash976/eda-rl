"""state_spec.py — the single source of truth for the 22-dim FunnelEnv state.

Every component that reads or writes the promotion-policy state vector imports
the layout from here so funnel.py (which *builds* the vector), promotion_agent.py
(which *reads* it), and benchmark_funnel.py (which wires agents) can never drift
apart.  Previously the layout was documented three different ways — funnel.py
emitted one encoding, promotion_agent.py's header documented another
(lanes/32, acc_w/32, recipe one-hot, unrun = -1), and benchmark_funnel.py had a
third dead builder — a correctness trap (audit H3/H4).

CANONICAL LAYOUT (what FunnelEnv._build_state actually emits; unrun slots = 0.0):

    [0]  log2(lanes)/5                 # 1 (sentinel) for non-RTL-param designs
    [1]  (acc_w - 16)/16
    [2]  clock_norm                    # generic designs: (clk-lo)/(hi-lo) from the
                                       # design's own platforms[platform]
                                       # ["clock_range_ns"] → ~[0,1] over its range
                                       # (F10).  TinyVAD/legacy/no-design: fixed
                                       # platform table (nangate45: (clk-3)/5;
                                       # asap7: (clk-0.3)/1.2) for bit-compat.
    [3]  recipe_idx/2                   # orfs_speed=0, orfs_area=1, plain=2  → 0/0.5/1
    [4]  platform ordinal              # 0.0=nangate45, 0.5=sky130hd, 1.0=asap7
                                       # (F10; was a binary asap7-vs-rest flag that
                                       # encoded sky130hd as nangate45)
    [5]  F0 cycles_norm                 # log2(SW_BASELINE/cycles)/10 (0 for generic)
    [6]  F0 accuracy                    # 0..1 from analytic table (0.0 = no-data sentinel)
    [7]  F1 cycles_norm                 # 0 if F1 unrun/skipped
    [8]  F1 accuracy                    # 0 if F1 unrun/skipped
    [9]  F2 proxy_area_norm             # area/20000 clip[0,3]; 0 if unrun
    [10] F2 wns_norm                    # clip(wns/scale, -2, 2); 0 if unrun.
                                       # scale = clock period for generic designs
                                       # (state[10]=wns/clk, so FixedGate's F11
                                       # kill rule wns<-0.5·clk is state[10]<-0.5);
                                       # scale = 5 ns for TinyVAD/legacy (compat).
    [11] F2 ff_count/1000 clip[0,3]    # NOTE: the synth+STA proxy reports no
                                       # separate FF count, so live/table F2 set
                                       # ff_count = total cell count — slots [11]
                                       # and [12] carry the same signal (audit M2).
                                       # (F3 obs DOES carry a real, distinct FF
                                       # count from 6_report.json, but that is in
                                       # the log/views, not this F2-time state.)
    [12] F2 cell_count/10000 clip[0,3]
    [13] F2 logic_levels/50 clip[0,2]
    [14] surrogate mu/4.5               # 0 if no surrogate
    [15] surrogate sigma               # 0 if no surrogate
    [16] incumbent best reward/4.5     # 0 if none
    [17] remaining budget fraction
    [18..21] depth one-hot (highest fidelity already run: F0, F1, F2, F3)

The unrun convention is 0.0 (NOT -1).  Slot [3] is a single scaled recipe index,
NOT a one-hot pair, and slot [4] is the platform ordinal (0.0/0.5/1.0 for
nangate45/sky130hd/asap7).  FixedGateAgent reads only slots [6], [8], [10],
[18..21] (it does NOT branch on [4], so the ordinal change is transparent to it);
LinUCB reads all of them, so the encoding above is the contract both must honour.
"""

from __future__ import annotations

STATE_DIM = 22

# ── State vector indices (the single source of truth) ─────────────────────────
IDX_LANES_NORM   = 0
IDX_ACCW_NORM    = 1
IDX_CLK_NORM     = 2
IDX_RECIPE       = 3   # recipe_idx/2 (orfs_speed=0, orfs_area=0.5, plain=1.0)
IDX_PLATFORM     = 4   # platform ordinal (0.0=nangate45, 0.5=sky130hd, 1.0=asap7)
IDX_F0_CYCLES    = 5
IDX_F0_ACC       = 6
IDX_F1_CYCLES    = 7
IDX_F1_ACC       = 8
IDX_F2_AREA      = 9
IDX_F2_WNS       = 10
IDX_F2_FF        = 11
IDX_F2_CELLS     = 12
IDX_F2_LEVELS    = 13
IDX_SURR_MU      = 14
IDX_SURR_SIG     = 15
IDX_INCUMBENT    = 16
IDX_BUDGET_FRAC  = 17
IDX_DEPTH_F0     = 18
IDX_DEPTH_F1     = 19
IDX_DEPTH_F2     = 20
IDX_DEPTH_F3     = 21

# Back-compat aliases (older modules referenced these names for [3]/[4]).
IDX_RECIPE_SPD   = IDX_RECIPE     # was "recipe one-hot orfs_speed"; now recipe_idx/2
IDX_RECIPE_AREA  = IDX_PLATFORM   # was "recipe one-hot orfs_area"; slot is platform ordinal

# Fidelity depth labels in promotion order.
DEPTH_ORDER = ["F0", "F1", "F2", "F3"]
