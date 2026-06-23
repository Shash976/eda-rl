"""eda_rl — multi-fidelity RL/DSE optimizer for RTL→GDS chip design-space exploration.

Drop in any design (RTL + a small YAML ``DesignSpec``) and the optimizer searches the
ORFS (OpenROAD-flow-scripts) flow for configs that trade off area / Fmax / power.

Layout:
  common/  shared plumbing — DesignSpec, ORFS runner, knob registry, rewards, constants
  gen2/    the multi-fidelity funnel (F0→F3), surrogate, candidate gen, promotion policies
  gen1/    legacy single-step black-box DSE (drives a behavioral sim; design-specific)
  viz/     campaign reporting + comparison renders
"""

__version__ = "0.1.0"
