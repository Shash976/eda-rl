"""eda_rl/common/designs.py — design registry for the chip-config optimizer.

Any design (RTL files + top module, possibly with macros) becomes an input by
creating a YAML spec in eda_rl/designs/<name>.yaml and loading it via
DesignSpec.load(name_or_path).

PINNED interface (concurrent agents code against this exactly):

    @dataclass
    class DesignSpec:
        name: str                      # registry key and ORFS DESIGN_NAME
        top: str                       # verilog top module
        rtl_files: list[str]           # absolute paths after load()
        clock_port: str                # for the SDC template
        params: dict[str, dict]        # RTL chparam axes; may be {}
        platforms: dict[str, dict]     # {"nangate45": {"clock_range_ns":[3.0,8.0], ...}, ...}
        has_macros: bool | None        # None = auto-detect at first F2
        functional_eval: dict | None   # {"kind":"tinyvad_sim"} or {"kind":"none"} or None
        @classmethod
        def load(cls, name_or_path: str) -> "DesignSpec"
        def sdc_text(self, platform: str, clock_value_native: float,
                     uncertainty: float | None = None,
                     io_delay: float | None = None) -> str

Design YAML spec format (see eda_rl/designs/tinymac_accel.yaml for full example):
    name: <str>
    top: <str>
    rtl_files: [<path>, ...]   # relative to repo root OR absolute
    clock_port: <str>
    params:
        PARAM_NAME:
            choices: [v1, v2, ...]   # OR range: [lo, hi]
            default: <value>
    platforms:
        nangate45:
            clock_range_ns: [lo, hi]
            default_clock_ns: <float>
        asap7:
            clock_range_ns: [lo, hi]
            default_clock_ns: <float>
    has_macros: false   # or true, or omit for auto-detect (None)
    functional_eval:
        kind: tinyvad_sim   # or: none
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root: eda_rl/common/designs.py → ../../../ = repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Designs YAML directory
_DESIGNS_DIR = Path(__file__).resolve().parent.parent / "designs"

# name/top become path components and get embedded in shell strings downstream
# (physical_runner.py _stage_inputs/run_physical/run_elaborate/run_synth_sta),
# so they must be safe plain identifiers — no slashes, quotes, or shell metachars.
# clock_port is interpolated into the generated SDC's `set clk_port_name <v>` /
# `get_ports <v>` (TCL), so it is an identifier by construction and validated
# with the same rule (audit F14).
_SAFE_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# knobs.fix / knobs.override.{range,choices,default} / params values flow into
# the generated SDC (TCL) and config.mk `export NAME = value` lines, where make
# would expand `$(shell …)` and TCL would evaluate `[...]` / `;` (audit F14).
# Numeric YAML scalars are always safe; string scalars must be limited to a
# conservative alphabet with no shell/TCL/make metacharacters.
_SAFE_VALUE_RE = re.compile(r"[A-Za-z0-9_ .+-]+")


def _check_yaml_value(field_label: str, value: Any, yaml_path: Any) -> None:
    """Reject a knob/param YAML value that is an injection risk.

    Numeric scalars (int/float/bool) always pass.  String scalars must match
    _SAFE_VALUE_RE.  Lists/tuples are checked element-wise; nested dicts are
    checked value-wise (with the key appended to the label).  Anything else
    (e.g. a mapping-typed default the schema doesn't expect) is rejected.
    """
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return
    if value is None:
        return
    if isinstance(value, str):
        if not _SAFE_VALUE_RE.fullmatch(value):
            raise ValueError(
                f"DesignSpec.load: {field_label} value {value!r} is not "
                f"injection-safe (must match {_SAFE_VALUE_RE.pattern!r}) in {yaml_path}"
            )
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_yaml_value(f"{field_label}[{i}]", item, yaml_path)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _check_yaml_value(f"{field_label}.{k}", v, yaml_path)
        return
    raise ValueError(
        f"DesignSpec.load: {field_label} has unsupported value type "
        f"{type(value).__name__} ({value!r}) in {yaml_path}"
    )


def _validate_design_values(raw: dict, yaml_path: Any) -> None:
    """Validate all design-YAML knob/param values at load time (audit F14).

    Covers: knobs.fix values, knobs.override {range/choices/default/low/high}
    entries, and every params[...] choices/range/default/values entry.
    """
    knobs = raw.get("knobs") or {}
    if isinstance(knobs, dict):
        for name, val in (knobs.get("fix") or {}).items():
            _check_yaml_value(f"knobs.fix.{name}", val, yaml_path)
        override = knobs.get("override") or {}
        if isinstance(override, dict):
            for axis, ov in override.items():
                if isinstance(ov, dict):
                    for key in ("range", "choices", "default", "low", "high"):
                        if key in ov:
                            _check_yaml_value(f"knobs.override.{axis}.{key}",
                                              ov[key], yaml_path)
                else:
                    _check_yaml_value(f"knobs.override.{axis}", ov, yaml_path)

    params = raw.get("params") or {}
    if isinstance(params, dict):
        for pname, pspec in params.items():
            if not isinstance(pspec, dict):
                _check_yaml_value(f"params.{pname}", pspec, yaml_path)
                continue
            for key in ("choices", "range", "default", "values"):
                if key in pspec:
                    _check_yaml_value(f"params.{pname}.{key}", pspec[key], yaml_path)

# SDC generation — generic; physical_runner.py applies PLATFORM_TIME_UNIT
# conversion before writing so clock_value_native/uncertainty/io_delay are
# always already in the platform's native unit.
#
# uncertainty/io_delay are None by default, which reproduces the original
# static template byte-for-byte (no set_clock_uncertainty line; io delay is
# the fixed clk_period*0.2 fraction) — every design that doesn't opt into the
# CLOCK_UNCERTAINTY/IO_DELAY knobs (see common/knobs.py) is unaffected.
def _sdc_text(top: str, clock_port: str, clock_period: float,
              uncertainty: float | None = None,
              io_delay: float | None = None) -> str:
    lines = [
        f"current_design {top}",
        "",
        "set clk_name      core_clock",
        f"set clk_port_name {clock_port}",
        f"set clk_period    {clock_period}",
    ]
    if io_delay is None:
        lines.append("set clk_io_pct    0.2")
    lines += [
        "",
        "set clk_port [get_ports $clk_port_name]",
        "create_clock -name $clk_name -period $clk_period $clk_port",
    ]
    if uncertainty is not None:
        # AutoTuner equivalent: _SDC_UNCERTAINTY.
        lines.append(f"set_clock_uncertainty {uncertainty} [get_clocks $clk_name]")
    lines += [
        "",
        "set non_clock_inputs [all_inputs -no_clocks]",
    ]
    if io_delay is None:
        lines += [
            "set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs",
            "set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]",
        ]
    else:
        # AutoTuner equivalent: _SDC_IO_DELAY — an absolute delay value, not a
        # clk_period fraction (matches ORFS's own convention, e.g.
        # flow/designs/asap7/mock-cpu/constraint.sdc).
        lines += [
            f"set_input_delay  {io_delay} -clock $clk_name $non_clock_inputs",
            f"set_output_delay {io_delay} -clock $clk_name [all_outputs]",
        ]
    return "\n".join(lines) + "\n"


@dataclass
class DesignSpec:
    """Immutable description of one chip design for the optimizer."""

    name: str
    top: str
    rtl_files: list[str]           # absolute paths (resolved on load)
    clock_port: str
    params: dict[str, dict]        # RTL chparam axes; may be {}
    platforms: dict[str, dict]     # per-platform clock ranges and defaults
    has_macros: bool | None        # None = auto-detect at first F2
    functional_eval: dict | None   # {"kind": "tinyvad_sim"} or {"kind": "none"} or None
    reward: dict | None = None     # optional reward config: weights + PPA anchors
                                   # (area_ref_um2 / power_ref_mw / fmax_ref_mhz).
                                   # When absent for a generic design, FunnelEnv
                                   # auto-anchors from the first F3 build.
    knobs: dict | None = None      # optional per-design control of the ORFS knob
                                   # search space (so a design is fully described
                                   # by its own YAML — no need to edit
                                   # search_space_funnel.yaml). Schema:
                                   #   knobs:
                                   #     fix:      {KNOB: value}   # pin + drop from space
                                   #     exclude:  [KNOB, ...]     # drop (use tool default)
                                   #     override: {AXIS: {range: [lo,hi] | choices: [...]
                                   #                       | low:/high:/default:}}
                                   # Omit entirely → every knob up to --max-tier is
                                   # optimized (nothing fixed).

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, name_or_path: str) -> "DesignSpec":
        """Load a DesignSpec from a YAML file.

        Resolution order:
            1. If name_or_path is an existing path (absolute or relative to cwd),
               load it directly.
            2. Otherwise, look up eda_rl/designs/<name_or_path>.yaml.

        RTL file paths in the YAML are resolved as:
            - If absolute: used as-is.
            - If relative: resolved against the YAML file's own directory, so a
              design bundle (RTL + YAML) is portable and works from anywhere.
              Override the base for relative paths with the EDA_RL_DESIGN_ROOT env
              var — e.g. point a voiceAI tinymac YAML (rtl_files: rtl/accel/...) at
              an external checkout via EDA_RL_DESIGN_ROOT=/path/to/voiceAI.

        Raises FileNotFoundError if the YAML is not found.
        Raises ValueError if required fields are missing.
        """
        import os
        import yaml  # type: ignore[import]

        yaml_path = Path(name_or_path)
        if not yaml_path.is_absolute():
            yaml_path = Path.cwd() / yaml_path
        if not yaml_path.exists():
            # Try the designs registry directory
            yaml_path = _DESIGNS_DIR / f"{name_or_path}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"DesignSpec.load: cannot find design '{name_or_path}'. "
                f"Tried: '{name_or_path}' (direct path) and "
                f"'{_DESIGNS_DIR / name_or_path}.yaml' (registry)."
            )
        yaml_path = yaml_path.resolve()

        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Required fields
        for required in ("name", "top", "rtl_files", "clock_port"):
            if required not in raw:
                raise ValueError(
                    f"DesignSpec.load: required field '{required}' missing in {yaml_path}"
                )

        # name/top/clock_port flow into filesystem paths, shell command strings
        # (physical_runner.py), and the generated SDC's TCL, so reject anything
        # that isn't a plain identifier (blocks path traversal via name and
        # shell/TCL injection via quotes/metachars). clock_port added per F14.
        for field_name in ("name", "top", "clock_port"):
            value = str(raw[field_name])
            if not _SAFE_IDENT_RE.fullmatch(value):
                raise ValueError(
                    f"DesignSpec.load: '{field_name}' must match "
                    f"{_SAFE_IDENT_RE.pattern!r} (got {value!r} in {yaml_path})"
                )

        # Validate design-YAML knob/param values (config.mk / SDC injection sinks).
        _validate_design_values(raw, yaml_path)

        # Resolve RTL file paths: relative paths anchor to the YAML's own directory,
        # or to EDA_RL_DESIGN_ROOT when set (for external/voiceAI design trees).
        _env_root = os.environ.get("EDA_RL_DESIGN_ROOT")
        design_root = Path(_env_root).resolve() if _env_root else yaml_path.parent
        rtl_files = []
        for p in raw["rtl_files"]:
            pp = Path(p)
            if not pp.is_absolute():
                pp = design_root / pp
            rtl_files.append(str(pp))

        # Optional fields with defaults
        params = raw.get("params") or {}
        platforms = raw.get("platforms") or {
            "nangate45": {"clock_range_ns": [3.0, 8.0], "default_clock_ns": 5.0}
        }
        has_macros_raw = raw.get("has_macros")
        has_macros: bool | None
        if has_macros_raw is None:
            has_macros = None        # auto-detect at first F2
        else:
            has_macros = bool(has_macros_raw)

        functional_eval = raw.get("functional_eval")
        reward = raw.get("reward")
        knobs = raw.get("knobs")

        return cls(
            name=str(raw["name"]),
            top=str(raw["top"]),
            rtl_files=rtl_files,
            clock_port=str(raw["clock_port"]),
            params=params,
            platforms=platforms,
            has_macros=has_macros,
            functional_eval=functional_eval,
            reward=reward,
            knobs=knobs,
        )

    # ── SDC generation ────────────────────────────────────────────────────────

    def sdc_text(self, platform: str, clock_value_native: float,
                 uncertainty: float | None = None,
                 io_delay: float | None = None) -> str:
        """Return the SDC content for the given platform clock value.

        clock_value_native/uncertainty/io_delay must already be in the
        platform's native unit (ns for nangate45, ps for asap7).  The caller
        (physical_runner) applies the PLATFORM_TIME_UNIT conversion BEFORE
        calling this method.

        uncertainty/io_delay are optional (CLOCK_UNCERTAINTY/IO_DELAY knobs,
        see common/knobs.py); omitting them reproduces the original SDC
        exactly (no set_clock_uncertainty line; io delay = clk_period*0.2).

        The SDC uses the design's top module name and clock port.
        """
        return _sdc_text(
            top=self.top,
            clock_port=self.clock_port,
            clock_period=clock_value_native,
            uncertainty=uncertainty,
            io_delay=io_delay,
        )

    # ── RTL content hash ──────────────────────────────────────────────────────

    def rtl_hash(self) -> str:
        """Return an 8-hex-digit SHA-256 digest of the RTL source files.

        Files are hashed in sorted-path order for determinism.  If a file is
        missing, its path string is hashed as a placeholder (consistent with the
        legacy _rtl_hash() in physical_runner.py for tinymac).
        """
        h = hashlib.sha256()
        for p in sorted(self.rtl_files):
            try:
                h.update(Path(p).read_bytes())
            except OSError:
                h.update(p.encode())   # placeholder for missing files
        return h.hexdigest()[:8]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def verilog_top_params_str(self, config: dict) -> str:
        """Build the VERILOG_TOP_PARAMS string from a config dict.

        For each param in self.params, if the canonical param name is present in
        config, emit it using the RTL chparam name (rtl_param_name field if present,
        otherwise the param name itself).  Returns "" if the design has no params.

        Example (new-style with rtl_param_name):
            design.params = {"mac_lanes": {"rtl_param_name": "LANES", ...},
                             "accumulator_width": {"rtl_param_name": "ACC_W", ...}}
            config = {"mac_lanes": 4, "accumulator_width": 24, "clock_period_ns": 5.0}
            → "LANES 4 ACC_W 24"

        Example (legacy: param name IS the RTL chparam name):
            design.params = {"LANES": ..., "ACC_W": ...}
            config = {"LANES": 4, "ACC_W": 24, "clock_period_ns": 5.0}
            → "LANES 4 ACC_W 24"
        """
        if not self.params:
            return ""
        parts = []
        for param_name, param_spec in self.params.items():
            if param_name in config:
                # Emit the RTL chparam name, not the canonical search-space name
                rtl_name = (
                    param_spec.get("rtl_param_name", param_name)
                    if isinstance(param_spec, dict)
                    else param_name
                )
                parts += [rtl_name, str(config[param_name])]
        return " ".join(parts)

    def functional_model(self):
        """Return the FunctionalModel this design opts into, or None.

        A design opts in via ``functional_eval.kind`` in its YAML; the registry
        (``common.functional_models``) maps that kind to a plugin.  ``None`` means
        a generic design — pure-PPA reward, F1 skipped, default report rendering.
        """
        from eda_rl.common import functional_models
        return functional_models.for_design(self)

    def has_functional_model(self) -> bool:
        """True if this design declares a registered functional model."""
        return self.functional_model() is not None

    def __repr__(self) -> str:
        return (
            f"DesignSpec(name={self.name!r}, top={self.top!r}, "
            f"rtl_files={[Path(f).name for f in self.rtl_files]}, "
            f"has_macros={self.has_macros!r}, "
            f"functional_eval={self.functional_eval!r})"
        )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== designs.py self-test ===")

    # 0. Injection-sink validation (F14) — run FIRST so it is exercised even if
    # the tinymac RTL is un-vendored in this checkout (which aborts test 1).
    import tempfile as _tf

    def _write_yaml(body: str) -> str:
        f = _tf.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(body)
        f.close()
        return f.name

    _base = (
        "name: victim\n"
        "top: victim\n"
        "rtl_files: [victim.v]\n"
    )
    # 0a. malicious clock_port must raise
    bad_clk = _write_yaml(_base + 'clock_port: "clk]; exec touch /tmp/pwned;#"\n')
    try:
        DesignSpec.load(bad_clk)
        raise AssertionError("malicious clock_port was accepted (F14 regression)")
    except ValueError as e:
        assert "clock_port" in str(e), f"wrong error for clock_port: {e}"
        print("  F14: malicious clock_port rejected  PASS")

    # 0b. malicious knobs.override choice must raise
    bad_choice = _write_yaml(
        _base + "clock_port: clk\n"
        "knobs:\n  override:\n    ABC_AREA:\n"
        '      choices: ["0", "1; exec rm -rf /"]\n'
    )
    try:
        DesignSpec.load(bad_choice)
        raise AssertionError("malicious override choice was accepted (F14 regression)")
    except ValueError as e:
        assert "knobs.override" in str(e), f"wrong error for override choice: {e}"
        print("  F14: malicious knobs.override choice rejected  PASS")

    # 0c. a benign design with safe values still loads (no false positives)
    ok_yaml = _write_yaml(
        _base + "clock_port: clk\n"
        "knobs:\n  fix: {CORE_UTILIZATION: 15}\n"
        "  override:\n    PLACE_DENSITY_LB_ADDON: {range: [0.2, 0.4], default: 0.3}\n"
    )
    _ok = DesignSpec.load(ok_yaml)
    assert _ok.clock_port == "clk"
    print("  F14: benign knob/param values still load  PASS")

    # 1. tinymac_accel loads
    try:
        tm = DesignSpec.load("tinymac_accel")
        print(f"  tinymac_accel loaded: {tm}")
        assert tm.name == "tinymac_accel"
        assert tm.top == "tinymac_accel"
        assert len(tm.rtl_files) == 3, f"Expected 3 RTL files, got {len(tm.rtl_files)}"
        assert all(Path(f).exists() for f in tm.rtl_files), \
            f"Some RTL files missing: {[f for f in tm.rtl_files if not Path(f).exists()]}"
        assert tm.clock_port == "clk"
        # Canonical param names (not RTL chparam names)
        assert "mac_lanes" in tm.params, "mac_lanes param not in tinymac_accel spec"
        assert "accumulator_width" in tm.params, "accumulator_width param not in tinymac_accel spec"
        # RTL chparam names carried in rtl_param_name field
        assert tm.params["mac_lanes"].get("rtl_param_name") == "LANES", \
            "mac_lanes should have rtl_param_name='LANES'"
        assert tm.params["accumulator_width"].get("rtl_param_name") == "ACC_W", \
            "accumulator_width should have rtl_param_name='ACC_W'"
        assert "nangate45" in tm.platforms
        assert "asap7" in tm.platforms
        assert tm.has_macros is False
        assert tm.has_functional_model()
        assert tm.functional_model().kind == "tinyvad_sim"
        print(f"  tinymac_accel: {len(tm.rtl_files)} RTL files, "
              f"params={list(tm.params.keys())}, has_macros={tm.has_macros}  PASS")
    except FileNotFoundError as e:
        print(f"  SKIP tinymac_accel load (YAML not yet written): {e}")

    # 2. gcd loads
    try:
        gcd = DesignSpec.load("gcd")
        print(f"  gcd loaded: {gcd}")
        assert gcd.name == "gcd"
        assert gcd.top == "gcd"
        assert len(gcd.rtl_files) >= 1
        assert gcd.params == {} or gcd.params is not None
        # gcd has no functional eval (generic design)
        assert not gcd.has_functional_model()
        print(f"  gcd: {len(gcd.rtl_files)} RTL files, has_macros={gcd.has_macros}  PASS")
    except FileNotFoundError as e:
        print(f"  SKIP gcd load (YAML not yet written): {e}")

    # 3. sdc_text produces valid content
    try:
        tm = DesignSpec.load("tinymac_accel")
        sdc = tm.sdc_text("nangate45", 5.0)
        assert "tinymac_accel" in sdc, "top module not in SDC"
        assert "clk" in sdc, "clock port not in SDC"
        assert "5.0" in sdc, "clock period not in SDC"
        sdc_asap7 = tm.sdc_text("asap7", 5000.0)   # 5.0 ns × 1000 = 5000 ps
        assert "5000.0" in sdc_asap7, "asap7 native clock not in SDC"
        print(f"  sdc_text nangate45 / asap7  PASS")

        # 3b. uncertainty/io_delay default to None -> byte-identical to the
        # pre-CLOCK_UNCERTAINTY/IO_DELAY-knob SDC (backward-compat regression guard).
        sdc_default = tm.sdc_text("nangate45", 5.0)
        sdc_explicit_none = tm.sdc_text("nangate45", 5.0, uncertainty=None, io_delay=None)
        assert sdc_default == sdc_explicit_none
        assert "set_clock_uncertainty" not in sdc_default
        assert "clk_io_pct" in sdc_default
        assert "[expr $clk_period * $clk_io_pct]" in sdc_default

        # 3c. uncertainty/io_delay given -> new lines appear, old fraction form doesn't
        sdc_tuned = tm.sdc_text("nangate45", 5.0, uncertainty=0.05, io_delay=0.3)
        assert "set_clock_uncertainty 0.05 [get_clocks $clk_name]" in sdc_tuned
        assert "set_input_delay  0.3 -clock $clk_name" in sdc_tuned
        assert "set_output_delay 0.3 -clock $clk_name" in sdc_tuned
        assert "clk_io_pct" not in sdc_tuned
        print(f"  sdc_text uncertainty/io_delay (backward-compat + new lines)  PASS")
    except FileNotFoundError:
        print("  SKIP sdc_text test (tinymac YAML not yet written)")

    # 4. tinymac rtl_hash matches the legacy _rtl_hash() from physical_runner
    try:
        tm = DesignSpec.load("tinymac_accel")
        # Compute the hash the same way physical_runner._rtl_hash() does:
        # sorted RTL_FILES (just file names), reading from RTL_DIR
        h = hashlib.sha256()
        rtl_dir = _REPO_ROOT / "rtl" / "accel"
        rtl_fnames = ("int8_mac_array.v", "requantize.v", "tinymac_accel.v")
        for fname in sorted(rtl_fnames):
            p = rtl_dir / fname
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(fname.encode())
        legacy_hash = h.hexdigest()[:8]

        ds_hash = tm.rtl_hash()
        assert ds_hash == legacy_hash, \
            f"rtl_hash mismatch: DesignSpec={ds_hash!r}, legacy={legacy_hash!r}"
        print(f"  rtl_hash matches legacy _rtl_hash(): {ds_hash!r}  PASS")
    except FileNotFoundError as e:
        print(f"  SKIP rtl_hash test: {e}")

    # 5. verilog_top_params_str — uses canonical param names (mac_lanes/accumulator_width)
    # as config dict keys; emits RTL chparam names (LANES/ACC_W) in the output string.
    try:
        tm = DesignSpec.load("tinymac_accel")
        # Config uses canonical names (mac_lanes/accumulator_width)
        vtp = tm.verilog_top_params_str({"mac_lanes": 4, "accumulator_width": 24,
                                          "clock_period_ns": 5.0})
        # VERILOG_TOP_PARAMS string must use RTL chparam names
        assert "LANES 4" in vtp and "ACC_W 24" in vtp, f"vtp={vtp!r}"
        print(f"  verilog_top_params_str (canonical config keys): {vtp!r}  PASS")
    except FileNotFoundError:
        pass

    print("\n=== designs.py self-test PASSED ===")
    sys.exit(0)
