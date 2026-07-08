"""physical_runner.py — run one real ORFS RTL→GDS trial and parse its metrics.

The Stage-6 analogue of runner.py (which drives Verilator).  Where runner.py
returns cycle counts from the behavioral sim, this drives the *classic
OpenROAD-flow-scripts make flow* on a machine that has a real OpenROAD (the
company VM at /opt/OpenROAD-flow-scripts) and returns real physical metrics:
area, WNS/TNS, Fmax, power.

It reuses the exact per-config mechanism of physical/orfs/make/sweep.sh:
  - VERILOG_TOP_PARAMS="LANES n ACC_W w"   → ORFS chparam overrides RTL params
  - FLOW_VARIANT=<variant>                 → each config gets its own results dir
  - a generated SDC                        → sets the clock period

Report strings parsed below were confirmed against a real VM run:
  reports/<plat>/<design>/<variant>/6_finish.rpt :
      "wns max -1.72"      (value is the LAST field)
      "tns max -25.86"
      "core_clock period_min = 3.72 fmax = 268.64"
      "setup violation count 40"
      report_power "Total  5.46e-01 4.69e-01 4.38e-04 1.02e+00 100.0%"  (total W = field 5)
  logs/<plat>/<design>/<variant>/6_report.log :
      "Design area 19738 um^2 48% utilization."

Set PHYSICAL_MOCK=1 to skip OpenROAD and return synthetic-but-plausible metrics
(for testing the optimizer loop on a machine without the tools).

DESIGN-AGNOSTIC EXTENSION (V12):
  run_physical() and run_synth_sta() now accept:
    design: DesignSpec | None = None
      None → DesignSpec.load("tinymac_accel"), preserving every existing call.
    knob_values: dict | None = None
      ORFS knob key→value dict from KnobRegistry; emit lines appended to config.mk
      after validate_config(). Knobs not present in knob_values are not emitted
      (use the ORFS defaults).

  When design is provided:
    - DESIGN_NAME/top, VERILOG_FILES, SDC clock port come from the DesignSpec.
    - VERILOG_TOP_PARAMS is assembled from design.params ∩ config keys.
    - The RTL content hash is computed from the DesignSpec's rtl_files (not the
      hardcoded RTL_FILES tuple), so any design gets automatic cache invalidation
      on RTL changes.
    - Variant names gain a design-name prefix only for non-tinymac designs
      (existing tinymac result dirs stay reachable unchanged).

  Macro auto-detect:
    After synthesis (run_synth_sta), the synth stat is parsed for macro/instance
    counts. When the design's has_macros is None (auto-detect), the result carries
    a has_macros_detected bool. Callers should propagate this back to the DesignSpec
    so tier-4 knobs become active for the next round.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from eda_rl.common.recipe import (
    config_mk_lines,
    recipe_suffix,
    resolve_recipe,
    write_abc_constr,
    yosys_abc_line,
)

_REPO    = Path(__file__).resolve().parent.parent.parent
# ORFS WORK_HOME — all per-variant staging and build output (src/, <platform>/<design>/,
# proxy/, elaborate/, results/, logs/, reports/) lives here.  Configurable so the tool
# runs against any design from any working directory; defaults to ./eda_rl_runs in CWD.
# Set EDA_RL_WORK to relocate (e.g. EDA_RL_WORK=physical/orfs/runs inside a voiceAI checkout).
MAKE_DIR = Path(os.environ.get("EDA_RL_WORK", Path.cwd() / "eda_rl_runs")).resolve()
RUN_DIR  = MAKE_DIR
# Legacy tinymac-only constants — used only by the design=None fallback path
# (run_elaborate / legacy _config_mk).  Unused when a DesignSpec is passed.
RTL_DIR  = _REPO / "rtl" / "accel"
DESIGN   = "tinymac_accel"
RTL_FILES = ("tinymac_accel.v", "int8_mac_array.v", "requantize.v")

# DesignSpec is imported lazily inside _config_mk, _stage_inputs, etc. to avoid
# circular imports and keep backward compatibility with callers that don't use design.

# V1: single source of truth for asap7 vs nangate45 time units.
# asap7 SDCs are in picoseconds; nangate45 in nanoseconds.
# Multiply the optimizer's ns clock by this factor when writing the SDC.
# Divide parsed report time values by this factor before storing in *_ns keys.
PLATFORM_TIME_UNIT: dict[str, float] = {"nangate45": 1.0, "asap7": 1000.0}

ORFS_DIR     = Path(os.environ.get("ORFS_DIR", "/opt/OpenROAD-flow-scripts"))
ORFS_TIMEOUT = int(os.environ.get("ORFS_TIMEOUT", "2400"))   # seconds; P&R is slow
PROXY_TIMEOUT = int(os.environ.get("PROXY_TIMEOUT", "300"))  # synth+STA is fast

# Std-cell liberty file(s) per platform, for the fast synth+STA proxy.
# nangate45/sky130hd each ship one merged .lib; asap7 has no merged liberty —
# its cells are split across per-cell-type NLDM libs (4 gzipped + the plain
# SEQ lib) that must all be loaded. yosys' abc/stat and OpenSTA's read_liberty
# accept repeated -liberty flags (and read .gz natively), so we pass the whole
# list, mirroring ORFS' synth_preamble.tcl `foreach lib $LIB_FILES`.
# asap7 corner: ORFS' asap7/config.mk defaults CORNER ?= BC (best = FF) and
# PRIMARY_VT = RVT (tag R), so we use the RVT/FF set — matching what the full
# F3 flow synthesises with.
_LIBERTY: dict[str, list[str]] = {
    "nangate45": ["platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib"],
    "sky130hd":  ["platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"],
    "asap7": [
        "platforms/asap7/lib/NLDM/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz",
        "platforms/asap7/lib/NLDM/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz",
        "platforms/asap7/lib/NLDM/asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz",
        "platforms/asap7/lib/NLDM/asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz",
        "platforms/asap7/lib/NLDM/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib",
    ],
}
# yosys' `dfflibmap` accepts only ONE liberty (see ORFS synth.tcl: "dfflibmap
# only supports one liberty file"). For split-lib platforms the sequential
# cells live in the SEQ lib; single-lib platforms map DFFs from their one lib.
# Absent an entry, run_synth_sta falls back to the platform's first _LIBERTY.
_DFF_LIB: dict[str, str] = {
    "asap7": "platforms/asap7/lib/NLDM/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib",
}
# Tech + std-cell LEF (OpenROAD's link_design needs a technology, not just
# liberty).  Curated — deliberately excludes the SRAM/fakeram macro LEFs.
_LEF = {
    "nangate45": ["platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef",
                  "platforms/nangate45/lef/NangateOpenCellLibrary.macro.lef"],
    "sky130hd":  ["platforms/sky130hd/lef/sky130_fd_sc_hd.tlef",
                  "platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef"],
    # asap7: tech LEF + primary-VT (R) std-cell LEF, per asap7/config.mk
    # (TECH_LEF, SC_LEF with PRIMARY_VT_TAG=R).
    "asap7": ["platforms/asap7/lef/asap7_tech_1x_201209.lef",
              "platforms/asap7/lef/asap7sc7p5t_28_R_1x_220121a.lef"],
}
# Synth cell area → estimated post-P&R "Design area" inflation (CTS/repair
# buffers). Calibrated on nangate45 LANES=4: 19738 / 14589 ≈ 1.35.
_PLACE_INFLATION = 1.35

# V3: RTL content hash — computed once per process (lazy cache).
# Encodes the concatenated contents of the three RTL files (sorted by name)
# so any RTL edit produces a new 8-hex digest → old cached results dirs become
# naturally unreachable without explicit invalidation.
_rtl_hash_cache: str | None = None


def _rtl_hash() -> str:
    """Return an 8-hex-digit sha256 digest of the three RTL source files."""
    global _rtl_hash_cache
    if _rtl_hash_cache is None:
        h = hashlib.sha256()
        for fname in sorted(RTL_FILES):          # sorted for determinism
            p = RTL_DIR / fname
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(fname.encode())         # file missing: use name as placeholder
        _rtl_hash_cache = h.hexdigest()[:8]
    return _rtl_hash_cache


# Explicit result cache replacing lru_cache.
# V10: only cache results whose status is "ok" or "mock" — never cache TIMEOUT,
# FAIL, or PARSE_FAIL so a transient build failure doesn't poison future calls.
_physical_cache: dict[tuple, dict] = {}
_synth_sta_cache: dict[tuple, dict] = {}


# ── validate_config warning rate-limiter (audit F17/R8) ───────────────────────
# A single sagar campaign emitted the same PLACE_DENSITY/LB_ADDON exclusivity
# warning to stderr on ~1,400 episodes.  Print each DISTINCT warning once per
# process; count repeats and summarise the suppressed counts at process exit.
_seen_knob_warnings: dict[str, int] = {}
_knob_warn_atexit_registered = False


def _flush_knob_warning_counts() -> None:
    import sys as _sys
    for msg, n in _seen_knob_warnings.items():
        if n > 1:
            print(f"[physical_runner] knob warning (repeated {n}×, "
                  f"{n - 1} suppressed after the first): {msg}", file=_sys.stderr)


def _warn_knob_once(msg: str) -> None:
    """Print a validate_config warning the first time it's seen; suppress repeats
    (counted and summarised at exit)."""
    import sys as _sys
    global _knob_warn_atexit_registered
    n = _seen_knob_warnings.get(msg, 0)
    _seen_knob_warnings[msg] = n + 1
    if n == 0:
        print(f"[physical_runner] knob warning: {msg}", file=_sys.stderr)
        if not _knob_warn_atexit_registered:
            import atexit
            atexit.register(_flush_knob_warning_counts)
            _knob_warn_atexit_registered = True


# ── Subprocess helpers (process-group-safe) ───────────────────────────────────

def _killpg(proc: "subprocess.Popen") -> None:
    """SIGKILL the whole process group led by `proc` (best-effort).

    `proc` must have been started with start_new_session=True so it leads its
    own group; killing the group reaps tool grandchildren (yosys/openroad under
    the `bash -c` wrapper) that a plain proc.kill() would leave detached.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run_capture(cmd: "list[str]", cwd: str, timeout: int) -> subprocess.CompletedProcess:
    """Drop-in for `subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
    timeout=timeout)` that kills the entire process group on timeout (or any
    other communicate() failure).

    A plain subprocess.run(timeout=…) only kills the direct child — the `bash`
    wrapper — so a `yosys`/`openroad` grandchild keeps running detached and
    accumulates over a long unattended campaign (audit F12).  This reuses the
    exact Popen(start_new_session=True) + communicate(timeout) + os.killpg
    pattern run_physical was hardened with in the second audit.

    Return/exception semantics match subprocess.run: a CompletedProcess with
    .returncode/.stdout/.stderr, and subprocess.TimeoutExpired re-raised on
    timeout (so callers that let it propagate, or catch SubprocessError, are
    unaffected).
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        # TimeoutExpired, decode error, OSError, KeyboardInterrupt, …: the child
        # is in its own session, so nothing else will reap the tool grandchildren
        # unless we kill the group here before propagating.
        _killpg(proc)
        proc.wait()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _staged_src_subdir(design: "Any") -> str:
    """Content-addressed RTL staging subdir under src/ (audit F13).

    Staging into ``src/<design>_<rtlhash8>/`` (not the shared ``src/<design>/``)
    makes the staged source immutable per RTL-content hash.  Before this, two
    campaigns on the same design name but different RTL content (an edited working
    tree vs an older checkout sharing one EDA_RL_WORK) both wrote
    ``src/<design>/<file>``; campaign B's copy would poison campaign A's next
    build, which then synthesised B's RTL into a variant dir whose name claims
    A's hash — a permanently poisoned cache.  The variant name already embeds the
    same ``design.rtl_hash()``, so the staging dir and the results dir stay in
    lockstep.  (Legacy design=None tinymac path keeps the flat ``src/<DESIGN>/``.)
    """
    return f"{design.name}_{design.rtl_hash()}"


@contextmanager
def _variant_build_lock(var: str, gds: "Path"):
    """Exclusive per-variant build lock around the ORFS make invocation (F13).

    Yields True if we acquired the lock (the caller should build), or False if
    another process already holds it — meaning an identical config from another
    seed/campaign sharing this EDA_RL_WORK is building the same variant, whose
    ``config_<var>.mk`` / ``results/<plat>/<design>/<var>/`` paths we'd otherwise
    race on.  When the lock is contended we DON'T double-build: we poll (every
    2 s, up to ORFS_TIMEOUT) until either the result GDS appears (reuse it) or the
    other builder releases the lock (it finished or failed — then we take the lock
    and build).  Non-blocking throughout, so a stuck peer can't wedge us forever.
    """
    lock_dir = RUN_DIR / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = open(lock_dir / f"{var}.lock", "w")
    have_lock = False
    try:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            have_lock = True
        except OSError:
            # Held by another process building this exact variant — wait it out.
            deadline = time.time() + ORFS_TIMEOUT
            while time.time() < deadline:
                if gds.exists():
                    break                       # peer produced the GDS → reuse
                try:
                    fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    have_lock = True
                    break                       # peer released → we build
                except OSError:
                    time.sleep(2.0)
        yield have_lock
    finally:
        if have_lock:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def variant_name(lanes: int, acc_w: int, clk_ns: float,
                 util: int = 40, density: float = 0.60, abc: str | None = None,
                 abc_recipe: str | None = None,
                 design: "Any | None" = None,
                 knob_values: "dict | None" = None) -> str:
    """Compute a unique, deterministic variant identifier.

    abc_recipe wins over legacy abc kwarg (resolve_recipe handles both).
    The recipe suffix is appended before the RTL hash, in the slot formerly
    occupied by the raw '_area' suffix (V8 fix extended to all recipes):
        orfs_speed → no suffix      (default, matches pre-recipe naming)
        orfs_area  → "_area"
        plain      → "_plain"

    V12: when design is a DesignSpec:
      - The RTL hash comes from design.rtl_hash() rather than the hardcoded
        tinymac RTL_FILES.
      - For non-tinymac designs, the variant name is prefixed with the design
        name so result dirs stay in a separate namespace (existing tinymac dirs
        remain at the same path).

    V13: knob_values (tier-2/3 ORFS knobs) are now included in the variant name
      as a short SHA-256 hash suffix when non-empty.  Without this, two configs
      differing only in PLACE_DENSITY_LB_ADDON would map to the same variant dir
      and the second run would silently reuse the first run's GDS — a stale-cache
      bug that this codebase was already burned by (see CLAUDE.md Gotchas).
    """
    # V11: use {:.4g} so 1.25 and 1.2 produce distinct tokens ("1p25" vs "1p2");
    # the old :.1f caused variant_name(4,24,1.25)==variant_name(4,24,1.2).
    clk_str = f"{float(clk_ns):.4g}".replace(".", "p")
    # V14: only embed the LANES/ACC_W tokens for designs that actually have
    # those RTL params.  The legacy tinymac path (design is None or the tinymac
    # spec) keeps the exact old "L{lanes}_A{acc_w}" prefix so existing result
    # dirs/caches stay reachable; generic designs (gcd, aes) without mac_lanes/
    # accumulator_width get a clean "c{clk}" name instead of a misleading
    # L4_A24 leak.
    if design is None or getattr(design, "name", None) == "tinymac_accel":
        name = f"L{lanes}_A{acc_w}_c{clk_str}"
    else:
        params = getattr(design, "params", None) or {}
        pa_parts = []
        if "mac_lanes" in params:
            pa_parts.append(f"L{lanes}")
        if "accumulator_width" in params:
            pa_parts.append(f"A{acc_w}")
        pa_parts.append(f"c{clk_str}")
        name = "_".join(pa_parts)
    # Flow-param suffixes are appended ONLY when non-default, so configs that use
    # the original util=40/density=0.60 keep the exact old variant name and reuse
    # any GDS already built by run.sh / sweep.sh.
    if util != 40:
        name += f"_u{util}"
    if abs(float(density) - 0.60) > 1e-9:
        name += f"_d{f'{float(density):.2f}'.replace('.', 'p')}"
    # Stage-B recipe axis (replaces the raw V8 abc=='area' suffix check):
    # resolve_recipe() handles legacy abc='speed'/'area' and new abc_recipe= kwarg.
    resolved = resolve_recipe(abc_recipe=abc_recipe, abc=abc)
    name += recipe_suffix(resolved)

    # V13: knob_values hash — ensures tier-2/3 knob differences produce distinct
    # variant dirs, preventing stale-cache reuse across configs that differ only
    # in knob_values.  Only appended when knob_values is non-empty so all existing
    # (no-knob-values) variant names stay unchanged.
    if knob_values:
        import json as _json
        kv_str = _json.dumps({k: knob_values[k] for k in sorted(knob_values)},
                             sort_keys=True, separators=(",", ":"))
        kv_hash = hashlib.sha256(kv_str.encode()).hexdigest()[:6]
        name += f"_k{kv_hash}"

    # V3/V12: append RTL content hash.
    # For tinymac (design is None or design.name == "tinymac_accel"), use the
    # legacy _rtl_hash() so existing result dirs are still reachable.
    if design is None or design.name == "tinymac_accel":
        name += f"_r{_rtl_hash()}"
    else:
        # Non-tinymac: prefix the design name to avoid namespace collision.
        name = f"{design.name}_{name}_r{design.rtl_hash()}"

    return name


# ── Report parsing ────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _last_float_on_line(text: str, prefix: str) -> float | None:
    """Return the last numeric token of the first line starting with `prefix`.
    Matches OpenROAD lines like 'wns max -1.72' / 'tns max -25.86'."""
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", line)
            if nums:
                return float(nums[-1])
    return None


def _parse_metrics(work: Path, platform: str, variant: str, clk_ns: float,
                   design_name: str = DESIGN) -> dict:
    rpt  = work / "reports" / platform / design_name / variant / "6_finish.rpt"
    rlog = work / "logs"    / platform / design_name / variant / "6_report.log"
    rjson = work / "logs"   / platform / design_name / variant / "6_report.json"
    gds  = work / "results" / platform / design_name / variant / "6_final.gds"

    rpt_txt, rlog_txt = _read(rpt), _read(rlog)

    # V1: divide all time-valued report fields by the platform time unit so that
    # stored *_ns keys are always in nanoseconds regardless of the platform's
    # native SDC/report unit (asap7 uses picoseconds → divide by 1000).
    time_div = PLATFORM_TIME_UNIT.get(platform, 1.0)

    out: dict = {
        "area_um2": None, "util_pct": None, "wns_ns": None, "tns_ns": None,
        "setup_viol": None, "power_mw": None, "fmax_mhz": None,
        "period_min_ns": None, "timing_met": None,
        "cell_count": None, "ff_count": None,
        "gds": str(gds) if gds.exists() else None,
        "report": str(rpt) if rpt.exists() else None,
    }

    # Post-PnR cell counts from 6_report.json (richer than any regex): total
    # standard cells + sequential (flip-flop) instances.  Same keys fit_surrogate
    # mines.  Non-essential — never let a missing JSON fail the parse.
    if rjson.exists():
        try:
            import json as _json
            jd = _json.loads(rjson.read_text())
            sc = jd.get("finish__design__instance__count__stdcell")
            ff = jd.get("finish__design__instance__count__class:sequential_cell")
            if sc is not None:
                out["cell_count"] = int(sc)
            # F17: a purely combinational design (e.g. likith) has no sequential
            # cells, so 6_report.json omits this key — ff_count legitimately stays
            # None (its initialised value), no warning.  The AGENTS.md "F3 carries
            # a real FF count" invariant holds "when present".
            if ff is not None:
                out["ff_count"] = int(ff)
        except (OSError, ValueError, TypeError):
            pass

    # area + utilisation (from the report LOG): "Design area 19738 um^2 48% utilization."
    m = re.search(r"Design area\s+([\d.]+)\s+um\^2\s+([\d.]+)%", rlog_txt)
    if m:
        out["area_um2"] = float(m.group(1))
        out["util_pct"] = float(m.group(2))

    # timing (from the finish RPT): "wns max -1.72" / "tns max -25.86"
    # Divide raw values by time_div to normalise to nanoseconds.
    raw_wns = _last_float_on_line(rpt_txt, "wns")
    raw_tns = _last_float_on_line(rpt_txt, "tns")
    out["wns_ns"] = (raw_wns / time_div) if raw_wns is not None else None
    out["tns_ns"] = (raw_tns / time_div) if raw_tns is not None else None

    # Fmax + min period: "core_clock period_min = 3.72 fmax = 268.64"
    # period_min is in the platform's native unit; divide to get ns.
    # fmax is in MHz regardless of unit (it's derived as 1/period by ORFS),
    # but for asap7 reported fmax is 1000/period_ps = MHz directly, so no
    # conversion needed for fmax_mhz.
    m = re.search(r"period_min\s*=\s*([\d.]+).*?fmax\s*=\s*([\d.]+)", rpt_txt)
    if m:
        out["period_min_ns"] = float(m.group(1)) / time_div
        out["fmax_mhz"]      = float(m.group(2))

    m = re.search(r"setup violation count\s+(\d+)", rpt_txt)
    if m:
        out["setup_viol"] = int(m.group(1))

    # total power (report_power "Total" row, 5th column = Total Watts)
    for line in rpt_txt.splitlines():
        if line.strip().startswith("Total"):
            nums = re.findall(r"\d+\.?\d*(?:[eE][-+]?\d+)?", line)
            if len(nums) >= 4:                      # internal, switching, leakage, total[, %]
                out["power_mw"] = float(nums[3]) * 1000.0
            break

    # timing met: prefer the explicit violation count, else sign of WNS
    if out["setup_viol"] is not None:
        out["timing_met"] = (out["setup_viol"] == 0)
    elif out["wns_ns"] is not None:
        out["timing_met"] = (out["wns_ns"] >= 0.0)

    # V2: if we couldn't parse the essential metrics, flag as PARSE_FAIL so the
    # reward function never silently substitutes reference values for missing data.
    # Required: area_um2 must be present, AND at least one of fmax_mhz / wns_ns.
    if out["area_um2"] is None or (out["fmax_mhz"] is None and out["wns_ns"] is None):
        out["status"] = "PARSE_FAIL"
    else:
        out["status"] = "ok"

    return out


def _abort_risk_errors(knob_values: dict) -> list[str]:
    """Subset of knobs.validate_config() findings severe enough to abort a
    build before spending ORFS_TIMEOUT on a run the placer is known to reject.

    Only "ABORT RISK"/"ERROR" entries gate; plain "WARNING" entries are left
    to the existing print-only path in _config_mk (informational, not fatal).
    """
    try:
        from eda_rl.common.knobs import validate_config as _vc
    except ImportError:
        return []
    return [w for w in _vc(knob_values) if w.startswith(("ABORT RISK", "ERROR"))]


# ── ORFS config generation (mirrors sweep.sh) ─────────────────────────────────

def _config_mk(platform: str, variant: str, lanes: int, acc_w: int,
               util: int = 40, density: float = 0.60, abc: str | None = None,
               abc_recipe: str | None = None,
               design: "Any | None" = None,
               knob_values: "dict | None" = None,
               fastroute_tcl_rel: "str | None" = None) -> str:
    """Generate the per-variant ORFS config.mk content.

    When `design` is None, falls back to the legacy hardcoded tinymac_accel
    behaviour (preserves all existing call sites unchanged).

    When `design` is a DesignSpec:
      - DESIGN_NAME and VERILOG_FILES come from the design spec.
      - VERILOG_TOP_PARAMS is built from design.params ∩ config (lanes/acc_w
        are passed as top-level args for backward compat; the design bridge
        assembles the full string).
      - fastroute_tcl_rel: relative path (DESIGN_HOME-relative content already
        included) to a per-variant fastroute.tcl written by _stage_inputs when
        the GR_SEED knob is active; emits FASTROUTE_TCL. None (the GR_SEED
        knob absent/not sampled) omits the export entirely, so ORFS falls back
        to its own platform-default fastroute.tcl — today's behavior.
      - knob_values: additional ORFS knob k→v pairs emitted after the standard
        lines.  validate_config() is called first; any errors are printed as
        warnings but do not abort (let the flow fail naturally for hard conflicts).

    The abc recipe is written as ABC_AREA=1 for orfs_area (the only ORFS
    mechanism); orfs_speed uses the default (omit the variable).
    config_mk_lines() from recipe.py raises ValueError for 'plain' (not
    reachable in the full flow without patching ORFS — use the proxy instead).
    """
    resolved = resolve_recipe(abc_recipe=abc_recipe, abc=abc)
    # config_mk_lines raises ValueError for 'plain' — let it propagate so
    # run_physical callers see a clear error message.
    abc_lines = "\n".join(config_mk_lines(resolved))
    if abc_lines:
        abc_lines += "\n"

    if design is None:
        # ── Legacy tinymac-hardcoded path ────────────────────────────────────
        return (
            "export DESIGN_HOME = .\n"
            f"export DESIGN_NAME = {DESIGN}\n"
            f"export PLATFORM    = {platform}\n"
            "export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/tinymac_accel.v \\\n"
            "                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/int8_mac_array.v \\\n"
            "                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/requantize.v\n"
            f"export SDC_FILE      = $(DESIGN_HOME)/{platform}/$(DESIGN_NAME)/constraint_{variant}.sdc\n"
            f"export CORE_UTILIZATION      = {util}\n"
            f"export PLACE_DENSITY          = {density}\n"
            f"{abc_lines}"
            "export SYNTH_REPEATABLE_BUILD = 1\n"
            f"export VERILOG_TOP_PARAMS = LANES {lanes} ACC_W {acc_w}\n"
        )

    # ── Design-agnostic path ─────────────────────────────────────────────────
    design_name = design.name
    # Build VERILOG_FILES lines.  Use $(DESIGN_HOME)/src/<design>/<file> for
    # files that have already been copied into the work tree (see _stage_inputs).
    # For external absolute paths (e.g. gcd from ORFS), we must use the
    # absolute path directly since we copy to src/<design>/<filename>.
    rtl_fnames = [Path(f).name for f in design.rtl_files]
    # F13: point VERILOG_FILES at the content-addressed staging dir that
    # _stage_inputs copied into (src/<design>_<rtlhash8>/), never the shared
    # src/<design>/ that a concurrent differing-RTL campaign could overwrite.
    src_sub = _staged_src_subdir(design)
    if len(rtl_fnames) == 1:
        vfile_lines = (
            f"export VERILOG_FILES = $(DESIGN_HOME)/src/{src_sub}/{rtl_fnames[0]}\n"
        )
    else:
        first = f"$(DESIGN_HOME)/src/{src_sub}/{rtl_fnames[0]}"
        rest  = " \\\n".join(
            f"                       $(DESIGN_HOME)/src/{src_sub}/{fn}"
            for fn in rtl_fnames[1:]
        )
        vfile_lines = f"export VERILOG_FILES = {first} \\\n{rest}\n"

    # VERILOG_TOP_PARAMS: build from design.params + caller args.
    # Each param entry may carry a 'rtl_param_name' key giving the Verilog chparam
    # name (e.g. mac_lanes → LANES).  When absent, the param name itself is used.
    # For backward compat with callers that pass lanes/acc_w directly, we resolve
    # the legacy positional args via the canonical param names too.
    vtp_parts = []
    if design.params:
        # Build a lookup: canonical_param_name -> (rtl_name, value_from_caller)
        # Caller's positional lanes/acc_w feed the canonical names mac_lanes/accumulator_width
        # (and also the legacy names LANES/ACC_W for pre-rename designs).
        param_vals: dict[str, Any] = {}
        for pname, pspec in design.params.items():
            rtl_name = pspec.get("rtl_param_name", pname) if isinstance(pspec, dict) else pname
            # Resolve caller positional args for well-known RTL names
            if rtl_name == "LANES" or pname in ("LANES", "mac_lanes"):
                param_vals[pname] = lanes
            elif rtl_name == "ACC_W" or pname in ("ACC_W", "accumulator_width"):
                param_vals[pname] = acc_w
        # knob_values may carry overrides for any param (by canonical name)
        if knob_values:
            for pk in design.params:
                if pk in knob_values:
                    param_vals[pk] = knob_values[pk]
        # Emit VERILOG_TOP_PARAMS using the RTL param names (not canonical names)
        for pname, pspec in design.params.items():
            if pname in param_vals:
                rtl_name = pspec.get("rtl_param_name", pname) if isinstance(pspec, dict) else pname
                vtp_parts += [rtl_name, str(param_vals[pname])]
    vtp_line = ""
    if vtp_parts:
        vtp_line = f"export VERILOG_TOP_PARAMS = {' '.join(vtp_parts)}\n"

    # knob_values: validate, then emit
    extra_lines = ""
    if knob_values:
        try:
            from eda_rl.common.knobs import validate_config as _vc, KnobRegistry
            warnings = _vc(knob_values)
            for w in warnings:
                _warn_knob_once(w)   # F17/R8: rate-limit repeated warnings
            reg = KnobRegistry.load()
            # Only emit knobs that are not already emitted above (util/density/abc/vtp).
            # CLOCK_UNCERTAINTY/IO_DELAY are SDC-owned (written by _stage_inputs via
            # design.sdc_text()); GR_SEED is fastroute.tcl-owned (written by
            # _stage_inputs directly) — none belong in config.mk.
            _SKIP_EMIT = {"CORE_UTILIZATION", "PLACE_DENSITY", "VERILOG_TOP_PARAMS",
                          "ABC_AREA", "CLOCK_PERIOD", "CLOCK_UNCERTAINTY", "IO_DELAY",
                          "GR_SEED"}
            for kname, kval in knob_values.items():
                if kname in _SKIP_EMIT:
                    continue
                knob = reg.get(kname)
                if knob is not None:
                    lines = knob.emit_lines(kval)
                    extra_lines += "\n".join(lines) + ("\n" if lines else "")
                # Unknown knob names are silently dropped (future-proofing)
        except ImportError:
            pass  # knobs.py not yet available; skip knob emission

    fastroute_line = (
        f"export FASTROUTE_TCL = $(DESIGN_HOME)/{fastroute_tcl_rel}\n"
        if fastroute_tcl_rel is not None else ""
    )

    return (
        "export DESIGN_HOME = .\n"
        # ORFS uses DESIGN_NAME as the Verilog top module (synth.tcl:
        # `hierarchy -check -top $::env(DESIGN_NAME)`), and DESIGN_NICKNAME for
        # the results/logs directory name.  These differ whenever a design's
        # top module name != its registry name (e.g. aes → aes_cipher_top), so
        # we must emit both: nickname keeps results where _parse_metrics looks
        # (results/<platform>/<design.name>/), DESIGN_NAME points yosys at the
        # real top.  For name == top designs (gcd/tinymac) this is a no-op.
        f"export DESIGN_NICKNAME = {design_name}\n"
        f"export DESIGN_NAME = {design.top}\n"
        f"export PLATFORM    = {platform}\n"
        f"{vfile_lines}"
        f"export SDC_FILE      = $(DESIGN_HOME)/{platform}/{design_name}/constraint_{variant}.sdc\n"
        f"{fastroute_line}"
        f"export CORE_UTILIZATION      = {util}\n"
        f"export PLACE_DENSITY          = {density}\n"
        f"{abc_lines}"
        "export SYNTH_REPEATABLE_BUILD = 1\n"
        f"{vtp_line}"
        f"{extra_lines}"
    )


def _stage_inputs(platform: str, variant: str, lanes: int, acc_w: int, clk_ns: float,
                  util: int = 40, density: float = 0.60, abc: str | None = None,
                  abc_recipe: str | None = None,
                  design: "Any | None" = None,
                  knob_values: "dict | None" = None) -> Path:
    """Copy RTL into the work tree and write the per-variant config.mk + SDC.
    Returns the path to the generated config.mk.

    When design is None, uses the legacy tinymac-hardcoded behaviour.
    When design is a DesignSpec, copies design.rtl_files to the work tree and
    uses the DesignSpec to generate the SDC and config.mk.
    """
    if design is None:
        # ── Legacy path ──────────────────────────────────────────────────────
        cfgdir = MAKE_DIR / platform / DESIGN
        srcdir = MAKE_DIR / "src" / DESIGN
        srcdir.mkdir(parents=True, exist_ok=True)
        cfgdir.mkdir(parents=True, exist_ok=True)

        for f in RTL_FILES:
            shutil.copy(RTL_DIR / f, srcdir / f)

        base_sdc = cfgdir / "constraint.sdc"
        if not base_sdc.exists():
            raise FileNotFoundError(
                f"base SDC missing: {base_sdc} "
                f"(expected from physical/orfs/make/{platform}/{DESIGN}/)"
            )
        # V1: multiply the optimizer's ns clock by the platform time unit factor so
        # the SDC gets the value in the platform's native unit (ps for asap7).
        time_unit = PLATFORM_TIME_UNIT.get(platform, 1.0)
        sdc_clk_value = clk_ns * time_unit
        gen_sdc = cfgdir / f"constraint_{variant}.sdc"
        gen_sdc.write_text(
            re.sub(r"(?m)^set clk_period.*$",
                   f"set clk_period    {sdc_clk_value}",
                   base_sdc.read_text())
        )

        gen_cfg = cfgdir / f"config_{variant}.mk"
        gen_cfg.write_text(_config_mk(platform, variant, lanes, acc_w,
                                       util, density, abc, abc_recipe))
        return gen_cfg

    # ── Design-agnostic path ─────────────────────────────────────────────────
    design_name = design.name
    cfgdir = MAKE_DIR / platform / design_name
    # F13: stage RTL into a content-addressed dir (src/<design>_<rtlhash8>/) so
    # concurrent campaigns on the same design name but different RTL can't
    # overwrite each other's source.  _config_mk points VERILOG_FILES here.
    srcdir = MAKE_DIR / "src" / _staged_src_subdir(design)
    srcdir.mkdir(parents=True, exist_ok=True)
    cfgdir.mkdir(parents=True, exist_ok=True)

    # Copy RTL files from design spec into src/<design_name>/
    for src_path in design.rtl_files:
        fname = Path(src_path).name
        try:
            shutil.copy(src_path, srcdir / fname)
        except OSError as exc:
            raise FileNotFoundError(
                f"RTL file not found for design '{design_name}': {src_path}"
            ) from exc

    # V1: multiply ns clock by platform time unit factor
    time_unit = PLATFORM_TIME_UNIT.get(platform, 1.0)
    sdc_clk_value = clk_ns * time_unit

    # CLOCK_UNCERTAINTY/IO_DELAY (SDC-owned pseudo-knobs): convert ns -> native
    # unit the same way clk_ns is, above. Absent from knob_values (e.g. tier <2,
    # or a design that doesn't opt in) -> None -> sdc_text() reproduces the
    # original SDC unchanged (no uncertainty line; clk_period*0.2 io delay).
    uncertainty_native = None
    io_delay_native = None
    if knob_values:
        if knob_values.get("CLOCK_UNCERTAINTY") is not None:
            uncertainty_native = float(knob_values["CLOCK_UNCERTAINTY"]) * time_unit
        if knob_values.get("IO_DELAY") is not None:
            io_delay_native = float(knob_values["IO_DELAY"]) * time_unit

    # Generate SDC from DesignSpec template
    gen_sdc = cfgdir / f"constraint_{variant}.sdc"
    gen_sdc.write_text(design.sdc_text(platform, sdc_clk_value,
                                        uncertainty=uncertainty_native,
                                        io_delay=io_delay_native))

    # GR_SEED (fastroute.tcl-owned pseudo-knob): absent -> no custom fastroute.tcl
    # is written, ORFS uses its own platform-default file (original behavior).
    #
    # audit F2: setting FASTROUTE_TCL makes ORFS *replace* its entire else-branch
    # (floorplan.tcl:104-111), which is the ONLY place the sampled
    # ROUTING_LAYER_ADJUSTMENT reaches the global router.  Copying the platform's
    # fastroute.tcl verbatim (its previous behavior) reintroduces that file's
    # hardcoded adjustment literal (asap7 0.25, sky130hd 0.2), silently pinning
    # ROUTING_LAYER_ADJUSTMENT to a constant every episode that sampled GR_SEED.
    # Instead we reproduce the platform file but replace the hardcoded literal in
    # set_global_routing_layer_adjustment with $::env(ROUTING_LAYER_ADJUSTMENT)
    # (ORFS always defines it -- default 0.5 in scripts/variables.yaml -- and the
    # sampled value is emitted to config.mk), preserving the platform's -clock/
    # -signal set_routing_layers lines.  Then append the seed line.
    fastroute_tcl_rel = None
    gr_seed = knob_values.get("GR_SEED") if knob_values else None
    if gr_seed is not None:
        base_fr = ORFS_DIR / "flow" / "platforms" / platform / "fastroute.tcl"
        if base_fr.exists():
            base_text = re.sub(
                r"(set_global_routing_layer_adjustment\s+\S+)\s+[\d.]+",
                r"\1 $::env(ROUTING_LAYER_ADJUSTMENT)",
                base_fr.read_text(),
            ).rstrip("\n")
        else:
            # No platform fastroute.tcl: reproduce ORFS's else-branch directly.
            base_text = (
                "set_global_routing_layer_adjustment "
                "$::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER) "
                "$::env(ROUTING_LAYER_ADJUSTMENT)\n"
                "set_routing_layers -signal "
                "$::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)"
            )
        gen_fr = cfgdir / f"fastroute_{variant}.tcl"
        gen_fr.write_text(
            base_text + "\n"
            f"set_global_routing_random -seed {int(gr_seed)}\n"
        )
        fastroute_tcl_rel = f"{platform}/{design_name}/fastroute_{variant}.tcl"

    gen_cfg = cfgdir / f"config_{variant}.mk"
    gen_cfg.write_text(_config_mk(platform, variant, lanes, acc_w,
                                   util, density, abc, abc_recipe,
                                   design=design, knob_values=knob_values,
                                   fastroute_tcl_rel=fastroute_tcl_rel))
    return gen_cfg


# ── Public entry point ────────────────────────────────────────────────────────

def run_physical(lanes: int, acc_w: int, clk_ns: float, platform: str = "nangate45",
                 util: int = 40, density: float = 0.60,
                 abc: str | None = None, abc_recipe: str | None = None,
                 design: "Any | None" = None,
                 knob_values: "dict | None" = None) -> dict:
    """Run the full RTL→GDS flow for one config and return parsed metrics.

    Deterministic for a fixed RTL + PDK + flow params, so cached in an explicit
    dict.  V10: only successful ("ok") results are cached; TIMEOUT/FAIL/PARSE_FAIL
    results are NOT cached so a transient failure can be retried.  Reuses an
    already-built variant (skips the flow if its 6_final.gds exists).

    abc_recipe (Stage-B): canonical recipe name — {orfs_speed, orfs_area, plain}.
    abc (legacy): accepts 'speed'/'area'; mapped via resolve_recipe().
    abc_recipe wins when both are given.
    'plain' is proxy-only and raises ValueError here (cannot be reproduced
    through ORFS without patching synth_preamble.tcl).

    util / density / abc_recipe are ORFS flow knobs (CORE_UTILIZATION,
    PLACE_DENSITY, ABC_AREA); at their defaults (40, 0.60, orfs_speed) the
    variant name is identical to run.sh/sweep.sh so previously-built GDSs
    are reused.

    Returns a dict with: lanes, acc_w, clk_ns, platform, variant, status,
    area_um2, util_pct, wns_ns, tns_ns, setup_viol, power_mw, fmax_mhz,
    period_min_ns, timing_met, gds, report.
    """
    resolved_recipe = resolve_recipe(abc_recipe=abc_recipe, abc=abc)
    if resolved_recipe == "plain":
        raise ValueError(
            "run_physical does not support abc_recipe='plain'. "
            "The 'plain' recipe (bare abc -liberty) cannot be reproduced in the "
            "ORFS full flow without patching synth_preamble.tcl. "
            "Use run_synth_sta(..., abc_recipe='plain') for the F2 proxy instead."
        )

    # V12: design default is tinymac_accel; None is allowed for backward compat
    # (callers that don't pass design get identical behaviour to before V12).
    eff_design = design  # may be None; _stage_inputs / variant_name handle None

    # Include design name and knob_values in cache key to avoid cross-design collisions.
    design_name_key = getattr(eff_design, "name", "tinymac_accel")
    kv_key: tuple = tuple(sorted((knob_values or {}).items()))
    cache_key = (design_name_key, lanes, acc_w, float(clk_ns), platform,
                 util, float(density), resolved_recipe, kv_key)
    if cache_key in _physical_cache:
        return _physical_cache[cache_key]

    var = variant_name(lanes, acc_w, clk_ns, util, density,
                       abc_recipe=resolved_recipe, design=eff_design,
                       knob_values=knob_values)  # V13: knob_values in dir name
    actual_design_name = design_name_key
    base = {"lanes": lanes, "acc_w": acc_w, "clk_ns": float(clk_ns),
            "platform": platform, "variant": var, "design": actual_design_name,
            "core_utilization": util, "place_density": density,
            "abc": abc, "abc_recipe": resolved_recipe}

    if os.environ.get("PHYSICAL_MOCK"):
        result = {**base, "status": "mock", **_mock_metrics(lanes, acc_w, clk_ns)}
        _physical_cache[cache_key] = result
        return result

    env_sh = ORFS_DIR / "env.sh"
    if not env_sh.exists():
        raise FileNotFoundError(
            f"ORFS not found at {ORFS_DIR} (set ORFS_DIR, or PHYSICAL_MOCK=1 to test offline)"
        )

    gen_cfg = _stage_inputs(platform, var, lanes, acc_w, clk_ns, util, density,
                            abc_recipe=resolved_recipe,
                            design=eff_design, knob_values=knob_values)
    gds = RUN_DIR / "results" / platform / actual_design_name / var / "6_final.gds"

    flow_status = "ok"
    if not gds.exists():
        # Known-doomed placer configs (e.g. CORE_UTILIZATION>60 + large
        # CELL_PAD) abort deep into the flow after burning most of
        # ORFS_TIMEOUT; catch them up front instead (validate_config's
        # ABORT RISK/ERROR findings — audit finding #5).
        merged_knobs = {"CORE_UTILIZATION": util, "PLACE_DENSITY": density,
                        "clock_period_ns": clk_ns, **(knob_values or {})}
        abort_errs = _abort_risk_errors(merged_knobs)
        if abort_errs:
            import sys as _sys
            for e in abort_errs:
                print(f"[physical_runner] config_abort: {e}", file=_sys.stderr)
            log_path = RUN_DIR / f"opt_{var}.log"
            log_path.write_text("[config_abort]\n" + "\n".join(abort_errs) + "\n")
            return {**base, "status": "config_abort",
                    "area_um2": None, "util_pct": None, "wns_ns": None,
                    "tns_ns": None, "setup_viol": None, "power_mw": None,
                    "fmax_mhz": None, "period_min_ns": None,
                    "timing_met": None, "gds": None, "report": str(log_path)}

        make_cmd = (
            f"source '{env_sh}' && "
            f"make --file='{ORFS_DIR}/flow/Makefile' "
            f"FLOW_HOME='{ORFS_DIR}/flow' WORK_HOME='{RUN_DIR}' "
            f"DESIGN_CONFIG='{gen_cfg}' FLOW_VARIANT='{var}'"
        )
        # F13: hold an exclusive per-variant lock across the make invocation so
        # two identical configs (e.g. two seeds sharing this EDA_RL_WORK) don't
        # race on config_<var>.mk / results/<var>/.  When the lock is contended
        # _variant_build_lock polls for the peer's GDS instead of double-building;
        # have_lock=False + a re-checked existing GDS means "reuse the peer's".
        with _variant_build_lock(var, gds) as have_lock:
            if have_lock and not gds.exists():
                # V10: use Popen + communicate(timeout) + start_new_session so we can
                # kill the entire process group on TimeoutExpired (not just the bash wrapper).
                proc = subprocess.Popen(
                    ["bash", "-c", make_cmd], cwd=str(MAKE_DIR),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = proc.communicate(timeout=ORFS_TIMEOUT)
                except subprocess.TimeoutExpired:
                    # Kill the entire process group (yosys/openroad children included).
                    _killpg(proc)
                    proc.wait()
                    log_path = RUN_DIR / f"opt_{var}.log"
                    log_path.write_text(f"[TIMEOUT after {ORFS_TIMEOUT}s]\n")
                    # V10: return TIMEOUT status (never raise); do NOT cache so the config
                    # can be retried and the agent can update on it.  (The lock is
                    # released by the context manager on return.)
                    return {**base, "status": "TIMEOUT",
                            "area_um2": None, "util_pct": None, "wns_ns": None,
                            "tns_ns": None, "setup_viol": None, "power_mw": None,
                            "fmax_mhz": None, "period_min_ns": None,
                            "timing_met": None, "gds": None, "report": str(log_path)}
                except BaseException:
                    # Any other failure during communicate() (decode error, OSError,
                    # KeyboardInterrupt, ...) — start_new_session=True detaches the
                    # child from our process group, so nothing else will reap the
                    # yosys/openroad grandchildren unless we kill them here too.
                    _killpg(proc)
                    proc.wait()
                    raise

                # Persist the flow output so a failure is debuggable.
                (RUN_DIR / f"opt_{var}.log").write_text(
                    (stdout or "") + "\n--- stderr ---\n" + (stderr or "")
                )
                if proc.returncode != 0:
                    flow_status = "FAIL"
            # else: a concurrent peer built (or is building) this variant and its
            # GDS now exists — fall through and parse the shared results.

    metrics = _parse_metrics(RUN_DIR, platform, var, clk_ns,
                             design_name=actual_design_name)
    # V2: _parse_metrics now sets status="PARSE_FAIL" if essential metrics are
    # missing; honour that over the flow_status, and also catch the case where
    # the report file is absent.
    parse_status = metrics.pop("status", "ok")
    if flow_status == "ok" and metrics.get("report") is None:
        flow_status = "FAIL"
    final_status = parse_status if parse_status != "ok" else flow_status

    result = {**base, "status": final_status, **metrics}

    # audit F1 — "fixed ruler": re-time the final netlist under a reference SDC
    # (io 0.2 fraction, no uncertainty) so the reward metrics can't be gamed by
    # the sampled IO_DELAY / CLOCK_UNCERTAINTY.  Only on a successful build.
    #   - design given  → run the reference STA; on failure leave the ref keys
    #     absent so the caller (FunnelEnv._terminal_reward) warns and falls back
    #     to the sampled metrics (never discards a successful build).
    #   - design is None (legacy tinymac) → its sampled SDC already IS the
    #     reference ruler (io 0.2, no uncertainty), so mirror the sampled metrics
    #     into the ref keys — no extra STA, no warning.
    if final_status == "ok":
        ref = _reference_sta(platform, actual_design_name, var, clk_ns, eff_design)
        if ref is None and eff_design is None:
            ref = {"wns_ref_ns": metrics.get("wns_ns"),
                   "fmax_ref_mhz": metrics.get("fmax_mhz"),
                   "period_ref_ns": metrics.get("period_min_ns"),
                   "comb_delay_ns": None, "combinational_ref": False}
        if ref is not None:
            result.update(ref)

    # V10: only cache successful results.
    if final_status == "ok":
        _physical_cache[cache_key] = result
    return result


# ── Gate 2: RTL elaboration check (Yosys hierarchy, ~1-2 s) ────────────────────

_elaborate_cache: dict[tuple, dict] = {}


def run_elaborate(lanes: int, acc_w: int, platform: str = "nangate45") -> dict:
    """Cheap structural gate: does the parameterised RTL elaborate cleanly?

    Reads the Verilog, applies the LANES/ACC_W chparams (exactly as ORFS will),
    builds the hierarchy and runs Yosys `check`. Catches parameter combinations
    that don't elaborate BEFORE paying for synthesis/STA/P&R. Seconds, not minutes.

    Returns {"ok": bool, "stage": "elaborate", "log": <path or note>}.
    """
    cache_key = (lanes, acc_w, platform)
    if cache_key in _elaborate_cache:
        return _elaborate_cache[cache_key]

    if os.environ.get("PHYSICAL_MOCK"):
        # The current RTL elaborates for any positive LANES/ACC_W; model that.
        ok = lanes >= 1 and acc_w >= 1
        result = {"ok": ok, "stage": "elaborate", "log": "mock"}
        _elaborate_cache[cache_key] = result
        return result

    env_sh = ORFS_DIR / "env.sh"
    if not env_sh.exists():
        raise FileNotFoundError(
            f"ORFS not found at {ORFS_DIR} (set ORFS_DIR, or PHYSICAL_MOCK=1 to test offline)"
        )

    rtl = " ".join(str(RTL_DIR / f) for f in RTL_FILES)
    work = RUN_DIR / "elaborate" / variant_name(lanes, acc_w, 1.0)
    work.mkdir(parents=True, exist_ok=True)
    ys = work / "elaborate.ys"
    ys.write_text("\n".join([
        f"read_verilog -sv {rtl}",
        f"chparam -set LANES {lanes} {DESIGN}",
        f"chparam -set ACC_W {acc_w} {DESIGN}",
        f"hierarchy -check -top {DESIGN}",
        "proc",
        "check -assert",
        "",
    ]))
    p = _run_capture(
        ["bash", "-c", f"source '{env_sh}' && yosys '{ys}'"],
        cwd=str(MAKE_DIR), timeout=PROXY_TIMEOUT,
    )
    (work / "elaborate.log").write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
    result = {"ok": p.returncode == 0, "stage": "elaborate", "log": str(work / "elaborate.log")}
    # Always cache elaborate results (elaboration is deterministic for given RTL hash)
    _elaborate_cache[cache_key] = result
    return result


# ── Fast proxy: Yosys synthesis + OpenROAD STA (no place & route) ──────────────

def _yosys_synth_script(lanes: int, acc_w: int, libs: "list[Path]", netlist: Path,
                        abc_recipe: str = "orfs_speed", platform: str = "nangate45",
                        clk_ns: float | None = None, work_dir: Path | None = None,
                        design: "Any | None" = None,
                        dff_lib: "Path | None" = None) -> str:
    """Return a Yosys synthesis script string.

    Stage-B recipe extension: the abc command is now recipe-aware.
      plain      → bare `abc -liberty <lib>`   (current/legacy proxy behaviour)
      orfs_speed → `abc -script abc_speed.script -liberty ... -constr ... [-D ...]`
      orfs_area  → `abc -script abc_area.script -liberty ... -constr ... [-D ...]`

    For orfs_speed/area, a abc.constr file is written to work_dir (or a temp
    location) so that -constr resolves to an actual file.  This replicates what
    ORFS's synth_preamble.tcl writes to $OBJECTS_DIR/abc.constr (BUF_X1, 3.898 fF
    for nangate45), ensuring F2 proxy and F3 full flow see identical cell
    constraints.

    Written to a .ys file (not passed via -p): yosys's command tokenizer does
    NOT strip shell quotes, so paths are left UNQUOTED — safe here as no path
    in this flow contains spaces.  Same chparam mechanism ORFS uses.

    V12: when design is a DesignSpec, uses design.rtl_files and design.top;
    also emits chparam for each axis in design.params using the caller's
    lanes/acc_w as the LANES/ACC_W values respectively.
    """
    resolved = resolve_recipe(abc_recipe)

    # Repeated `-liberty <path>` args for the multi-command yosys flow. abc and
    # stat accept many; dfflibmap accepts exactly one (the DFF/SEQ lib).
    lib_flags = " ".join(f"-liberty {one}" for one in libs)
    dff = dff_lib if dff_lib is not None else libs[0]

    if design is None:
        # Legacy tinymac path
        rtl = " ".join(str(RTL_DIR / f) for f in RTL_FILES)
        top = DESIGN
        chparam_lines = [
            f"chparam -set LANES {lanes} {top}",
            f"chparam -set ACC_W {acc_w} {top}",
        ]
    else:
        rtl = " ".join(design.rtl_files)
        top = design.top
        # Emit chparam for each param the design has; resolve RTL name via rtl_param_name.
        # Each pspec may carry 'rtl_param_name'; if absent the param key is the RTL name.
        chparam_lines = []
        for pname, pspec in (design.params or {}).items():
            rtl_name = pspec.get("rtl_param_name", pname) if isinstance(pspec, dict) else pname
            if rtl_name == "LANES" or pname in ("LANES", "mac_lanes"):
                chparam_lines.append(f"chparam -set LANES {lanes} {top}")
            elif rtl_name == "ACC_W" or pname in ("ACC_W", "accumulator_width"):
                chparam_lines.append(f"chparam -set ACC_W {acc_w} {top}")

    if resolved == "plain":
        abc_cmd = f"abc {lib_flags}"
    else:
        # Write the abc.constr file so -constr has a real target.
        constr_dir = work_dir if work_dir is not None else dff.parent
        constr_path = write_abc_constr(constr_dir, platform)
        abc_cmd = yosys_abc_line(resolved, platform, clk_ns, list(libs),
                                 constr_path=constr_path)

    return "\n".join([
        f"read_verilog -sv {rtl}",
        *chparam_lines,
        f"synth -top {top} -flatten",
        f"dfflibmap -liberty {dff}",
        abc_cmd,
        "opt_clean -purge",
        f"stat {lib_flags}",
        f"write_verilog -noattr {netlist}",
        "",
    ])


def _sta_script(lefs: list[Path], libs: "list[Path]", netlist: Path, sdc: Path,
                top: str = DESIGN) -> str:
    # Pre-layout STA in OpenROAD: cell delays only (optimistic, no net RC), but
    # fast and — unlike the routed flow — target-clock-independent, so Fmax is a
    # fair cross-config speed metric.  OpenROAD's link_design needs a technology,
    # so read the tech + std-cell LEF first.  report_clock_min_period prints the
    # same "period_min = X fmax = Y" line the full-flow parser already handles.
    lef_lines = "".join(f"read_lef {lef}\n" for lef in lefs)
    lib_lines = "".join(f"read_liberty {one}\n" for one in libs)
    return (
        lef_lines
        + lib_lines
        + f"read_verilog {netlist}\n"
        + f"link_design {top}\n"
        + f"read_sdc {sdc}\n"
        + "report_clock_min_period\n"
        + "report_wns\n"
        + "report_tns\n"
    )


# ── Tool-output parsers (extracted for golden-log regression tests, R5) ────────
#
# These are the exact parsing rules run_synth_sta applies to real yosys/OpenSTA
# output. They live as standalone functions so tests/test_parsers.py can feed
# captured real-tool fixtures through them — the two silent parser deaths the
# third audit found (F3: yosys stat format change; F4: 'fmax = inf') were both
# invisible to mock-based self-tests, which fabricate exactly these fields.

def _parse_synth_stat(stdout: str) -> tuple[int | None, float | None]:
    """Parse (cell_count, cell_area_um2) from a yosys synth log.

    Takes the LAST stat (after opt_clean), not intermediate synth/abc stats.
    The installed yosys `stat` prints a tabular total line, one of:
        36  231.472 cells     (with per-cell liberty area)
        36            cells    (no liberty / pre-map stat)
    older yosys printed "Number of cells:  36".  Match both formats so the
    cell count is not silently None on the current toolchain (audit F3).
    Each `stat` block prints exactly one such total for its (flattened) top
    module; the LAST match is the post-opt_clean top-module total, never a
    sub-block (per-cell-type lines like "1  0.204  AND3x4..." don't end in
    "cells" so never match).
    """
    cell_matches = re.findall(
        r"Number of cells:\s+(\d+)|^\s*(\d+)\s+(?:[\d.]+\s+)?cells\s*$",
        stdout, re.MULTILINE)
    # each match is a 2-tuple (old-format group, new-format group); take the
    # non-empty one from the last match.
    cell_count = None
    if cell_matches:
        last = cell_matches[-1]
        cell_count = int(last[0] or last[1])
    chip_all  = re.findall(r"Chip area for module.*?:\s+([\d.]+)", stdout)
    cell_area = float(chip_all[-1]) if chip_all else None
    return cell_count, cell_area


def _parse_sta_timing(sta_out: str, time_div: float, clk_ns: float) -> dict:
    """Parse pre-layout STA output into timing metrics (native unit → ns).

    audit F4: report_clock_min_period prints "period_min = 0.00 fmax = inf" for
    a design with no reg-to-reg (or reg-to-clock) path — every combinational
    block, and any I/O-dominated one.  The old regex required both groups to be
    [\\d.]+, so "inf" failed the match entirely and the slack fallback below
    silently substituted fmax = 1000/(clk-wns), i.e. a pure function of the
    sampled clock carrying ZERO design information.  Parse inf/0.00 explicitly
    and record fmax_mhz = None + combinational=True instead, so downstream
    consumers (and F1's reference STA) treat it as "no clocked-path speed",
    not a plausible-looking constraint echo.

    The slack fallback fires ONLY when report_clock_min_period gave no usable
    period, the design is not combinational, AND wns is a genuine violation
    figure (wns < 0); it flags fmax_inferred so consumers can tell measured
    from inferred.
    """
    fmax = period_min = None
    combinational = False
    fmax_inferred = False
    m = re.search(r"period_min\s*=\s*([\d.]+|inf).*?fmax\s*=\s*([\d.]+|inf)", sta_out)
    if m:
        pmin_s, fmax_s = m.group(1), m.group(2)
        if fmax_s == "inf" or pmin_s == "inf" or float(pmin_s) == 0.0:
            combinational = True          # no clocked path → no min-period metric
        else:
            period_min = float(pmin_s) / time_div
            fmax = float(fmax_s)
    wns_raw = _last_float_on_line(sta_out, "wns")
    tns_raw = _last_float_on_line(sta_out, "tns")
    wns = (wns_raw / time_div) if wns_raw is not None else None
    tns = (tns_raw / time_div) if tns_raw is not None else None
    if fmax is None and not combinational and wns is not None and wns < 0.0 \
            and clk_ns - wns > 0:
        period_min = round(clk_ns - wns, 3)
        fmax = round(1000.0 / period_min, 2)
        fmax_inferred = True

    timing_met = (wns >= 0.0) if wns is not None else None
    return {
        "fmax_mhz": fmax, "period_min_ns": period_min,
        "wns_ns": wns, "tns_ns": tns,
        "combinational": combinational, "fmax_inferred": fmax_inferred,
        "timing_met": timing_met,
    }


def run_synth_sta(lanes: int, acc_w: int, clk_ns: float, platform: str = "nangate45",
                  abc: str | None = None, abc_recipe: str | None = None,
                  design: "Any | None" = None,
                  knob_values: "dict | None" = None) -> dict:
    """Fast proxy: synthesise (Yosys) + static-timing (OpenROAD), no P&R.

    Stage-B recipe extension:
      abc_recipe (new): canonical recipe {orfs_speed, orfs_area, plain}.
      abc (legacy):  'speed'/'area' aliases; abc_recipe wins when both given.
      Default: orfs_speed (ORFS default — same script as the full flow uses).

    V12: design (DesignSpec | None): when None, uses tinymac_accel behaviour.
    When provided, uses the design's RTL files, top module, and clock port.
    knob_values: ORFS knob dict — validated and emitted into config.mk.

    Also detects macros from synth stat when design.has_macros is None:
    sets result["has_macros_detected"] = True/False.

    For orfs_speed/area the proxy uses the same ORFS abc_{speed,area}.script
    with the same -constr file as the full flow (F2 and F3 share recipes),
    so the proxy's cell area distribution matches the full-flow's, improving
    the ρ correlation.  'plain' keeps the legacy bare `abc -liberty` behaviour
    (useful for calibration; not reachable in F3).

    Seconds instead of minutes.  Returns the SAME metric shape as run_physical
    (area_um2, fmax_mhz, wns_ns, tns_ns, timing_met, …) so the reward/env are
    unchanged — area is the synth cell area scaled by the placement-inflation
    factor to approximate die area; timing is pre-layout (optimistic).
    """
    resolved_recipe = resolve_recipe(abc_recipe=abc_recipe, abc=abc)

    eff_design = design
    design_name_key = getattr(eff_design, "name", "tinymac_accel")
    kv_key: tuple = tuple(sorted((knob_values or {}).items()))
    cache_key = (design_name_key, lanes, acc_w, float(clk_ns), platform,
                 resolved_recipe, kv_key)
    if cache_key in _synth_sta_cache:
        return _synth_sta_cache[cache_key]

    var = variant_name(lanes, acc_w, clk_ns, abc_recipe=resolved_recipe,
                       design=eff_design,
                       knob_values=knob_values)  # V13: knob_values in dir name
    base = {"lanes": lanes, "acc_w": acc_w, "clk_ns": float(clk_ns),
            "platform": platform, "variant": var, "design": design_name_key,
            "abc": abc, "abc_recipe": resolved_recipe}

    if os.environ.get("PHYSICAL_MOCK"):
        result = {**base, "status": "mock-proxy", **_mock_metrics(lanes, acc_w, clk_ns)}
        _synth_sta_cache[cache_key] = result
        return result

    env_sh = ORFS_DIR / "env.sh"
    if not env_sh.exists():
        raise FileNotFoundError(
            f"ORFS not found at {ORFS_DIR} (set ORFS_DIR, or PHYSICAL_MOCK=1 to test offline)"
        )
    lib_rels = _LIBERTY.get(platform)
    if not lib_rels or platform not in _LEF:
        raise ValueError(f"no proxy lib/lef for platform '{platform}' "
                         f"(try nangate45 / sky130hd / asap7)")
    libs = [ORFS_DIR / "flow" / rel for rel in lib_rels]
    dff_rel = _DFF_LIB.get(platform)
    dff_lib = (ORFS_DIR / "flow" / dff_rel) if dff_rel else libs[0]
    lefs = [ORFS_DIR / "flow" / rel for rel in _LEF[platform]]

    gen_cfg = _stage_inputs(platform, var, lanes, acc_w, clk_ns,
                            design=eff_design, knob_values=knob_values)
    sdc = gen_cfg.parent / f"constraint_{var}.sdc"

    work = RUN_DIR / "proxy" / var
    work.mkdir(parents=True, exist_ok=True)
    netlist = work / "netlist.v"
    synth_ys = work / "synth.ys"
    sta_tcl = work / "sta.tcl"

    # Determine the top module for the STA script
    top_module = getattr(eff_design, "top", DESIGN) if eff_design is not None else DESIGN

    # Pass recipe, platform, clk_ns, design, and work_dir so the constr file is written
    # inside the proxy work directory (same isolation as the full flow).
    synth_ys.write_text(_yosys_synth_script(lanes, acc_w, libs, netlist,
                                            abc_recipe=resolved_recipe,
                                            platform=platform, clk_ns=clk_ns,
                                            work_dir=work, design=eff_design,
                                            dff_lib=dff_lib))
    sta_tcl.write_text(_sta_script(lefs, libs, netlist, sdc, top=top_module))

    # 1) synthesis (script FILE, not -p, to avoid shell-quoting issues; no -q,
    #    which would suppress the `stat` output we parse).
    p1 = _run_capture(
        ["bash", "-c", f"source '{env_sh}' && yosys '{synth_ys}'"],
        cwd=str(MAKE_DIR), timeout=PROXY_TIMEOUT,
    )
    (work / "synth.log").write_text((p1.stdout or "") + "\n--- stderr ---\n" + (p1.stderr or ""))
    if p1.returncode != 0 or not netlist.exists():
        return {**base, "status": "FAIL", "stage": "synth",
                "area_um2": None, "fmax_mhz": None, "wns_ns": None, "tns_ns": None,
                "timing_met": None, "power_mw": None}

    cell_count, cell_area = _parse_synth_stat(p1.stdout)
    est_area   = round(cell_area * _PLACE_INFLATION, 1) if cell_area else None

    # Macro auto-detect: check for "$" instances in synth stat (ORFS macro pattern).
    # When design.has_macros is None, record detected value in the result.
    # Research note: Yosys stat output includes "Number of cells: N" for std cells;
    # macro instances appear as cells with "/" in their name or in hierarchy lines.
    # More reliably: check for "DFFRAM" or "fakeram" or "$" cell type prefixes.
    has_macros_detected: bool | None = None
    if eff_design is not None and eff_design.has_macros is None:
        synth_stdout = p1.stdout or ""
        # Macro cells in Yosys stat are listed as "   <name>    N" under "Chip area..."
        # AutoTuner research: macro presence detected by any cell with underscore-
        # prefixed type (e.g. DFFRAM, fakeram, user_* macros). Simple heuristic:
        # check if any line in stat matches a non-standard-cell pattern.
        # We use a conservative check: any "$techmap" or named cell not in a std-cell
        # lib (ie non-DFF/BUF/AND etc) implies macros. Simpler: check for "SRAM" or
        # "RAM" or "fakeram" in the output.
        macro_keywords = ("sram", "ram", "dffram", "fakeram", "macro")
        has_macros_detected = any(
            kw in synth_stdout.lower() for kw in macro_keywords
        )

    # 2) static timing (OpenROAD, pre-layout)
    p2 = _run_capture(
        ["bash", "-c", f"source '{env_sh}' && openroad -no_init -exit '{sta_tcl}'"],
        cwd=str(MAKE_DIR), timeout=PROXY_TIMEOUT,
    )
    sta_out = (p2.stdout or "")
    (work / "sta.log").write_text(sta_out + "\n--- stderr ---\n" + (p2.stderr or ""))

    # V1: parse STA timing in the platform's native unit (ns for nangate45, ps for asap7).
    time_div = PLATFORM_TIME_UNIT.get(platform, 1.0)
    timing = _parse_sta_timing(sta_out, time_div, clk_ns)

    result = {
        **base, "status": "ok", "stage": "proxy",
        "cells": cell_count,
        "cell_area_um2": cell_area,
        "area_um2": est_area,            # die-area estimate (cell × inflation)
        "util_pct": None,
        "wns_ns": timing["wns_ns"], "tns_ns": timing["tns_ns"],
        "fmax_mhz": timing["fmax_mhz"], "period_min_ns": timing["period_min_ns"],
        "combinational": timing["combinational"],   # True → no reg-to-reg path (fmax is None)
        "fmax_inferred": timing["fmax_inferred"],   # True → fmax from slack fallback, not measured
        "setup_viol": None, "power_mw": None,
        "timing_met": timing["timing_met"],
        "gds": None, "report": str(work / "sta.log"),
    }
    # Include macro auto-detect result if applicable
    if has_macros_detected is not None:
        result["has_macros_detected"] = has_macros_detected

    # V10: only cache successful proxy results
    if result["status"] == "ok":
        _synth_sta_cache[cache_key] = result
    return result


# ── Reward "fixed ruler": reference-SDC re-time of the final netlist (F1) ──────

def _reference_sta(platform: str, design_name: str, variant: str, clk_ns: float,
                   design: "Any") -> dict | None:
    """Re-time the routed netlist under a REFERENCE SDC to get gaming-proof
    reward metrics (audit F1).

    The campaign samples IO_DELAY / CLOCK_UNCERTAINTY, which rewrite the SDC the
    build's own wns/fmax are measured against — so "loosen the constraints" reads
    as "faster chip" and the reported optimum is not a better chip.  To break
    that, we re-time the final netlist (results/<plat>/<design>/<variant>/
    6_final.v) with a *reference* SDC: the exact SDC the design gets with NO
    constraint knobs (io delay = 0.2 x clock period, no uncertainty), at the same
    sampled clock.  These reference metrics feed the reward; the sampled-SDC
    metrics stay in the obs for flow visibility.

    Speed metric:
      - Sequential design → report_clock_min_period gives a real fmax/period;
        use them directly (fmax_ref_mhz, period_ref_ns).
      - Purely combinational design → report_clock_min_period returns fmax = inf
        (no reg-to-reg path), so instead we take the max in->out path from
        report_checks -path_delay max and record the pure logic delay
        comb_delay_ns = (data arrival time) - (input external delay).  Subtracting
        the reference io delay leaves the netlist's intrinsic critical-path delay,
        independent of both the sampled io_delay AND the clock; fmax_ref_mhz is
        its reciprocal (1000 / comb_delay_ns).  This is a real measurement, never
        the 1000/clk constraint echo.

    Returns {wns_ref_ns, fmax_ref_mhz, period_ref_ns, comb_delay_ns,
    combinational_ref} or None if the netlist/tools/STA are unavailable (caller
    then falls back to the sampled-SDC metrics — a successful build is never
    discarded).
    """
    if design is None:
        return None   # legacy tinymac SDC already IS the reference (io 0.2, no unc.)
    netlist = RUN_DIR / "results" / platform / design_name / variant / "6_final.v"
    lib_rels = _LIBERTY.get(platform)
    env_sh = ORFS_DIR / "env.sh"
    if not netlist.exists() or not lib_rels or platform not in _LEF or not env_sh.exists():
        return None

    time_unit = PLATFORM_TIME_UNIT.get(platform, 1.0)
    libs = [ORFS_DIR / "flow" / rel for rel in lib_rels]
    lefs = [ORFS_DIR / "flow" / rel for rel in _LEF[platform]]
    top = design.top

    work = RUN_DIR / "refsta" / variant
    work.mkdir(parents=True, exist_ok=True)
    # Reference SDC: no uncertainty, default 0.2 io fraction — the fixed ruler.
    ref_sdc = work / "ref.sdc"
    ref_sdc.write_text(design.sdc_text(platform, clk_ns * time_unit))
    sta_tcl = work / "ref_sta.tcl"
    sta_tcl.write_text(
        _sta_script(lefs, libs, netlist, ref_sdc, top=top)
        + "report_checks -path_delay max\n"
    )

    try:
        # audit F12: process-group-safe capture so a hung reference-STA openroad
        # doesn't leak its group; TimeoutExpired (a SubprocessError) is caught here.
        p = _run_capture(
            ["bash", "-c", f"source '{env_sh}' && openroad -no_init -exit '{sta_tcl}'"],
            cwd=str(MAKE_DIR), timeout=PROXY_TIMEOUT,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    out = p.stdout or ""
    (work / "ref_sta.log").write_text(out + "\n--- stderr ---\n" + (p.stderr or ""))

    wns_raw = _last_float_on_line(out, "wns")
    wns_ref = (wns_raw / time_unit) if wns_raw is not None else None

    fmax_ref = period_ref = comb_delay_ns = None
    combinational_ref = False
    m = re.search(r"period_min\s*=\s*([\d.]+|inf).*?fmax\s*=\s*([\d.]+|inf)", out)
    if m and m.group(1) != "inf" and m.group(2) != "inf" and float(m.group(1)) != 0.0:
        period_ref = float(m.group(1)) / time_unit
        fmax_ref = float(m.group(2))
    else:
        # Combinational: derive the honest speed metric from the critical in->out
        # path.  data arrival time - input external delay = pure logic delay.
        combinational_ref = True
        m_arr = re.search(r"([\d.]+)\s+data arrival time", out)
        m_in = re.search(r"([\d.]+)\s+[v^]\s+input external delay", out)
        if m_arr:
            arrival = float(m_arr.group(1))
            input_ext = float(m_in.group(1)) if m_in else 0.0
            comb_native = max(arrival - input_ext, 1e-9)
            comb_delay_ns = comb_native / time_unit
            period_ref = comb_delay_ns
            if comb_delay_ns > 0:
                fmax_ref = round(1000.0 / comb_delay_ns, 2)

    return {"wns_ref_ns": wns_ref, "fmax_ref_mhz": fmax_ref,
            "period_ref_ns": period_ref, "comb_delay_ns": comb_delay_ns,
            "combinational_ref": combinational_ref}


# ── Mock mode (no OpenROAD) ───────────────────────────────────────────────────

def _mock_metrics(lanes: int, acc_w: int, clk_ns: float) -> dict:
    """Plausible synthetic metrics for offline testing of the optimizer loop.
    Anchored to the real nangate45 LANES=4 numbers; area grows sub-linearly with
    lanes (per the measured synth sweep), Fmax ~constant (requantize-limited)."""
    # synth-sweep-shaped cell area, inflated to ~P&R scale (×1.35)
    cell_area = {1: 12331, 2: 13120, 4: 14589, 8: 17354, 16: 22949}.get(lanes, 14589)
    area = round(cell_area * 1.353, 1)                      # → ~19738 at lanes=4
    fmax = 269.0                                            # requantize-limited, lane-independent
    period_min = round(1000.0 / fmax, 2)
    met = clk_ns >= period_min
    wns = round(clk_ns - 3.82, 3)                           # crit path ≈ 3.82 ns
    power = round(900.0 + 30.0 * lanes, 1)                  # rough lane scaling, mW
    cell_count = round(cell_area / 1.4)                     # ~µm²/cell at nangate45
    ff_count = round(cell_count * 0.15)                     # plausible sequential share
    return {
        "area_um2": area, "util_pct": 47.0,
        "wns_ns": wns, "tns_ns": round(min(wns, 0.0) * 15, 2),
        "setup_viol": 0 if met else 40,
        "power_mw": power, "fmax_mhz": fmax, "period_min_ns": period_min,
        "timing_met": met,
        # audit F1: fabricate the fixed-ruler reference metrics so mock-mode
        # self-tests exercise the same reward path as real runs.  Mock is
        # TinyMAC-shaped (sequential), so the reference equals the sampled values.
        "wns_ref_ns": wns, "fmax_ref_mhz": fmax, "period_ref_ns": period_min,
        "comb_delay_ns": None, "combinational_ref": False,
        # cell_count: consumed by run_physical F3 path; cells: the key the
        # synth proxy (run_synth_sta) / FunnelEnv._run_f2 read at F2.
        "cell_count": cell_count, "ff_count": ff_count, "cells": cell_count,
        "gds": None, "report": "mock",
    }


if __name__ == "__main__":
    import json
    import sys
    # Usage: physical_runner.py [--proxy] LANES ACC_W CLK_NS [PLATFORM]
    #   --proxy → fast synth+STA; otherwise full RTL→GDS.
    argv = sys.argv[1:]
    proxy = "--proxy" in argv
    argv = [a for a in argv if a != "--proxy"]
    L = int(argv[0]) if len(argv) > 0 else 4
    A = int(argv[1]) if len(argv) > 1 else 24
    C = float(argv[2]) if len(argv) > 2 else 5.0
    P = argv[3] if len(argv) > 3 else "nangate45"
    fn = run_synth_sta if proxy else run_physical
    print(json.dumps(fn(L, A, C, P), indent=2))
