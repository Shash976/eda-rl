#!/usr/bin/env python3
"""collect_best.py — harvest the best configs from a campaign.

Reads a campaign log, ranks the F3 (full RTL→GDS) results, and for the best
configurations:
  * copies each one's 6_final.gds (and its ORFS report) into an output folder,
  * writes a machine-readable manifest (best_configs.json),
  * renders a self-contained before/after comparison page (best_configs.html),
    optionally with KLayout-rendered layout thumbnails.

Design-agnostic: works for any design that reached F3 — the metrics and the GDS
path are read straight from each episode's logged ``obs`` (area_um2, fmax_mhz,
power_mw, timing_met, gds), so no variant names are re-derived.

Baseline comparison: by default collect also builds a **stock-default** version
of the design — every knob pinned to its registry/YAML default (the "original,
unoptimized" configuration) — runs it through the full F3 flow, and renders it
as a BASELINE card. Every best-config card then shows its %-delta vs that
baseline for area / Fmax / power. The baseline build needs ORFS (or
PHYSICAL_MOCK=1 for a metrics-only baseline with no GDS); if it can't be built
the page is produced without it. Disable with --no-baseline.

    eda-rl collect                                  # latest campaign, best picks + baseline
    eda-rl collect --design gcd --platform nangate45  # pick a campaign by design/platform
    eda-rl collect --campaign all --top 5 --open
    eda-rl collect --out /tmp/best --render         # render layout PNGs (needs klayout)
    eda-rl collect --no-baseline                    # skip the stock-default build
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import webbrowser
from pathlib import Path

from eda_rl.viz.campaign_data import load_campaign_rows, episode_value, resolve_log_path


# ── metric accessors ──────────────────────────────────────────────────────────

def _obs(r: dict) -> dict:
    return r.get("obs") or {}

def _area(r: dict):  return _obs(r).get("area_um2")
def _fmax(r: dict):  return _obs(r).get("fmax_mhz")
def _power(r: dict): return _obs(r).get("power_mw")
def _cells(r: dict): return _obs(r).get("cell_count")
def _ffs(r: dict):   return _obs(r).get("ff_count")
def _gds(r: dict):   return _obs(r).get("gds")
def _timing(r: dict): return _obs(r).get("timing_met")


def _is_buildable(r: dict) -> bool:
    """An F3 result we can actually harvest: ok status with area + a GDS path."""
    return (
        r.get("fidelity") == "F3"
        and _obs(r).get("status") in ("ok", "mock")
        and _area(r) is not None
        and _fmax(r) is not None
    )


# Mock F3 metrics are synthetic (TinyMAC-shaped, design-agnostic); a real ("ok")
# build measures the actual chip.  The two live on different rulers, so a %-delta
# between a mock and a real number is meaningless — collect refuses to draw it.
_MOCK_STATUS = frozenset({"mock", "mock-proxy"})

def _ruler(r: dict) -> str:
    """'mock' if this F3 row's metrics are synthetic, else 'real'."""
    return "mock" if _obs(r).get("status") in _MOCK_STATUS else "real"


def _variant_of(r: dict) -> str:
    """Identify a build: the GDS's parent dir is the ORFS FLOW_VARIANT; fall back
    to a compact config string."""
    g = _gds(r)
    if g:
        return Path(g).parent.name
    cfg = r.get("config") or {}
    return "_".join(f"{k}{cfg[k]}" for k in sorted(cfg))[:48] or "config"


def _cfg_label(r: dict) -> str:
    cfg = r.get("config") or {}
    if "mac_lanes" in cfg:
        return f"L{cfg.get('mac_lanes')}_A{cfg.get('accumulator_width')}"
    return _variant_of(r)


# ── selection ─────────────────────────────────────────────────────────────────

def select_best(rows: list[dict], top: int = 3) -> list[dict]:
    """Return an ordered, de-duplicated list of standout F3 results.

    Each returned row is annotated with a ``_badge`` and ``_sublabel``. Picks:
    best overall score, max Fmax, min area, min power (if logged), and the top-N
    by score — then de-duplicates by build variant, keeping the first (highest
    priority) badge.
    """
    f3 = [r for r in rows if _is_buildable(r)]
    if not f3:
        return []

    picks: list[tuple[str, str, dict]] = []

    best_score = max(f3, key=lambda r: (episode_value(r) if episode_value(r) is not None else float("-inf")))
    picks.append(("BEST OVERALL", "highest optimizer score", best_score))
    picks.append(("MAX FMAX", "fastest clock", max(f3, key=lambda r: _fmax(r))))
    picks.append(("MIN AREA", "smallest die", min(f3, key=lambda r: _area(r))))
    if any(_power(r) is not None for r in f3):
        picks.append(("MIN POWER", "lowest total power",
                      min((r for r in f3 if _power(r) is not None), key=lambda r: _power(r))))

    ranked = sorted(f3, key=lambda r: (episode_value(r) if episode_value(r) is not None else float("-inf")),
                    reverse=True)
    for i, r in enumerate(ranked[:top]):
        picks.append((f"TOP-{i+1}", "by optimizer score", r))

    # De-duplicate by variant, keeping the first (highest-priority) badge.
    seen: set[str] = set()
    out: list[dict] = []
    for badge, sub, r in picks:
        v = _variant_of(r)
        if v in seen:
            continue
        seen.add(v)
        rr = dict(r)
        rr["_badge"], rr["_sublabel"], rr["_variant"] = badge, sub, v
        out.append(rr)
    return out


# ── stock-default baseline build ──────────────────────────────────────────────

def _default_config(space: dict) -> dict:
    """Construct the 'original, unoptimized' config: every axis at its default.

    Mirrors run_funnel_optimizer._surrogate_covers' probe config — default →
    first choice → range low — so the result is a valid point in the design's
    own space (including YAML overrides like likith's PDN-safe CORE_UTILIZATION
    default). Int axes are rounded so the sampler/log/emission agree.
    """
    cfg: dict = {}
    for axis, spec in space.items():
        if "default" in spec:
            val = spec["default"]
        elif spec.get("choices"):
            val = spec["choices"][0]
        elif spec.get("range"):
            val = spec["range"][0]
        else:
            continue
        if spec.get("type") == "int" and isinstance(val, (int, float)):
            val = int(round(float(val)))
        cfg[axis] = val
    return cfg


def build_baseline(design: str | None, platform: str, max_tier: int,
                   results_path: Path) -> dict | None:
    """Build the stock-default design and return it as a collect-style pick row.

    Runs one real F3 (RTL→GDS) build of the all-defaults config via FunnelEnv's
    ``commit`` action and reads back ``env.terminal_obs`` — the same obs shape
    (area_um2/fmax_mhz/power_mw/gds/…) the campaign log carries, so it flows
    through the existing copy/render/manifest path unchanged.

    Returns None (and prints why) if the funnel is unavailable, the knob space
    can't be built, or the F3 build produced no usable metrics — the caller then
    renders the page without a baseline. Needs ORFS unless PHYSICAL_MOCK=1 (which
    yields a metrics-only baseline with no GDS).
    """
    try:
        from eda_rl.funnel.env import FunnelEnv
        from eda_rl.funnel.run_funnel_optimizer import _build_space
    except Exception as exc:  # noqa: BLE001
        print(f"  [baseline] funnel env unavailable ({exc}); omitting baseline.", file=sys.stderr)
        return None

    try:
        space = _build_space(design, platform, max_tier)
    except Exception as exc:  # noqa: BLE001
        print(f"  [baseline] could not build knob space ({exc}); omitting baseline.", file=sys.stderr)
        return None

    cfg = _default_config(space)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [baseline] building stock-default config (max_tier={max_tier}) — a full F3 flow, this can take minutes…")
    try:
        env = FunnelEnv(
            platform=platform, design=design, max_tier=max_tier,
            active_space=space, budget_s=1e9, results_path=results_path,
        )
        env.reset(cfg)
        env.step("commit")          # jump straight to F3 (terminal)
        obs = env.terminal_obs
    except Exception as exc:  # noqa: BLE001
        print(f"  [baseline] F3 build failed ({exc}); omitting baseline.", file=sys.stderr)
        return None

    if (obs.get("status") not in ("ok", "mock", "mock-proxy")
            or obs.get("area_um2") is None or obs.get("fmax_mhz") is None):
        print(f"  [baseline] build produced no usable metrics "
              f"(status={obs.get('status')!r}); omitting baseline.", file=sys.stderr)
        return None

    row = {"config": cfg, "obs": obs, "fidelity": "F3", "status": obs.get("status")}
    row["_badge"], row["_sublabel"], row["_variant"] = (
        "BASELINE", "stock default (all knobs at default)", _variant_of(row))
    return row


# ── GDS rendering (optional) ──────────────────────────────────────────────────

def _render_gds(gds: Path, platform: str, out_png: Path, size: int = 1400) -> bool:
    import os
    import subprocess
    import shutil as _sh
    if not _sh.which("klayout") or not gds.exists():
        return False
    orfs = Path(os.environ.get("ORFS_DIR", "/opt/OpenROAD-flow-scripts"))
    lyp = orfs / "flow" / "platforms" / platform / "KLayout" / f"{platform}.lyp"
    rb = (
        "view = RBA::LayoutView.new\n"
        "view.load_layer_props($lyp) if $lyp && File.exist?($lyp)\n"
        "view.load_layout($gds)\n"
        "view.max_hier\nview.zoom_fit\n"
        "view.save_image($out, Integer($w), Integer($h))\n"
    )
    script = out_png.parent / "_render.rb"
    script.write_text(rb)
    subprocess.run(
        ["klayout", "-z", "-rd", f"gds={gds}", "-rd", f"lyp={lyp}",
         "-rd", f"out={out_png}", "-rd", f"w={size}", "-rd", f"h={size}",
         "-r", str(script)],
        capture_output=True, text=True,
    )
    script.unlink(missing_ok=True)
    return out_png.exists()


# ── comparison page ───────────────────────────────────────────────────────────

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0}
header{background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:26px 32px 18px;border-bottom:1px solid #1e293b}
header h1{font-size:23px;font-weight:800;letter-spacing:-.5px;color:#f1f5f9}
header .sub{font-size:13px;color:#64748b;margin-top:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;padding:20px 32px 32px}
.card{background:#1e293b;border-radius:12px;overflow:hidden;border:2px solid #2563eb;display:flex;flex-direction:column}
.card.best{border-color:#059669}
.card.base{border-color:#d97706}
.chead{padding:14px 16px 12px;background:#0f172a}
.badge{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.08em;color:#fff;background:#1e3a8a;padding:3px 8px;border-radius:4px;margin-bottom:8px}
.card.best .badge{background:#065f46}
.card.base .badge{background:#b45309}
.dlt{font-size:10px;font-weight:700;margin-left:6px}
.dlt.good{color:#34d399}
.dlt.bad{color:#f87171}
.dlt.neu{color:#64748b}
.title{font-size:16px;font-weight:700;color:#f8fafc}
.csub{font-size:11px;color:#64748b;margin-top:2px}
.limg{padding:10px 12px 4px}
.limg img{width:100%;height:auto;border-radius:6px;border:1px solid #334155;image-rendering:pixelated;display:block}
.noimg{width:100%;aspect-ratio:1;background:#1e293b;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:12px;border:1px dashed #334155}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}
td{padding:4px 12px;color:#cbd5e1}
.k{color:#64748b;width:46%}
.v{font-family:'SF Mono',monospace;font-size:11.5px;color:#e2e8f0}
tr:hover td{background:rgba(255,255,255,.03)}
.foot{padding:14px 32px;border-top:1px solid #1e293b;color:#475569;font-size:12px}
.warn{margin:16px 32px 0;padding:11px 15px;background:#3b2410;border:1px solid #b45309;border-radius:8px;color:#fbbf24;font-size:12.5px;line-height:1.5}
"""


def _delta_span(v, base, lower_is_better: bool) -> str:
    """Return a `<span>` with the %-delta of v vs the baseline, coloured
    good/bad by whether the change is an improvement. Empty string when either
    value is missing or the baseline is zero."""
    if v is None or base is None or base == 0:
        return ""
    pct = (v - base) / abs(base) * 100.0
    if abs(pct) < 0.05:
        return " <span class='dlt neu'>±0.0%</span>"
    improved = (pct < 0) if lower_is_better else (pct > 0)
    cls = "good" if improved else "bad"
    return f" <span class='dlt {cls}'>{pct:+.1f}%</span>"


def _card(r: dict, img_b64: str | None, baseline: dict | None = None) -> str:
    cfg = r.get("config") or {}
    is_base = r["_badge"] == "BASELINE"
    best = r["_badge"] in ("BEST OVERALL", "TOP-1")
    cls = "base" if is_base else ("best" if best else "")
    # Deltas are shown on every non-baseline card once a baseline exists.
    bo = (baseline or {}) if not is_base else {}
    img = (f'<img src="data:image/png;base64,{img_b64}" alt="layout">' if img_b64
           else '<div class="noimg">layout not rendered<br>(run with --render + klayout)</div>')
    title = "stock default" if is_base else _cfg_label(r)
    rows = [("Config", _cfg_label(r)), ("Variant", r["_variant"])]
    # show a few salient config knobs
    for k in ("mac_lanes", "accumulator_width", "clock_period_ns", "abc_recipe"):
        if k in cfg:
            v = cfg[k]
            rows.append((k, f"{v:.3f}" if isinstance(v, float) else str(v)))
    rows += [
        ("Area", f"{_area(r):,.0f} µm²" + _delta_span(_area(r), bo.get("area_um2"), True)),
        ("Fmax", f"{_fmax(r):,.0f} MHz" + _delta_span(_fmax(r), bo.get("fmax_mhz"), False)),
    ]
    if _cells(r) is not None:
        rows.append(("Cells", f"{_cells(r):,.0f}" + _delta_span(_cells(r), bo.get("cell_count"), True)))
    if _ffs(r) is not None:
        rows.append(("FFs", f"{_ffs(r):,.0f}"))
    if _power(r) is not None:
        rows.append(("Power", f"{_power(r):.1f} mW" + _delta_span(_power(r), bo.get("power_mw"), True)))
    rows.append(("Timing", "✅ met" if _timing(r) else "❌ not met"))
    sc = episode_value(r)
    if sc is not None:
        rows.append(("Score", f"{sc:.3f}"))
    body = "".join(f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>" for k, v in rows)
    return (
        f'<div class="card{" " + cls if cls else ""}">'
        f'<div class="chead"><span class="badge">{r["_badge"]}</span>'
        f'<div class="title">{title}</div><div class="csub">{r["_sublabel"]}</div></div>'
        f'<div class="limg">{img}</div><table><tbody>{body}</tbody></table></div>'
    )


def build_page(picks: list[dict], design: str, platform: str, imgs: dict[str, str | None],
               baseline: dict | None = None, deltas: bool = True,
               ruler_warning: str | None = None) -> str:
    # Deltas are only drawn when the baseline and the campaign were measured on
    # the same ruler (both real, or both mock); `deltas=False` keeps the baseline
    # card but drops the misleading %-annotations.
    baseline_obs = _obs(baseline) if (baseline and deltas) else None
    cards = "".join(_card(r, imgs.get(r["_variant"]), baseline_obs) for r in picks)
    sub = (f'{len(picks)} standout designs harvested from the funnel optimizer · '
           f'GDS + reports collected alongside this page')
    if baseline is not None:
        sub += (' · deltas vs the stock-default baseline' if deltas
                else ' · baseline shown for reference (deltas suppressed)')
    banner = (f'<div class="warn">⚠ {ruler_warning}</div>' if ruler_warning else '')
    foot = ("Each card's 6_final.gds and ORFS report were copied into this folder. "
            "See best_configs.json for the full manifest.")
    if baseline is not None and deltas:
        foot = ("The BASELINE card is the stock-default design (all knobs at default); "
                "every other card's %-deltas are measured against it. ") + foot
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{design} — best optimized configs</title><style>{_CSS}</style></head><body>'
        f'<header><h1>{design} · {platform} · best optimized configurations</h1>'
        f'<p class="sub">{sub}</p></header>'
        f'{banner}'
        f'<div class="grid">{cards}</div>'
        f'<div class="foot">{foot}</div></body></html>'
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect the best configs from a campaign: GDS + report + comparison page")
    ap.add_argument("--design", default=None,
                    help="design name, e.g. 'sagar' — resolves the campaign log for you "
                         "(pair with --platform; preferred over --log)")
    ap.add_argument("--platform", default=None,
                    help="platform name, e.g. 'sky130hd' (pair with --design)")
    ap.add_argument("--log", default=None,
                    help="campaign JSONL path (overrides --design/--platform; "
                         "default: most-recently-modified log under eda_rl/campaigns/)")
    ap.add_argument("--campaign", default="latest", help="campaign_id | 'latest' | 'all'")
    ap.add_argument("--out", default=None, help="output directory (default: best_configs/<design>_<platform>)")
    ap.add_argument("--top", type=int, default=3, help="how many top-by-score configs to include (default 3)")
    ap.add_argument("--render", action="store_true", help="render layout PNGs with KLayout (needs klayout + ORFS_DIR)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip building the stock-default baseline (no F3 baseline build / delta comparison)")
    ap.add_argument("--max-tier", type=int, default=None,
                    help="knob tier for the baseline's default config (default: read from the campaign log)")
    ap.add_argument("--open", action="store_true", help="open the comparison page in a browser")
    args = ap.parse_args()
    args.log = str(resolve_log_path(args.log, args.design, args.platform))

    log_path = resolve_log_path(args.log, args.design, args.platform)
    rows = load_campaign_rows(log_path, args.campaign)
    if not rows:
        print(f"No episodes found in {log_path} for campaign={args.campaign!r}", file=sys.stderr)
        sys.exit(1)

    picks = select_best(rows, top=args.top)
    if not picks:
        print("No buildable F3 results in this campaign (need an F3 run with area/Fmax/GDS logged).",
              file=sys.stderr)
        sys.exit(1)

    platform = log_path.parent.name
    design = log_path.parent.parent.name
    out_dir = Path(args.out) if args.out else Path.cwd() / "best_configs" / f"{design}_{platform}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── stock-default baseline (built here, prepended so it flows through the
    # same copy/render/manifest path and lands as the first card) ──────────────
    baseline: dict | None = None
    if not args.no_baseline:
        # The campaign log records the design identifier and knob tier it ran
        # under; prefer them (design.name from the path may be the yosys top,
        # not the DesignSpec.load key the campaign used).
        meta = next((r for r in rows if r.get("max_tier") is not None), {})
        design_id = meta.get("design") or design
        max_tier = args.max_tier if args.max_tier is not None else int(meta.get("max_tier", 1))
        baseline = build_baseline(
            design_id, platform, max_tier,
            out_dir / "_baseline_build" / "funnel_baseline.jsonl",
        )

    render_picks = ([baseline] if baseline is not None else []) + picks

    manifest = []
    imgs: dict[str, str | None] = {}
    print(f"Collecting {len(render_picks)} configs → {out_dir}")
    for r in render_picks:
        v = r["_variant"]
        cdir = out_dir / f"{r['_badge'].replace(' ', '_')}__{v}"
        cdir.mkdir(parents=True, exist_ok=True)

        # copy the GDS + report if they still exist on disk
        gds_src = Path(_gds(r)) if _gds(r) else None
        gds_dst = None
        if gds_src and gds_src.exists():
            gds_dst = cdir / gds_src.name
            shutil.copy(gds_src, gds_dst)
        rpt_src = _obs(r).get("report")
        if rpt_src and Path(rpt_src).exists():
            shutil.copy(rpt_src, cdir / Path(rpt_src).name)

        # optional layout render
        if args.render and gds_src and gds_src.exists():
            png = cdir / "layout.png"
            if _render_gds(gds_src, platform, png):
                imgs[v] = base64.b64encode(png.read_bytes()).decode()
        imgs.setdefault(v, None)

        entry = {
            "badge": r["_badge"], "variant": v, "config": r.get("config"),
            "area_um2": _area(r), "fmax_mhz": _fmax(r), "power_mw": _power(r),
            "timing_met": _timing(r), "score": episode_value(r),
            "gds_source": str(gds_src) if gds_src else None,
            "gds_collected": str(gds_dst) if gds_dst else None,
        }
        manifest.append(entry)
        status = "gds copied" if gds_dst else "GDS missing on disk (work dir cleared?)"
        print(f"  [{r['_badge']:<12}] {_cfg_label(r):<14} "
              f"area={_area(r):>8,.0f}µm²  fmax={_fmax(r):>6,.0f}MHz  → {status}")

    # Deltas are only meaningful when the baseline and the campaign's best
    # configs were measured the same way.  The baseline is always freshly built
    # here (real ORFS, or mock under PHYSICAL_MOCK=1); the picks come from the
    # log and may be the other ruler (e.g. a mock-generated campaign vs a real
    # baseline).  Mixing them yields nonsense %-deltas — the exact "measure the
    # chip, not the ruler" trap — so detect the mismatch and suppress deltas.
    deltas = True
    ruler_warning: str | None = None
    if baseline is not None:
        base_ruler = _ruler(baseline)
        pick_rulers = {_ruler(p) for p in picks}
        if pick_rulers != {base_ruler}:
            deltas = False
            ruler_warning = (
                f"Deltas suppressed: the baseline was built on the "
                f"'{base_ruler}' ruler but the campaign's best configs are "
                f"'{'/'.join(sorted(pick_rulers))}'. A real-vs-mock (or "
                f"cross-ruler) %-delta is meaningless. Re-run the campaign with "
                f"real ORFS — or collect with PHYSICAL_MOCK=1 to match a mock "
                f"campaign — for comparable numbers."
            )
            print(f"  [WARNING] {ruler_warning}", file=sys.stderr)

    (out_dir / "best_configs.json").write_text(json.dumps(manifest, indent=2))
    html = build_page(render_picks, design, platform, imgs, baseline=baseline,
                      deltas=deltas, ruler_warning=ruler_warning)
    html_path = out_dir / "best_configs.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\nManifest → {out_dir / 'best_configs.json'}")
    print(f"Comparison page → {html_path}")
    n_gds = sum(1 for e in manifest if e["gds_collected"])
    print(f"Collected {n_gds}/{len(manifest)} GDS files.")
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
