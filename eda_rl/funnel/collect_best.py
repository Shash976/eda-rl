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

    eda-rl collect                                  # latest campaign, best picks
    eda-rl collect --campaign all --top 5 --open
    eda-rl collect --out /tmp/best --render         # render layout PNGs (needs klayout)
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
.chead{padding:14px 16px 12px;background:#0f172a}
.badge{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.08em;color:#fff;background:#1e3a8a;padding:3px 8px;border-radius:4px;margin-bottom:8px}
.card.best .badge{background:#065f46}
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
"""


def _card(r: dict, img_b64: str | None) -> str:
    o = _obs(r)
    cfg = r.get("config") or {}
    best = r["_badge"] in ("BEST OVERALL", "TOP-1")
    img = (f'<img src="data:image/png;base64,{img_b64}" alt="layout">' if img_b64
           else '<div class="noimg">layout not rendered<br>(run with --render + klayout)</div>')
    rows = [("Config", _cfg_label(r)), ("Variant", r["_variant"])]
    # show a few salient config knobs
    for k in ("mac_lanes", "accumulator_width", "clock_period_ns", "abc_recipe"):
        if k in cfg:
            v = cfg[k]
            rows.append((k, f"{v:.3f}" if isinstance(v, float) else str(v)))
    rows += [
        ("Area", f"{_area(r):,.0f} µm²"),
        ("Fmax", f"{_fmax(r):,.0f} MHz"),
    ]
    if _cells(r) is not None:
        rows.append(("Cells", f"{_cells(r):,.0f}"))
    if _ffs(r) is not None:
        rows.append(("FFs", f"{_ffs(r):,.0f}"))
    if _power(r) is not None:
        rows.append(("Power", f"{_power(r):.1f} mW"))
    rows.append(("Timing", "✅ met" if _timing(r) else "❌ not met"))
    sc = episode_value(r)
    if sc is not None:
        rows.append(("Score", f"{sc:.3f}"))
    body = "".join(f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>" for k, v in rows)
    return (
        f'<div class="card{" best" if best else ""}">'
        f'<div class="chead"><span class="badge">{r["_badge"]}</span>'
        f'<div class="title">{_cfg_label(r)}</div><div class="csub">{r["_sublabel"]}</div></div>'
        f'<div class="limg">{img}</div><table><tbody>{body}</tbody></table></div>'
    )


def build_page(picks: list[dict], design: str, platform: str, imgs: dict[str, str | None]) -> str:
    cards = "".join(_card(r, imgs.get(r["_variant"])) for r in picks)
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{design} — best optimized configs</title><style>{_CSS}</style></head><body>'
        f'<header><h1>{design} · {platform} · best optimized configurations</h1>'
        f'<p class="sub">{len(picks)} standout designs harvested from the funnel optimizer · '
        f'GDS + reports collected alongside this page</p></header>'
        f'<div class="grid">{cards}</div>'
        f'<div class="foot">Each card\'s 6_final.gds and ORFS report were copied into this folder. '
        f'See best_configs.json for the full manifest.</div></body></html>'
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
    ap.add_argument("--open", action="store_true", help="open the comparison page in a browser")
    args = ap.parse_args()
    args.log = str(resolve_log_path(args.log, args.design, args.platform))

    rows = load_campaign_rows(args.log, args.campaign)
    if not rows:
        print(f"No episodes found in {args.log} for campaign={args.campaign!r}", file=sys.stderr)
        sys.exit(1)

    picks = select_best(rows, top=args.top)
    if not picks:
        print("No buildable F3 results in this campaign (need an F3 run with area/Fmax/GDS logged).",
              file=sys.stderr)
        sys.exit(1)

    log_path = Path(args.log)
    platform = log_path.parent.name
    design = log_path.parent.parent.name
    out_dir = Path(args.out) if args.out else Path.cwd() / "best_configs" / f"{design}_{platform}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    imgs: dict[str, str | None] = {}
    print(f"Collecting {len(picks)} configs → {out_dir}")
    for r in picks:
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

    (out_dir / "best_configs.json").write_text(json.dumps(manifest, indent=2))
    html = build_page(picks, design, platform, imgs)
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
