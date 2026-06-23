"""eda_rl.cli — single entry point for the eda-rl tool.

Dispatches subcommands to the existing module ``main()`` functions, each of which
parses its own arguments with argparse.  Usage:

    eda-rl optimize    --design <yaml> --platform <plat> --budget-hours N [...]
    eda-rl build-table --design <yaml> [...]
    eda-rl benchmark   [...]
    eda-rl report      [--campaign all | --log <jsonl>] [...]

Run ``eda-rl <subcommand> --help`` for per-command options.
"""

from __future__ import annotations

import sys
from importlib import import_module

# subcommand -> "module:function"
_COMMANDS: dict[str, str] = {
    "optimize":    "eda_rl.gen2.run_funnel_optimizer:main",
    "report":      "eda_rl.viz.report:main",
    "collect":     "eda_rl.gen2.collect_best:main",
    "dashboard":   "eda_rl.viz.dashboard:main",
    "build-table": "eda_rl.gen2.build_table:main",
    "benchmark":   "eda_rl.gen2.benchmark_funnel:main",
}


def _usage() -> str:
    lines = ["eda-rl — multi-fidelity RTL→GDS design-space optimizer", "",
             "usage: eda-rl <command> [options]", "", "commands:"]
    width = max(len(c) for c in _COMMANDS)
    blurbs = {
        "optimize":    "run an optimization campaign on a design (the main pipeline)",
        "report":      "render the graphical HTML analysis dashboard from a campaign",
        "collect":     "harvest best configs: copy their GDS + a comparison page",
        "dashboard":   "launch the live/interactive Optuna dashboard (needs [dashboard] extra)",
        "build-table": "pre-build an offline F0–F2 evaluation table (resumable)",
        "benchmark":   "compare promotion/candidate strategies on the table simulator",
    }
    for c in _COMMANDS:
        lines.append(f"  {c.ljust(width)}  {blurbs.get(c, '')}")
    lines += ["", "run 'eda-rl <command> --help' for per-command options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return
    if argv[0] in ("-V", "--version"):
        from importlib.metadata import version, PackageNotFoundError
        try:
            print(version("eda-rl"))
        except PackageNotFoundError:
            print("0.1.0 (dev)")
        return

    cmd, rest = argv[0], argv[1:]
    target = _COMMANDS.get(cmd)
    if target is None:
        print(f"eda-rl: unknown command '{cmd}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        sys.exit(2)

    mod_name, func_name = target.split(":")
    func = getattr(import_module(mod_name), func_name)
    # Hand the subcommand its own argv so its argparse sees the right prog + args.
    sys.argv = [f"eda-rl {cmd}", *rest]
    func()


if __name__ == "__main__":
    main()
