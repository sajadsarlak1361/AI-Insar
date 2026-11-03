"""Command line for the workbench.

With no subcommand, or with ``gui``, this opens the desktop window.
``check`` scores the built-in subsidence bowl, ``run`` writes products
without a window, and ``package-windows`` writes ``dist/ai-insar-windows.zip``.
``ai-insar --help`` lists the same commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_insar import __version__
from ai_insar.check import run_self_check
from ai_insar.io_stack import load_slc, save_products
from ai_insar.pipeline.engine import Workspace, run_all
from ai_insar.pipeline.synthetic import generate_demo_stack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-insar",
        description=(
            "SLC-first InSAR workbench. With no command, open the desktop GUI. "
            "Interferograms are later × conjugate(earlier)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"ai-insar {__version__}")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("gui", help="Open the desktop workbench (also the default)")
    commands.add_parser(
        "check",
        help="Process the built-in synthetic subsidence bowl and print PASS or FAIL",
    )

    run_parser = commands.add_parser("run", help="Run the pipeline without opening a window")
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for velocity, mask, LOS series, and products.json",
    )
    source = run_parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="Use the built-in synthetic SLC stack")
    source.add_argument("--slc", type=Path, help="complex64 .npy SLC stack shaped (dates, height, width)")
    run_parser.add_argument("--meta", type=Path, help="JSON sidecar with dates and wavelength_m")

    commands.add_parser(
        "package-windows",
        help="Write dist/ai-insar-windows.zip (source plus run_windows.bat)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "gui"):
        return _gui()
    if args.command == "check":
        return _check()
    if args.command == "run":
        return _run(args)
    if args.command == "package-windows":
        return _package()
    parser.error(f"unknown command {args.command}")
    return 2


def _check() -> int:
    result = run_self_check()
    print(result.render())
    return 0 if result.passed else 1


def _run(args: argparse.Namespace) -> int:
    if args.slc is None:
        stack = generate_demo_stack()
        print("Using the built-in synthetic C-band SLC stack.")
    else:
        stack = load_slc(args.slc, args.meta)
        print(f"Loaded SLC {stack.data.shape} {stack.data.dtype} from {args.slc}.")
    workspace = Workspace(stack=stack)

    def _printer(message: str) -> None:
        print(message)

    run_all(workspace, log=_printer)
    save_products(workspace, args.output)
    print(f"Wrote products to {args.output}")
    return 0


def _package() -> int:
    from ai_insar.packaging import make_windows_zip

    path = make_windows_zip()
    print(path)
    return 0


def _gui() -> int:
    try:
        from ai_insar.gui.app import main as gui_main
    except ImportError as exc:
        print(
            "The desktop GUI could not be imported. Install PySide6 "
            f"(pip install PySide6) and retry. ({exc})",
            file=sys.stderr,
        )
        return 1
    return int(gui_main())
