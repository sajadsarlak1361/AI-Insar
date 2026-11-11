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
    run_parser.add_argument("--aoi", help="Processing-area WKT polygon, POLYGON ((lon lat, ...))")
    run_parser.add_argument("--aoi-config", type=Path, help="JSON AOI saved by the Processing area dialog")

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
    _apply_cli_aoi(workspace, args)

    def _printer(message: str) -> None:
        print(message)

    run_all(workspace, log=_printer)
    save_products(workspace, args.output)
    print(f"Wrote products to {args.output}")
    return 0


def _apply_cli_aoi(workspace: Workspace, args: argparse.Namespace) -> None:
    """Store a typed WKT or a saved AOI config on the workspace before the chain runs."""
    from ai_insar.pipeline.aoi import AoiError, chain_aoi, load_aoi_config, parse_wkt, prepare_aoi

    config = None
    if getattr(args, "aoi_config", None):
        config = load_aoi_config(args.aoi_config)
    wkt = str(getattr(args, "aoi", "") or "").strip()
    if not wkt and config is not None:
        wkt = config.wkt
    if not wkt:
        return
    try:
        report = prepare_aoi(parse_wkt(wkt))
    except AoiError as exc:
        raise SystemExit(str(exc)) from exc
    if config is not None:
        report.subswath = config.subswath or report.subswath
        report.polarisation = config.polarisation or report.polarisation
        report.burst_first = config.burst_first
        report.burst_last = config.burst_last
        report.burst_ids = tuple(config.burst_ids)
        report.esd_overlap_bursts = config.esd_overlap_bursts
        report.pixel_region = config.region
        report.sub_sampling_x = config.sub_sampling_x
        report.sub_sampling_y = config.sub_sampling_y
        report.source_bands = tuple(config.source_bands)
        report.copy_metadata = config.copy_metadata
        if config.area_km2 is not None:
            report.area_km2 = config.area_km2
        if config.centre_latitude is not None:
            report.centre_latitude = config.centre_latitude
            report.centre_longitude = config.centre_longitude
    workspace.config.aoi_wkt = report.clipped_wkt or report.requested_wkt
    workspace.aoi = report.to_dict()
    workspace.aoi_stages = chain_aoi(report)
    print(report.message)


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
