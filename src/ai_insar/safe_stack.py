"""Time-sorted ``complex64`` stack from many Sentinel-1 SAFE zips.

The product ``.zip`` is the primary input. Nothing is unpacked for the user.
A ``.SAFE`` folder or ``manifest.safe`` is accepted as well. Every date uses
the same swath, polarisation, and crop window, then the dates are ordered by
the manifest start time.

There is no coregistration. The window is the same row and column rectangle
in each measurement raster. That is a teaching crop-window alignment, not a
resampled stack.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ai_insar.pipeline.synthetic import SLCStack
from ai_insar.sentinel import (
    _dates_for_sources,
    _order_sources,
    _read_source,
    _safe_source,
    _stack_wavelength,
)

ProgressFn = Callable[[int, int, str], bool | None]


class SafeStackCancelled(Exception):
    """The progress callback asked to stop before every product was read."""


def is_safe_input(path: Path | str) -> bool:
    """True for a SAFE ``.zip``, a ``.SAFE`` folder, or a ``manifest.safe`` file."""
    candidate = Path(path)
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        return True
    if candidate.is_file() and candidate.name.lower() == "manifest.safe":
        return True
    return candidate.is_dir() and (candidate / "manifest.safe").is_file()


def stack_safe_zips(
    paths: Sequence[Path | str],
    *,
    swath: str = "IW1",
    pol: str = "VV",
    row: int = 0,
    col: int = 0,
    height: int = 256,
    width: int = 256,
    wavelength_m: float | None = None,
    progress: ProgressFn | None = None,
) -> SLCStack:
    """Read many SAFE zips or folders into one ``(n_dates, height, width)`` ``complex64`` stack.

    ``progress(step, steps, label)`` is called twice per zip: once while the
    manifest is opened and once while the crop is read. ``steps`` is ``2 * N``.
    Return ``False`` from the callback to cancel. The label says which file
    of ``N`` is opening or being read.

    The crop, swath, and polarisation are identical for every date. Dates are
    sorted by manifest start time and must be strictly increasing.
    """
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("no Sentinel-1 SAFE products were selected")
    for path in files:
        if not is_safe_input(path):
            raise ValueError(
                f"{path.name} is not a SAFE zip or a SAFE folder. "
                "Select product .zip files or a .SAFE folder (manifest.safe). "
                "Zips are read in place and do not need to be unzipped."
            )
    if not swath.strip() or not pol.strip():
        raise ValueError("swath and polarisation are required")

    total = len(files)
    steps = total * 2
    opened = []
    for index, path in enumerate(files, start=1):
        _report(progress, index, steps, f"Opening {index} of {total}: {path.name}")
        opened.append(_safe_source(path, swath=swath, pol=pol))
    missing = [source.label for source in opened if source.date is None]
    if missing:
        raise ValueError(
            "SAFE product has no acquisition date in the manifest start time: " + ", ".join(missing)
        )
    ordered = _order_sources(opened)
    planes: list[np.ndarray] = []
    for index, source in enumerate(ordered, start=1):
        _report(progress, total + index, steps, f"Reading {index} of {total}: {source.label}")
        planes.append(_read_source(source, int(row), int(col), int(height), int(width)))
    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        raise ValueError(f"cropped SLC dates do not share a shape: {sorted(shapes)}")
    data = np.stack(planes, axis=0).astype(np.complex64, copy=False)
    band = next(source.band_note for source in ordered if source.band_note)
    if any(source.band_note != band for source in ordered):
        raise ValueError(f"SAFE stack bands differ: {sorted({source.band_note for source in ordered})}")
    return SLCStack(
        data=data,
        dates=_dates_for_sources(ordered, count=data.shape[0]),
        wavelength_m=_stack_wavelength(ordered, wavelength_m),
        sensor="Sentinel-1 SLC",
        description=_stack_description(
            [source.label for source in ordered],
            band=band or f"{swath.strip().upper()} {pol.strip().upper()}",
            row=int(row),
            col=int(col),
            height=int(data.shape[1]),
            width=int(data.shape[2]),
        ),
    )


def _stack_description(
    labels: list[str],
    *,
    band: str,
    row: int,
    col: int,
    height: int,
    width: int,
) -> str:
    names = ", ".join(labels[:4])
    if len(labels) > 4:
        names += f", +{len(labels) - 4} more"
    return (
        f"SAFE zip stack, {band}, crop rows {row}:{row + height}, cols {col}:{col + width}, "
        f"from {names}. Time-sorted by manifest start time. "
        "Teaching crop-window alignment: the same row and column window in each "
        "measurement. There is no coregistration."
    )


def _report(progress: ProgressFn | None, step: int, steps: int, label: str) -> None:
    if progress is None:
        return
    if progress(step, steps, label) is False:
        raise SafeStackCancelled("SAFE stack read was cancelled")
