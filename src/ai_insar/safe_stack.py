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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_insar.pipeline.synthetic import SLCStack
from ai_insar.safe import resolve_safe, select_measurement
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
            container=_container_label(files),
        ),
    )


def _container_label(files: Sequence[Path]) -> str:
    """Name the container that was actually opened. A folder is not a zip."""
    zips = 0
    for path in files:
        if path.is_file() and path.suffix.lower() == ".zip":
            zips += 1
    if zips == len(files):
        return "SAFE zip stack"
    if zips == 0:
        return "SAFE folder stack"
    return "SAFE stack"


def _stack_description(
    labels: list[str],
    *,
    band: str,
    row: int,
    col: int,
    height: int,
    width: int,
    container: str,
) -> str:
    names = ", ".join(labels[:4])
    if len(labels) > 4:
        names += f", +{len(labels) - 4} more"
    return (
        f"{container}, {band}, crop rows {row}:{row + height}, cols {col}:{col + width}, "
        f"from {names}. Time-sorted by manifest start time. "
        "Teaching crop-window alignment: the same row and column window in each "
        "measurement. There is no coregistration."
    )


@dataclass(frozen=True)
class SafeInput:
    """One SAFE folder or zip kept on the workspace after a multi-product load."""

    path: str
    file_name: str
    product_id: str
    product_type: str
    mode: str
    acquisition: str
    track: str
    orbit: str
    swath: str
    polarisation: str
    burst_count: int
    burst_first: int | None
    burst_last: int | None
    is_reference: bool


def expand_safe_selection(paths: Sequence[Path | str]) -> list[Path]:
    """Turn a chooser selection into SAFE products.

    A selected SAFE folder or ``.zip`` stays itself. A plain directory that
    contains several SAFE folders or zips contributes every child, not the
    first one. Anything else is kept so validation can name it.
    """
    chosen: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        for item in _expand_one(Path(raw)):
            key = item.resolve(strict=False)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(item)
    return chosen


def _expand_one(path: Path) -> list[Path]:
    if path.is_file() and path.name.lower() == "manifest.safe":
        return [path.parent]
    if is_safe_input(path):
        return [path]
    if path.is_dir():
        children: list[Path] = []
        try:
            entries = sorted(path.iterdir(), key=lambda item: item.name)
        except OSError:
            return [path]
        for child in entries:
            if child.is_file() and child.name.lower() == "manifest.safe":
                children.append(path)
            elif is_safe_input(child):
                children.append(child)
        if children:
            unique: list[Path] = []
            seen: set[Path] = set()
            for child in children:
                key = child.resolve(strict=False)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(child)
            return unique
    return [path]


def inspect_safe_products(
    paths: Sequence[Path | str],
    *,
    swath: str,
    pol: str,
    reference: Path | str | None = None,
    burst_first: int = 0,
    burst_last: int = 0,
) -> list[SafeInput]:
    """Check every product, then return one record per path.

    Mode and product type must match across the set. The requested swath and
    polarisation must exist on each product. A burst range, when set, must
    fit every product's burst list. The reference must be one of the paths.
    """
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("no Sentinel-1 SAFE products were selected")
    bounds = _burst_bounds(burst_first, burst_last)
    opened = []
    for path in files:
        if not is_safe_input(path):
            raise ValueError(
                f"{path.name} is not a SAFE zip or a SAFE folder. "
                "Select product .zip files or a .SAFE folder (manifest.safe)."
            )
        product = resolve_safe(path)
        opened.append((path, product))
    modes: dict[str, list[str]] = {}
    types: dict[str, list[str]] = {}
    for _path, product in opened:
        modes.setdefault(product.mode or "?", []).append(product.product_id)
        types.setdefault(product.product_type or "?", []).append(product.product_id)
    if len(modes) > 1:
        detail = "; ".join(f"{mode}: {', '.join(names)}" for mode, names in sorted(modes.items()))
        raise ValueError(f"Incompatible SAFE products (mode differs). {detail}")
    if len(types) > 1:
        detail = "; ".join(f"{kind}: {', '.join(names)}" for kind, names in sorted(types.items()))
        raise ValueError(f"Incompatible SAFE products (product type differs). {detail}")
    kind = next(iter(types))
    if kind != "SLC":
        names = ", ".join(types[kind])
        raise ValueError(f"The SLC stack requires SLC products. {names} is {kind}.")
    ref_key = None if reference is None else Path(reference).resolve(strict=False)
    records: list[SafeInput] = []
    for index, (path, product) in enumerate(opened):
        measurement = select_measurement(product, swath, pol)
        burst_count = len(measurement.bursts)
        if bounds is not None:
            first, last = bounds
            if burst_count < 1:
                raise ValueError(
                    f"{product.product_id} {swath.strip().upper()} {pol.strip().upper()} "
                    "has no burst list, so a burst range cannot be checked."
                )
            if last > burst_count:
                raise ValueError(
                    f"{product.product_id} {swath.strip().upper()} {pol.strip().upper()} "
                    f"has {burst_count} bursts; requested {first} to {last}."
                )
        key = path.resolve(strict=False)
        is_reference = (ref_key is None and index == 0) or (ref_key is not None and key == ref_key)
        records.append(
            SafeInput(
                path=str(path),
                file_name=path.name,
                product_id=product.product_id,
                product_type=product.product_type or "",
                mode=product.mode or "",
                acquisition=product.start_utc or "",
                track="" if product.relative_orbit is None else str(product.relative_orbit),
                orbit="" if product.absolute_orbit is None else str(product.absolute_orbit),
                swath=measurement.swath,
                polarisation=measurement.polarisation,
                burst_count=burst_count,
                burst_first=None if bounds is None else bounds[0],
                burst_last=None if bounds is None else bounds[1],
                is_reference=is_reference,
            )
        )
    if ref_key is not None and not any(item.is_reference for item in records):
        raise ValueError("The reference product is not in the selected list.")
    return records


def _burst_bounds(burst_first: int, burst_last: int) -> tuple[int, int] | None:
    first = int(burst_first)
    last = int(burst_last)
    if first == 0 and last == 0:
        return None
    if first < 1 or last < first:
        raise ValueError("Burst range must be 0 and 0, or a first..last range with first >= 1.")
    return first, last


def _report(progress: ProgressFn | None, step: int, steps: int, label: str) -> None:
    if progress is None:
        return
    if progress(step, steps, label) is False:
        raise SafeStackCancelled("SAFE stack read was cancelled")
