"""Export a velocity or displacement map as NumPy, a PNG preview, and a TIFF.

The arrays stay on the radar grid. The TIFF has no CRS and no geotransform:
it is a single-band float image that desktop GIS tools can open, not a
geocoded product. The PNG is a display preview (the same blue–red ramp the
window uses, plus a scale strip). The ``.npy`` file is the data.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from ai_insar.gui.render import colorize, symmetric_limits

_PREVIEW_GAP = 4
_PREVIEW_BAR = 12


def export_velocity_map(velocity_m_yr: np.ndarray, directory: Path) -> dict[str, Path]:
    """Write ``velocity_m_yr.npy``, ``.png``, and ``.tif`` when tifffile imports."""
    field = np.asarray(velocity_m_yr, dtype=np.float64)
    vmin, vmax = symmetric_limits(field)
    return export_raster(
        field,
        directory,
        "velocity_m_yr",
        cmap="velocity",
        vmin=vmin,
        vmax=vmax,
        unit="m/yr",
    )


def export_displacement_map(
    displacement_m: np.ndarray,
    directory: Path,
    *,
    stem: str = "los_m",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, Path]:
    """Write one displacement date. ``vmin``/``vmax`` keep the preview on the viewer's scale."""
    field = np.asarray(displacement_m, dtype=np.float64)
    if vmin is None or vmax is None:
        vmin, vmax = symmetric_limits(field)
    return export_raster(
        field,
        directory,
        stem,
        cmap="velocity",
        vmin=float(vmin),
        vmax=float(vmax),
        unit="m",
    )


def export_raster(
    values: np.ndarray,
    directory: Path,
    stem: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    unit: str,
) -> dict[str, Path]:
    """Write ``stem.npy``, ``stem.png``, and ``stem.tif`` (TIFF only if tifffile imports)."""
    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"a map must be 2-D; got shape {field.shape}")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    npy_path = folder / f"{stem}.npy"
    np.save(npy_path, field)
    preview = map_preview(field, cmap=cmap, vmin=vmin, vmax=vmax)
    png_path = folder / f"{stem}.png"
    write_png(png_path, preview)
    written: dict[str, Path] = {"npy": npy_path, "png": png_path}
    tif_path = _try_tiff(folder / f"{stem}.tif", field, unit)
    if tif_path is not None:
        written["tif"] = tif_path
    return written


def map_preview(
    values: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """RGB preview: the colorized map, a gap, and a horizontal scale strip."""
    field = np.asarray(values, dtype=np.float64)
    rgb = colorize(field, vmin=vmin, vmax=vmax, cmap=cmap)
    width = rgb.shape[1]
    gap = np.full((_PREVIEW_GAP, width, 3), 27, dtype=np.uint8)
    scale = np.linspace(vmin, vmax, width, dtype=np.float64)[None, :]
    bar = colorize(scale, vmin=vmin, vmax=vmax, cmap=cmap)
    bar = np.repeat(bar, _PREVIEW_BAR, axis=0)
    return np.concatenate([rgb, gap, bar], axis=0)


def write_png(path: Path | object, rgb: np.ndarray) -> None:
    """Write an 8-bit RGB or RGBA PNG. RGBA alpha 0 is fully transparent."""
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("PNG preview expects an RGB or RGBA uint8 image")
    height, width, channels = array.shape
    colour_type = 2 if channels == 3 else 6
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )
    if hasattr(path, "write"):
        path.write(payload)
        return
    Path(path).write_bytes(payload)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _try_tiff(path: Path, values: np.ndarray, unit: str) -> Path | None:
    """Write a float32 TIFF when tifffile is installed. No CRS is stored."""
    try:
        import tifffile
    except ImportError:
        return None
    description = (
        f"AI Insar radar-grid map. unit={unit}. "
        "Not geocoded: no CRS and no geotransform."
    )
    tifffile.imwrite(
        path,
        np.asarray(values, dtype=np.float32),
        photometric="minisblack",
        description=description,
    )
    return path
