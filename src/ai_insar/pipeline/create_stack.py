"""SNAP Create Stack collocation for one shared crop.

Resampling ``NONE`` copies the same row/column window from every product.
Any other kernel shifts each secondary by one initial offset, then resamples
it onto the reference grid. Output extents are Master, the union (Maximum),
or the intersection (Minimum).
"""

from __future__ import annotations

import math

import numpy as np

from ai_insar.pipeline.s1_annotation import SwathModel

RESAMPLING = (
    "NONE",
    "NEAREST_NEIGHBOUR",
    "BILINEAR_INTERPOLATION",
    "CUBIC_CONVOLUTION",
    "BICUBIC_INTERPOLATION",
    "BISINC_5_POINT_INTERPOLATION",
    "BISINC_11_POINT_INTERPOLATION",
    "BISINC_21_POINT_INTERPOLATION",
)
OFFSET_METHODS = ("Orbit", "Product Geolocation", "GCP")
EXTENTS = ("Master", "Maximum", "Minimum")


def kernel_margin(method: str) -> int:
    name = str(method).strip().upper()
    if name in {"", "NONE", "NEAREST_NEIGHBOUR"}:
        return 0
    if name == "BILINEAR_INTERPOLATION":
        return 1
    if name in {"CUBIC_CONVOLUTION", "BICUBIC_INTERPOLATION"}:
        return 2
    halves = {
        "BISINC_5_POINT_INTERPOLATION": 2,
        "BISINC_11_POINT_INTERPOLATION": 5,
        "BISINC_21_POINT_INTERPOLATION": 10,
    }
    if name not in halves:
        raise ValueError(f"{method} is not a Create Stack resampling type.")
    return halves[name]


def output_window(height: int, width: int, offsets: list[tuple[float, float]], extent: str) -> tuple[int, int, int, int]:
    """Return ``(row0, col0, out_height, out_width)`` in reference-crop coordinates.

    Each offset is ``(line, sample)`` added to a reference coordinate to reach
    the same point in that product. The reference offset is ``(0, 0)``.
    """
    mode = str(extent).strip()
    if mode not in EXTENTS:
        raise ValueError(f"{extent} is not a Create Stack output extent.")
    boxes = [(-float(dy), -float(dx), int(height) - float(dy), int(width) - float(dx)) for dy, dx in offsets]
    if mode == "Master":
        row0, col0, row1, col1 = 0, 0, int(height), int(width)
    elif mode == "Maximum":
        row0 = math.floor(min(box[0] for box in boxes))
        col0 = math.floor(min(box[1] for box in boxes))
        row1 = math.ceil(max(box[2] for box in boxes))
        col1 = math.ceil(max(box[3] for box in boxes))
    else:
        row0 = math.ceil(max(box[0] for box in boxes))
        col0 = math.ceil(max(box[1] for box in boxes))
        row1 = math.floor(min(box[2] for box in boxes))
        col1 = math.floor(min(box[3] for box in boxes))
    if row1 <= row0 or col1 <= col0:
        raise ValueError(
            "Output Extents Minimum is empty after the initial offset. "
            "The products do not share this crop once they are shifted."
        )
    return int(row0), int(col0), int(row1 - row0), int(col1 - col0)


def resample_crop(
    image: np.ndarray,
    offset: tuple[float, float],
    *,
    origin: tuple[int, int],
    shape: tuple[int, int],
    method: str,
) -> np.ndarray:
    """Resample ``image`` onto an output grid.

    ``image`` is the product crop whose local ``(0, 0)`` is the dialog's
    row and column. ``offset`` is added to a reference-crop coordinate to
    land in this image.
    """
    from ai_insar.pipeline.tops_ops import resample_complex

    out_h, out_w = shape
    row0, col0 = origin
    dy, dx = offset
    rows = np.arange(out_h, dtype=np.float64)[:, None] + float(row0) + float(dy)
    cols = np.arange(out_w, dtype=np.float64)[None, :] + float(col0) + float(dx)
    line = np.broadcast_to(rows, (out_h, out_w))
    sample = np.broadcast_to(cols, (out_h, out_w))
    return resample_complex(np.asarray(image), line, sample, method)


def initial_offset(
    reference: SwathModel,
    secondary: SwathModel,
    *,
    row: int,
    col: int,
    height: int,
    width: int,
    method: str,
) -> tuple[float, float]:
    """Line and sample to add to a reference coordinate to reach the secondary."""
    name = str(method).strip()
    if name == "GCP":
        raise ValueError(
            "These SAFE products have no ground control points. "
            "Choose Orbit or Product Geolocation for the initial offset."
        )
    centre_line = float(row) + (int(height) - 1) / 2.0
    centre_sample = float(col) + (int(width) - 1) / 2.0
    if name == "Product Geolocation":
        return _geolocation_offset(reference, secondary, centre_line, centre_sample)
    if name != "Orbit":
        raise ValueError(f"{method} is not an initial offset method.")
    return _orbit_offset(reference, secondary, centre_line, centre_sample)


def _orbit_offset(reference: SwathModel, secondary: SwathModel, line: float, sample: float) -> tuple[float, float]:
    from ai_insar.pipeline.geometry import ground_point_at_height

    when = reference.line_time(int(round(line)))
    slant = reference.slant_range_m(sample)
    height = float(reference.terrain_height_m or 0.0)
    ground = ground_point_at_height(reference.orbit, when, slant, height, degree=3)
    guess = secondary.bursts[min(len(secondary.bursts) - 1, max(int(line) // max(secondary.lines_per_burst, 1), 0))].azimuth_time
    sec_when = secondary.orbit.zero_doppler_time(ground, guess, 3)
    sec_line = _line_of_time(secondary, sec_when)
    position, _velocity = secondary.orbit.interpolate(sec_when, 3)
    sec_range = float(np.linalg.norm(np.asarray(ground, dtype=np.float64) - position))
    sec_sample = _sample_of_range(secondary, sec_range)
    return float(sec_line - line), float(sec_sample - sample)


def _line_of_time(model: SwathModel, when) -> float:
    target = int(np.datetime64(when, "ns").astype(np.int64))
    nearest = None
    nearest_gap = None
    for index, burst in enumerate(model.bursts):
        start = int(np.datetime64(burst.azimuth_time, "ns").astype(np.int64))
        local = (target - start) * 1e-9 / float(model.azimuth_time_interval_s)
        if 0.0 <= local < float(model.lines_per_burst):
            return index * float(model.lines_per_burst) + local
        gap = min(abs(local), abs(local - (model.lines_per_burst - 1)))
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest = index * float(model.lines_per_burst) + float(np.clip(local, 0, model.lines_per_burst - 1))
    if nearest is None:
        raise ValueError("the secondary orbit time is not on this swath")
    return float(nearest)


def _sample_of_range(model: SwathModel, slant_range_m: float) -> float:
    speed = 299_792_458.0
    delay = 2.0 * float(slant_range_m) / speed
    spacing = float(model.range_pixel_spacing_m)
    if spacing == 0.0:
        raise ValueError("range pixel spacing is zero")
    return (delay - float(model.slant_range_time_s)) * speed / (2.0 * spacing)


def _geolocation_offset(
    reference: SwathModel,
    secondary: SwathModel,
    line: float,
    sample: float,
) -> tuple[float, float]:
    lat, lon = _grid_latlon(reference, line, sample)
    sec_line, sec_sample = _nearest_grid_point(secondary, lat, lon)
    return float(sec_line - line), float(sec_sample - sample)


def _grid_latlon(model: SwathModel, line: float, sample: float) -> tuple[float, float]:
    if model.geolocation_latitude.size < 1:
        raise ValueError("the annotation has no geolocation grid for Product Geolocation")
    index = int(
        np.argmin(
            (model.geolocation_line - float(line)) ** 2 + (model.geolocation_pixel - float(sample)) ** 2
        )
    )
    return float(model.geolocation_latitude[index]), float(model.geolocation_longitude[index])


def _nearest_grid_point(model: SwathModel, lat: float, lon: float) -> tuple[float, float]:
    if model.geolocation_latitude.size < 1:
        raise ValueError("the secondary annotation has no geolocation grid")
    index = int(np.argmin((model.geolocation_latitude - lat) ** 2 + (model.geolocation_longitude - lon) ** 2))
    return float(model.geolocation_line[index]), float(model.geolocation_pixel[index])
