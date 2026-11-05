"""Integer multilook by averaging whole look cells.

Range looks run along the last axis and azimuth looks along the axis before
it. Remainder rows and columns that do not fill a cell are left out.
Complex samples are averaged as complex numbers unless intensity output is
requested. This is not a resample onto a new grid.
"""

from __future__ import annotations

import numpy as np


def multilook_array(
    data: np.ndarray,
    range_looks: int,
    azimuth_looks: int,
    *,
    output_intensity: bool = False,
) -> np.ndarray:
    """Average ``data`` in non-overlapping look cells.

    ``data`` is 2-D or ``(n, height, width)``.
    """
    array = np.asarray(data)
    squeeze = array.ndim == 2
    if squeeze:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("multilook input must be 2-D or (n, height, width)")
    range_count = int(range_looks)
    azimuth_count = int(azimuth_looks)
    if range_count < 1 or azimuth_count < 1:
        raise ValueError("range and azimuth looks must be at least 1")
    count, height, width = array.shape
    used_height = (height // azimuth_count) * azimuth_count
    used_width = (width // range_count) * range_count
    if used_height < azimuth_count or used_width < range_count:
        raise ValueError(
            f"image {height}×{width} is smaller than {azimuth_count} azimuth by {range_count} range looks"
        )
    trimmed = array[:, :used_height, :used_width]
    if output_intensity:
        if np.iscomplexobj(trimmed):
            values = (trimmed.real.astype(np.float64) ** 2) + (trimmed.imag.astype(np.float64) ** 2)
        else:
            values = trimmed.astype(np.float64)
    else:
        values = trimmed
    blocks = values.reshape(
        count,
        used_height // azimuth_count,
        azimuth_count,
        used_width // range_count,
        range_count,
    )
    reduced = blocks.mean(axis=(2, 4))
    if output_intensity:
        reduced = reduced.astype(np.float32)
    elif np.iscomplexobj(reduced):
        reduced = reduced.astype(np.complex64)
    if squeeze:
        return reduced[0]
    return reduced
