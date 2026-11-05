"""Sentinel-1 TOPS burst window for Split.

Bursts in a measurement raster are contiguous blocks of ``linesPerBurst``
lines. Split keeps those lines. It does not remove the overlap, so it is
not a deburst. If the raster is smaller than the annotation window, the
read is refused instead of returning a different crop.
"""

from __future__ import annotations

import numpy as np

from ai_insar.safe import SafeMeasurement, SafeProduct, open_measurement, select_measurement


def burst_pixel_window(*, lines_per_burst: int, first: int, last: int, samples: int) -> tuple[int, int, int]:
    """Return ``(row, height, width)`` for 1-based inclusive burst indexes."""
    if int(lines_per_burst) < 1 or int(samples) < 1:
        raise ValueError("linesPerBurst and the sample count must be positive")
    if int(first) < 1 or int(last) < int(first):
        raise ValueError("burst range must satisfy 1 <= first <= last")
    row = (int(first) - 1) * int(lines_per_burst)
    height = (int(last) - int(first) + 1) * int(lines_per_burst)
    return row, height, int(samples)


def split_swath(
    product: SafeProduct,
    *,
    swath: str,
    pol: str,
    burst_first: int,
    burst_last: int,
) -> tuple[np.ndarray, SafeMeasurement, tuple[int, int, int]]:
    """Read one subswath and polarisation for the requested bursts.

    Returns the complex window, the measurement, and ``(row, height, width)``.
    """
    measurement = select_measurement(product, swath, pol)
    if not measurement.lines_per_burst or not measurement.samples:
        raise ValueError(
            f"{product.product_id} {swath} {pol} has no linesPerBurst or sample count in the annotation"
        )
    burst_count = len(measurement.bursts)
    if burst_count < 1:
        raise ValueError(f"{product.product_id} {swath} {pol} has no burst list")
    if int(burst_last) > burst_count:
        raise ValueError(
            f"{product.product_id} {swath} {pol} has {burst_count} bursts; "
            f"requested {int(burst_first)} to {int(burst_last)}"
        )
    row, height, width = burst_pixel_window(
        lines_per_burst=int(measurement.lines_per_burst),
        first=int(burst_first),
        last=int(burst_last),
        samples=int(measurement.samples),
    )
    with open_measurement(product, measurement) as handle:
        raster_lines, raster_samples = handle.shape
        if row + height > raster_lines or width > raster_samples:
            raise ValueError(
                f"{product.product_id} {measurement.swath} {measurement.polarisation} "
                f"raster is {raster_lines}×{raster_samples}, but bursts "
                f"{int(burst_first)}–{int(burst_last)} need rows {row}:{row + height} "
                f"and {width} samples (linesPerBurst={measurement.lines_per_burst}). "
                "A smaller crop is not a TOPS split."
            )
        data = handle.read_window(row, 0, height, width)
    return np.asarray(data, dtype=np.complex64), measurement, (row, height, width)
