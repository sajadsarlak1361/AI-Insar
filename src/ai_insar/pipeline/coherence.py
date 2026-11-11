"""Windowed interferometric coherence estimated from the SLC pair itself."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

COHERENCE_LABEL = "Coherence"

COHERENCE_SUMMARY = (
    "Coherence is the magnitude of the windowed mean of "
    "later × conjugate(earlier), divided by the geometric mean of the "
    "windowed SLC powers. The window is a boxcar on the two SLC dates. "
    "It does not use the Goldstein-filtered interferogram, so a stronger "
    "filter cannot raise the coherence."
)

# Menu presets. Each entry is (key, label, window pixels).
COHERENCE_PRESETS: tuple[tuple[str, str, int], ...] = (
    ("fine", "3×3 window", 3),
    ("default", "5×5 window", 5),
    ("wide", "7×7 window", 7),
)


def coherence_map(
    early: np.ndarray,
    late: np.ndarray,
    window: int,
    *,
    azimuth_window: int | None = None,
    product: np.ndarray | None = None,
) -> np.ndarray:
    """Coherence of one pair.

    ``gamma = |mean(product)| / sqrt(mean(|early|^2) * mean(|late|^2))``.
    ``product`` is the complex interferogram. When it is omitted the product
    is ``late × conj(early)``. Pass the flat-earth and topo removed
    interferogram so the average is taken after those phases, which is what
    SNAP does. The window is SNAP's: half the size, integer division, and
    the box runs from ``centre - half`` through ``centre + half`` inclusive.
    An even size therefore covers one extra pixel. Edges are clipped, not
    reflected.
    """
    s1 = np.asarray(early)
    s2 = np.asarray(late)
    if s1.shape != s2.shape or s1.ndim != 2:
        raise ValueError("early and late SLCs must be 2-D and the same shape")
    if not np.iscomplexobj(s1) or not np.iscomplexobj(s2):
        raise ValueError("coherence requires complex SLCs")
    azimuth, range_window = _window_size(window, azimuth_window)
    cross = s2 * np.conj(s1) if product is None else np.asarray(product)
    if cross.shape != s1.shape:
        raise ValueError("flattened interferogram must match the SLC shape")
    return _coherence_plane(cross, s1, s2, azimuth, range_window)


def snap_square_windows(model, range_window: int) -> tuple[int, int]:
    """Azimuth and range sizes for SNAP Square Pixel.

    ``azimuth = round(range * ground_range_spacing / azimuth_spacing)``.
    When that count is below 1, SNAP uses azimuth 2 and a range size of
    ``round(azimuth_spacing / ground_range_spacing)``.
    """
    ground_range = float(model.ground_range_spacing_m())
    azimuth_spacing = float(model.azimuth_ground_spacing_m())
    if ground_range <= 0.0 or azimuth_spacing <= 0.0:
        raise ValueError("pixel spacing must be positive")
    azimuth = int(round(int(range_window) * ground_range / azimuth_spacing))
    range_size = int(range_window)
    if azimuth < 1:
        azimuth = 2
        range_size = max(1, int(round(azimuth_spacing / ground_range)))
    return max(1, azimuth), max(1, range_size)


def snap_window_bounds(index: int, size: int, length: int) -> tuple[int, int]:
    """Inclusive ``[centre - size//2, centre + size//2]``, clipped to the axis."""
    half = int(size) // 2
    start = max(0, int(index) - half)
    stop = min(int(length) - 1, int(index) + half)
    return start, stop


def coherence_stack(
    slc: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    window: int,
    *,
    azimuth_window: int | None = None,
) -> np.ndarray:
    """Coherence for every ``(earlier, later)`` pair. Shape ``(n_pairs, height, width)``."""
    data = np.asarray(slc)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("SLC stack must be complex with shape (dates, height, width)")
    if len(pairs) == 0:
        raise ValueError("pairs is empty")
    azimuth, range_window = _window_size(window, azimuth_window)
    out = np.empty((len(pairs), data.shape[1], data.shape[2]), dtype=np.float32)
    for index, (earlier, later) in enumerate(pairs):
        early = data[int(earlier)]
        late = data[int(later)]
        out[index] = _coherence_plane(late * np.conj(early), early, late, azimuth, range_window)
    return out


def _window_size(window: int, azimuth_window: int | None) -> tuple[int, int]:
    """Return ``(azimuth, range)`` pixel sizes. Rows are azimuth, columns range."""
    range_window = int(window)
    azimuth = range_window if azimuth_window is None else int(azimuth_window)
    if range_window < 1 or azimuth < 1:
        raise ValueError("coherence windows must be at least 1")
    return azimuth, range_window


def _coherence_plane(
    product: np.ndarray,
    early: np.ndarray,
    late: np.ndarray,
    azimuth: int,
    range_window: int,
) -> np.ndarray:
    """Integral-image boxcar. Even sizes use ``size // 2`` on both sides."""
    cross = np.asarray(product)
    real = _box_sum(np.real(cross).astype(np.float64), azimuth, range_window)
    imag = _box_sum(np.imag(cross).astype(np.float64), azimuth, range_window)
    power_early = _box_sum(
        np.real(early).astype(np.float64) ** 2 + np.imag(early).astype(np.float64) ** 2,
        azimuth,
        range_window,
    )
    power_late = _box_sum(
        np.real(late).astype(np.float64) ** 2 + np.imag(late).astype(np.float64) ** 2,
        azimuth,
        range_window,
    )
    gamma = np.abs(real + 1j * imag) / np.sqrt(np.maximum(power_early * power_late, 0.0) + 1e-30)
    return np.clip(gamma, 0.0, 1.0).astype(np.float32)


def _box_sum(values: np.ndarray, azimuth: int, range_window: int) -> np.ndarray:
    height, width = values.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    np.cumsum(np.cumsum(values, axis=0), axis=1, out=integral[1:, 1:])
    half_y = int(azimuth) // 2
    half_x = int(range_window) // 2
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    y1 = np.clip(y - half_y, 0, height)
    y2 = np.clip(y + half_y + 1, 0, height)
    x1 = np.clip(x - half_x, 0, width)
    x2 = np.clip(x + half_x + 1, 0, width)
    return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
