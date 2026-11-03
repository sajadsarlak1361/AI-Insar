"""Windowed interferometric coherence estimated from the SLC pair itself."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import uniform_filter


def coherence_map(early: np.ndarray, late: np.ndarray, window: int) -> np.ndarray:
    """Coherence of one pair.

    ``gamma = |mean(late * conj(early))| / sqrt(mean(|early|^2) * mean(|late|^2))``
    with a ``window`` by ``window`` boxcar. This uses the SLCs, not the
    Goldstein-filtered interferogram, so filtering cannot inflate coherence.
    """
    s1 = np.asarray(early)
    s2 = np.asarray(late)
    if s1.shape != s2.shape or s1.ndim != 2:
        raise ValueError("early and late SLCs must be 2-D and the same shape")
    if not np.iscomplexobj(s1) or not np.iscomplexobj(s2):
        raise ValueError("coherence requires complex SLCs")
    if int(window) < 1:
        raise ValueError("window must be at least 1")
    ifg = s2 * np.conj(s1)
    size = int(window)
    real = uniform_filter(ifg.real.astype(np.float64), size=size, mode="reflect")
    imag = uniform_filter(ifg.imag.astype(np.float64), size=size, mode="reflect")
    power1 = uniform_filter((s1.real.astype(np.float64) ** 2 + s1.imag.astype(np.float64) ** 2), size=size)
    power2 = uniform_filter((s2.real.astype(np.float64) ** 2 + s2.imag.astype(np.float64) ** 2), size=size)
    gamma = np.abs(real + 1j * imag) / np.sqrt(power1 * power2 + 1e-12)
    return np.clip(gamma, 0.0, 1.0).astype(np.float32)


def coherence_stack(
    slc: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    window: int,
) -> np.ndarray:
    """Coherence for every ``(earlier, later)`` pair. Shape ``(n_pairs, height, width)``."""
    data = np.asarray(slc)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("SLC stack must be complex with shape (dates, height, width)")
    if len(pairs) == 0:
        raise ValueError("pairs is empty")
    out = np.empty((len(pairs), data.shape[1], data.shape[2]), dtype=np.float32)
    for index, (earlier, later) in enumerate(pairs):
        out[index] = coherence_map(data[int(earlier)], data[int(later)], window)
    return out
