"""Coherence-gated displacement mask from the SBAS velocity."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def detect_displacement(
    velocity_m_yr: np.ndarray,
    coherence: np.ndarray,
    *,
    sigma_k: float = 3.5,
    peak_fraction: float = 0.28,
    min_coherence: float = 0.3,
    min_pixels: int = 12,
) -> np.ndarray:
    """Boolean mask of pixels whose LOS velocity is large for this scene.

    A pixel is kept when all of the following hold:

    - mean coherence is at least ``min_coherence``
    - the velocity is at least ``sigma_k`` robust standard deviations from the
      iteratively clipped background
    - the absolute velocity is at least ``peak_fraction`` times the peak
      absolute velocity among coherent pixels
    - the pixel belongs to a connected component of at least ``min_pixels``

    The test is two-sided, so uplift and subsidence are both marked. The scale
    is estimated from the scene itself; there is no trained classifier.
    """
    velocity = np.asarray(velocity_m_yr, dtype=np.float64)
    coh = np.asarray(coherence, dtype=np.float64)
    if velocity.shape != coh.shape or velocity.ndim != 2:
        raise ValueError("velocity and coherence must be 2-D and the same shape")
    if sigma_k <= 0:
        raise ValueError("sigma_k must be positive")
    if not 0.0 < peak_fraction < 1.0:
        raise ValueError("peak_fraction must be between 0 and 1")
    valid = np.isfinite(velocity) & np.isfinite(coh) & (coh >= float(min_coherence))
    mask = np.zeros(velocity.shape, dtype=bool)
    if int(valid.sum()) < 10:
        return mask

    center, scale = _robust_center_scale(velocity[valid])
    z_score = np.zeros_like(velocity)
    z_score[valid] = (velocity[valid] - center) / scale
    absolute = np.abs(velocity)
    peak = float(np.max(absolute[valid]))
    floor = float(peak_fraction) * peak
    raw = valid & (np.abs(z_score) >= float(sigma_k)) & (absolute >= floor)
    return _keep_components(raw, int(min_pixels))


def _robust_center_scale(sample: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based sigma, with the extreme values clipped out."""
    values = np.asarray(sample, dtype=np.float64)
    values = values[np.isfinite(values)]
    center = float(np.median(values))
    for _ in range(5):
        deviation = np.abs(values - center)
        mad = float(np.median(deviation))
        scale = 1.4826 * mad
        if scale < 1e-8:
            return center, 1e-8
        keep = deviation <= 3.5 * scale
        if int(keep.sum()) < max(10, int(0.2 * values.size)):
            break
        values = values[keep]
        center = float(np.median(values))
    deviation = np.abs(values - center)
    scale = max(1.4826 * float(np.median(deviation)), 1e-8)
    return center, scale


def _keep_components(raw: np.ndarray, min_pixels: int) -> np.ndarray:
    labeled, count = ndimage.label(raw)
    if count == 0:
        return np.zeros(raw.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= max(1, min_pixels)
    keep[0] = False
    kept = keep[labeled]
    return ndimage.binary_fill_holes(kept)
