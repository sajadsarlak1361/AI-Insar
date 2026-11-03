"""Topographic phase for one interferogram grid.

This is the small-baseline approximation

    phase = -4π / wavelength * B_perp / (slant_range * sin(incidence)) * height

used by the MAKE TOPO_RA step of a pair chain. The formula is reimplemented
here. GMTSAR source is not copied. With no DEM, or with a zero perpendicular
baseline, the interferogram is left unchanged.
"""

from __future__ import annotations

import numpy as np


def topographic_phase_rad(
    dem_m: np.ndarray,
    *,
    wavelength_m: float,
    baseline_perp_m: float,
    slant_range_m: float,
    incidence_rad: float,
) -> np.ndarray:
    """Topographic phase in radians, same shape as ``dem_m``."""
    heights = np.asarray(dem_m, dtype=np.float64)
    if heights.ndim != 2:
        raise ValueError("DEM must be a 2-D array of heights in meters")
    if wavelength_m <= 0 or slant_range_m <= 0:
        raise ValueError("wavelength_m and slant_range_m must be positive")
    sine = float(np.sin(incidence_rad))
    if abs(sine) < 1e-8:
        raise ValueError("incidence is too close to 0 or 180 degrees")
    scale = -4.0 * np.pi / float(wavelength_m) * float(baseline_perp_m) / (float(slant_range_m) * sine)
    return scale * heights


def apply_topographic_phase(interferograms: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    """Multiply each interferogram by ``exp(-i * phase)``. Returns ``complex64``."""
    data = np.asarray(interferograms)
    phase = np.asarray(phase_rad, dtype=np.float64)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("interferograms must be complex with shape (pairs, height, width)")
    if phase.shape != data.shape[1:]:
        raise ValueError("topographic phase shape must match one interferogram")
    factor = np.exp(-1j * phase).astype(np.complex64)
    return np.asarray(data * factor, dtype=np.complex64)
