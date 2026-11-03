"""Synthetic C-band SLC stack with a known subsidence bowl.

No scene is downloaded. Each date is a complex image whose shared speckle
cancels in the interferogram, leaving deformation phase, a per-date plane,
and partial temporal decorrelation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from ai_insar.conventions import (
    C_BAND_WAVELENGTH_M,
    DEFAULT_MASK_FRACTION,
    phase_from_los,
)
from ai_insar.io_stack import validate_slc


@dataclass
class SLCStack:
    """Complex SLC cube and the dates that index it.

    ``data`` has shape ``(n_dates, height, width)`` and dtype ``complex64``.
    Date index increases with time. Interferograms use that order.
    """

    data: np.ndarray
    dates: np.ndarray
    wavelength_m: float
    sensor: str = "synthetic C-band"
    description: str = ""
    truth_velocity_m_yr: np.ndarray | None = None
    truth_mask: np.ndarray | None = None
    truth_los_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data = validate_slc(self.data)
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        if self.dates.ndim != 1 or self.dates.shape[0] != self.data.shape[0]:
            raise ValueError(
                f"dates length ({self.dates.shape}) must match the SLC date axis "
                f"({self.data.shape[0]})"
            )
        if self.wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        if np.any(np.diff(self.dates) <= np.timedelta64(0, "D")):
            raise ValueError("SLC dates must be strictly increasing")

    @property
    def n_dates(self) -> int:
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    def times_year(self) -> np.ndarray:
        """Acquisition times in years relative to the first date."""
        days = (self.dates - self.dates[0]) / np.timedelta64(1, "D")
        return np.asarray(days, dtype=np.float64) / 365.25


def generate_demo_stack() -> SLCStack:
    """Synthetic stack used by the GUI sample button and ``ai-insar check``."""
    return generate_subsidence_stack()


def generate_subsidence_stack(
    *,
    n_dates: int = 8,
    height: int = 72,
    width: int = 72,
    spacing_days: int = 36,
    start: str = "2020-01-08",
    rate_m_yr: float = 0.22,
    sigma_px: float = 11.0,
    center_yx: tuple[float, float] | None = None,
    temporal_coherence: float = 0.9,
    ramp_rad: float = 0.35,
    wavelength_m: float = C_BAND_WAVELENGTH_M,
    seed: int = 7,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> SLCStack:
    """Build a complex64 SLC stack whose LOS velocity is a Gaussian bowl.

    Parameters
    ----------
    rate_m_yr:
        Peak LOS rate. Positive values subside (move away from the radar)
        and are stored as a negative velocity, matching :func:`phase_from_los`.
    temporal_coherence:
        Magnitude of the speckle component that is common to every date.
        ``1`` is a noise-free persistent target; lower values add decorrelation.
    ramp_rad:
        Standard deviation of the per-date piston/tilt, in radians.
    """
    if n_dates < 2:
        raise ValueError("n_dates must be at least 2")
    if height < 8 or width < 8:
        raise ValueError("height and width must be at least 8")
    if spacing_days <= 0:
        raise ValueError("spacing_days must be positive")
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    if not 0.0 <= temporal_coherence <= 1.0:
        raise ValueError("temporal_coherence must be between 0 and 1")
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    start_day = np.datetime64(start, "D")
    dates = start_day + np.arange(n_dates, dtype=np.int64) * np.timedelta64(int(spacing_days), "D")
    times = (np.arange(n_dates, dtype=np.float64) * float(spacing_days)) / 365.25

    yy, xx = np.mgrid[0:height, 0:width]
    if center_yx is None:
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
    else:
        cy, cx = center_yx
    bowl = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_px**2))
    truth_velocity = -float(rate_m_yr) * bowl
    truth_los = truth_velocity[None, :, :] * times[:, None, None]
    defo_phase = phase_from_los(truth_los, wavelength_m)

    texture = gaussian_filter(rng.random((height, width)), sigma=2.2)
    backscatter = 0.45 + texture
    backscatter = backscatter / np.mean(backscatter)

    x_n = (xx - xx.mean()) / (width / 2.0)
    y_n = (yy - yy.mean()) / (height / 2.0)
    gamma = float(temporal_coherence)
    shared = _unit_complex(rng, (height, width))
    slc = np.empty((n_dates, height, width), dtype=np.complex128)
    for index in range(n_dates):
        indep = _unit_complex(rng, (height, width))
        speckle = np.sqrt(gamma) * shared + np.sqrt(max(0.0, 1.0 - gamma)) * indep
        piston, tilt_x, tilt_y = rng.normal(scale=float(ramp_rad), size=3)
        ramp = piston + tilt_x * x_n + tilt_y * y_n
        slc[index] = backscatter * speckle * np.exp(1j * (defo_phase[index] + ramp))

    peak = float(np.max(np.abs(truth_velocity)))
    truth_mask = np.abs(truth_velocity) >= mask_fraction * peak
    description = (
        f"Synthetic C-band SLC stack, {n_dates} dates every {spacing_days} days, "
        f"Gaussian subsidence bowl sigma={sigma_px:g} px, peak LOS rate "
        f"{-float(rate_m_yr):.4f} m/yr (negative = away from the radar), seed={seed}."
    )
    return SLCStack(
        data=slc,
        dates=dates,
        wavelength_m=float(wavelength_m),
        sensor="synthetic C-band",
        description=description,
        truth_velocity_m_yr=truth_velocity.astype(np.float64),
        truth_mask=truth_mask,
        truth_los_m=truth_los.astype(np.float64),
    )


def _unit_complex(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Circular complex Gaussian noise with unit expected power."""
    scale = 1.0 / np.sqrt(2.0)
    return rng.normal(scale=scale, size=shape) + 1j * rng.normal(scale=scale, size=shape)
