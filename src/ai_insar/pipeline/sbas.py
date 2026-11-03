"""Small-baseline inversion of unwrapped interferograms into LOS displacement."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai_insar.conventions import los_from_phase


def design_matrix(pairs: Sequence[tuple[int, int]], n_dates: int) -> np.ndarray:
    """Build ``G`` such that ``G @ d[1:] = phase[later] - phase[earlier]``.

    The unknown vector is the unwrapped phase at dates ``1 .. n_dates-1``.
    The first date is the reference and is fixed at zero, so ``G`` has
    ``n_dates - 1`` columns. A connected span-1 network has full column rank.
    """
    if n_dates < 2:
        raise ValueError("n_dates must be at least 2")
    n_unknown = n_dates - 1
    matrix = np.zeros((len(pairs), n_unknown), dtype=np.float64)
    for row, pair in enumerate(pairs):
        earlier, later = int(pair[0]), int(pair[1])
        if not (0 <= earlier < later < n_dates):
            raise ValueError(f"pair {(earlier, later)} is not a valid date index pair")
        if later > 0:
            matrix[row, later - 1] += 1.0
        if earlier > 0:
            matrix[row, earlier - 1] -= 1.0
    return matrix


def linear_velocity(times_year: np.ndarray, los_m: np.ndarray) -> np.ndarray:
    """Per-pixel slope of LOS displacement against time, in meters per year."""
    times = np.asarray(times_year, dtype=np.float64)
    los = np.asarray(los_m, dtype=np.float64)
    if los.ndim != 3 or los.shape[0] != times.shape[0]:
        raise ValueError("los_m must have shape (n_dates, height, width)")
    centered = times - float(times.mean())
    denom = float(np.sum(centered**2))
    if denom <= 0:
        raise ValueError("acquisition times do not span a positive interval")
    los_centered = los - np.mean(los, axis=0, keepdims=True)
    slope = np.sum(centered[:, None, None] * los_centered, axis=0) / denom
    return slope.astype(np.float64)


def invert_sbas(
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    weights: np.ndarray | None = None,
    *,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a stack of unwrapped interferograms.

    Parameters
    ----------
    unwrapped_phase:
        Radians, shape ``(n_pairs, height, width)``. NaNs are dropped by
        setting their weight to zero.
    weights:
        Non-negative weights with the same shape. ``None`` uses ones.

    Returns
    -------
    los_m:
        Displacement in meters, shape ``(n_dates, height, width)``, with the
        first date equal to zero.
    velocity_m_yr:
        Linear LOS velocity, shape ``(height, width)``. Negative is motion
        away from the radar.
    """
    observations = np.asarray(unwrapped_phase, dtype=np.float64)
    if observations.ndim != 3:
        raise ValueError("unwrapped_phase must have shape (n_pairs, height, width)")
    n_pairs, height, width = observations.shape
    if len(pairs) != n_pairs:
        raise ValueError("pairs and unwrapped_phase are different lengths")
    times = np.asarray(times_year, dtype=np.float64)
    n_dates = int(times.shape[0])
    matrix = design_matrix(pairs, n_dates)
    n_unknown = n_dates - 1
    if weights is None:
        used_weights = np.ones_like(observations, dtype=np.float64)
    else:
        used_weights = np.asarray(weights, dtype=np.float64)
        if used_weights.shape != observations.shape:
            raise ValueError("weights must match unwrapped_phase")

    normal = np.zeros((n_unknown, n_unknown, height, width), dtype=np.float64)
    right = np.zeros((n_unknown, height, width), dtype=np.float64)
    for index in range(n_pairs):
        row = matrix[index]
        finite = np.isfinite(observations[index]) & np.isfinite(used_weights[index])
        weight = np.where(finite, np.clip(used_weights[index], 0.0, None), 0.0)
        obs = np.where(finite, observations[index], 0.0)
        for col_a in range(n_unknown):
            if row[col_a] == 0.0:
                continue
            right[col_a] += row[col_a] * weight * obs
            for col_b in range(n_unknown):
                if row[col_b] == 0.0:
                    continue
                normal[col_a, col_b] += (row[col_a] * row[col_b]) * weight

    system = np.moveaxis(normal, (0, 1), (-2, -1))
    target = np.moveaxis(right, 0, -1)
    diag = np.arange(n_unknown)
    system[..., diag, diag] += float(ridge)
    # A trailing column keeps the right-hand side a matrix. NumPy 2.x
    # ``solve`` rejects a vector whose batch shape can be read as (M, K).
    solved = np.linalg.solve(system, target[..., None])[..., 0]
    phase_unknown = np.moveaxis(solved, -1, 0)
    phase = np.concatenate([np.zeros((1, height, width), dtype=np.float64), phase_unknown], axis=0)
    los_m = los_from_phase(phase, wavelength_m)
    velocity = linear_velocity(times, los_m)
    return los_m, velocity
