"""Time-series methods that sit beside SBAS.

All of them turn unwrapped interferograms into a LOS displacement cube
``(dates, height, width)`` and a linear velocity map. Negative velocity is
motion away from the radar, the same convention as ``invert_sbas``. None of
these is a StaMPS or MintPy replacement: one linear rate, radar grid, small
stacks in memory.

Methods
-------
sbas
    Coherence-weighted small-baseline inversion (the original stage).
stacking
    Average each interferogram's LOS rate, weighted by coherence squared.
nsbas
    The SBAS system plus a light constraint that displacement follows one
    velocity (an educational NSBAS-style regularization).
ps
    Persistent-scatterer selection from amplitude dispersion or mean
    coherence, then the stacking velocity on the pixels that pass.
linking
    Sequential coherence-weighted phase linking on the unwrapped network:
    each date is the weighted mean of ``phase[earlier] + unwrapped pair``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai_insar.conventions import los_from_phase
from ai_insar.pipeline.sbas import design_matrix, invert_sbas, linear_velocity

TIMESERIES_METHODS: tuple[str, ...] = ("sbas", "stacking", "nsbas", "ps", "linking")

TIMESERIES_LABELS: dict[str, str] = {
    "sbas": "SBAS",
    "stacking": "Stacking",
    "nsbas": "NSBAS",
    "ps": "PS linear",
    "linking": "Phase linking",
}


def coherence_weights(coherence: np.ndarray, unwrapped_phase: np.ndarray) -> np.ndarray:
    """Coherence squared, with non-finite unwrapped samples dropped."""
    weights = np.square(np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0))
    return np.where(np.isfinite(unwrapped_phase), weights, 0.0)


def stack_interferograms(
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coherence-weighted stack of interferogram LOS rates.

    Each pair contributes ``(los_later - los_earlier) / Δt``. The stacked
    velocity is the weighted mean of those rates. The displacement series is
    that velocity times time since the first date, so the result is linear
    by construction.
    """
    observations = np.asarray(unwrapped_phase, dtype=np.float64)
    times = np.asarray(times_year, dtype=np.float64)
    used = np.asarray(weights, dtype=np.float64)
    _check_observations(observations, pairs, used)
    height, width = observations.shape[1:]
    numerator = np.zeros((height, width), dtype=np.float64)
    denominator = np.zeros((height, width), dtype=np.float64)
    for index, (earlier, later) in enumerate(pairs):
        span = float(times[later] - times[earlier])
        if span <= 0:
            raise ValueError("pair times must increase")
        increment = los_from_phase(observations[index], wavelength_m)
        weight = np.where(np.isfinite(increment), np.clip(used[index], 0.0, None), 0.0)
        rate = np.where(weight > 0, increment / span, 0.0)
        numerator += weight * rate
        denominator += weight
    velocity = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), np.nan)
    elapsed = (times - times[0])[:, None, None]
    los_m = velocity[None, :, :] * elapsed
    los_m[:, denominator <= 0] = np.nan
    return los_m, velocity


def invert_nsbas(
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    weights: np.ndarray,
    *,
    constraint_weight: float = 0.25,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """SBAS inversion with an extra linear-displacement constraint.

    Unknowns are the phase at dates ``1 .. N-1`` and one phase rate. Each
    date adds the equation ``phase[k] - rate * t[k] = 0`` with weight
    ``constraint_weight``. A small weight stays close to SBAS. A larger
    weight pulls gappy pixels toward a single velocity, which is the role of
    the linear term in NSBAS. This is not a seasonal NSBAS model.
    """
    if constraint_weight <= 0:
        raise ValueError("constraint_weight must be positive")
    observations = np.asarray(unwrapped_phase, dtype=np.float64)
    times = np.asarray(times_year, dtype=np.float64)
    used = np.asarray(weights, dtype=np.float64)
    _check_observations(observations, pairs, used)
    n_pairs, height, width = observations.shape
    n_dates = int(times.shape[0])
    matrix = design_matrix(pairs, n_dates)
    n_phase = n_dates - 1
    n_unknown = n_phase + 1
    normal = np.zeros((n_unknown, n_unknown, height, width), dtype=np.float64)
    right = np.zeros((n_unknown, height, width), dtype=np.float64)
    for index in range(n_pairs):
        row = matrix[index]
        finite = np.isfinite(observations[index]) & np.isfinite(used[index])
        weight = np.where(finite, np.clip(used[index], 0.0, None), 0.0)
        obs = np.where(finite, observations[index], 0.0)
        for col_a in range(n_phase):
            if row[col_a] == 0.0:
                continue
            right[col_a] += row[col_a] * weight * obs
            for col_b in range(n_phase):
                if row[col_b] == 0.0:
                    continue
                normal[col_a, col_b] += (row[col_a] * row[col_b]) * weight
    penalty = float(constraint_weight)
    for date_index in range(1, n_dates):
        unknown = date_index - 1
        elapsed = float(times[date_index] - times[0])
        normal[unknown, unknown] += penalty
        normal[unknown, n_phase] += -penalty * elapsed
        normal[n_phase, unknown] += -penalty * elapsed
        normal[n_phase, n_phase] += penalty * elapsed * elapsed
    system = np.moveaxis(normal, (0, 1), (-2, -1))
    target = np.moveaxis(right, 0, -1)
    diag = np.arange(n_unknown)
    system[..., diag, diag] += float(ridge)
    solved = np.linalg.solve(system, target[..., None])[..., 0]
    phase_unknown = np.moveaxis(solved[..., :n_phase], -1, 0)
    phase = np.concatenate([np.zeros((1, height, width), dtype=np.float64), phase_unknown], axis=0)
    los_m = los_from_phase(phase, wavelength_m)
    return los_m, linear_velocity(times, los_m)


def persistent_scatterer_mask(
    slc: np.ndarray,
    mean_coherence: np.ndarray,
    *,
    da_max: float = 0.4,
    min_coherence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels with stable amplitude or high mean coherence.

    Amplitude dispersion is ``std(|SLC|) / mean(|SLC|)`` over dates, the
    Ferretti-style index (lower is more point-like). A pixel is kept when
    dispersion is at most ``da_max`` or mean coherence is at least
    ``min_coherence``. Returns the boolean mask and the dispersion image.
    """
    if da_max <= 0:
        raise ValueError("da_max must be positive")
    amplitude = np.abs(np.asarray(slc))
    if amplitude.ndim != 3:
        raise ValueError("slc must have shape (dates, height, width)")
    mean = np.mean(amplitude, axis=0)
    dispersion = np.std(amplitude, axis=0) / np.maximum(mean, 1e-6)
    coherence = np.asarray(mean_coherence, dtype=np.float64)
    if coherence.shape != dispersion.shape:
        raise ValueError("mean coherence shape must match the SLC image")
    selected = (dispersion <= float(da_max)) | (coherence >= float(min_coherence))
    selected &= np.isfinite(dispersion) & np.isfinite(coherence)
    return selected, dispersion.astype(np.float64)


def persistent_scatterer_velocity(
    slc: np.ndarray,
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    weights: np.ndarray,
    mean_coherence: np.ndarray,
    *,
    da_max: float = 0.4,
    min_coherence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacking velocity restricted to persistent-scatterer candidates.

    Pixels that fail the amplitude or coherence test are NaN in both the
    displacement series and the velocity. The third return is the boolean
    selection mask.
    """
    los_m, velocity = stack_interferograms(
        unwrapped_phase, pairs, times_year, wavelength_m, weights
    )
    selected, _dispersion = persistent_scatterer_mask(
        slc, mean_coherence, da_max=da_max, min_coherence=min_coherence
    )
    los_m = np.where(selected[None, :, :], los_m, np.nan)
    velocity = np.where(selected, velocity, np.nan)
    return los_m, velocity, selected


def sequential_phase_link(
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coherence-weighted sequential linking of the unwrapped network.

    Date 0 is zero. Each later date is the weighted mean of
    ``phase[earlier] + unwrapped(earlier, later)`` over pairs that touch it.
    Short-baseline pairs that were already unwrapped supply the increments, so
    a step larger than π is not re-wrapped. This is the sequential estimator
    idea (one date at a time, coherence as the weight), not the EMI eigenvalue
    decomposition of a full coherence matrix.
    """
    observations = np.asarray(unwrapped_phase, dtype=np.float64)
    used = np.asarray(weights, dtype=np.float64)
    times = np.asarray(times_year, dtype=np.float64)
    _check_observations(observations, pairs, used)
    n_dates = int(times.shape[0])
    height, width = observations.shape[1:]
    by_later: dict[int, list[tuple[int, int]]] = {date: [] for date in range(n_dates)}
    for index, (earlier, later) in enumerate(pairs):
        by_later[int(later)].append((int(earlier), index))
    phase = np.zeros((n_dates, height, width), dtype=np.float64)
    for date in range(1, n_dates):
        numerator = np.zeros((height, width), dtype=np.float64)
        denominator = np.zeros((height, width), dtype=np.float64)
        for earlier, index in by_later[date]:
            estimate = phase[earlier] + observations[index]
            weight = np.clip(used[index], 0.0, None)
            finite = np.isfinite(estimate) & np.isfinite(weight) & np.isfinite(phase[earlier])
            weight = np.where(finite, weight, 0.0)
            numerator += weight * np.where(finite, estimate, 0.0)
            denominator += weight
        phase[date] = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), np.nan)
    los_m = los_from_phase(phase, wavelength_m)
    return los_m, _finite_velocity(times, los_m)


def invert_timeseries(
    method: str,
    *,
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    coherence: np.ndarray,
    slc: np.ndarray | None = None,
    da_max: float = 0.4,
    ps_min_coherence: float = 0.5,
    nsbas_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one named method. ``method`` is one of :data:`TIMESERIES_METHODS`."""
    if method not in TIMESERIES_METHODS:
        known = ", ".join(TIMESERIES_METHODS)
        raise ValueError(f"unknown time-series method {method!r}; expected one of {known}")
    weights = coherence_weights(coherence, unwrapped_phase)
    if method == "sbas":
        return invert_sbas(unwrapped_phase, pairs, times_year, wavelength_m, weights)
    if method == "stacking":
        return stack_interferograms(unwrapped_phase, pairs, times_year, wavelength_m, weights)
    if method == "nsbas":
        return invert_nsbas(
            unwrapped_phase,
            pairs,
            times_year,
            wavelength_m,
            weights,
            constraint_weight=nsbas_weight,
        )
    if method == "ps":
        if slc is None:
            raise ValueError("PS selection needs the SLC stack")
        mean_coherence = np.nanmean(np.asarray(coherence, dtype=np.float64), axis=0)
        los_m, velocity, _selected = persistent_scatterer_velocity(
            slc,
            unwrapped_phase,
            pairs,
            times_year,
            wavelength_m,
            weights,
            mean_coherence,
            da_max=da_max,
            min_coherence=ps_min_coherence,
        )
        return los_m, velocity
    los_m, velocity = sequential_phase_link(
        unwrapped_phase, pairs, times_year, wavelength_m, weights
    )
    return los_m, velocity


def _finite_velocity(times_year: np.ndarray, los_m: np.ndarray) -> np.ndarray:
    """Linear slope, ignoring non-finite dates at a pixel."""
    los = np.asarray(los_m, dtype=np.float64)
    if np.isfinite(los).all():
        return linear_velocity(times_year, los)
    times = np.asarray(times_year, dtype=np.float64)[:, None, None]
    finite = np.isfinite(los)
    count = finite.sum(axis=0)
    safe = np.where(finite, los, 0.0)
    time_sum = np.sum(np.where(finite, times, 0.0), axis=0)
    time_mean = time_sum / np.maximum(count, 1)
    value_mean = np.sum(safe, axis=0) / np.maximum(count, 1)
    centered_time = np.where(finite, times - time_mean, 0.0)
    centered_value = np.where(finite, los - value_mean, 0.0)
    denom = np.sum(centered_time**2, axis=0)
    slope = np.sum(centered_time * centered_value, axis=0) / np.maximum(denom, 1e-12)
    return np.where(count >= 2, slope, np.nan)


def _check_observations(
    observations: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    weights: np.ndarray,
) -> None:
    if observations.ndim != 3:
        raise ValueError("unwrapped_phase must have shape (n_pairs, height, width)")
    if len(pairs) != observations.shape[0]:
        raise ValueError("pairs and unwrapped_phase are different lengths")
    if weights.shape != observations.shape:
        raise ValueError("weights must match unwrapped_phase")
