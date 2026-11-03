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
    EMI principal-eigenvector phase linking (batched ``eigh``), lifted with
    the unwrapped pairs so steps larger than π keep their sign.
seasonal
    One velocity plus a one-year sine and cosine. A 3×3 weighted least
    squares on the unwrapped interferograms. The reported velocity is the
    linear coefficient; the displacement series also contains the annual term.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai_insar.conventions import los_from_phase
from ai_insar.pipeline.sbas import design_matrix, invert_sbas, linear_velocity, weighted_normal

TIMESERIES_METHODS: tuple[str, ...] = ("sbas", "stacking", "nsbas", "ps", "linking", "seasonal")

TIMESERIES_LABELS: dict[str, str] = {
    "sbas": "SBAS",
    "stacking": "Stacking",
    "nsbas": "NSBAS",
    "ps": "PS linear",
    "linking": "Phase linking",
    "seasonal": "Seasonal",
}

TIMESERIES_SUMMARIES: dict[str, str] = {
    "sbas": (
        "SBAS solves a displacement at each date from the unwrapped "
        "short-baseline interferograms, weighted by coherence squared. "
        "The first date is zero. A linear velocity is then fit to that series. "
        "Negative velocity is motion away from the radar."
    ),
    "stacking": (
        "Stacking averages each interferogram's LOS rate, weighted by "
        "coherence squared. The displacement series is that single rate "
        "times time since the first date."
    ),
    "nsbas": (
        "NSBAS here is the SBAS system plus a light constraint that "
        "displacement follows one velocity. There is no seasonal term. "
        "It is an educational network inversion, not a full NSBAS package."
    ),
    "ps": (
        "PS linear keeps pixels whose amplitude dispersion is at most 0.4 "
        "or whose mean coherence is at least 0.5, then uses the stacking "
        "velocity. Other pixels are left empty. This is not StaMPS."
    ),
    "linking": (
        "Phase linking runs a batched EMI eigendecomposition of the coherence "
        "matrix (NumPy linalg.eigh), then lifts those wrapped phases with the "
        "unwrapped pairs so a step larger than π keeps its sign. "
        "Dates are solved sequentially and the pixel work stays in NumPy."
    ),
    "seasonal": (
        "Seasonal least squares fits one velocity plus a one-year sine and "
        "cosine to the unwrapped interferograms. The fit is a 3×3 system per "
        "pixel. InSARdev/core decomposes series with STL; that code is "
        "source-available and is not included here."
    ),
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
    earlier = np.array([pair[0] for pair in pairs], dtype=np.int64)
    later = np.array([pair[1] for pair in pairs], dtype=np.int64)
    span = times[later] - times[earlier]
    if np.any(span <= 0):
        raise ValueError("pair times must increase")
    increment = los_from_phase(observations, wavelength_m)
    finite = np.isfinite(increment) & np.isfinite(used)
    weight = np.where(finite, np.clip(used, 0.0, None), 0.0)
    rate = np.zeros_like(increment)
    np.divide(increment, span[:, None, None], out=rate, where=weight > 0)
    numerator = np.sum(weight * rate, axis=0)
    denominator = np.sum(weight, axis=0)
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
    _n_pairs, height, width = observations.shape
    n_dates = int(times.shape[0])
    matrix = design_matrix(pairs, n_dates)
    n_phase = n_dates - 1
    n_unknown = n_phase + 1
    phase_normal, phase_right = weighted_normal(matrix, observations, used)
    normal = np.zeros((n_unknown, n_unknown, height, width), dtype=np.float64)
    right = np.zeros((n_unknown, height, width), dtype=np.float64)
    normal[:n_phase, :n_phase] = phase_normal
    right[:n_phase] = phase_right
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


def emi_principal_phase(
    coherence: np.ndarray,
    interferometric_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    n_dates: int,
) -> np.ndarray:
    """Wrapped phase series from one batched Hermitian eigendecomposition.

    This is the small-stack EMI / principal-eigenvector linker: the complex
    coherence matrix is built per pixel and ``numpy.linalg.eigh`` returns its
    dominant eigenvector. The phase of that vector is the linked wrapped phase,
    referenced so date 0 has phase 0. Increments larger than π are not stored
    in the complex matrix; :func:`sequential_phase_link` restores them from
    the unwrapped pairs.
    """
    gamma = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
    phase = np.asarray(interferometric_phase, dtype=np.float64)
    if gamma.shape != phase.shape or gamma.ndim != 3:
        raise ValueError("coherence and phase must have shape (n_pairs, height, width)")
    if len(pairs) != gamma.shape[0]:
        raise ValueError("pairs and coherence are different lengths")
    height, width = gamma.shape[1:]
    covariance = np.zeros((height, width, n_dates, n_dates), dtype=np.complex128)
    diagonal = np.arange(n_dates)
    covariance[:, :, diagonal, diagonal] = 1.0
    for index, (earlier, later) in enumerate(pairs):
        sample = gamma[index] * np.exp(1j * phase[index])
        sample = np.where(np.isfinite(phase[index]), sample, 0.0)
        # C[later, earlier] = γ exp(i(φ_later-φ_earlier)) matches v vᴴ.
        covariance[:, :, later, earlier] = sample
        covariance[:, :, earlier, later] = np.conj(sample)
    covariance = 0.5 * (covariance + np.conjugate(np.swapaxes(covariance, -1, -2)))
    _values, vectors = np.linalg.eigh(covariance)
    principal = vectors[..., -1]
    reference = principal[..., :1]
    gauge = np.exp(-1j * np.angle(reference))
    gauge = np.where(np.abs(reference) > 0, gauge, 1.0)
    linked = np.angle(principal * gauge)
    return np.moveaxis(linked, -1, 0)


def sequential_phase_link(
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    weights: np.ndarray,
    coherence: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """EMI phase link, lifted onto the unwrapped network.

    ``numpy.linalg.eigh`` estimates a wrapped phase at every date from the
    complex coherence matrix. Each later date is then the coherence-weighted
    mean of ``phase[earlier] + unwrapped pair`` so increments larger than π
    keep the sign already unwrapped. If a date has no unwrapped link, the
    wrapped EMI step from the previous date is used instead.
    """
    observations = np.asarray(unwrapped_phase, dtype=np.float64)
    used = np.asarray(weights, dtype=np.float64)
    times = np.asarray(times_year, dtype=np.float64)
    _check_observations(observations, pairs, used)
    n_dates = int(times.shape[0])
    gamma = np.sqrt(np.clip(used, 0.0, None)) if coherence is None else np.asarray(coherence, dtype=np.float64)
    emi = emi_principal_phase(gamma, observations, pairs, n_dates)
    by_later: dict[int, list[tuple[int, int]]] = {date: [] for date in range(n_dates)}
    for index, (earlier, later) in enumerate(pairs):
        by_later[int(later)].append((int(earlier), index))
    phase = np.zeros((n_dates, observations.shape[1], observations.shape[2]), dtype=np.float64)
    for date in range(1, n_dates):
        links = by_later[date]
        if links:
            earlier_index = [item[0] for item in links]
            pair_index = [item[1] for item in links]
            estimate = phase[earlier_index] + observations[pair_index]
            weight = np.clip(used[pair_index], 0.0, None)
            finite = np.isfinite(estimate) & np.isfinite(weight) & np.isfinite(phase[earlier_index])
            weight = np.where(finite, weight, 0.0)
            numerator = np.sum(weight * np.where(finite, estimate, 0.0), axis=0)
            denominator = np.sum(weight, axis=0)
            linked = numerator / np.maximum(denominator, 1e-12)
        else:
            denominator = np.zeros(observations.shape[1:], dtype=np.float64)
            linked = np.zeros(observations.shape[1:], dtype=np.float64)
        emi_step = np.angle(np.exp(1j * (emi[date] - emi[date - 1])))
        fallback = phase[date - 1] + emi_step
        phase[date] = np.where(denominator > 0, linked, fallback)
    los_m = los_from_phase(phase, wavelength_m)
    return los_m, _finite_velocity(times, los_m)


def seasonal_least_squares(
    unwrapped_phase: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    times_year: np.ndarray,
    wavelength_m: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One velocity plus an annual harmonic, solved for every pixel at once.

    Each interferogram observes the change in
    ``v t + b sin(2π t) + c cos(2π t)``. Three unknowns, weighted by
    coherence squared. The reported velocity is the linear coefficient.
    """
    observations = np.asarray(unwrapped_phase, dtype=np.float64)
    times = np.asarray(times_year, dtype=np.float64)
    used = np.asarray(weights, dtype=np.float64)
    _check_observations(observations, pairs, used)
    earlier = np.array([pair[0] for pair in pairs], dtype=np.int64)
    later = np.array([pair[1] for pair in pairs], dtype=np.int64)
    elapsed = times - times[0]
    sine = np.sin(2.0 * np.pi * elapsed)
    cosine = np.cos(2.0 * np.pi * elapsed)
    design = np.column_stack(
        [
            elapsed[later] - elapsed[earlier],
            sine[later] - sine[earlier],
            cosine[later] - cosine[earlier],
        ]
    )
    los_increment = los_from_phase(observations, wavelength_m)
    normal, right = weighted_normal(design, los_increment, used)
    system = np.moveaxis(normal, (0, 1), (-2, -1))
    target = np.moveaxis(right, 0, -1)
    diag = np.arange(3)
    system[..., diag, diag] += 1e-8
    solved = np.linalg.solve(system, target[..., None])[..., 0]
    velocity = solved[..., 0]
    sine_amp = solved[..., 1]
    cosine_amp = solved[..., 2]
    los_m = (
        velocity[None, :, :] * elapsed[:, None, None]
        + sine_amp[None, :, :] * (sine - sine[0])[:, None, None]
        + cosine_amp[None, :, :] * (cosine - cosine[0])[:, None, None]
    )
    return los_m, velocity


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
    if method == "seasonal":
        return seasonal_least_squares(
            unwrapped_phase, pairs, times_year, wavelength_m, weights
        )
    los_m, velocity = sequential_phase_link(
        unwrapped_phase, pairs, times_year, wavelength_m, weights, coherence
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
