"""Enhanced spectral diversity for Sentinel-1 TOPS overlaps.

The equations are the published ones (Prats-Iraola and the SNAP operator
description): an early and a late overlap interferogram, a coherence weight,
and an azimuth shift from the double-difference phase. Nothing here is copied
from SNAP, ISCE, MintPy, or dolphin. Those programs are optional external
callers when they are installed; the numbers this module returns are computed
here.

Periodogram maximises the coherence-weighted real part. Average takes the
argument of the weighted complex sum and divides by ``2π Δf Δt``. Both are
in pixels. The applied pair shift is the negated weight-mean, which is the
value SNAP writes after the block list.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from ai_insar.pipeline.tops_ops import (
    esd_applied_shift_px,
    overlap_block_shifts,
    resample_complex,
)


@dataclass(frozen=True)
class EsdEstimate:
    """One overlap, or one pair assembled from several overlaps."""

    block_shifts_px: np.ndarray
    block_weights: np.ndarray
    periodogram_mean_px: float
    applied_shift_px: float
    range_shift_px: float
    range_correlation: float
    estimator: str
    backend: str


def double_difference(master_early, master_late, slave_early, slave_late) -> np.ndarray:
    """``(m_early · conj(s_early)) · conj(m_late · conj(s_late))`` as complex64."""
    early = np.asarray(master_early, dtype=np.complex64) * np.conj(np.asarray(slave_early, dtype=np.complex64))
    late = np.asarray(master_late, dtype=np.complex64) * np.conj(np.asarray(slave_late, dtype=np.complex64))
    return (early * np.conj(late)).astype(np.complex64)


def average_shift_px(
    cross: np.ndarray,
    weight: np.ndarray,
    separation_hz: np.ndarray,
    azimuth_time_interval_s: float,
) -> float:
    """``arg(Σ w · Δ) / (2π · f̄ · Δt)``, with ``f̄`` the weighted spectral separation."""
    weights = np.asarray(weight, dtype=np.float64).ravel()
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    phase = float(np.angle(np.sum(np.asarray(cross, dtype=np.complex64).ravel() * weights)))
    separation = np.asarray(separation_hz, dtype=np.float64).ravel()
    if separation.size == 1:
        frequency = float(separation[0])
    else:
        frequency = float(np.sum(weights * separation) / total)
    interval = float(azimuth_time_interval_s)
    if frequency == 0.0 or interval <= 0.0:
        return 0.0
    return phase / (2.0 * np.pi * frequency * interval)


def spectral_phase_ramp(separation_hz: np.ndarray, shift_px: float, azimuth_time_interval_s: float) -> np.ndarray:
    """``exp(-j 2π Δf_DC · shift_px · Δt)``.

    ``shift_px`` is the periodogram shift to remove. The negated mean that
    SNAP prints as the applied shift is ``-shift_px`` for one weight group.
    """
    separation = np.asarray(separation_hz, dtype=np.float64)
    phase = -2.0 * np.pi * separation * float(shift_px) * float(azimuth_time_interval_s)
    return np.exp(1j * phase).astype(np.complex64)


def apply_azimuth_phase_ramp(
    image: np.ndarray,
    separation_hz: np.ndarray,
    shift_px: float,
    azimuth_time_interval_s: float,
) -> np.ndarray:
    """Multiply a burst look by the spectral-diversity phase ramp."""
    ramp = spectral_phase_ramp(separation_hz, shift_px, azimuth_time_interval_s)
    data = np.asarray(image, dtype=np.complex64)
    if ramp.ndim == 1:
        return (data * ramp[None, :]).astype(np.complex64)
    return (data * ramp).astype(np.complex64)


def range_cross_correlation_shift(
    master: np.ndarray,
    slave: np.ndarray,
    *,
    oversample: int = 128,
    search_px: int = 16,
    threshold: float = 0.1,
) -> tuple[float, float]:
    """Constant range shift, in pixels, from an FFT amplitude correlation.

    The coarse peak uses ``scipy.fft.rfft``. A short DFT oversamples a
    ``±search_px`` neighbourhood by ``oversample``. The return is
    ``(shift, correlation)``. A correlation below ``threshold`` yields a
    shift of 0 and the measured correlation.
    """
    from scipy.fft import irfft, next_fast_len, rfft

    reference = np.abs(np.asarray(master, dtype=np.complex64)).astype(np.float64)
    repeat = np.abs(np.asarray(slave, dtype=np.complex64)).astype(np.float64)
    if reference.shape != repeat.shape or reference.ndim != 2:
        raise ValueError("range correlation looks must share one 2-D shape")
    reference = reference - np.mean(reference, axis=1, keepdims=True)
    repeat = repeat - np.mean(repeat, axis=1, keepdims=True)
    length = next_fast_len(reference.shape[1] * 2)
    workers = -1
    try:
        spec_ref = rfft(reference, n=length, axis=1, workers=workers)
        spec_rep = rfft(repeat, n=length, axis=1, workers=workers)
        coarse = irfft(spec_ref * np.conj(spec_rep), n=length, axis=1, workers=workers)
    except TypeError:
        spec_ref = rfft(reference, n=length, axis=1)
        spec_rep = rfft(repeat, n=length, axis=1)
        coarse = irfft(spec_ref * np.conj(spec_rep), n=length, axis=1)
    profile = np.sum(coarse, axis=0)
    # Zero lag sits at index 0. Negative lags are wrapped to the end.
    center = int(np.argmax(profile))
    if center > length // 2:
        center -= length
    half = max(1, int(search_px))
    lags = np.linspace(center - half, center + half, half * 2 * max(int(oversample), 1) + 1)
    # Matrix DFT at the refined lags, summed over azimuth and the coarse bins.
    frequency = 2.0 * np.pi * np.arange(spec_ref.shape[1]) / length
    phase = np.exp(1j * frequency[:, None] * lags[None, :])
    refined = np.sum(np.real(np.sum(spec_ref * np.conj(spec_rep), axis=0)[:, None] * phase), axis=0)
    best = int(np.argmax(refined))
    # Positive means the secondary content sits at a higher range sample, so
    # resampling adds this many pixels to the source sample.
    shift = float(-lags[best])
    energy = float(np.sqrt(np.sum(reference**2) * np.sum(repeat**2)))
    correlation = 0.0 if energy == 0.0 else float(refined[best] / energy)
    if correlation < float(threshold):
        return 0.0, correlation
    return shift, correlation


def apply_range_shift(image: np.ndarray, shift_px: float, method: str = "BICUBIC_INTERPOLATION") -> np.ndarray:
    """Resample so a positive ``shift_px`` reads the slave from the right."""
    data = np.asarray(image)
    rows, cols = data.shape
    line = np.broadcast_to(np.arange(rows, dtype=np.float64)[:, None], (rows, cols)).copy()
    sample = np.broadcast_to(np.arange(cols, dtype=np.float64)[None, :], (rows, cols)).copy()
    sample += float(shift_px)
    return resample_complex(data, line, sample, method)


def estimate_overlap(
    master_early,
    master_late,
    slave_early,
    slave_late,
    *,
    separation_hz,
    azimuth_time_interval_s: float,
    estimator: str = "Periodogram",
    weight: str = "Inv Quadratic",
    n_blocks: int = 10,
    coherence_threshold: float = 0.3,
    range_oversample: int = 128,
    range_search_px: int = 16,
    range_threshold: float = 0.1,
    estimate_range: bool = True,
) -> EsdEstimate:
    """Block shifts for one overlap, plus an optional range correlation."""
    name = str(estimator).strip().lower()
    if name not in {"periodogram", "average"}:
        raise ValueError(f"{estimator} is not an ESD estimator. Use Average or Periodogram.")
    weight_name = str(weight).strip().lower()
    if weight_name not in {"inv quadratic", "none", "linear", "quadratic"}:
        raise ValueError(f"{weight} is not an ESD weight. Use Inv Quadratic, None, Linear, or Quadratic.")
    reports = overlap_block_shifts(
        master_early,
        master_late,
        slave_early,
        slave_late,
        separation_hz=separation_hz,
        azimuth_time_interval_s=azimuth_time_interval_s,
        n_blocks=int(n_blocks),
        coherence_threshold=float(coherence_threshold),
    )
    if weight_name != "inv quadratic":
        reports = [_reweight(item, weight_name) for item in reports]
    shifts = np.array([float(item["shift_px"]) for item in reports], dtype=np.float64)
    weights = np.array([float(item["weight"]) for item in reports], dtype=np.float64)
    if name == "average":
        cross = double_difference(master_early, master_late, slave_early, slave_late)
        from ai_insar.pipeline.tops_ops import early_look_coherence

        look = early_look_coherence(master_early, slave_early)
        pixel_weight = _pixel_weight(look, weight_name, float(coherence_threshold))
        separation = np.atleast_1d(np.asarray(separation_hz, dtype=np.float64))
        if separation.size == 1:
            separation = np.full(cross.shape[1], float(separation[0]))
        block_width = cross.shape[1] // int(n_blocks)
        # Average is one number for the overlap. The block list stays the periodogram.
        mean_shift = average_shift_px(
            cross[:, : block_width * int(n_blocks)],
            pixel_weight[:, : block_width * int(n_blocks)],
            np.broadcast_to(separation[: block_width * int(n_blocks)], (cross.shape[0], block_width * int(n_blocks))),
            azimuth_time_interval_s,
        )
        periodogram_mean = float(mean_shift)
        applied = float(-mean_shift)
    else:
        total = float(np.sum(weights))
        periodogram_mean = 0.0 if total == 0.0 else float(np.sum(shifts * weights) / total)
        applied = esd_applied_shift_px(reports)
    range_shift = 0.0
    correlation = 0.0
    if estimate_range:
        range_shift, correlation = range_cross_correlation_shift(
            master_early,
            slave_early,
            oversample=int(range_oversample),
            search_px=int(range_search_px),
            threshold=float(range_threshold),
        )
    return EsdEstimate(
        block_shifts_px=shifts,
        block_weights=weights,
        periodogram_mean_px=periodogram_mean,
        applied_shift_px=applied,
        range_shift_px=range_shift,
        range_correlation=correlation,
        estimator=name,
        backend=active_backend(),
    )


def network_shifts(
    observations: list[tuple[int, int, float, float]],
    n_images: int,
    *,
    method: str = "L2",
    max_temporal_baseline: int = 4,
) -> np.ndarray:
    """Image azimuth shifts in pixels, image 0 fixed at 0.

    Each observation is ``(reference_index, secondary_index, shift_px, weight)``
    and means ``x[secondary] - x[reference] = shift_px``. Pairs whose index
    gap exceeds ``max_temporal_baseline`` are dropped. ``L2`` is weighted
    least squares. ``L1`` is a few reweighted least-squares passes. ``L1 and
    L2`` uses L1 when the network has more pairs than unknowns, and L2 when
    it does not. This is the small-baseline network equation, written here;
    MintPy is not imported.
    """
    if int(n_images) < 2:
        raise ValueError("a network needs at least two images")
    kept = [
        item
        for item in observations
        if abs(int(item[1]) - int(item[0])) <= int(max_temporal_baseline) and float(item[3]) > 0.0
    ]
    unknowns = int(n_images) - 1
    solution = np.zeros(int(n_images), dtype=np.float64)
    if not kept or unknowns < 1:
        return solution
    design = np.zeros((len(kept), unknowns), dtype=np.float64)
    target = np.zeros(len(kept), dtype=np.float64)
    weight = np.zeros(len(kept), dtype=np.float64)
    for row, (master, slave, shift, pair_weight) in enumerate(kept):
        if int(master) > 0:
            design[row, int(master) - 1] -= 1.0
        if int(slave) > 0:
            design[row, int(slave) - 1] += 1.0
        target[row] = float(shift)
        weight[row] = float(pair_weight)
    name = str(method).strip().lower()
    use_l1 = name == "l1" or (name in {"l1 and l2", "l1andl2"} and len(kept) > unknowns)
    values = _solve_weighted(design, target, weight)
    if use_l1:
        for _ in range(8):
            residual = np.abs(design @ values - target)
            robust = weight / np.maximum(residual, 1e-6)
            values = _solve_weighted(design, target, robust)
    solution[1:] = values
    return solution


def active_backend() -> str:
    """Name the optional accelerator. The arithmetic stays in this module."""
    if importlib.util.find_spec("numba") is not None:
        return "numpy+numba"
    return "numpy"


def optional_libraries() -> dict[str, bool]:
    """Which external InSAR packages import. None of them are required."""
    names = ("insardev_pygmtsar", "isce3", "dolphin", "mintpy", "compass")
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _pixel_weight(coherence: np.ndarray, weight: str, threshold: float) -> np.ndarray:
    look = np.asarray(coherence, dtype=np.float64)
    name = weight.strip().lower()
    if name == "none":
        base = np.ones_like(look)
    elif name == "linear":
        base = look
    elif name == "quadratic":
        base = look**2
    else:
        base = np.sqrt(np.clip(look, 0.0, None))
    return np.where(look > float(threshold), base, 0.0)


def _reweight(report: dict, weight_name: str) -> dict:
    updated = dict(report)
    coherence = float(report["coherence"])
    if weight_name == "none":
        updated["weight"] = 1.0 if coherence > 0.0 else 0.0
    elif weight_name == "linear":
        updated["weight"] = coherence
    elif weight_name == "quadratic":
        updated["weight"] = coherence**2
    updated["accepted"] = float(updated["weight"]) > 0.0
    return updated


def _solve_weighted(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(weight, 0.0))
    weighted_design = design * scale[:, None]
    weighted_target = target * scale
    values, *_rest = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    return np.asarray(values, dtype=np.float64)
