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

from ai_insar.pipeline.tops_ops import esd_applied_shift_px, overlap_block_shifts


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
    window: int = 512,
    lines_per_burst: int | None = None,
    samples_per_burst: int | None = None,
    array_row0: int = 0,
    array_col0: int = 0,
    mask: np.ndarray | None = None,
    max_shift_px: float = 1.0,
) -> tuple[float, float]:
    """Constant range shift, one power-of-two window at the centre of each burst.

    The window is 512 by 512 unless the array is smaller. Magnitude images are
    de-meaned, cross-correlated without wrap-around, and the peak is read on a
    grid spaced by ``1/oversample`` inside the accuracy box. Bursts below the
    coherence threshold or farther than one pixel are left out. The returned
    shift is the mean of the remaining bursts, the same aggregate a two-image
    pair uses. A positive value is the Fourier delay applied to the secondary.
    """
    reference = np.asarray(master)
    repeat = np.asarray(slave)
    if reference.shape != repeat.shape or reference.ndim != 2:
        raise ValueError("range correlation looks must share one 2-D shape")
    windows = _burst_windows(
        reference.shape,
        window=int(window),
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
        array_row0=int(array_row0),
        array_col0=int(array_col0),
    )
    shifts: list[float] = []
    coefficients: list[float] = []
    best_coef = 0.0
    for row0, row1, col0, col1 in windows:
        if mask is not None and not np.all(mask[row0:row1, col0:col1]):
            continue
        chip_m = reference[row0:row1, col0:col1]
        chip_s = repeat[row0:row1, col0:col1]
        azimuth, rng, coef = fine_magnitude_offset(
            chip_m, chip_s, oversample=int(oversample), accuracy=int(search_px)
        )
        best_coef = max(best_coef, coef)
        if coef < float(threshold) or abs(rng) > float(max_shift_px):
            continue
        shifts.append(rng)
        coefficients.append(coef)
        del azimuth
    if not shifts:
        return 0.0, best_coef
    return float(np.mean(shifts)), float(np.mean(coefficients))


def _burst_windows(
    shape: tuple[int, int],
    *,
    window: int,
    lines_per_burst: int | None,
    samples_per_burst: int | None,
    array_row0: int,
    array_col0: int,
) -> list[tuple[int, int, int, int]]:
    """Power-of-two rectangles centred on each burst. A window that does not fit is skipped."""
    height, width = shape
    side_y = _power_of_two_at_most(min(window, height))
    side_x = _power_of_two_at_most(min(window, width))
    if side_y < 8 or side_x < 8:
        return []
    burst_width = int(samples_per_burst) if samples_per_burst else width + array_col0
    centre_col = burst_width // 2 - array_col0
    if lines_per_burst is None or lines_per_burst <= 0:
        box = _centred_window(height // 2, width // 2, side_y, side_x, height, width)
        return [box] if box is not None else []
    windows = []
    burst0 = max(array_row0 // int(lines_per_burst), 0)
    last = (array_row0 + height - 1) // int(lines_per_burst)
    for burst in range(burst0, last + 1):
        centre_row = burst * int(lines_per_burst) + int(lines_per_burst) // 2 - array_row0
        box = _centred_window(centre_row, centre_col, side_y, side_x, height, width)
        if box is not None:
            windows.append(box)
    return windows


def _power_of_two_at_most(value: int) -> int:
    size = 1
    while size * 2 <= int(value):
        size *= 2
    return size


def _centred_window(centre_row: int, centre_col: int, height: int, width: int, limit_row: int, limit_col: int):
    row0 = int(centre_row) - height // 2
    col0 = int(centre_col) - width // 2
    row1 = row0 + height
    col1 = col0 + width
    if row0 < 0 or col0 < 0 or row1 > limit_row or col1 > limit_col:
        return None
    return row0, row1, col0, col1


def fine_magnitude_offset(
    master: np.ndarray,
    slave: np.ndarray,
    *,
    oversample: int = 128,
    accuracy: int = 16,
) -> tuple[float, float, float]:
    """Azimuth and range offset of one window, plus its normalised peak.

    Both looks are converted to zero-mean magnitude. The master sits in the
    corner of a doubled array and the secondary sits in the centre, so the
    correlation does not wrap. The integer peak of the normalised surface is
    then refined by Fourier upsampling of the ``2·accuracy`` box around it.
    The returned offsets are negated, which is the delay that recentres the
    secondary on the reference. ``-0.0625`` is eight bins of a 128× grid.
    """
    reference = np.abs(np.asarray(master)).astype(np.float64, copy=False)
    repeat = np.abs(np.asarray(slave)).astype(np.float64, copy=False)
    if reference.shape != repeat.shape or reference.ndim != 2:
        raise ValueError("a correlation window must be one shared 2-D rectangle")
    lines, pixels = reference.shape
    if lines < 8 or pixels < 8 or (lines & (lines - 1)) or (pixels & (pixels - 1)):
        raise ValueError("a correlation window must be a power of two on both axes")
    reference = reference - float(np.mean(reference))
    repeat = repeat - float(np.mean(repeat))
    surface = _normalised_covariance(reference, repeat)
    peak = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak_line = int(peak[0])
    peak_pixel = int(peak[1])
    half_line = lines // 2
    half_pixel = pixels // 2
    acc_line = min(int(accuracy), half_line // 2)
    acc_pixel = min(int(accuracy), half_pixel // 2)
    line0 = int(np.clip(peak_line, acc_line, lines - acc_line))
    pixel0 = int(np.clip(peak_pixel, acc_pixel, pixels - acc_pixel))
    chip = surface[line0 - acc_line : line0 + acc_line, pixel0 - acc_pixel : pixel0 + acc_pixel]
    zoom = _oversample_real(chip, int(oversample))
    fine = np.unravel_index(int(np.argmax(zoom)), zoom.shape)
    offset_line = -half_line + line0 - acc_line + fine[0] / float(oversample)
    offset_pixel = -half_pixel + pixel0 - acc_pixel + fine[1] / float(oversample)
    coefficient = float(surface[peak_line, peak_pixel])
    return -float(offset_line), -float(offset_pixel), coefficient


def _normalised_covariance(master: np.ndarray, slave: np.ndarray) -> np.ndarray:
    """Normalised covariance on the unwrapped lag grid, size ``(lines+1, pixels+1)``."""
    lines, pixels = master.shape
    padded_lines = lines * 2
    padded_pixels = pixels * 2
    master_pad = np.zeros((padded_lines, padded_pixels), dtype=np.complex128)
    slave_pad = np.zeros((padded_lines, padded_pixels), dtype=np.complex128)
    master_pad[:lines, :pixels] = master
    slave_pad[lines // 2 : lines // 2 + lines, pixels // 2 : pixels // 2 + pixels] = slave
    cross = np.fft.ifft2(np.conj(np.fft.fft2(master_pad)) * np.fft.fft2(slave_pad))
    power = np.zeros((padded_lines, padded_pixels), dtype=np.complex128)
    power[lines:, pixels:] = master[::-1, ::-1] ** 2 + 1j * slave**2
    ones = np.zeros((padded_lines, padded_pixels), dtype=np.complex128)
    ones[lines // 2 : lines // 2 + lines, pixels // 2 : pixels // 2 + pixels] = 1.0
    norm = np.fft.ifft2(np.fft.fft2(power) * np.conj(np.fft.fft2(ones)))
    denom = np.sqrt(np.maximum(norm.real, 0.0) * np.maximum(norm.imag, 0.0))
    covar = np.zeros((lines + 1, pixels + 1), dtype=np.float64)
    usable = denom[: lines + 1, : pixels + 1] > 0
    covar[usable] = cross.real[: lines + 1, : pixels + 1][usable] / denom[: lines + 1, : pixels + 1][usable]
    return covar


def _oversample_real(image: np.ndarray, factor: int) -> np.ndarray:
    """Harmonic interpolation of a real chip by an integer factor on each axis."""
    factor = max(int(factor), 1)
    if factor == 1:
        return np.asarray(image, dtype=np.float64)
    lines, pixels = image.shape
    spectrum = np.fft.fft2(image)
    spectrum = spectrum.copy()
    spectrum[:, pixels // 2] *= 0.5
    spectrum[lines // 2, :] *= 0.5
    out_lines = lines * factor
    out_pixels = pixels * factor
    placed = np.zeros((out_lines, out_pixels), dtype=np.complex128)
    half_line = lines // 2
    half_pixel = pixels // 2
    placed[: half_line + 1, : half_pixel + 1] = spectrum[: half_line + 1, : half_pixel + 1]
    placed[: half_line + 1, -half_pixel:] = spectrum[: half_line + 1, half_pixel:]
    placed[-half_line:, : half_pixel + 1] = spectrum[half_line:, : half_pixel + 1]
    placed[-half_line:, -half_pixel:] = spectrum[half_line:, half_pixel:]
    return np.real(np.fft.ifft2(placed)) * float(factor * factor)


def apply_range_shift(image: np.ndarray, shift_px: float, method: str = "FOURIER") -> np.ndarray:
    """Move the secondary by ``shift_px`` range samples with a Fourier ramp.

    ``method`` is accepted for older callers and does not change the ramp.
    """
    del method
    data = np.asarray(image, dtype=np.complex64)
    if shift_px == 0.0:
        return data
    spectrum = np.fft.fft(data, axis=1)
    frequency = np.fft.fftfreq(data.shape[1])
    spectrum *= np.exp(-1j * 2.0 * np.pi * frequency[None, :] * float(shift_px))
    return np.fft.ifft(spectrum, axis=1).astype(np.complex64)


def apply_global_shift(image: np.ndarray, azimuth_px: float, range_px: float, azimuth_time_interval_s: float) -> np.ndarray:
    """Fourier-shift the whole secondary in azimuth and range."""
    from ai_insar.pipeline.tops_ops import apply_azimuth_pixel_shift

    shifted = np.asarray(image, dtype=np.complex64)
    if azimuth_px:
        shifted = apply_azimuth_pixel_shift(shifted, float(azimuth_px), float(azimuth_time_interval_s))
    if range_px:
        shifted = apply_range_shift(shifted, float(range_px))
    return shifted


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
    """Vectorized NumPy and ``scipy.fft``. Numba is not on this path."""
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
