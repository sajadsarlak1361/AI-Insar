"""Constant-offset SLC alignment by amplitude cross-correlation.

The shift is the peak of the normalized FFT cross-correlation of the
amplitudes (phase correlation). The secondary image is moved back onto the
reference with an FFT phase ramp. That pair of ideas is the same one used
by the MIT InSARdev/edu Sentinel-1 example. This file does not copy that
script, and it does not contain GMTSAR source.

The shift is one constant for the whole image (circular FFT). It is the
ALIGN SLC step in the workbench. It is not a patch-wise geometric
coregistration and it does not use orbits.
"""

from __future__ import annotations

import numpy as np


def estimate_amplitude_shift(reference: np.ndarray, secondary: np.ndarray) -> tuple[float, float]:
    """Return ``(azimuth, range)`` shift of ``secondary`` relative to ``reference``.

    A positive azimuth shift means the secondary content is lower in the
    array than the same content in the reference. The estimate is sub-pixel
    when the correlation peak has two finite neighbors.
    """
    ref = np.asarray(reference)
    sec = np.asarray(secondary)
    if ref.shape != sec.shape or ref.ndim != 2:
        raise ValueError("reference and secondary must be 2-D and the same shape")
    ref_amp = _amplitude(ref)
    sec_amp = _amplitude(sec)
    cross = np.fft.fft2(ref_amp) * np.conj(np.fft.fft2(sec_amp))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    frac_y, frac_x = _parabolic_offset(correlation, int(peak_y), int(peak_x))
    # The phase-correlation peak sits at the negative of the content shift.
    shift_y = -_signed_index(peak_y + frac_y, correlation.shape[0])
    shift_x = -_signed_index(peak_x + frac_x, correlation.shape[1])
    return float(shift_y), float(shift_x)


def fft_shift_complex(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Move complex content by ``(shift_y, shift_x)`` pixels with an FFT ramp.

    Positive ``shift_y`` moves content toward larger row indexes. The shift
    is circular.
    """
    data = np.asarray(image)
    if data.ndim != 2 or not np.iscomplexobj(data):
        raise ValueError("image must be a 2-D complex array")
    if shift_y == 0.0 and shift_x == 0.0:
        return np.asarray(data, dtype=np.complex64)
    freq_y = np.fft.fftfreq(data.shape[0]).reshape(-1, 1)
    freq_x = np.fft.fftfreq(data.shape[1]).reshape(1, -1)
    ramp = np.exp(-2j * np.pi * (float(shift_y) * freq_y + float(shift_x) * freq_x))
    shifted = np.fft.ifft2(np.fft.fft2(data) * ramp)
    return np.asarray(shifted, dtype=np.complex64)


def align_slc_stack(
    slc: np.ndarray,
    *,
    reference: int = 0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Align every date onto ``reference``. The reference date is copied.

    Returns the ``complex64`` stack and one ``(azimuth, range)`` shift per
    date. The shift is how far that date's content sat from the reference
    before the correction. The reference shift is ``(0, 0)``.
    """
    data = np.asarray(slc)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("SLC stack must be complex with shape (dates, height, width)")
    n_dates = int(data.shape[0])
    ref_index = int(reference)
    if not 0 <= ref_index < n_dates:
        raise ValueError(f"reference date {ref_index} is outside 0..{n_dates - 1}")
    aligned = np.empty(data.shape, dtype=np.complex64)
    shifts: list[tuple[float, float]] = []
    reference_image = np.asarray(data[ref_index], dtype=np.complex64)
    for index in range(n_dates):
        if index == ref_index:
            aligned[index] = reference_image
            shifts.append((0.0, 0.0))
            continue
        shift_y, shift_x = estimate_amplitude_shift(reference_image, data[index])
        shifts.append((shift_y, shift_x))
        aligned[index] = fft_shift_complex(data[index], -shift_y, -shift_x)
    return aligned, shifts


def _amplitude(image: np.ndarray) -> np.ndarray:
    amplitude = np.abs(np.asarray(image, dtype=np.complex64)).astype(np.float64)
    amplitude -= float(amplitude.mean())
    return amplitude


def _signed_index(position: float, length: int) -> float:
    half = length / 2.0
    if position > half:
        return position - length
    if position < -half:
        return position + length
    return position


def _parabolic_offset(correlation: np.ndarray, y: int, x: int) -> tuple[float, float]:
    height, width = correlation.shape

    def _axis(center: float, before: float, after: float) -> float:
        denominator = before - 2.0 * center + after
        if abs(denominator) < 1e-12:
            return 0.0
        offset = 0.5 * (before - after) / denominator
        if abs(offset) > 1.0:
            return 0.0
        return float(offset)

    center = float(correlation[y, x])
    dy = _axis(center, float(correlation[(y - 1) % height, x]), float(correlation[(y + 1) % height, x]))
    dx = _axis(center, float(correlation[y, (x - 1) % width]), float(correlation[y, (x + 1) % width]))
    return dy, dx
