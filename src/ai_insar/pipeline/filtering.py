"""Goldstein spectral filter for a complex interferogram.

With ``alpha = 0`` the overlap-add of the analysis windows reconstructs the
input. Larger ``alpha`` narrows the patch spectrum and smooths the phase.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

GOLDSTEIN_LABEL = "Goldstein"

GOLDSTEIN_SUMMARY = (
    "The Goldstein filter overlap-adds windowed FFTs of the complex "
    "interferogram and raises the smoothed power spectrum to alpha. "
    "Alpha 0 reconstructs the input. The default is alpha 0.5 with a "
    "32-pixel window and 50% overlap. Unwrapping uses this phase. "
    "Coherence is still estimated from the SLC pair, so filtering cannot "
    "inflate it."
)

# Menu presets. Each entry is (key, label, alpha). The window stays whatever
# Pipeline → Settings last stored (default 32).
GOLDSTEIN_PRESETS: tuple[tuple[str, str, float], ...] = (
    ("light", "Light (α = 0.3)", 0.3),
    ("default", "Default (α = 0.5)", 0.5),
    ("strong", "Strong (α = 0.8)", 0.8),
)


def goldstein_filter(
    interferogram: np.ndarray,
    *,
    alpha: float = 0.5,
    window: int = 32,
    overlap: float = 0.5,
    spectrum_smooth: int = 3,
) -> np.ndarray:
    """Filter one complex interferogram. Returns ``complex64`` of the same shape."""
    ifg = np.asarray(interferogram)
    if ifg.ndim != 2 or not np.iscomplexobj(ifg):
        raise ValueError("interferogram must be a 2-D complex array")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    height, width = ifg.shape
    win = int(window)
    if win < 8 or min(height, width) < 8:
        return np.asarray(ifg, dtype=np.complex64)
    win = min(win, height, width)
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(round(win * (1.0 - overlap))))

    source = np.asarray(ifg, dtype=np.complex128)
    hann = _analysis_window(win)
    acc = np.zeros((height, width), dtype=np.complex128)
    weight = np.zeros((height, width), dtype=np.float64)
    smooth = max(1, int(spectrum_smooth))

    for y0 in _starts(height, win, step):
        for x0 in _starts(width, win, step):
            patch = source[y0 : y0 + win, x0 : x0 + win] * hann
            spectrum = np.fft.fftshift(np.fft.fft2(patch))
            power = np.abs(spectrum) ** 2
            smoothed = uniform_filter(power, size=smooth, mode="nearest")
            scale = smoothed / (float(smoothed.max()) + 1e-12)
            filtered = spectrum * (scale ** float(alpha))
            spatial = np.fft.ifft2(np.fft.ifftshift(filtered))
            acc[y0 : y0 + win, x0 : x0 + win] += spatial
            weight[y0 : y0 + win, x0 : x0 + win] += hann

    covered = weight > 1e-8
    out = source.copy()
    out[covered] = acc[covered] / weight[covered]
    return out.astype(np.complex64)


def filter_stack(
    interferograms: np.ndarray,
    *,
    alpha: float = 0.5,
    window: int = 32,
    overlap: float = 0.5,
) -> np.ndarray:
    """Apply :func:`goldstein_filter` to each interferogram in a stack."""
    data = np.asarray(interferograms)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("interferograms must have shape (pairs, height, width)")
    out = np.empty_like(data, dtype=np.complex64)
    for index in range(data.shape[0]):
        out[index] = goldstein_filter(data[index], alpha=alpha, window=window, overlap=overlap)
    return out


def _analysis_window(win: int) -> np.ndarray:
    """Separable Hann window floored so image edges keep a non-zero weight.

    A pure Hann is zero on its border, which would leave the outer pixels of
    the interferogram unfiltered. Flooring the window does not change the
    alpha = 0 reconstruction: overlap-add divides by the sum of the windows.
    """
    if win <= 1:
        return np.ones((win, win), dtype=np.float64)
    k = np.arange(win, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (win - 1))
    hann = 0.25 + 0.75 * hann
    return np.outer(hann, hann)


def _starts(length: int, win: int, step: int) -> list[int]:
    if length <= win:
        return [0]
    starts = list(range(0, length - win + 1, step))
    last = length - win
    if starts[-1] != last:
        starts.append(last)
    return starts
