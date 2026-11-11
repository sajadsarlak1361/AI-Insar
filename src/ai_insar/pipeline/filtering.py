"""Goldstein spectral filter for a complex interferogram.

The patch FFT, ``|spectrum| ** alpha`` weight, and triangular overlap-add
follow ``goldstein_numpy`` / ``goldstein_pytorch`` in InSARdev
``utils_goldstein.py`` (copyright (c) 2025 Alexey Pechnikov). The repository
owner authorized that port. This module does not import ``insardev`` and does
not require ``pip install insardev``. See NOTICE.

With ``alpha = 0`` the overlap-add reconstructs the input. Larger ``alpha``
narrows the patch spectrum and smooths the phase. Patches are transformed
together with ``scipy.fft`` (or PyTorch when requested), not one Python
``fft2`` per window.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as scipy_fft
from scipy.ndimage import uniform_filter1d

GOLDSTEIN_LABEL = "Goldstein"

GOLDSTEIN_SUMMARY = (
    "The Goldstein filter overlap-adds triangular windows of complex "
    "interferogram patches. Each patch spectrum is multiplied by "
    "|FFT| raised to alpha, optionally after a small spectral smooth. "
    "Alpha 0 reconstructs the input. The default is alpha 0.5 with a "
    "32-pixel window and 50% overlap. The batched FFT is the InSARdev "
    "goldstein_numpy kernel (Alexey Pechnikov), ported in-tree, with an "
    "optional PyTorch path. Unwrapping uses this phase. Coherence is still "
    "estimated from the SLC pair, so filtering cannot inflate it."
)

# Menu presets. Each entry is (key, label, alpha). The window stays whatever
# Pipeline → Settings last stored (default 32).
GOLDSTEIN_PRESETS: tuple[tuple[str, str, float], ...] = (
    ("light", "Light (α = 0.3)", 0.3),
    ("default", "Default (α = 0.5)", 0.5),
    ("strong", "Strong (α = 0.8)", 0.8),
)

# One batched FFT stays near this size. Larger scenes are split by patch row.
_PATCH_BYTES = 64 * 1024 * 1024

_NUMBA_ACCUM = None
_NUMBA_TRIED = False
_FFT_WORKERS: int | None = -1


def goldstein_filter(
    interferogram: np.ndarray,
    *,
    alpha: float = 0.5,
    window: int = 32,
    overlap: float = 0.5,
    spectrum_smooth: int = 3,
    backend: str | None = None,
    coherence: np.ndarray | None = None,
    alpha_min: float = 0.2,
    alpha_max: float = 1.0,
) -> np.ndarray:
    """Filter one complex interferogram. Returns ``complex64`` of the same shape.

    ``backend`` is ``numpy`` (default) or ``torch``. The environment variable
    ``AI_INSAR_FILTER`` selects the same names when ``backend`` is omitted.
    Torch is used only when it imports; otherwise the NumPy FFT runs.
    ``AI_INSAR_FILTER_DEVICE`` chooses the torch device (default ``cpu``).
    """
    ifg = np.asarray(interferogram)
    if ifg.ndim != 2 or not np.iscomplexobj(ifg):
        raise ValueError("interferogram must be a 2-D complex array")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    height, width = ifg.shape
    win = int(window)
    if win < 8 or min(height, width) < 8:
        goldstein_filter.last_backend = "numpy"
        return np.asarray(ifg, dtype=np.complex64)
    win = min(win, height, width)
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be in [0, 1)")
    smooth = max(1, int(spectrum_smooth))
    coh = None
    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
        if coh.shape != ifg.shape:
            raise ValueError("coherence shape must match the interferogram")
        if not 0.0 <= float(alpha_min) <= float(alpha_max) <= 1.0:
            raise ValueError("adaptive alpha bounds must satisfy 0 <= min <= max <= 1")
    chosen = _resolve_backend(backend)
    source = np.array(ifg, dtype=np.complex64, copy=True)
    invalid = ~np.isfinite(source)
    saved = source[invalid].copy() if np.any(invalid) else None
    if saved is not None:
        source[invalid] = 0

    filtered = None
    used = "numpy"
    if chosen == "torch" and coh is None:
        try:
            filtered = _goldstein_torch(source, float(alpha), win, float(overlap), smooth)
        except Exception:
            filtered = None
        else:
            used = "torch"
    if filtered is None:
        filtered = _goldstein_numpy(
            source,
            float(alpha),
            win,
            float(overlap),
            smooth,
            coherence=coh,
            alpha_min=float(alpha_min),
            alpha_max=float(alpha_max),
        )
        used = "numpy"
    if saved is not None:
        filtered[invalid] = saved
    goldstein_filter.last_backend = used
    return filtered


goldstein_filter.last_backend = "numpy"


def goldstein_snap(
    interferogram: np.ndarray,
    *,
    alpha: float = 1.0,
    fft_size: int = 64,
    window_size: int = 3,
    coherence: np.ndarray | None = None,
    adaptive: bool = False,
    alpha_min: float = 0.2,
    alpha_max: float = 1.0,
    coherence_mask: bool = False,
    coherence_threshold: float = 0.2,
) -> np.ndarray:
    """SNAP Goldstein filter: FFT blocks of ``fft_size``, step ``fft_size/4``.

    The spectrum magnitude is averaged in a ``window_size`` box (3 means the
    bin and its neighbours), zeros are left out of that average, and the mean
    is raised to ``alpha``. A crop smaller than the FFT is zero-padded so the
    block still runs, then cropped back. A sample that is NaN, or 0 on either
    component, stays no-data. The overlap-add weight is SNAP's separable
    triangle. The phase is the argument of that weighted sum.
    """
    ifg = np.asarray(interferogram)
    if ifg.ndim != 2 or not np.iscomplexobj(ifg):
        raise ValueError("interferogram must be a 2-D complex array")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    fft = int(fft_size)
    if fft < 8 or fft % 2 != 0:
        raise ValueError("FFT size must be an even integer of at least 8")
    half_window = int(window_size) // 2
    height, width = ifg.shape
    real = np.real(ifg).astype(np.float64, copy=False)
    imag = np.imag(ifg).astype(np.float64, copy=False)
    valid = np.isfinite(real) & np.isfinite(imag) & (real != 0.0) & (imag != 0.0)
    pad_y = max(0, fft - height)
    pad_x = max(0, fft - width)
    padded_h = height + pad_y
    padded_w = width + pad_x
    real_p = np.zeros((padded_h, padded_w), dtype=np.float64)
    imag_p = np.zeros((padded_h, padded_w), dtype=np.float64)
    valid_p = np.zeros((padded_h, padded_w), dtype=bool)
    real_p[:height, :width] = np.where(valid, real, 0.0)
    imag_p[:height, :width] = np.where(valid, imag, 0.0)
    valid_p[:height, :width] = valid
    acc_i = np.zeros((height, width), dtype=np.float64)
    acc_q = np.zeros((height, width), dtype=np.float64)
    covered = np.zeros((height, width), dtype=bool)
    step = max(1, fft // 4)
    half = fft // 2
    for y0 in _snap_origins(0, padded_h - fft, step):
        for x0 in _snap_origins(0, padded_w - fft, step):
            block_valid = valid_p[y0 : y0 + fft, x0 : x0 + fft]
            if not np.any(block_valid):
                continue
            filtered_i, filtered_q = _snap_filter_block(
                real_p[y0 : y0 + fft, x0 : x0 + fft],
                imag_p[y0 : y0 + fft, x0 : x0 + fft],
                float(alpha),
                half_window,
                coherence,
                y0,
                x0,
                height,
                width,
                adaptive=adaptive,
                alpha_min=float(alpha_min),
                alpha_max=float(alpha_max),
            )
            _snap_accumulate(
                acc_i, acc_q, covered, filtered_i, filtered_q, block_valid, y0, x0, height, width, half
            )
    out = np.zeros((height, width), dtype=np.complex64)
    out[covered] = (acc_i[covered] + 1j * acc_q[covered]).astype(np.complex64)
    out[~valid] = 0
    if coherence_mask and coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
        if coh.shape != ifg.shape:
            raise ValueError("coherence shape must match the interferogram")
        restore = valid & np.isfinite(coh) & (coh >= 0.0) & (coh <= 1.0) & (coh < float(coherence_threshold))
        out[restore] = np.asarray(ifg[restore], dtype=np.complex64)
    return out


def _snap_origins(start: int, last: int, step: int) -> list[int]:
    if last < start:
        return []
    origins = list(range(start, last + 1, step))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _snap_filter_block(
    block_i,
    block_q,
    alpha,
    half_window,
    coherence,
    y0,
    x0,
    height,
    width,
    *,
    adaptive,
    alpha_min,
    alpha_max,
):
    packed = np.asarray(block_q, dtype=np.float64) + 1j * np.asarray(block_i, dtype=np.float64)
    spectrum = np.fft.fft2(packed)
    smooth = _box_mean_skip_zeros(np.abs(spectrum), int(half_window))
    if adaptive and coherence is not None:
        y1 = min(height, y0 + block_i.shape[0])
        x1 = min(width, x0 + block_i.shape[1])
        block_alpha = float(alpha)
        if y0 < height and x0 < width and y1 > max(y0, 0) and x1 > max(x0, 0):
            view = np.asarray(coherence)[max(y0, 0) : y1, max(x0, 0) : x1]
            finite = view[np.isfinite(view) & (view >= 0.0) & (view <= 1.0)]
            if finite.size:
                block_alpha = float(np.clip(1.0 - np.mean(finite), alpha_min, alpha_max))
    else:
        block_alpha = float(alpha)
    weight = np.ones(smooth.shape, dtype=np.float64)
    if block_alpha != 0.0:
        weight = np.zeros(smooth.shape, dtype=np.float64)
        nonzero = smooth > 0.0
        weight[nonzero] = np.power(smooth[nonzero], block_alpha)
    restored = np.fft.ifft2(spectrum * weight)
    return restored.imag, restored.real


def _box_mean_skip_zeros(power: np.ndarray, half: int) -> np.ndarray:
    """Mean of the non-zero bins in a clamped ``(2*half+1)`` box."""
    if half <= 0:
        return np.array(power, dtype=np.float64, copy=True)
    height, width = power.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    count = np.zeros_like(integral)
    present = power != 0.0
    np.cumsum(np.cumsum(np.where(present, power, 0.0), axis=0), axis=1, out=integral[1:, 1:])
    np.cumsum(np.cumsum(present.astype(np.float64), axis=0), axis=1, out=count[1:, 1:])
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    y1 = np.clip(y - half, 0, height)
    y2 = np.clip(y + half + 1, 0, height)
    x1 = np.clip(x - half, 0, width)
    x2 = np.clip(x + half + 1, 0, width)
    total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    number = count[y2, x2] - count[y1, x2] - count[y2, x1] + count[y1, x1]
    mean = np.zeros(power.shape, dtype=np.float64)
    good = number > 0
    mean[good] = total[good] / number[good]
    return mean


def _snap_accumulate(acc_i, acc_q, covered, filtered_i, filtered_q, block_valid, y0, x0, height, width, half):
    fft_n = filtered_i.shape[0]
    y_start = max(y0, 0)
    x_start = max(x0, 0)
    y_stop = min(y0 + fft_n, height)
    x_stop = min(x0 + fft_n, width)
    if y_stop <= y_start or x_stop <= x_start:
        return
    yy = np.arange(y_start, y_stop)
    xx = np.arange(x_start, x_stop)
    weight_y = 1.0 - np.abs(yy - y0 - half + 0.5) / half
    weight_x = 1.0 - np.abs(xx - x0 - half + 0.5) / half
    weight = weight_y[:, None] * weight_x[None, :]
    local_y = yy - y0
    local_x = xx - x0
    take = block_valid[np.ix_(local_y, local_x)]
    acc_i[np.ix_(yy, xx)] += np.where(take, filtered_i[np.ix_(local_y, local_x)] * weight, 0.0)
    acc_q[np.ix_(yy, xx)] += np.where(take, filtered_q[np.ix_(local_y, local_x)] * weight, 0.0)
    covered[np.ix_(yy, xx)] |= take


def filter_stack(
    interferograms: np.ndarray,
    *,
    alpha: float = 0.5,
    window: int = 32,
    overlap: float = 0.5,
    spectrum_smooth: int = 3,
    backend: str | None = None,
) -> np.ndarray:
    """Apply :func:`goldstein_filter` to each interferogram in a stack."""
    data = np.asarray(interferograms)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("interferograms must have shape (pairs, height, width)")
    out = np.empty_like(data, dtype=np.complex64)
    for index in range(data.shape[0]):
        out[index] = goldstein_filter(
            data[index],
            alpha=alpha,
            window=window,
            overlap=overlap,
            spectrum_smooth=spectrum_smooth,
            backend=backend,
        )
    return out


def _resolve_backend(backend: str | None) -> str:
    if backend is None:
        backend = os.environ.get("AI_INSAR_FILTER", "numpy")
    name = str(backend).strip().lower()
    if name in {"", "auto", "numpy", "numba"}:
        return "numpy"
    if name in {"torch", "pytorch"}:
        return "torch"
    raise ValueError("goldstein backend must be 'numpy' or 'torch'")


def _pad_and_count(length: int, win: int, step: int) -> tuple[int, int]:
    """Pad the far edge so every pixel sits in a window on a regular ``step`` grid."""
    if length <= win:
        return 0, 1
    last = length - win
    new_last = int(np.ceil(last / step)) * step
    return new_last + win - length, new_last // step + 1


def _triangular_weight(win: int) -> np.ndarray:
    """Separable triangular window used as the overlap-add weight.

    Even sizes use the quadrant construction from InSARdev
    ``goldstein_numpy``. A small floor keeps image corners finite: a pure
    triangle is zero on its border, and this project's tests expect a finite
    reconstruction there. The floor does not change an ``alpha = 0`` result,
    because overlap-add divides by the sum of the same weights.
    """
    if win % 2 == 0 and win >= 2:
        half = win // 2
        taper = 1.0 - np.abs(np.arange(half) - (win / 2.0 - 1.0)) / (win / 2.0 - 1.0)
        quadrant = np.outer(taper, taper)
        weight = np.block(
            [
                [quadrant, np.flip(quadrant, axis=1)],
                [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)],
            ]
        )
    else:
        coord = np.arange(win, dtype=np.float64)
        taper = 1.0 - np.abs(coord - (win - 1) / 2.0) / max((win - 1) / 2.0, 1.0)
        weight = np.outer(taper, taper)
    weight = np.asarray(weight, dtype=np.float32)
    np.maximum(weight, 1e-3, out=weight)
    return weight


def _goldstein_numpy(
    source: np.ndarray,
    alpha: float,
    win: int,
    overlap: float,
    spectrum_smooth: int,
    coherence: np.ndarray | None = None,
    alpha_min: float = 0.2,
    alpha_max: float = 1.0,
) -> np.ndarray:
    height, width = source.shape
    step = max(1, int(round(win * (1.0 - overlap))))
    pad_y, _n_y = _pad_and_count(height, win, step)
    pad_x, _n_x = _pad_and_count(width, win, step)
    padded = source
    padded_coh = None
    if pad_y or pad_x:
        padded = np.pad(source, ((0, pad_y), (0, pad_x)), mode="edge")
        if coherence is not None:
            padded_coh = np.pad(coherence, ((0, pad_y), (0, pad_x)), mode="edge")
    elif coherence is not None:
        padded_coh = coherence
    view = sliding_window_view(padded, (win, win))[::step, ::step]
    coh_view = None
    if padded_coh is not None:
        coh_view = sliding_window_view(padded_coh, (win, win))[::step, ::step]
    n_rows = int(view.shape[0])
    n_cols = int(view.shape[1])
    weight = _triangular_weight(win)
    acc = np.zeros(padded.shape, dtype=np.complex64)
    count = np.zeros(padded.shape, dtype=np.float32)
    row_bytes = max(1, n_cols * win * win * 8)
    rows_per = max(1, _PATCH_BYTES // row_bytes)
    for row0 in range(0, n_rows, rows_per):
        row1 = min(n_rows, row0 + rows_per)
        patches = np.array(view[row0:row1], dtype=np.complex64, copy=True).reshape(-1, win, win)
        patch_alpha: float | np.ndarray = alpha
        if coh_view is not None:
            means = np.nanmean(coh_view[row0:row1].reshape(-1, win * win), axis=1)
            means = np.where(np.isfinite(means), means, 0.0)
            patch_alpha = np.clip(1.0 - means, alpha_min, alpha_max).astype(np.float32)
        spatial = _filter_patch_batch_numpy(patches, patch_alpha, spectrum_smooth)
        spatial *= weight
        spatial = np.ascontiguousarray(spatial.reshape(row1 - row0, n_cols, win, win))
        _accumulate(acc, count, spatial, weight, row0 * step, step)
    covered_acc = acc[:height, :width]
    covered_count = count[:height, :width]
    out = source.copy()
    covered = covered_count > 1e-6
    out[covered] = covered_acc[covered] / covered_count[covered]
    return out


def _filter_patch_batch_numpy(
    patches: np.ndarray,
    alpha: float | np.ndarray,
    spectrum_smooth: int,
) -> np.ndarray:
    """Batched ``ifft(|FFT|**alpha * FFT)`` for patches shaped ``(n, win, win)``."""
    spectrum = _fft2(patches, inverse=False)
    magnitude = np.abs(spectrum)
    if spectrum_smooth > 1:
        magnitude = uniform_filter1d(magnitude, size=spectrum_smooth, axis=-1, mode="nearest")
        magnitude = uniform_filter1d(magnitude, size=spectrum_smooth, axis=-2, mode="nearest")
    np.maximum(magnitude, 1e-10, out=magnitude)
    exponent = np.asarray(alpha, dtype=np.float32)
    if exponent.ndim == 0:
        if float(exponent) != 0.0:
            spectrum *= np.power(magnitude, exponent)
    else:
        spectrum *= np.power(magnitude, exponent.reshape(-1, 1, 1))
    spatial = _fft2(spectrum, inverse=True)
    return np.ascontiguousarray(spatial, dtype=np.complex64)


def _fft2(array: np.ndarray, *, inverse: bool) -> np.ndarray:
    """``scipy.fft`` on the last two axes. Drop flags this SciPy build rejects."""
    global _FFT_WORKERS
    transform = scipy_fft.ifft2 if inverse else scipy_fft.fft2
    data = array if array.flags.writeable else np.array(array, copy=True)
    if _FFT_WORKERS is None:
        return transform(data, axes=(-2, -1))
    try:
        return transform(data, axes=(-2, -1), workers=_FFT_WORKERS, overwrite_x=True)
    except TypeError:
        _FFT_WORKERS = None
        return transform(data, axes=(-2, -1))
    except ValueError:
        return transform(np.array(data, copy=True), axes=(-2, -1), workers=_FFT_WORKERS)


def _accumulate(
    acc: np.ndarray,
    count: np.ndarray,
    spatial: np.ndarray,
    weight: np.ndarray,
    y_origin: int,
    step: int,
) -> None:
    kernel = _numba_accumulate()
    if kernel is not None:
        win = int(spatial.shape[-1])
        group = max(1, (win + int(step) - 1) // int(step))
        n_rows = int(spatial.shape[0])
        try:
            for phase in range(group):
                if phase >= n_rows:
                    break
                kernel(
                    acc,
                    count,
                    spatial,
                    weight,
                    int(y_origin),
                    int(step),
                    int(phase),
                    int(group),
                )
        except Exception:
            _disable_numba()
        else:
            return
    _accumulate_python(acc, count, spatial, weight, y_origin, step)


def _accumulate_python(
    acc: np.ndarray,
    count: np.ndarray,
    spatial: np.ndarray,
    weight: np.ndarray,
    y_origin: int,
    step: int,
) -> None:
    n_rows, n_cols, win, _win_x = spatial.shape
    for i in range(n_rows):
        y = y_origin + i * step
        row = spatial[i]
        for j in range(n_cols):
            x = j * step
            acc[y : y + win, x : x + win] += row[j]
            count[y : y + win, x : x + win] += weight


def _disable_numba() -> None:
    global _NUMBA_ACCUM, _NUMBA_TRIED
    _NUMBA_ACCUM = None
    _NUMBA_TRIED = True


def _numba_accumulate():
    """Compile one non-overlapping phase of the overlap-add, or return None.

    ``prange`` only accepts a constant step, so the phase stride is applied
    inside the loop: row ``phase + k * group``. Those rows do not overlap
    when ``group * step >= window``.
    """
    global _NUMBA_ACCUM, _NUMBA_TRIED
    if _NUMBA_TRIED:
        return _NUMBA_ACCUM
    _NUMBA_TRIED = True
    try:
        from numba import njit, prange

        @njit(cache=True, parallel=True)
        def _kernel(acc, count, spatial, weight, y_origin, step, phase, group):
            n_rows, n_cols, win, _win_x = spatial.shape
            n_this = (n_rows - phase + group - 1) // group
            for k in prange(n_this):
                i = phase + k * group
                y = y_origin + i * step
                for j in range(n_cols):
                    x = j * step
                    for yy in range(win):
                        row = y + yy
                        for xx in range(win):
                            acc[row, x + xx] += spatial[i, j, yy, xx]
                            count[row, x + xx] += weight[yy, xx]
    except Exception:
        _NUMBA_ACCUM = None
        return None
    _NUMBA_ACCUM = _kernel
    return _NUMBA_ACCUM


def _goldstein_torch(
    source: np.ndarray,
    alpha: float,
    win: int,
    overlap: float,
    spectrum_smooth: int,
) -> np.ndarray:
    """Same patch filter as :func:`_goldstein_numpy`, with the FFT on PyTorch."""
    import torch

    device_name = os.environ.get("AI_INSAR_FILTER_DEVICE", "cpu")
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    height, width = source.shape
    step = max(1, int(round(win * (1.0 - overlap))))
    pad_y, _n_y = _pad_and_count(height, win, step)
    pad_x, _n_x = _pad_and_count(width, win, step)
    padded = source
    if pad_y or pad_x:
        padded = np.pad(source, ((0, pad_y), (0, pad_x)), mode="edge")
    view = sliding_window_view(padded, (win, win))[::step, ::step]
    n_rows = int(view.shape[0])
    n_cols = int(view.shape[1])
    weight = _triangular_weight(win)
    acc = np.zeros(padded.shape, dtype=np.complex64)
    count = np.zeros(padded.shape, dtype=np.float32)
    row_bytes = max(1, n_cols * win * win * 8)
    rows_per = max(1, _PATCH_BYTES // row_bytes)
    for row0 in range(0, n_rows, rows_per):
        row1 = min(n_rows, row0 + rows_per)
        patches = np.array(view[row0:row1], dtype=np.complex64, copy=True).reshape(-1, win, win)
        spatial = _filter_patch_batch_torch(patches, alpha, spectrum_smooth, device)
        spatial *= weight
        spatial = np.ascontiguousarray(spatial.reshape(row1 - row0, n_cols, win, win))
        _accumulate(acc, count, spatial, weight, row0 * step, step)
    covered_acc = acc[:height, :width]
    covered_count = count[:height, :width]
    out = source.copy()
    covered = covered_count > 1e-6
    out[covered] = covered_acc[covered] / covered_count[covered]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return out


def _filter_patch_batch_torch(patches: np.ndarray, alpha: float, spectrum_smooth: int, device) -> np.ndarray:
    import torch

    spectrum = torch.from_numpy(np.ascontiguousarray(patches)).to(device)
    spectrum = torch.fft.fft2(spectrum)
    magnitude = spectrum.abs().clamp_min(1e-10)
    if spectrum_smooth > 1:
        host = magnitude.detach().cpu().numpy()
        host = uniform_filter1d(host, size=spectrum_smooth, axis=-1, mode="nearest")
        host = uniform_filter1d(host, size=spectrum_smooth, axis=-2, mode="nearest")
        magnitude = torch.from_numpy(np.ascontiguousarray(host)).to(device)
    if alpha != 0.0:
        spectrum = spectrum * magnitude.pow(float(alpha))
    spatial = torch.fft.ifft2(spectrum).cpu().numpy()
    return np.ascontiguousarray(spatial, dtype=np.complex64)
