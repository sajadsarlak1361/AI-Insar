"""Quality-guided phase unwrapping and a border plane removal.

The growth walk is a Numba loop when ``numba`` is installed
(``pip install "ai-insar[fast]"``) and the same algorithm in Python otherwise.
Both integrate wrapped neighbor differences, so a residue-free field matches
the continuous phase up to one constant.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# Fast unwrap is first: it is the performance default for Run all stages.
UNWRAP_METHODS: tuple[str, ...] = ("separable", "quality")
UNWRAP_LABELS: dict[str, str] = {
    "quality": "Quality-guided",
    "separable": "Fast unwrap",
}
UNWRAP_SUMMARIES: dict[str, str] = {
    "quality": (
        "Quality-guided unwrap grows from the best coherence pixel and adds "
        "wrapped neighbor differences. Numba runs the walk when installed; "
        "otherwise it is a Python heap. Pixels below the coherence floor stay empty."
    ),
    "separable": (
        "Fast unwrap is the default for Run all stages and ai-insar check. "
        "It smooths the complex interferogram with a Gaussian, then uses "
        "NumPy's one-dimensional unwrap along range and azimuth, starting "
        "from the middle of the image. The idea follows the MIT educational "
        "example in InSARdev/edu (NISAR/nisar_numpy.py). It is meant for "
        "smooth fields. Quality-guided is still available when you need the "
        "slower coherence-ordered walk."
    ),
}


def reliability(phase: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    """Unwrap priority: high coherence and a small local phase second difference."""
    wrapped = np.asarray(phase, dtype=np.float64)
    coh = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
    if wrapped.shape != coh.shape:
        raise ValueError("phase and coherence shapes differ")
    dx = np.zeros_like(wrapped)
    dy = np.zeros_like(wrapped)
    dx[:, 1:] = np.angle(np.exp(1j * np.diff(wrapped, axis=1)))
    dy[1:, :] = np.angle(np.exp(1j * np.diff(wrapped, axis=0)))
    ddx = np.zeros_like(wrapped)
    ddy = np.zeros_like(wrapped)
    ddx[:, 1:] = dx[:, 1:] - dx[:, :-1]
    ddy[1:, :] = dy[1:, :] - dy[:-1, :]
    roughness = uniform_filter(ddx**2 + ddy**2, size=3, mode="reflect")
    return coh / (1.0 + roughness)


def unwrap_quality_guided(
    phase: np.ndarray,
    quality: np.ndarray,
    *,
    min_quality: float = 0.0,
) -> np.ndarray:
    """Unwrap a 2-D wrapped phase by growing from the best quality pixel.

    Each newly visited pixel is placed ``wrap(phase - phase_parent)`` away
    from an already unwrapped neighbor. On a residue-free field the result
    matches the true continuous phase up to one global constant. Pixels below
    ``min_quality`` stay NaN and act as barriers so a noisy pixel is not a bridge.
    Numba runs this walk when it is installed; the Python heap is the fallback.
    """
    wrapped = np.asarray(phase, dtype=np.float64)
    score = np.asarray(quality, dtype=np.float64)
    if wrapped.shape != score.shape or wrapped.ndim != 2:
        raise ValueError("phase and quality must be 2-D and the same shape")
    valid = np.isfinite(wrapped) & np.isfinite(score) & (score >= float(min_quality))
    if not np.any(valid):
        return np.full(wrapped.shape, np.nan, dtype=np.float64)
    kernel = _numba_unwrap()
    if kernel is not None:
        return kernel(np.ascontiguousarray(wrapped), np.ascontiguousarray(score), np.ascontiguousarray(valid))
    return _unwrap_python(wrapped, score, valid)


def _unwrap_python(wrapped: np.ndarray, score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height, width = wrapped.shape
    unwrapped = np.full((height, width), np.nan, dtype=np.float64)
    flat = np.where(valid, score, -np.inf)
    seed_y, seed_x = np.unravel_index(int(np.argmax(flat)), (height, width))
    unwrapped[seed_y, seed_x] = wrapped[seed_y, seed_x]
    visited = np.zeros((height, width), dtype=bool)
    visited[seed_y, seed_x] = True
    heap: list[tuple[float, int, int, int, int, int]] = []
    counter = 0

    def push_neighbors(y: int, x: int) -> None:
        nonlocal counter
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = y + dy
            nx = x + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width:
                continue
            if not valid[ny, nx] or visited[ny, nx]:
                continue
            counter += 1
            heapq.heappush(heap, (-float(score[ny, nx]), counter, ny, nx, y, x))

    push_neighbors(int(seed_y), int(seed_x))
    while heap:
        _neg, _order, y, x, parent_y, parent_x = heapq.heappop(heap)
        if visited[y, x]:
            continue
        delta = wrapped[y, x] - wrapped[parent_y, parent_x]
        step = math.atan2(math.sin(delta), math.cos(delta))
        unwrapped[y, x] = unwrapped[parent_y, parent_x] + step
        visited[y, x] = True
        push_neighbors(y, x)
    return unwrapped


_NUMBA_UNWRAP = None
_NUMBA_TRIED = False


def _numba_unwrap():
    """Compile the growth walk once, or return None when Numba is absent."""
    global _NUMBA_UNWRAP, _NUMBA_TRIED
    if _NUMBA_TRIED:
        return _NUMBA_UNWRAP
    _NUMBA_TRIED = True
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def _kernel(phase, quality, valid):
        height, width = phase.shape
        unwrapped = np.empty((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                unwrapped[y, x] = np.nan
        visited = np.zeros((height, width), dtype=np.bool_)
        best = -1.0e300
        seed_y = 0
        seed_x = 0
        found = False
        for y in range(height):
            for x in range(width):
                if valid[y, x] and quality[y, x] >= best:
                    best = quality[y, x]
                    seed_y = y
                    seed_x = x
                    found = True
        if not found:
            return unwrapped
        unwrapped[seed_y, seed_x] = phase[seed_y, seed_x]
        visited[seed_y, seed_x] = True
        capacity = height * width * 4 + 8
        neg_q = np.empty(capacity, dtype=np.float64)
        order = np.empty(capacity, dtype=np.int64)
        hy = np.empty(capacity, dtype=np.int32)
        hx = np.empty(capacity, dtype=np.int32)
        py = np.empty(capacity, dtype=np.int32)
        px = np.empty(capacity, dtype=np.int32)
        size = 0
        counter = 0
        neighbors_y = np.array((-1, 1, 0, 0), dtype=np.int32)
        neighbors_x = np.array((0, 0, -1, 1), dtype=np.int32)

        def push(nq, seq, y, x, parent_y, parent_x):
            nonlocal size
            i = size
            size += 1
            neg_q[i] = nq
            order[i] = seq
            hy[i] = y
            hx[i] = x
            py[i] = parent_y
            px[i] = parent_x
            while i > 0:
                parent = (i - 1) // 2
                if neg_q[i] < neg_q[parent] or (neg_q[i] == neg_q[parent] and order[i] < order[parent]):
                    neg_q[i], neg_q[parent] = neg_q[parent], neg_q[i]
                    order[i], order[parent] = order[parent], order[i]
                    hy[i], hy[parent] = hy[parent], hy[i]
                    hx[i], hx[parent] = hx[parent], hx[i]
                    py[i], py[parent] = py[parent], py[i]
                    px[i], px[parent] = px[parent], px[i]
                    i = parent
                else:
                    break

        def pop():
            nonlocal size
            nq = neg_q[0]
            seq = order[0]
            y = hy[0]
            x = hx[0]
            parent_y = py[0]
            parent_x = px[0]
            size -= 1
            if size > 0:
                neg_q[0] = neg_q[size]
                order[0] = order[size]
                hy[0] = hy[size]
                hx[0] = hx[size]
                py[0] = py[size]
                px[0] = px[size]
                i = 0
                while True:
                    left = 2 * i + 1
                    right = left + 1
                    smallest = i
                    if left < size and (
                        neg_q[left] < neg_q[smallest]
                        or (neg_q[left] == neg_q[smallest] and order[left] < order[smallest])
                    ):
                        smallest = left
                    if right < size and (
                        neg_q[right] < neg_q[smallest]
                        or (neg_q[right] == neg_q[smallest] and order[right] < order[smallest])
                    ):
                        smallest = right
                    if smallest == i:
                        break
                    neg_q[i], neg_q[smallest] = neg_q[smallest], neg_q[i]
                    order[i], order[smallest] = order[smallest], order[i]
                    hy[i], hy[smallest] = hy[smallest], hy[i]
                    hx[i], hx[smallest] = hx[smallest], hx[i]
                    py[i], py[smallest] = py[smallest], py[i]
                    px[i], px[smallest] = px[smallest], px[i]
                    i = smallest
            return nq, seq, y, x, parent_y, parent_x

        def consider(y, x):
            nonlocal counter
            for step in range(4):
                ny = y + neighbors_y[step]
                nx = x + neighbors_x[step]
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if not valid[ny, nx] or visited[ny, nx]:
                    continue
                counter += 1
                push(-quality[ny, nx], counter, ny, nx, y, x)

        consider(seed_y, seed_x)
        while size > 0:
            _nq, _seq, y, x, parent_y, parent_x = pop()
            if visited[y, x]:
                continue
            delta = phase[y, x] - phase[parent_y, parent_x]
            step = math.atan2(math.sin(delta), math.cos(delta))
            unwrapped[y, x] = unwrapped[parent_y, parent_x] + step
            visited[y, x] = True
            consider(y, x)
        return unwrapped

    _NUMBA_UNWRAP = _kernel
    return _NUMBA_UNWRAP


def unwrap_separable(
    phase: np.ndarray,
    coherence: np.ndarray,
    *,
    sigma: float = 2.0,
    min_coherence: float = 0.2,
) -> np.ndarray:
    """Gaussian-smoothed separable unwrap.

    Smooth the coherence-weighted complex phase, then call ``numpy.unwrap``
    along range and along azimuth. The image is split at mid-azimuth and each
    half is unwrapped outward so a bad edge does not cross the whole frame.
    This follows the structure of the MIT-licensed educational unwrapper in
    InSARdev/edu ``NISAR/nisar_numpy.py`` and is reimplemented here for stacks
    that are already coregistered. It does not copy that file.
    """
    wrapped = np.asarray(phase, dtype=np.float64)
    coh = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
    if wrapped.shape != coh.shape or wrapped.ndim != 2:
        raise ValueError("phase and coherence must be 2-D and the same shape")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    carrier = np.exp(1j * wrapped) * coh
    smooth = gaussian_filter(carrier.real, sigma) + 1j * gaussian_filter(carrier.imag, sigma)
    angle = np.angle(smooth)
    mid = angle.shape[0] // 2
    if mid == 0 or mid == angle.shape[0]:
        field = np.unwrap(np.unwrap(angle, axis=1), axis=0)
    else:
        bottom = np.unwrap(np.unwrap(angle[mid:], axis=1), axis=0)
        top = np.unwrap(np.unwrap(angle[:mid][::-1], axis=1), axis=0)[::-1]
        shift = float(np.nanmedian(bottom[0]) - np.nanmedian(top[-1]))
        top = top + shift
        field = np.concatenate([top, bottom], axis=0)
    field = np.where(coh >= float(min_coherence), field, np.nan)
    return field.astype(np.float64)


def residue_count(phase: np.ndarray) -> int:
    """Count 2π residues on the wrapped phase (absolute charge, summed)."""
    wrapped = np.asarray(phase, dtype=np.float64)
    east = np.angle(np.exp(1j * (wrapped[:-1, 1:] - wrapped[:-1, :-1])))
    south = np.angle(np.exp(1j * (wrapped[1:, 1:] - wrapped[:-1, 1:])))
    west = np.angle(np.exp(1j * (wrapped[1:, :-1] - wrapped[1:, 1:])))
    north = np.angle(np.exp(1j * (wrapped[:-1, :-1] - wrapped[1:, :-1])))
    charge = np.rint((east + south + west + north) / (2.0 * np.pi))
    return int(np.sum(np.abs(charge)))


def remove_ramp(
    phase: np.ndarray,
    weight: np.ndarray,
    *,
    border_px: int = 4,
    min_weight: float = 0.3,
) -> np.ndarray:
    """Subtract a plane fit on the border so the stable frame is the phase reference.

    The fit uses only a frame of pixels. A centered deformation bowl is nearly
    invisible there, so the plane tracks orbit ramps and the piston rather than
    the bowl itself. The returned array is NaN wherever ``phase`` is NaN.
    """
    field = np.asarray(phase, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    if field.shape != weights.shape or field.ndim != 2:
        raise ValueError("phase and weight must be 2-D and the same shape")
    height, width = field.shape
    frame = max(1, min(int(border_px), height // 4, width // 4))
    border = np.zeros((height, width), dtype=bool)
    border[:frame, :] = True
    border[-frame:, :] = True
    border[:, :frame] = True
    border[:, -frame:] = True
    good = border & np.isfinite(field) & np.isfinite(weights) & (weights >= float(min_weight))
    if int(good.sum()) < 8:
        reference = np.nanmedian(field)
        if not np.isfinite(reference):
            return field.copy()
        out = field - reference
        out[~np.isfinite(field)] = np.nan
        return out

    ys, xs = np.nonzero(good)
    design = np.column_stack(
        [np.ones(xs.shape[0], dtype=np.float64), xs.astype(np.float64), ys.astype(np.float64)]
    )
    sample_w = np.clip(weights[ys, xs], 0.05, None)
    coef, *_ = np.linalg.lstsq(design * sample_w[:, None], field[ys, xs] * sample_w, rcond=None)
    yy, xx = np.indices((height, width))
    ramp = coef[0] + coef[1] * xx + coef[2] * yy
    out = field - ramp
    out[~np.isfinite(field)] = np.nan
    return out
