"""Quality-guided phase unwrapping and a border plane removal."""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import uniform_filter


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
    """
    wrapped = np.asarray(phase, dtype=np.float64)
    score = np.asarray(quality, dtype=np.float64)
    if wrapped.shape != score.shape or wrapped.ndim != 2:
        raise ValueError("phase and quality must be 2-D and the same shape")
    height, width = wrapped.shape
    valid = np.isfinite(wrapped) & np.isfinite(score) & (score >= float(min_quality))
    unwrapped = np.full((height, width), np.nan, dtype=np.float64)
    if not np.any(valid):
        return unwrapped

    flat = np.where(valid, score, -np.inf)
    seed_y, seed_x = np.unravel_index(int(np.argmax(flat)), (height, width))
    unwrapped[seed_y, seed_x] = wrapped[seed_y, seed_x]
    visited = np.zeros((height, width), dtype=bool)
    visited[seed_y, seed_x] = True

    # (-quality, insertion order, y, x, parent_y, parent_x)
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
