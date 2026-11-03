"""Array-to-RGB rendering for the desktop viewer. No matplotlib dependency."""

from __future__ import annotations

import numpy as np


def colorize(
    values: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    invalid: np.ndarray | None = None,
) -> np.ndarray:
    """Return an ``uint8`` RGB image. Non-finite samples are painted dark."""
    field = np.asarray(values, dtype=np.float64)
    span = float(vmax) - float(vmin)
    if span <= 0:
        normalized = np.zeros(field.shape, dtype=np.float64)
    else:
        normalized = (field - float(vmin)) / span
    # np.clip leaves NaN in place, and a later uint8 cast warns on those pixels.
    finite = np.isfinite(normalized)
    normalized = np.where(finite, np.clip(normalized, 0.0, 1.0), 0.0)
    rgb = _apply(cmap, normalized)
    bad = ~np.isfinite(field)
    if invalid is not None:
        bad = bad | np.asarray(invalid, dtype=bool)
    rgb[bad] = np.array([28, 34, 44], dtype=np.uint8)
    return rgb


def amplitude_rgb(amplitude: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Percentile-stretched grayscale amplitude."""
    field = np.asarray(amplitude, dtype=np.float64)
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return colorize(field, vmin=0.0, vmax=1.0, cmap="gray"), 0.0, 1.0
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return colorize(field, vmin=vmin, vmax=vmax, cmap="gray"), vmin, vmax


def symmetric_limits(values: np.ndarray, percentile: float = 98.0) -> tuple[float, float]:
    """Limits of the form ``(-peak, peak)`` from a percentile of the absolute values."""
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    peak = float(np.percentile(finite, percentile))
    if peak <= 0:
        peak = float(finite.max()) if finite.size else 1.0
    if peak <= 0:
        peak = 1.0
    return -peak, peak


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 140, 40)) -> np.ndarray:
    """Blend ``color`` onto ``base_rgb`` where ``mask`` is true."""
    out = np.array(base_rgb, dtype=np.float64, copy=True)
    selected = np.asarray(mask, dtype=bool)
    paint = np.array(color, dtype=np.float64)
    out[selected] = 0.45 * out[selected] + 0.55 * paint
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply(name: str, normalized: np.ndarray) -> np.ndarray:
    if name == "phase":
        return _hsv(normalized, saturation=0.72, value=0.95)
    if name == "velocity":
        return _diverging(normalized)
    if name == "coherence":
        return _stops(
            normalized,
            (
                (0.00, (12, 8, 40)),
                (0.25, (60, 30, 120)),
                (0.55, (30, 130, 140)),
                (0.80, (140, 200, 80)),
                (1.00, (245, 230, 90)),
            ),
        )
    if name == "mask":
        return _stops(normalized, ((0.0, (30, 36, 48)), (1.0, (255, 150, 40))))
    return _stops(normalized, ((0.0, (8, 10, 16)), (1.0, (235, 238, 244))))


def _hsv(hue: np.ndarray, saturation: float, value: float) -> np.ndarray:
    """Cyclic map. Hue 0 and hue 1 are the same red, so wrapped phase joins up."""
    h = np.mod(hue, 1.0)
    i = np.floor(h * 6.0).astype(np.int64) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = np.full(h.shape, value * (1.0 - saturation), dtype=np.float64)
    q = value * (1.0 - f * saturation)
    t = value * (1.0 - (1.0 - f) * saturation)
    v = np.full(h.shape, value, dtype=np.float64)
    channels = (
        np.stack((v, t, p), axis=-1),
        np.stack((q, v, p), axis=-1),
        np.stack((p, v, t), axis=-1),
        np.stack((p, q, v), axis=-1),
        np.stack((t, p, v), axis=-1),
        np.stack((v, p, q), axis=-1),
    )
    rgb = np.zeros(h.shape + (3,), dtype=np.float64)
    for index, plane in enumerate(channels):
        rgb[i == index] = plane[i == index]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _diverging(normalized: np.ndarray) -> np.ndarray:
    """Blue (negative LOS, away from the radar) through white to red (toward the radar)."""
    return _stops(
        normalized,
        (
            (0.00, (32, 92, 196)),
            (0.50, (244, 244, 246)),
            (1.00, (196, 52, 48)),
        ),
    )


def _stops(normalized: np.ndarray, stops: tuple[tuple[float, tuple[int, int, int]], ...]) -> np.ndarray:
    positions = np.array([item[0] for item in stops], dtype=np.float64)
    colors = np.array([item[1] for item in stops], dtype=np.float64)
    flat = normalized.ravel()
    right = np.searchsorted(positions, flat, side="right")
    right = np.clip(right, 1, len(positions) - 1)
    left = right - 1
    span = positions[right] - positions[left]
    span = np.where(span <= 0, 1.0, span)
    mix = ((flat - positions[left]) / span)[:, None]
    rgb = colors[left] + (colors[right] - colors[left]) * mix
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(normalized.shape + (3,))
