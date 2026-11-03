"""Interferogram formation and the small-baseline date network.

Interferograms are ``slc[later] * conj(slc[earlier])``. The multiply is a
chunked NumPy product so a long pair list does not allocate three full
cubes at once. When Numba is installed (``pip install "ai-insar[fast]"``)
the same product is a compiled loop. That loop is the same complex product
used by the MIT educational examples in InSARdev/edu (``NISAR/nisar_numpy.py``
and ``Sentinel-1/s1.py``). Those files are not copied here.

When ``insardev`` imports (``pip install "ai-insar[insardev]"``), that call
is tried first: ``Stack.pairs().interferogram()``. A returned complex cube
is the result, including when its values differ from NumPy. NumPy/Numba
runs when the package is missing, the call fails, or
``AI_INSAR_BACKEND=numpy``. The SAL sources are not copied here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai_insar.insardev_optional import try_insardev_interferogram, use_insardev_backend

# One indexed date-cube kept under this size. The conjugate reuses that copy.
_CHUNK_BYTES = 32 * 1024 * 1024


def date_pairs(n_dates: int, max_span: int) -> list[tuple[int, int]]:
    """Return ``(earlier, later)`` index pairs with ``1 <= later - earlier <= max_span``.

    The sequential pairs (span 1) keep the SBAS network connected. Extra
    pairs up to ``max_span`` are the short-baseline interferograms.
    """
    if n_dates < 2:
        raise ValueError("need at least two SLC dates to form an interferogram")
    if max_span < 1:
        raise ValueError("max_span must be at least 1")
    span = min(int(max_span), n_dates - 1)
    pairs: list[tuple[int, int]] = []
    for earlier in range(n_dates):
        last = min(n_dates - 1, earlier + span)
        for later in range(earlier + 1, last + 1):
            pairs.append((earlier, later))
    return pairs


def form_interferograms(
    slc: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    *,
    backend: str | None = None,
) -> np.ndarray:
    """Form interferograms as ``slc[later] * conj(slc[earlier])``.

    Parameters
    ----------
    slc:
        Complex stack shaped ``(n_dates, height, width)``. Date index must
        increase with time; ``later`` is the larger index.
    pairs:
        ``(earlier, later)`` indexes with ``later > earlier``.
    backend:
        ``auto`` (the default, also ``AI_INSAR_BACKEND``) uses an installed
        ``insardev`` single-look interferogram when that call returns a
        complex cube. A difference from the NumPy product does not drop it.
        ``numpy`` skips the call. The in-tree product is used when the
        package is missing or the call fails.

    Returns
    -------
    complex64 array shaped ``(n_pairs, height, width)``.

    ``form_interferograms.last_backend`` is ``insardev``, ``numba``, or
    ``numpy`` after a successful call.
    """
    data = np.asarray(slc)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("SLC stack must be complex with shape (dates, height, width)")
    n_dates, height, width = data.shape
    earlier, later = _pair_index(n_dates, pairs)
    cube = np.ascontiguousarray(data, dtype=np.complex64)
    if use_insardev_backend(backend):
        alt = try_insardev_interferogram(cube, pairs)
        if alt is not None:
            form_interferograms.last_backend = "insardev"
            return np.ascontiguousarray(alt, dtype=np.complex64)
    formed, used = _form_mit(cube, earlier, later, height, width)
    form_interferograms.last_backend = used
    return formed


form_interferograms.last_backend = "numpy"


def _form_mit(
    cube: np.ndarray,
    earlier: np.ndarray,
    later: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, str]:
    """Numba loop when it is installed, otherwise the chunked NumPy product."""
    kernel = _numba_form()
    if kernel is not None:
        try:
            formed = kernel(cube, earlier, later)
        except Exception:
            _disable_numba()
        else:
            if formed.shape == (earlier.shape[0], height, width) and formed.dtype == np.complex64:
                return formed, "numba"
            _disable_numba()
    return _form_vectorized(cube, earlier, later), "numpy"


def _pair_index(n_dates: int, pairs: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if len(pairs) == 0:
        raise ValueError("pairs is empty")
    earlier = np.empty(len(pairs), dtype=np.int64)
    later = np.empty(len(pairs), dtype=np.int64)
    for index, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"pair {pair!r} must be (earlier, later)")
        early, late = int(pair[0]), int(pair[1])
        if not (0 <= early < late < n_dates):
            raise ValueError(
                f"pair {(early, late)} is outside 0..{n_dates - 1} or is not "
                "ordered as (earlier, later). Interferograms are later * conj(earlier)."
            )
        earlier[index] = early
        later[index] = late
    return earlier, later


def _chunk_pairs(n_pairs: int, height: int, width: int) -> int:
    pair_bytes = max(1, int(height) * int(width) * 8)
    return max(1, min(int(n_pairs), _CHUNK_BYTES // pair_bytes))


def _form_vectorized(data: np.ndarray, earlier: np.ndarray, later: np.ndarray) -> np.ndarray:
    """``later * conj(earlier)`` in vectorized chunks. ``data`` is complex64."""
    n_pairs = int(earlier.shape[0])
    height, width = int(data.shape[1]), int(data.shape[2])
    out = np.empty((n_pairs, height, width), dtype=np.complex64)
    step = _chunk_pairs(n_pairs, height, width)
    for start in range(0, n_pairs, step):
        stop = min(n_pairs, start + step)
        late = data[later[start:stop]]
        early = np.array(data[earlier[start:stop]], dtype=np.complex64, copy=True)
        np.conj(early, out=early)
        np.multiply(late, early, out=out[start:stop])
    return out


_NUMBA_FORM = None
_NUMBA_TRIED = False


def _disable_numba() -> None:
    global _NUMBA_FORM, _NUMBA_TRIED
    _NUMBA_FORM = None
    _NUMBA_TRIED = True


def _numba_form():
    """Compile ``later * conj(earlier)`` once, or return None when Numba is absent."""
    global _NUMBA_FORM, _NUMBA_TRIED
    if _NUMBA_TRIED:
        return _NUMBA_FORM
    _NUMBA_TRIED = True
    try:
        from numba import njit, prange

        @njit(cache=True, parallel=True)
        def _kernel(data, earlier, later):
            n_pairs = earlier.shape[0]
            height = data.shape[1]
            width = data.shape[2]
            out = np.empty((n_pairs, height, width), dtype=np.complex64)
            for pair in prange(n_pairs):
                early = earlier[pair]
                late = later[pair]
                for y in range(height):
                    for x in range(width):
                        sample = data[early, y, x]
                        out[pair, y, x] = data[late, y, x] * (sample.real - 1j * sample.imag)
            return out
    except Exception:
        _NUMBA_FORM = None
        return None
    _NUMBA_FORM = _kernel
    return _NUMBA_FORM
