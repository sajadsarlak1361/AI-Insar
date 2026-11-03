"""Interferogram formation and the small-baseline date network."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
) -> np.ndarray:
    """Form interferograms as ``slc[later] * conj(slc[earlier])``.

    Parameters
    ----------
    slc:
        Complex stack shaped ``(n_dates, height, width)``. Date index must
        increase with time; ``later`` is the larger index.
    pairs:
        ``(earlier, later)`` indexes with ``later > earlier``.

    Returns
    -------
    complex64 array shaped ``(n_pairs, height, width)``.
    """
    data = np.asarray(slc)
    if data.ndim != 3 or not np.iscomplexobj(data):
        raise ValueError("SLC stack must be complex with shape (dates, height, width)")
    n_dates, height, width = data.shape
    if len(pairs) == 0:
        raise ValueError("pairs is empty")
    out = np.empty((len(pairs), height, width), dtype=np.complex64)
    for index, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"pair {pair!r} must be (earlier, later)")
        earlier, later = int(pair[0]), int(pair[1])
        if not (0 <= earlier < later < n_dates):
            raise ValueError(
                f"pair {(earlier, later)} is outside 0..{n_dates - 1} or is not "
                "ordered as (earlier, later). Interferograms are later * conj(earlier)."
            )
        out[index] = data[later] * np.conj(data[earlier])
    return out
