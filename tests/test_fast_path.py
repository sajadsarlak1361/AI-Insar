"""Fast in-tree Goldstein, coherence, and separable unwrap on synthetic data.

These checks do not import ``insardev``. Timings for a larger stack are
printed by ``python scripts/bench_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_insar.pipeline.coherence import coherence_map, coherence_stack
from ai_insar.pipeline.filtering import goldstein_filter
from ai_insar.pipeline.interferogram import form_interferograms
from ai_insar.pipeline.synthetic import generate_subsidence_stack
from ai_insar.pipeline.unwrap import unwrap_separable


def test_batched_goldstein_reconstructs_at_alpha_zero_and_keeps_constant_phase():
    rng = np.random.default_rng(4)
    ifg = (rng.normal(size=(48, 40)) + 1j * rng.normal(size=(48, 40))).astype(np.complex64)
    rebuilt = goldstein_filter(ifg, alpha=0.0, window=16, overlap=0.5, backend="numpy")
    assert goldstein_filter.last_backend == "numpy"
    assert np.allclose(rebuilt, ifg, atol=1e-5)
    odd = goldstein_filter(ifg, alpha=0.0, window=15, overlap=0.25, backend="numpy")
    assert np.allclose(odd, ifg, atol=1e-5)
    constant = np.full((36, 36), np.exp(1j * 0.4), dtype=np.complex64)
    filtered = goldstein_filter(constant, alpha=0.7, window=16, backend="numpy")
    assert np.allclose(np.angle(filtered), 0.4, atol=1e-5)


def test_goldstein_rejects_a_real_image_and_an_unknown_backend():
    real = np.ones((16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        goldstein_filter(real)
    complex_image = np.ones((16, 16), dtype=np.complex64)
    with pytest.raises(ValueError):
        goldstein_filter(complex_image, backend="snap")


def test_coherence_stack_matches_the_single_pair_map():
    rng = np.random.default_rng(5)
    slc = (rng.normal(size=(3, 24, 22)) + 1j * rng.normal(size=(3, 24, 22))).astype(np.complex64)
    pairs = [(0, 1), (1, 2), (0, 2)]
    stacked = coherence_stack(slc, pairs, window=5)
    for index, (earlier, later) in enumerate(pairs):
        direct = coherence_map(slc[earlier], slc[later], 5)
        assert np.allclose(stacked[index], direct, atol=1e-6)


def test_fast_stack_forms_filters_and_unwraps_a_smooth_bowl():
    stack = generate_subsidence_stack(
        n_dates=4,
        height=96,
        width=96,
        spacing_days=36,
        rate_m_yr=0.15,
        sigma_px=12,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=11,
    )
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]
    formed = form_interferograms(stack.data, pairs, backend="numpy")
    assert formed.shape == (4, 96, 96)
    assert formed.dtype == np.complex64
    filtered = goldstein_filter(formed[0], alpha=0.5, window=16, backend="numpy")
    assert filtered.shape == formed[0].shape
    assert np.isfinite(filtered).all()
    phase = np.angle(filtered)
    coherence = coherence_map(stack.data[0], stack.data[1], window=5)
    unwrapped = unwrap_separable(phase, coherence, sigma=1.5, min_coherence=0.0)
    true = np.angle(formed[0])
    # The separable path tracks the smooth filtered phase up to a constant.
    finite = np.isfinite(unwrapped)
    assert float(finite.mean()) > 0.95
    offset = float(np.nanmedian(unwrapped - true))
    residual = np.abs(np.angle(np.exp(1j * (unwrapped - true - offset))))
    assert float(np.nanmedian(residual)) < 0.35


def test_torch_backend_falls_back_or_matches_numpy():
    rng = np.random.default_rng(6)
    ifg = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))).astype(np.complex64)
    numpy_filtered = goldstein_filter(ifg, alpha=0.4, window=16, backend="numpy")
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_filtered = goldstein_filter(ifg, alpha=0.4, window=16, backend="torch")
        assert goldstein_filter.last_backend == "numpy"
        assert np.allclose(torch_filtered, numpy_filtered, atol=1e-5)
        return
    torch_filtered = goldstein_filter(ifg, alpha=0.4, window=16, backend="torch")
    assert goldstein_filter.last_backend == "torch"
    assert np.allclose(np.angle(torch_filtered), np.angle(numpy_filtered), atol=1e-3)
