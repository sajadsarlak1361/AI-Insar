#!/usr/bin/env python3
"""Time the in-tree fast path on a synthetic stack.

No ``insardev`` install is required. Numba is used when it is importable
(``pip install "ai-insar[fast]"``). Goldstein stays on the SciPy FFT unless
``AI_INSAR_FILTER=torch``.

    python scripts/bench_pipeline.py
"""

from __future__ import annotations

import time

import numpy as np

from ai_insar.pipeline.coherence import coherence_stack
from ai_insar.pipeline.engine import (
    PipelineConfig,
    Workspace,
    compute_coherence,
    compute_filter,
    compute_interferograms,
    compute_unwrap,
)
from ai_insar.pipeline.filtering import goldstein_filter
from ai_insar.pipeline.interferogram import date_pairs, form_interferograms
from ai_insar.pipeline.synthetic import generate_subsidence_stack
from ai_insar.pipeline.unwrap import unwrap_quality_guided, unwrap_separable


def _seconds(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    rng = np.random.default_rng(0)
    image = (rng.normal(size=(1024, 1024)) + 1j * rng.normal(size=(1024, 1024))).astype(np.complex64)
    goldstein_filter(image[:64, :64], alpha=0.5, window=32, backend="numpy")
    goldstein_s = _seconds(lambda: goldstein_filter(image, alpha=0.5, window=32, backend="numpy"))

    stack = generate_subsidence_stack(
        n_dates=6,
        height=512,
        width=512,
        spacing_days=36,
        rate_m_yr=0.2,
        sigma_px=40,
        temporal_coherence=0.9,
        ramp_rad=0.2,
        seed=1,
    )
    pairs = date_pairs(stack.n_dates, 3)
    form_interferograms(stack.data, pairs[:1], backend="numpy")
    ifg_s = _seconds(lambda: form_interferograms(stack.data, pairs, backend="numpy"))
    formed = form_interferograms(stack.data, pairs, backend="numpy")
    coherence_stack(stack.data, pairs[:1], 5)
    coh_s = _seconds(lambda: coherence_stack(stack.data, pairs, 5))
    goldstein_filter(formed[0, :64, :64], alpha=0.5, window=32, backend="numpy")
    stack_filter_s = _seconds(
        lambda: [goldstein_filter(formed[index], alpha=0.5, window=32, backend="numpy") for index in range(formed.shape[0])]
    )

    phase = np.angle(formed[0])
    coherence = np.clip(np.abs(formed[0]) / (np.abs(formed[0]).max() + 1e-6), 0, 1)
    unwrap_separable(phase[:64, :64], coherence[:64, :64], sigma=2.0, min_coherence=0.0)
    sep_s = _seconds(lambda: unwrap_separable(phase, coherence, sigma=2.0, min_coherence=0.0))
    quality = np.ones_like(phase)
    unwrap_quality_guided(phase[:48, :48], quality[:48, :48])
    qual_s = _seconds(lambda: unwrap_quality_guided(phase, quality))

    workspace = Workspace(stack=stack, config=PipelineConfig(unwrap_method="separable"))
    pipeline_s = _seconds(
        lambda: (
            compute_interferograms(workspace),
            compute_filter(workspace),
            compute_coherence(workspace),
            compute_unwrap(workspace),
        )
    )
    print(f"Goldstein 1024x1024 batched FFT: {goldstein_s:.3f}s ({goldstein_filter.last_backend})")
    print(f"Interferograms {formed.shape} : {ifg_s:.3f}s ({form_interferograms.last_backend})")
    print(f"Coherence stack {len(pairs)} pairs of 512x512: {coh_s:.3f}s")
    print(f"Goldstein stack {formed.shape}: {stack_filter_s:.3f}s")
    print(f"Fast unwrap one 512x512: {sep_s:.3f}s")
    print(f"Quality-guided unwrap one 512x512: {qual_s:.3f}s")
    print(f"Pipeline ifg+filter+coherence+fast unwrap on {stack.data.shape}: {pipeline_s:.3f}s")


if __name__ == "__main__":
    main()
