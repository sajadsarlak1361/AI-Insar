"""Constant-offset alignment and topographic phase. No GMTSAR source."""

from __future__ import annotations

import numpy as np

from ai_insar.gmtsar_flow import FLOWCHART_TEXT, GMTSAR_STAGES
from ai_insar.pipeline.align import align_slc_stack, estimate_amplitude_shift, fft_shift_complex
from ai_insar.pipeline.topo import apply_topographic_phase, topographic_phase_rad


def test_fft_shift_roundtrip_and_amplitude_correlation():
    rng = np.random.default_rng(3)
    reference = (rng.normal(size=(48, 40)) + 1j * rng.normal(size=(48, 40))).astype(np.complex64)
    secondary = fft_shift_complex(reference, 4.0, -3.0)
    shift_y, shift_x = estimate_amplitude_shift(reference, secondary)
    assert abs(shift_y - 4.0) < 0.15
    assert abs(shift_x - (-3.0)) < 0.15
    aligned = fft_shift_complex(secondary, -shift_y, -shift_x)
    assert np.allclose(aligned, reference, atol=1e-4)


def test_align_stack_leaves_the_reference_date_and_repairs_the_other():
    rng = np.random.default_rng(4)
    reference = (rng.normal(size=(32, 36)) + 1j * rng.normal(size=(32, 36))).astype(np.complex64)
    other = fft_shift_complex(reference, -2.0, 5.0)
    stack = np.stack([reference, other], axis=0)
    aligned, shifts = align_slc_stack(stack, reference=0)
    assert shifts[0] == (0.0, 0.0)
    assert abs(shifts[1][0] - (-2.0)) < 0.15
    assert abs(shifts[1][1] - 5.0) < 0.15
    assert np.allclose(aligned[0], reference, atol=1e-6)
    assert np.allclose(aligned[1], reference, atol=1e-4)


def test_topographic_phase_is_zero_without_a_baseline_and_scales_with_height():
    dem = np.array([[0.0, 100.0], [50.0, -20.0]], dtype=np.float64)
    zero = topographic_phase_rad(
        dem,
        wavelength_m=0.055,
        baseline_perp_m=0.0,
        slant_range_m=800_000.0,
        incidence_rad=np.deg2rad(35.0),
    )
    assert np.allclose(zero, 0.0)
    phase = topographic_phase_rad(
        dem,
        wavelength_m=0.055,
        baseline_perp_m=100.0,
        slant_range_m=800_000.0,
        incidence_rad=np.deg2rad(35.0),
    )
    assert phase[0, 1] < 0.0
    assert abs(phase[0, 1] / phase[1, 0] - 100.0 / 50.0) < 1e-6
    interferogram = np.ones((1, 2, 2), dtype=np.complex64)
    removed = apply_topographic_phase(interferogram, phase)
    expected = np.exp(-1j * phase).astype(np.complex64)
    assert np.allclose(removed[0], expected, atol=1e-5)


def test_flowchart_names_the_six_gmtsar_stages():
    labels = [label for _key, label, _role in GMTSAR_STAGES]
    assert labels == [
        "PREPROCESS",
        "ALIGN SLC",
        "MAKE TOPO_RA",
        "MAKE INTERFEROGRAMS",
        "UNWRAP PHASE",
        "GEOCODE",
    ]
    for label in labels:
        assert label in FLOWCHART_TEXT
    assert "GPL" in FLOWCHART_TEXT
    assert "SBAS" in FLOWCHART_TEXT
