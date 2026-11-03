"""Core pipeline tests. These do not open a window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_insar import __version__
from ai_insar.check import recovery_metrics, run_self_check
from ai_insar.conventions import C_BAND_WAVELENGTH_M, los_from_phase, phase_from_los
from ai_insar.io_stack import load_slc, save_products, save_slc, validate_slc
from ai_insar.packaging import make_windows_zip
from ai_insar.pipeline.coherence import coherence_map
from ai_insar.pipeline.detect import detect_displacement
from ai_insar.pipeline.engine import PipelineConfig, PipelineError, Workspace, run_all
from ai_insar.pipeline.filtering import goldstein_filter
from ai_insar.pipeline.interferogram import date_pairs, form_interferograms
from ai_insar.pipeline.sbas import design_matrix, invert_sbas
from ai_insar.pipeline.synthetic import generate_subsidence_stack
from ai_insar.pipeline.unwrap import remove_ramp, residue_count, unwrap_quality_guided

ROOT = Path(__file__).resolve().parents[1]


def test_version_and_project_name():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "ai-insar"' in text
    assert f'version = "{__version__}"' in text


def test_phase_los_roundtrip_and_subsidence_sign():
    los = np.array([0.0, -0.01, -0.02])
    phase = phase_from_los(los, C_BAND_WAVELENGTH_M)
    assert np.allclose(los_from_phase(phase, C_BAND_WAVELENGTH_M), los)
    assert phase[1] - phase[0] > 0


def test_slc_must_be_complex64_stack():
    stack = generate_subsidence_stack(n_dates=3, height=16, width=16, temporal_coherence=1.0, ramp_rad=0.0)
    assert stack.data.dtype == np.complex64
    assert stack.data.shape == (3, 16, 16)
    with pytest.raises(ValueError, match="complex"):
        validate_slc(np.zeros((2, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="dates, height, width"):
        validate_slc(np.zeros((2, 8), dtype=np.complex64))


def test_interferogram_is_later_times_conj_earlier():
    rng = np.random.default_rng(4)
    slc = (rng.normal(size=(4, 5, 6)) + 1j * rng.normal(size=(4, 5, 6))).astype(np.complex64)
    pairs = [(0, 1), (1, 3), (0, 2)]
    formed = form_interferograms(slc, pairs)
    for index, (earlier, later) in enumerate(pairs):
        expected = slc[later] * np.conj(slc[earlier])
        assert formed[index].dtype == np.complex64
        assert np.allclose(formed[index], expected)
    with pytest.raises(ValueError, match="later"):
        form_interferograms(slc, [(2, 1)])


def test_noise_free_subsidence_phase_matches_los_model():
    stack = generate_subsidence_stack(
        n_dates=4,
        height=36,
        width=36,
        spacing_days=36,
        rate_m_yr=0.22,
        sigma_px=7,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=2,
    )
    true_phase = phase_from_los(stack.truth_los_m, stack.wavelength_m)
    row, col = np.unravel_index(int(np.argmax(np.abs(stack.truth_velocity_m_yr))), stack.truth_velocity_m_yr.shape)
    assert true_phase[1, row, col] - true_phase[0, row, col] > 0
    ifg = stack.data[1] * np.conj(stack.data[0])
    predicted = true_phase[1] - true_phase[0]
    strong = np.abs(ifg) > np.percentile(np.abs(ifg), 30)
    wrapped_error = np.abs(np.angle(np.exp(1j * (np.angle(ifg) - predicted))))
    assert float(np.max(wrapped_error[strong])) < 1e-3


def test_date_network_is_connected_and_full_rank():
    pairs = date_pairs(8, 3)
    assert (0, 1) in pairs and (6, 7) in pairs
    assert all(later > earlier for earlier, later in pairs)
    matrix = design_matrix(pairs, 8)
    assert matrix.shape == (len(pairs), 7)
    assert np.linalg.matrix_rank(matrix) == 7
    # Span-1 network is square and invertible.
    sequential = design_matrix(date_pairs(5, 1), 5)
    assert sequential.shape == (4, 4)
    assert np.linalg.matrix_rank(sequential) == 4


def test_sbas_recovers_known_displacement_exactly():
    stack = generate_subsidence_stack(
        n_dates=6,
        height=24,
        width=24,
        spacing_days=36,
        rate_m_yr=0.18,
        sigma_px=6,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=5,
    )
    phase = phase_from_los(stack.truth_los_m, stack.wavelength_m)
    pairs = date_pairs(stack.n_dates, 3)
    observations = np.stack([phase[later] - phase[earlier] for earlier, later in pairs], axis=0)
    los, velocity = invert_sbas(observations, pairs, stack.times_year(), stack.wavelength_m)
    assert np.allclose(los, stack.truth_los_m, atol=1e-8)
    assert np.allclose(velocity, stack.truth_velocity_m_yr, atol=1e-8)
    assert float(velocity[stack.height // 2, stack.width // 2]) < 0


def test_goldstein_alpha_zero_reconstructs_and_preserves_constant_phase():
    rng = np.random.default_rng(0)
    ifg = (rng.normal(size=(40, 36)) + 1j * rng.normal(size=(40, 36))).astype(np.complex128)
    rebuilt = goldstein_filter(ifg, alpha=0.0, window=16)
    assert np.allclose(rebuilt, ifg, atol=1e-5)
    constant = np.full((32, 32), np.exp(1j * 0.4), dtype=np.complex128)
    filtered = goldstein_filter(constant, alpha=0.7, window=16)
    assert np.allclose(np.angle(filtered), 0.4, atol=1e-5)


def test_coherence_of_identical_slc_is_one():
    rng = np.random.default_rng(1)
    slc = (rng.normal(size=(20, 18)) + 1j * rng.normal(size=(20, 18))).astype(np.complex64)
    gamma = coherence_map(slc, slc, window=5)
    assert gamma.shape == slc.shape
    assert float(gamma.min()) > 0.99


def test_unwrap_recovers_multi_fringe_bowl():
    height, width = 80, 80
    yy, xx = np.mgrid[0:height, 0:width]
    true = 18.0 * np.exp(-((yy - 40) ** 2 + (xx - 42) ** 2) / (2 * 12.0**2))
    true = true + 0.15 * (xx - xx.mean()) / width
    wrapped = np.angle(np.exp(1j * true))
    assert residue_count(wrapped) == 0
    quality = np.ones_like(true)
    unwrapped = unwrap_quality_guided(wrapped, quality)
    offset = float(np.nanmedian(unwrapped - true))
    assert float(np.nanmax(np.abs(unwrapped - true - offset))) < 1e-6


def test_remove_ramp_clears_a_plane():
    yy, xx = np.mgrid[0:30, 0:28]
    plane = 0.4 + 0.03 * xx - 0.02 * yy
    flat = remove_ramp(plane, np.ones_like(plane, dtype=np.float64), border_px=3)
    assert float(np.nanstd(flat)) < 1e-8


def test_zero_velocity_scene_is_not_masked():
    velocity = np.zeros((32, 32))
    coherence = np.ones((32, 32))
    mask = detect_displacement(velocity, coherence)
    assert int(mask.sum()) == 0


def test_noise_free_pipeline_recovers_the_bowl():
    stack = generate_subsidence_stack(
        n_dates=6,
        height=48,
        width=48,
        spacing_days=36,
        rate_m_yr=0.2,
        sigma_px=8,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=3,
    )
    workspace = Workspace(
        stack=stack,
        config=PipelineConfig(goldstein_alpha=0.35, goldstein_window=16, coherence_window=3),
    )
    run_all(workspace)
    result = recovery_metrics(stack, workspace)
    assert result.relative_error < 0.15, result.render()
    assert result.correlation > 0.95, result.render()
    assert result.core_recall > 0.8, result.render()
    assert result.passed, result.render()


def test_demo_self_check_passes():
    result = run_self_check()
    assert result.passed, result.render()
    assert result.estimated_peak_m_yr < 0
    assert result.mask_pixels > 0


def test_no_deformation_pipeline_stays_quiet():
    stack = generate_subsidence_stack(
        n_dates=6,
        height=40,
        width=40,
        spacing_days=36,
        rate_m_yr=0.0,
        sigma_px=8,
        temporal_coherence=0.95,
        ramp_rad=0.3,
        seed=9,
    )
    workspace = Workspace(stack=stack, config=PipelineConfig(goldstein_window=16))
    run_all(workspace)
    assert workspace.mask is not None and workspace.velocity_m_yr is not None
    assert float(workspace.mask.mean()) < 0.02
    assert float(np.percentile(np.abs(workspace.velocity_m_yr), 95)) < 0.02


def test_stages_refuse_to_run_out_of_order():
    stack = generate_subsidence_stack(n_dates=3, height=16, width=16, temporal_coherence=1.0, ramp_rad=0.0)
    workspace = Workspace(stack=stack)
    with pytest.raises(PipelineError):
        from ai_insar.pipeline.engine import compute_unwrap

        compute_unwrap(workspace)


def test_slc_roundtrip_sorts_dates_and_saves_products(tmp_path: Path):
    stack = generate_subsidence_stack(
        n_dates=4,
        height=16,
        width=18,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=4,
    )
    path = tmp_path / "slc.npy"
    # The constructor rejects unsorted dates, so write the cube and sidecar directly.
    np.save(path, stack.data[::-1])
    sidecar = path.with_suffix(".json")
    import json

    sidecar.write_text(
        json.dumps(
            {
                "wavelength_m": stack.wavelength_m,
                "dates": [str(day) for day in stack.dates[::-1]],
                "sensor": "synthetic C-band",
                "description": "reversed",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_slc(path)
    assert loaded.dates[0] == stack.dates[0]
    assert np.allclose(loaded.data, stack.data)
    assert "sorted" in loaded.description

    saved = save_slc(stack, tmp_path / "clean.npy")
    again = load_slc(saved)
    assert np.array_equal(again.data, stack.data)
    assert np.array_equal(again.dates, stack.dates)

    workspace = Workspace(stack=stack, config=PipelineConfig(goldstein_window=16, coherence_window=3))
    run_all(workspace)
    out = save_products(workspace, tmp_path / "products")
    velocity = np.load(out / "velocity_m_yr.npy")
    mask = np.load(out / "mask.npy")
    assert velocity.shape == (16, 18)
    assert mask.shape == (16, 18)
    assert (out / "products.json").is_file()


def test_windows_zip_contains_launcher(tmp_path: Path):
    archive = make_windows_zip(tmp_path / "ai-insar-windows.zip", root=ROOT)
    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "ai-insar/run_windows.bat" in names
    assert "ai-insar/WINDOWS.md" in names
    assert "ai-insar/README.md" in names
    assert "ai-insar/src/ai_insar/__main__.py" in names
    assert "ai-insar/pyproject.toml" in names


def test_windows_bat_is_crlf_and_launches_module():
    data = (ROOT / "run_windows.bat").read_bytes()
    assert b"\r\n" in data
    assert b"\n" in data
    assert b"python -m ai_insar" in data
    text = (ROOT / "WINDOWS.md").read_text(encoding="utf-8")
    assert "run_windows.bat" in text
    assert "dist/ai-insar-windows.zip" in text
