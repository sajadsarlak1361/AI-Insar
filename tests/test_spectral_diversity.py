"""Full-overlap ESD: Average, Periodogram, the phase ramp, and the network."""

from __future__ import annotations

import numpy as np

from ai_insar.pipeline.spectral_diversity import (
    apply_azimuth_phase_ramp,
    average_shift_px,
    double_difference,
    estimate_overlap,
    network_shifts,
    range_cross_correlation_shift,
)
from ai_insar.pipeline.tops_ops import early_look_coherence


def _tone(rows: int, cols: int, pixels: float, separation: float, interval: float):
    phase = 2.0 * np.pi * separation * interval * pixels
    master = np.ones((rows, cols), dtype=np.complex64)
    slave_early = master.copy()
    slave_late = (np.exp(-1j * phase) * master).astype(np.complex64)
    return master, slave_early, slave_late


def test_periodogram_on_a_full_overlap_recovers_the_tone():
    """Ten blocks across a burst-overlap-sized window, not an 8×16 crop."""
    separation = 6400.0
    interval = 2.055556e-3
    pixels = -0.000727
    rows, cols = 127, 2048
    master, slave_early, slave_late = _tone(rows, cols, pixels, separation, interval)
    estimate = estimate_overlap(
        master,
        master,
        slave_early,
        slave_late,
        separation_hz=separation,
        azimuth_time_interval_s=interval,
        estimator="Periodogram",
        n_blocks=10,
        estimate_range=False,
    )
    assert estimate.block_shifts_px.shape == (10,)
    # master·conj(slave) flips the sign of the injected tone.
    assert np.max(np.abs(estimate.block_shifts_px - (-pixels))) < 2e-4
    assert abs(estimate.applied_shift_px - pixels) < 2e-4


def test_average_matches_the_constant_frequency_periodogram():
    separation = 5000.0
    interval = 0.002
    pixels = 0.0012
    master, slave_early, slave_late = _tone(96, 640, pixels, separation, interval)
    cross = double_difference(master, master, slave_early, slave_late)
    weight = np.ones(cross.shape, dtype=np.float64)
    closed = average_shift_px(cross, weight, np.array([separation]), interval)
    estimate = estimate_overlap(
        master,
        master,
        slave_early,
        slave_late,
        separation_hz=separation,
        azimuth_time_interval_s=interval,
        estimator="Average",
        n_blocks=10,
        estimate_range=False,
    )
    assert abs(closed - (-pixels)) < 2e-4
    assert abs(estimate.applied_shift_px - pixels) < 2e-4


def test_phase_ramp_on_the_late_look_removes_the_tone():
    separation = 6400.0
    interval = 0.002
    pixels = -0.0011
    rows, cols = 64, 320
    master, slave_early, slave_late = _tone(rows, cols, pixels, separation, interval)
    estimate = estimate_overlap(
        master,
        master,
        slave_early,
        slave_late,
        separation_hz=np.full(cols, separation),
        azimuth_time_interval_s=interval,
        n_blocks=4,
        estimate_range=False,
    )
    corrected = apply_azimuth_phase_ramp(slave_late, np.full(cols, separation), estimate.periodogram_mean_px, interval)
    again = estimate_overlap(
        master,
        master,
        slave_early,
        corrected,
        separation_hz=separation,
        azimuth_time_interval_s=interval,
        n_blocks=4,
        estimate_range=False,
    )
    assert abs(float(again.periodogram_mean_px)) < 2e-4


def test_range_correlation_finds_a_one_pixel_shift():
    rng = np.random.default_rng(4)
    master = rng.normal(size=(64, 128)) + 1j * rng.normal(size=(64, 128))
    slave = np.zeros_like(master)
    slave[:, 1:] = master[:, :-1]
    shift, correlation = range_cross_correlation_shift(master, slave, oversample=32, search_px=4, threshold=0.1)
    # Content moved one sample to the right, so the Fourier correction is -1.
    assert 0.5 < correlation <= 1.0
    assert abs(shift - (-1.0)) < 1.0 / 32


def test_range_correlation_averages_one_window_per_burst():
    rng = np.random.default_rng(5)
    master = rng.normal(size=(128, 64)) + 1j * rng.normal(size=(128, 64))
    slave = master.copy()
    slave[64:, 1:] = master[64:, :-1]
    slave[64:, 0] = 0
    shift, correlation = range_cross_correlation_shift(
        master,
        slave,
        oversample=16,
        search_px=4,
        threshold=0.1,
        window=32,
        lines_per_burst=64,
    )
    # Burst 0 is aligned and burst 1 is one sample late, so the mean is -0.5.
    assert correlation > 0.5
    assert abs(shift - (-0.5)) < 1.0 / 16


def test_network_fixes_the_reference_and_rejects_a_long_baseline():
    observations = [
        (0, 1, -0.001511, 1.0),
        (1, 2, -0.001000, 1.0),
        (0, 2, -0.002511, 0.5),
        (0, 6, 9.0, 5.0),
    ]
    shifts = network_shifts(observations, 7, method="L2", max_temporal_baseline=4)
    assert shifts[0] == 0.0
    assert abs(shifts[1] - (-0.001511)) < 1e-6
    assert shifts[6] == 0.0
    assert abs(shifts[2] - (shifts[1] - 0.001000)) < 5e-4


def test_bisinc5_uses_a_normalised_hann_window_and_keeps_phase_in_float64():
    from ai_insar.pipeline.tops_ops import resample_complex, resample_real

    image = np.zeros((9, 9), dtype=np.complex64)
    image[4, 4] = 1
    on_pixel = resample_complex(image, np.array([[4.0]]), np.array([[4.0]]), "BISINC_5_POINT_INTERPOLATION")
    assert abs(on_pixel[0, 0] - 1) < 1e-5
    half = resample_complex(image, np.array([[4.5]]), np.array([[4.0]]), "BISINC_5_POINT_INTERPOLATION")

    def weight(offset: float) -> float:
        if abs(offset) > 2.0:
            return 0.0
        return float(np.sinc(offset) * 0.5 * (1.0 + np.cos(np.pi * offset / 2.0)))

    taps = np.array([weight(4.5 - index) for index in range(2, 7)])
    taps /= np.sum(taps)
    assert abs(float(half[0, 0].real) - float(taps[2])) < 1e-4
    phase = np.full((9, 9), 1000.0)
    restored = resample_real(phase, np.array([[4.25]]), np.array([[4.0]]), "BISINC_5_POINT_INTERPOLATION")
    assert restored.dtype == np.float64
    assert abs(float(restored[0, 0]) - 1000.0) < 1e-6


def test_coherence_box_matches_a_direct_five_by_five():
    rng = np.random.default_rng(1)
    master = (rng.normal(size=(11, 9)) + 1j * rng.normal(size=(11, 9))).astype(np.complex64)
    slave = (0.8 * master + 0.1 * (rng.normal(size=master.shape) + 1j * rng.normal(size=master.shape))).astype(np.complex64)
    got = early_look_coherence(master, slave)
    direct = np.zeros_like(got)
    for row in range(11):
        for col in range(9):
            r0, r1 = max(row - 2, 0), min(row + 2, 10) + 1
            c0, c1 = max(col - 2, 0), min(col + 2, 8) + 1
            box_m = master[r0:r1, c0:c1]
            box_s = slave[r0:r1, c0:c1]
            cross = np.sum(box_m * np.conj(box_s))
            power = np.sqrt(np.sum(np.abs(box_m) ** 2) * np.sum(np.abs(box_s) ** 2))
            direct[row, col] = 0.0 if power == 0.0 else np.abs(cross) / power
    assert np.max(np.abs(got - direct)) < 1e-6
