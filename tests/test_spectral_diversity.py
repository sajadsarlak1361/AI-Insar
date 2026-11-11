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


def test_average_uses_the_phase_not_the_amplitude():
    """A bright sample must not outvote a unit-weight neighbour."""
    phase = np.full((4, 4), 0.2)
    phase[0, 0] = 0.0
    cross = np.exp(1j * phase)
    cross[0, 0] *= 1.0e6
    weight = np.ones(phase.shape)
    separation = 4000.0
    interval = 0.002
    got = average_shift_px(cross, weight, np.array([separation]), interval)
    votes = np.exp(1j * phase.ravel())
    expected_phase = float(np.angle(np.sum(votes)))
    # Every weight is 1, so SNAP's /N equals the weighted mean of Δf.
    expected = expected_phase / (2.0 * np.pi * separation * interval)
    assert abs(got - expected) < 1e-9
    amplitude_weighted = float(np.angle(np.sum(cross)))
    assert abs(amplitude_weighted) < 0.01
    assert abs(expected_phase - 0.2) > 0.01


def test_a_masked_late_look_does_not_vote():
    separation = 5000.0
    interval = 0.002
    pixels = 0.01
    master, slave_early, slave_late = _tone(32, 80, pixels, separation, interval)
    slave_late = slave_late.copy()
    slave_late[:, :40] = 0
    estimate = estimate_overlap(
        master,
        master,
        slave_early,
        slave_late,
        separation_hz=separation,
        azimuth_time_interval_s=interval,
        n_blocks=2,
        estimate_range=False,
    )
    assert estimate.block_weights[0] == 0.0
    assert abs(estimate.block_shifts_px[1] - (-pixels)) < 2e-3
    assert abs(estimate.applied_shift_px - pixels) < 2e-3


def test_range_correlation_recovers_one_sixteenth_of_a_pixel():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(64, 64))
    spectrum = np.fft.fft2(base)
    keep = np.abs(np.fft.fftfreq(64)) < 0.15
    spectrum *= keep[:, None] * keep[None, :]
    smooth = np.fft.ifft2(spectrum).real
    master = smooth.astype(np.complex128)
    frequency = np.fft.fftfreq(64)
    slave = np.fft.ifft(np.fft.fft(master, axis=1) * np.exp(-1j * 2.0 * np.pi * frequency * (1.0 / 16.0)), axis=1)
    shift, correlation = range_cross_correlation_shift(
        master, slave, oversample=128, search_px=8, threshold=0.1, window=64
    )
    assert correlation > 0.1
    # One bin of the 128× grid is the oversampled peak spacing.
    assert abs(shift - (-1.0 / 16.0)) <= 1.0 / 128.0


def test_bisinc_edge_repeats_the_border_sample():
    """SNAP clamps a tap that falls outside the raster instead of dropping it."""
    from ai_insar.pipeline.tops_ops import resample_real

    image = np.arange(6, dtype=np.float64).reshape(1, 6)
    got = resample_real(image, np.array([[0.0]]), np.array([[0.25]]), "BISINC_5_POINT_INTERPOLATION")

    def weight(offset: float) -> float:
        if abs(offset) > 2.0:
            return 0.0
        return float(np.sinc(offset) * 0.5 * (1.0 + np.cos(np.pi * offset / 2.0)))

    # Sample 0.25, taps at -2..+2, clamped onto columns 0, 0, 0, 1, 2.
    deltas = np.array([2.25, 1.25, 0.25, -0.75, -1.75])
    taps = np.array([weight(delta) for delta in deltas])
    expected = np.dot(taps, [0, 0, 0, 1, 2]) / np.sum(taps)
    assert abs(float(got[0, 0]) - expected) < 1e-6
    # Dropping the two taps that start outside the raster renormalises the rest.
    kept = taps[2:]
    dropped_value = np.dot(kept, [0, 1, 2]) / np.sum(kept)
    assert abs(expected - dropped_value) > 1e-3


def test_esd_shift_uses_the_full_burst_tile_and_keeps_nodata():
    from ai_insar.pipeline.geometry import Orbit
    from ai_insar.pipeline.s1_annotation import BurstTiming, SwathModel
    from ai_insar.pipeline.tops_ops import RadarGrid, shift_secondary_in_overlap

    start = np.datetime64("2021-06-30T05:26:15.752882")
    interval = 0.002
    lines = 20
    following = start + np.timedelta64(int(round(12 * interval * 1e9)), "ns")
    orbit = Orbit(
        np.array([start, following]),
        np.array([[7_000_000.0, 0.0, 0.0], [7_000_000.0, 1_000.0, 0.0]]),
        np.array([[0.0, 7_000.0, 0.0], [0.0, 7_000.0, 0.0]]),
    )
    model = SwathModel(
        wavelength_m=0.055,
        azimuth_time_interval_s=interval,
        range_pixel_spacing_m=2.3,
        slant_range_time_s=0.005,
        lines_per_burst=lines,
        samples_per_burst=6,
        number_of_lines=2 * lines,
        incidence_rad=0.6,
        fm_rate=np.zeros(3),
        bursts=(
            BurstTiming(start, 0, lines - 1, 0, 5),
            BurstTiming(following, 0, lines - 1, 0, 5),
        ),
        orbit=orbit,
    )
    data = (np.arange(2 * lines, dtype=np.float64)[:, None] + 1j * np.arange(6)[None, :]).astype(np.complex64)
    data[3, 1] = 0
    grid = RadarGrid(0, 0, data, model)
    shifted = shift_secondary_in_overlap(grid, 0.25, 0.0, None)
    # Both bursts are transformed, including lines that are not in the overlap.
    assert np.max(np.abs(shifted[0] - data[0])) > 1e-4
    assert np.max(np.abs(shifted[lines] - data[lines])) > 1e-4
    assert shifted[3, 1] == 0
    # A one-line transform is a different Fourier support from the burst tile.
    short = RadarGrid(0, 0, data[:1].copy(), model)
    short_shifted = shift_secondary_in_overlap(short, 0.25, 0.0, None)
    assert np.max(np.abs(short_shifted[0] - shifted[0])) > 1e-4


def test_esd_dialog_uses_snap_parameter_names():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest = __import__("pytest")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ai_insar.gui.chain_dialogs import EsdDialog

    QApplication.instance() or QApplication([])
    dialog = EsdDialog(can_run=True)
    params = dialog.parameters()
    for name in (
        "fineWinWidthStr",
        "fineWinHeightStr",
        "fineWinAccAzimuth",
        "fineWinAccRange",
        "fineWinOversampling",
        "xCorrThreshold",
        "cohThreshold",
        "numBlocksPerOverlap",
        "esdEstimator",
        "weightFunc",
        "temporalBaselineType",
        "maxTemporalBaseline",
        "integrationMethod",
        "doNotWriteTargetBands",
        "useSuppliedRangeShift",
        "overallRangeShift",
        "useSuppliedAzimuthShift",
        "overallAzimuthShift",
    ):
        assert name in params
    assert params["fineWinWidthStr"] == "512"
    assert params["esdEstimator"] == "Periodogram"
    assert params["xCorrThreshold"] == 0.1
    dialog.close()


def test_snap_flat_earth_polynomial_has_twenty_one_coefficients_and_the_geometric_sign():
    from ai_insar.pipeline.tops_ops import _distribute_points, _snap_monomials, flat_earth_phase
    from test_flat_earth_coherence import _pair

    points = _distribute_points(501, 1507, 22848)
    assert points.shape == (501, 2)
    assert points[:, 0].min() >= 0 and points[:, 0].max() <= 1507
    assert points[:, 1].min() >= 0 and points[:, 1].max() <= 22848
    assert _snap_monomials(points[:, 0], points[:, 1], 5, 1507, 22848).shape[1] == 21
    early_orbit, later_orbit, early, later = _pair()
    dense = flat_earth_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, degree=3, polynomial_points=None
    )
    fitted = flat_earth_phase(
        early,
        later,
        early_orbit=early_orbit,
        later_orbit=later_orbit,
        degree=3,
        polynomial_degree=5,
        polynomial_points=501,
        orbit_states=8,
    )
    assert fitted.shape == dense.shape
    assert np.isfinite(fitted).all()
    # Same sign as -4π/λ (R_reference - R_secondary). The polynomial is not the dense field.
    assert fitted[4, 3] * dense[4, 3] > 0.0


def test_snap_dem_half_cell_shifts_the_query_by_half_a_post():
    from ai_insar.pipeline.dem import ElevationModel, GeoidGrid

    lat = np.array([10.0, 9.0, 8.0])
    lon = np.array([20.0, 21.0, 22.0])
    # North-up: dy = -1, dx = +1. SNAP reads the centre grid half a post southeast.
    heights = np.array([[0.0, 10.0, 20.0], [100.0, 110.0, 120.0], [200.0, 210.0, 220.0]])
    dem = ElevationModel(
        lat_deg=lat,
        lon_deg=lon,
        orthometric_m=heights,
        nodata=-32768.0,
        orthometric_datum="EGM2008",
        geoid=GeoidGrid(
            lat_deg=np.array([12.0, 6.0]),
            lon_deg=np.array([18.0, 24.0]),
            undulation_m=np.zeros((2, 2)),
            source="test",
            model="EGM2008",
        ),
        provenance="test",
    )
    centre, centre_ok = dem.ellipsoidal_height(9.0, 21.0, half_cell=False)
    shifted, shifted_ok = dem.ellipsoidal_height(9.0, 21.0, half_cell=True)
    assert centre_ok and shifted_ok
    assert abs(centre - 110.0) < 1e-6
    # lat 9 + (-1)/2 = 8.5, lon 21 + 0.5 = 21.5, bilinear between 110, 120, 210, 220.
    assert abs(shifted - centre) > 1.0


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
