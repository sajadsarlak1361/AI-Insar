"""SNAP flat-earth sign, even coherence windows, and topographic residual."""

from __future__ import annotations

import pytest
import numpy as np

from ai_insar.pipeline.coherence import coherence_map, snap_window_bounds
from ai_insar.pipeline.dem import ElevationModel, GeoidGrid
from ai_insar.pipeline.geometry import Orbit, ecef_to_geodetic, ground_point_at_height
from ai_insar.pipeline.tops_ops import RadarGrid, flat_earth_phase, geometric_phase, topographic_residual_phase
from test_locate_grid_batch import _models, _orbit


def _pair():
    orbit = _orbit()
    reference_model, secondary_model = _models(orbit)
    later = Orbit(
        orbit.time + np.timedelta64(12, "D"),
        orbit.position_m + np.array([40.0, -20.0, 30.0]),
        orbit.velocity_m_s,
    )
    data = np.ones((12, 10), dtype=np.complex64)
    reference = RadarGrid(0, 0, data, reference_model)
    secondary = RadarGrid(0, 0, data, secondary_model)
    return orbit, later, reference, secondary


def test_flat_earth_phase_is_minus_four_pi_over_lambda_times_reference_minus_secondary():
    early_orbit, later_orbit, early, later = _pair()
    phase = flat_earth_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, degree=3, max_solves=early.shape[0] * early.shape[1]
    )
    row, col = 4, 3
    azimuth = early.model.line_time(early.row + row)
    slant = early.model.slant_range_m(early.col + col)
    ground = ground_point_at_height(early_orbit, azimuth, slant, 0.0, degree=3)
    early_range = float(np.linalg.norm(ground - early_orbit.interpolate(azimuth, 3)[0]))
    from ai_insar.pipeline.tops_ops import repeat_azimuth_time

    guess = repeat_azimuth_time(early.model, early.row + row, later.model)
    when = later_orbit.zero_doppler_time(ground, guess, 3)
    later_range = float(np.linalg.norm(ground - later_orbit.interpolate(when, 3)[0]))
    expected = -4.0 * np.pi / early.model.wavelength_m * (early_range - later_range)
    assert phase[row, col] == np.float64(phase[row, col])
    assert phase[row, col] == expected or abs(phase[row, col] - expected) < 1e-6
    assert abs(phase[row, col] - (-expected)) > 1e-3


def test_deramp_polynomial_follows_the_burst_not_the_window_origin():
    from ai_insar.pipeline.s1_annotation import BurstTiming, RangePolynomial
    from ai_insar.pipeline.tops_ops import tops_phase

    orbit = _orbit()
    reference, _secondary = _models(orbit)
    lines = int(reference.lines_per_burst)
    start = reference.bursts[0].azimuth_time
    second = reference.bursts[1].azimuth_time
    bursts = (
        BurstTiming(start, 0, lines - 1, 0, 20),
        BurstTiming(second, 0, lines - 1, 0, 20),
    )
    polynomials = (
        RangePolynomial(start, 0.006, np.array([-1000.0, 0.0])),
        RangePolynomial(second, 0.006, np.array([-5000.0, 0.0])),
    )
    model = reference.__class__(
        wavelength_m=reference.wavelength_m,
        azimuth_time_interval_s=reference.azimuth_time_interval_s,
        range_pixel_spacing_m=reference.range_pixel_spacing_m,
        slant_range_time_s=reference.slant_range_time_s,
        lines_per_burst=lines,
        samples_per_burst=reference.samples_per_burst,
        number_of_lines=lines * 2,
        incidence_rad=reference.incidence_rad,
        fm_rate=np.array([-2000.0, 0.0, 0.0]),
        bursts=bursts,
        orbit=orbit,
        fm_rates=polynomials,
        dc_estimates=polynomials,
        azimuth_steering_rate_rad_s=0.0,
    )
    both = tops_phase(RadarGrid(0, 0, np.ones((lines * 2, 8), dtype=np.complex64), model))
    alone = tops_phase(RadarGrid(lines, 0, np.ones((lines, 8), dtype=np.complex64), model))
    assert np.max(np.abs(both[lines:] - alone)) < 1e-6
    assert np.max(np.abs(both[lines:] - both[:lines])) > 1.0


def test_interferogram_window_is_exactly_two_by_ten_and_anchored_forward():
    from ai_insar.pipeline.coherence import _box_sum, correct_coherence_bias

    assert snap_window_bounds(5, 2, 20) == (5, 6)
    assert snap_window_bounds(5, 10, 40) == (1, 10)
    field = np.zeros((6, 16), dtype=np.float64)
    field[3, 5] = 1.0
    summed = _box_sum(field, 2, 10)
    assert summed[3, 5] == 1.0
    assert summed[2, 5] == 1.0
    assert summed[4, 5] == 0.0
    assert summed[3, 9] == 1.0
    assert summed[3, 10] == 0.0
    corrected = correct_coherence_bias(np.array([0.5], dtype=np.float32), 30)
    assert corrected[0] == np.float32(np.float32(0.5) - (1.0 - 0.25) / 60.0)
    rows, cols = 8, 16
    early = np.ones((rows, cols), dtype=np.complex64)
    late = np.ones((rows, cols), dtype=np.complex64)
    ramp = np.exp(1j * np.linspace(0, 8, cols))[None, :]
    raw = coherence_map(early, late * ramp, 10, azimuth_window=2)
    flat = np.ones((rows, cols), dtype=np.complex64)
    flattened = coherence_map(early, late * ramp, 10, azimuth_window=2, product=flat)
    assert flattened.shape == (rows, cols)
    assert flattened[4, 8] > 0.99
    assert raw[4, 8] < flattened[4, 8]


def _dem_over_window(early, early_orbit, height_m: float) -> ElevationModel:
    """Fine posts around the ellipsoid footprint so radar coding lands on the window."""
    azimuth = early.model.line_time(early.row + early.shape[0] // 2)
    slant = early.model.slant_range_m(early.col + early.shape[1] // 2)
    ground = ground_point_at_height(early_orbit, azimuth, slant, 0.0, degree=3)
    lat, lon, _height = ecef_to_geodetic(np.atleast_2d(ground))
    lat0 = float(np.rad2deg(np.asarray(lat).reshape(-1)[0]))
    lon0 = float(np.rad2deg(np.asarray(lon).reshape(-1)[0]))
    span = 0.002
    count = 41
    return ElevationModel(
        lat_deg=np.linspace(lat0 + span, lat0 - span, count),
        lon_deg=np.linspace(lon0 - span, lon0 + span, count),
        orthometric_m=np.full((count, count), height_m),
        nodata=-32768.0,
        orthometric_datum="EGM2008",
        geoid=GeoidGrid(
            lat_deg=np.array([lat0 + 1.0, lat0 - 1.0]),
            lon_deg=np.array([lon0 - 1.0, lon0 + 1.0]),
            undulation_m=np.zeros((2, 2)),
            source="test",
            model="EGM2008",
        ),
        provenance="test",
    )


def test_topographic_phase_is_the_radar_coded_slave_range_difference():
    from ai_insar.pipeline.geometry import geodetic_to_ecef, ground_points_at_height
    from ai_insar.pipeline.tops_ops import _slave_ranges

    early_orbit, later_orbit, early, later = _pair()
    row, col = 6, 5
    height_m = 15.0
    dem = _dem_over_window(early, early_orbit, height_m)
    dense, _dense_height = geometric_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, dem=dem, degree=3, half_cell=True
    )
    polynomial = flat_earth_phase(
        early,
        later,
        early_orbit=early_orbit,
        later_orbit=later_orbit,
        degree=3,
        polynomial_degree=5,
        polynomial_points=501,
        orbit_states=8,
    )
    residual, heights = topographic_residual_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, dem=dem, degree=3
    )
    finite = np.isfinite(heights)
    assert int(np.count_nonzero(finite)) == early.shape[0] * early.shape[1]
    assert np.max(np.abs(heights - height_m)) < 1.0
    # Adding the flat-earth polynomial back does not recover Phi(H). The old
    # residual was exactly Phi(H) - P5, so this difference was zero.
    assert np.max(np.abs((polynomial + residual)[finite] - dense[finite])) > 1.0e-4
    azimuth = early.model.line_time(early.row + row)
    slant = early.model.slant_range_m(early.col + col)
    ground = ground_point_at_height(early_orbit, azimuth, slant, height_m, degree=3)
    lat, lon, _height = [np.asarray(item).reshape(-1) for item in ecef_to_geodetic(np.atleast_2d(ground))]
    xyz = geodetic_to_ecef(lat, lon, np.array([height_m]))
    ellip = ground_points_at_height(early_orbit, np.array([azimuth]), np.array([slant]), np.zeros(1), degree=3)
    guess = np.array([later.model.bursts[0].azimuth_time])
    slave_dem = float(_slave_ranges(later_orbit, xyz, guess, 3)[0])
    slave_ellip = float(_slave_ranges(later_orbit, ellip, guess, 3)[0])
    expected = -4.0 * np.pi / float(early.model.wavelength_m) * (slave_ellip - slave_dem)
    assert abs(float(residual[row, col]) - expected) < 0.01
    assert float(residual[row, col]) * expected > 0.0
    zero = _dem_over_window(early, early_orbit, 0.0)
    zero_phase, _zero_heights = topographic_residual_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, dem=zero, degree=3
    )
    assert np.max(np.abs(zero_phase)) < np.max(np.abs(residual))


def test_topo_post_geopos_is_the_gis_centre_minus_half_a_cell():
    from ai_insar.pipeline.tops_ops import _topo_post_geopos

    # Native geocoding of one Copernicus post: GIS centre minus half the step.
    dy = -0.00027777777777515666
    dx = 0.000277777777776933
    lat, lon = _topo_post_geopos(50.23027777777778, 13.276041666666666, dy, dx)
    assert abs(float(lat) - 50.23041666666667) < 1e-9
    assert abs(float(lon) - 13.275902777777777) < 1e-9


def test_height_aware_dem_posts_cover_every_radar_pixel():
    """A 400 m surface shifts the footprint by more than 20 DEM posts."""
    azimuth = None
    early_orbit, later_orbit, early, later = _pair()
    azimuth = early.model.line_time(early.row + early.shape[0] // 2)
    slant = early.model.slant_range_m(early.col + early.shape[1] // 2)
    ground = ground_point_at_height(early_orbit, azimuth, slant, 0.0, degree=3)
    lat, lon, _height = ecef_to_geodetic(np.atleast_2d(ground))
    lat0 = float(np.rad2deg(np.asarray(lat).reshape(-1)[0]))
    lon0 = float(np.rad2deg(np.asarray(lon).reshape(-1)[0]))
    dem = ElevationModel(
        lat_deg=np.linspace(lat0 + 0.001, lat0 - 0.012, 91),
        lon_deg=np.linspace(lon0 - 0.003, lon0 + 0.003, 25),
        orthometric_m=np.full((91, 25), 400.0),
        nodata=-32768.0,
        orthometric_datum="EGM2008",
        geoid=GeoidGrid(
            lat_deg=np.array([lat0 + 1.0, lat0 - 1.0]),
            lon_deg=np.array([lon0 - 1.0, lon0 + 1.0]),
            undulation_m=np.zeros((2, 2)),
            source="test",
            model="EGM2008",
        ),
        provenance="test",
    )
    _residual, heights = topographic_residual_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, dem=dem, degree=3
    )
    assert int(np.count_nonzero(np.isfinite(heights))) == heights.size


def test_missing_dem_coverage_raises_instead_of_zeroing_the_phase():
    early_orbit, later_orbit, early, later = _pair()
    dem = _dem_over_window(early, early_orbit, 15.0)
    dem = ElevationModel(
        lat_deg=dem.lat_deg + 20.0,
        lon_deg=dem.lon_deg + 20.0,
        orthometric_m=dem.orthometric_m,
        nodata=dem.nodata,
        orthometric_datum=dem.orthometric_datum,
        geoid=dem.geoid,
        provenance="elsewhere",
    )
    with pytest.raises(ValueError, match="not set to zero"):
        topographic_residual_phase(
            early, later, early_orbit=early_orbit, later_orbit=later_orbit, dem=dem, degree=3
        )


def test_coherence_is_zero_when_a_look_has_no_power_and_does_not_cross_bursts():
    early = np.zeros((3, 20), dtype=np.complex64)
    late = np.zeros((3, 20), dtype=np.complex64)
    early[1, 4:8] = 1
    late[1, 4:8] = 1
    gamma = coherence_map(early, late, 10, azimuth_window=2)
    assert gamma.dtype == np.float32
    assert float(gamma[0, 19]) == 0.0
    assert float(gamma[2, 6]) == 0.0
    assert float(gamma[1, 6]) > 0.5


def test_deramp_and_reramp_use_float64_and_snaps_signs():
    from ai_insar.pipeline.tops_ops import _deramp_iq, _reramp_iq

    value = 3.0 + 4.0j
    phase = np.array([[12_345.678]])
    real, imag = _deramp_iq(np.array([[value]]), phase)
    restored_real, restored_imag = _reramp_iq(real, imag, phase)
    restored = complex(restored_real[0, 0], restored_imag[0, 0])
    assert abs(restored - value) < 1e-9


def test_snaphu_export_is_unscaled_little_endian_float32():
    from ai_insar.pipeline.snaphu_io import export_snaphu

    phase = np.array([[0.5, np.nan], [-1.25, 0.0]], dtype=np.float64)
    coherence = np.array([[0.25, 1.0], [np.nan, 0.5]], dtype=np.float64)
    conf = export_snaphu("/tmp/snaphu-parity-9501", phase, coherence)
    folder = conf.parent
    written = np.fromfile(folder / "corr.img", dtype="<f4").reshape(2, 2)
    assert written.tolist() == [[0.25, 1.0], [0.0, 0.5]]
    header = (folder / "phase.hdr").read_text(encoding="utf-8")
    assert "data type = 4" in header
    assert "byte order = 0" in header
    phase_written = np.fromfile(folder / "phase.img", dtype="<f4")
    assert abs(float(phase_written[0]) - 0.5) < 1e-6
    assert float(phase_written[1]) == 0.0


def test_unremoved_topographic_phase_moves_coherence_past_one_hundredth():
    rows, cols = 8, 24
    early = np.ones((rows, cols), dtype=np.complex64)
    late = np.ones((rows, cols), dtype=np.complex64)
    flat = coherence_map(early, late, 10, azimuth_window=2)
    # One radian across the 10-sample range window. That is the size of slope
    # left in the interferogram when topographic phase is missing.
    ramp = np.exp(1j * (1.2 / 10.0) * np.arange(cols))[None, :]
    sloped = coherence_map(early, late, 10, azimuth_window=2, product=(late * np.conj(early) * ramp))
    assert float(flat[4, 12] - sloped[4, 12]) > 0.01


def test_esd_phase_shift_keeps_a_large_phase_in_float64():
    from ai_insar.pipeline.tops_ops import _azimuth_shift_real

    samples = np.arange(256, dtype=np.float64)
    phase = (1.0e5 * np.sin(2.0 * np.pi * samples / samples.size))[:, None] * np.ones((1, 4))

    def _complex64_shift(values: np.ndarray, pixels: float) -> np.ndarray:
        spectrum = np.fft.fft(np.asarray(values, dtype=np.complex64), axis=0)
        frequency = np.fft.fftfreq(values.shape[0])
        spectrum = spectrum * np.exp(-1j * 2.0 * np.pi * frequency[:, None] * pixels)
        return np.fft.ifft(spectrum, axis=0).real

    restored = _azimuth_shift_real(_azimuth_shift_real(phase, 0.25, 1.0), -0.25, 1.0)
    restored_64 = _complex64_shift(_complex64_shift(phase, 0.25), -0.25)
    float_error = float(np.max(np.abs(restored - phase)))
    complex_error = float(np.max(np.abs(restored_64 - phase)))
    assert float_error < 1.0e-6
    assert complex_error > 1.0e-3
    assert float_error < complex_error / 10.0


def test_snap_polynomial_keeps_the_geometric_sign_and_uses_twenty_one_terms():
    from ai_insar.pipeline.tops_ops import _distribute_points, _snap_monomials

    points = _distribute_points(501, 1507, 22848)
    assert points.shape == (501, 2)
    assert int(points[:, 0].min()) >= 0 and int(points[:, 0].max()) <= 1507
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
    assert fitted[4, 3] * dense[4, 3] > 0.0


def test_snap_half_cell_dem_is_not_the_pixel_centre_sample():
    lat = np.array([10.0, 9.0, 8.0])
    lon = np.array([20.0, 21.0, 22.0])
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
    assert abs(shifted - centre) > 1.0
