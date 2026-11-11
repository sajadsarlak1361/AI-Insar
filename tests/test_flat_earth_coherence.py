"""SNAP flat-earth sign, even coherence windows, and topographic residual."""

from __future__ import annotations

import numpy as np

from ai_insar.pipeline.coherence import coherence_map, snap_window_bounds
from ai_insar.pipeline.dem import ElevationModel, GeoidGrid
from ai_insar.pipeline.geometry import Orbit, ecef_to_geodetic, ground_point_at_height
from ai_insar.pipeline.tops_ops import RadarGrid, flat_earth_phase, topographic_residual_phase
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


def test_topographic_residual_is_the_dem_phase_minus_the_ellipsoid_phase():
    early_orbit, later_orbit, early, later = _pair()
    azimuth = early.model.line_time(early.row + 6)
    slant = early.model.slant_range_m(5)
    ground = ground_point_at_height(early_orbit, azimuth, slant, 0.0, degree=3)
    lat, lon, _height = ecef_to_geodetic(ground)
    lat_deg = float(np.rad2deg(lat))
    lon_deg = float(np.rad2deg(lon))
    dem = ElevationModel(
        lat_deg=np.linspace(lat_deg + 1.0, lat_deg - 1.0, 5),
        lon_deg=np.linspace(lon_deg - 1.0, lon_deg + 1.0, 5),
        orthometric_m=np.full((5, 5), 400.0),
        nodata=-32768.0,
        orthometric_datum="EGM2008",
        geoid=GeoidGrid(
            lat_deg=np.array([lat_deg + 2.0, lat_deg - 2.0]),
            lon_deg=np.array([lon_deg - 2.0, lon_deg + 2.0]),
            undulation_m=np.zeros((2, 2)),
            source="test",
            model="EGM2008",
        ),
        provenance="test",
    )
    flat = flat_earth_phase(early, later, early_orbit=early_orbit, later_orbit=later_orbit, degree=3)
    dem_phase = flat_earth_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, degree=3, dem=dem
    )
    residual, heights = topographic_residual_phase(
        early, later, early_orbit=early_orbit, later_orbit=later_orbit, dem=dem, degree=3
    )
    finite = np.isfinite(heights) & np.isfinite(dem_phase) & np.isfinite(flat)
    assert finite.any()
    assert np.max(np.abs((dem_phase - flat)[finite] - residual[finite])) < 1e-4
    assert np.max(np.abs(heights[finite] - 400.0)) < 5.0


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
