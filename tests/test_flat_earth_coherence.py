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


def test_even_coherence_window_uses_size_over_two_on_both_sides():
    assert snap_window_bounds(5, 2, 20) == (4, 6)
    assert snap_window_bounds(5, 10, 40) == (0, 10)
    rows, cols = 8, 16
    early = np.ones((rows, cols), dtype=np.complex64)
    late = np.ones((rows, cols), dtype=np.complex64)
    ramp = np.exp(1j * np.linspace(0, 8, cols))[None, :]
    raw = coherence_map(early, late * ramp, 10, azimuth_window=2)
    flat = np.ones((rows, cols), dtype=np.complex64)
    flattened = coherence_map(early, late * ramp, 10, azimuth_window=2, product=flat)
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
