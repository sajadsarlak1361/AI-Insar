"""DEM-aware terrain-correction bounds and the SNAP Goldstein kernel."""

from __future__ import annotations

import numpy as np
import pytest

from ai_insar.pipeline.dem import ElevationModel, GeoidGrid
from ai_insar.pipeline.filtering import _box_mean_skip_zeros, goldstein_snap
from ai_insar.pipeline.geometry import Orbit, ecef_to_geodetic, ground_point_at_height
from ai_insar.pipeline.tops_ops import RadarGrid, _raster_coordinates, terrain_correct
from test_locate_grid_batch import _models, _orbit


def _dem_at(height: float, lat0: float, lat1: float, lon0: float, lon1: float) -> ElevationModel:
    geoid = GeoidGrid(
        lat_deg=np.array([lat1, lat0]),
        lon_deg=np.array([lon0, lon1]),
        undulation_m=np.zeros((2, 2)),
        source="test",
        model="EGM2008",
    )
    return ElevationModel(
        lat_deg=np.linspace(lat1, lat0, 6),
        lon_deg=np.linspace(lon0, lon1, 6),
        orthometric_m=np.full((6, 6), height),
        nodata=-32768.0,
        orthometric_datum="EGM2008",
        geoid=geoid,
        provenance="test",
    )


def _window():
    orbit = _orbit()
    reference, _secondary = _models(orbit)
    data = np.ones((23, 9), dtype=np.complex64) * (1.0 + 0.5j)
    grid = RadarGrid(4, 6, data, reference)
    return orbit, grid


def test_raster_coordinates_use_the_subset_origin():
    orbit, grid = _window()
    azimuth = grid.model.line_time(grid.row)
    slant = grid.model.slant_range_m(grid.col)
    ground = ground_point_at_height(orbit, azimuth, slant, 367.0, degree=3)
    line, sample = _raster_coordinates(grid, orbit, ground, guess=azimuth, degree=3)
    assert abs(line) < 0.05
    assert abs(sample) < 0.05


def test_terrain_correction_uses_dem_height_and_writes_nan_outside():
    orbit, grid = _window()
    azimuth = grid.model.line_time(grid.row + grid.shape[0] // 2)
    slant = grid.model.slant_range_m(grid.col + grid.shape[1] // 2)
    ground = ground_point_at_height(orbit, azimuth, slant, 0.0, degree=3)
    lat, lon, _height = ecef_to_geodetic(ground)
    dem = _dem_at(367.0, float(np.rad2deg(lat)) - 2.0, float(np.rad2deg(lat)) + 2.0, float(np.rad2deg(lon)) - 2.0, float(np.rad2deg(lon)) + 2.0)
    image, bounds, heights = terrain_correct(grid, orbit, dem, out_rows=8, out_cols=8, degree=3)
    assert image.shape == (8, 8)
    assert np.isfinite(np.real(image)).any()
    assert np.isnan(np.real(image)).any() or np.isfinite(heights).all()
    # Invalid samples are NaN, not a silent zero.
    missing = ~np.isfinite(np.real(image))
    if np.any(missing):
        assert np.all(np.isnan(image.real[missing]))
        assert np.all(np.isnan(image.imag[missing]))
    finite_h = heights[np.isfinite(heights)]
    assert finite_h.size
    assert np.max(np.abs(finite_h - 367.0)) < 1.0
    west, south, east, north = bounds
    assert east > west and north > south


def test_terrain_correction_refuses_a_dem_that_misses_the_footprint():
    orbit, grid = _window()
    dem = _dem_at(367.0, 40.0, 42.0, 10.0, 12.0)
    with pytest.raises(ValueError, match="no valid output|no DEM height"):
        terrain_correct(grid, orbit, dem, out_rows=4, out_cols=4, degree=3)


def test_goldstein_snap_keeps_phase_at_alpha_zero_and_pads_a_small_crop():
    rng = np.random.default_rng(3)
    phase = 0.4
    image = (rng.random((16, 16)) + 0.2) * np.exp(1j * phase)
    filtered = goldstein_snap(image.astype(np.complex64), alpha=0.0, fft_size=16, window_size=3)
    got = np.angle(filtered[np.abs(filtered) > 0])
    assert np.max(np.abs(got - phase)) < 1e-6

    small = (rng.random((10, 12)) + 0.2) * np.exp(1j * phase)
    padded = goldstein_snap(small.astype(np.complex64), alpha=1.0, fft_size=16, window_size=3)
    assert padded.shape == small.shape
    assert np.isfinite(padded).any()
    assert not np.allclose(padded, small)

    zeroed = image.copy()
    zeroed[0, 0] = 0
    kept = goldstein_snap(zeroed.astype(np.complex64), alpha=0.0, fft_size=16, window_size=3)
    assert kept[0, 0] == 0


def test_goldstein_spectrum_window_is_a_3x3_mean_that_skips_zeros():
    power = np.array(
        [
            [0.0, 0.0, 4.0],
            [0.0, 2.0, 6.0],
            [1.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    mean = _box_mean_skip_zeros(power, 1)
    # Centre bin sees 4, 2, 6, 1, 3 (the zeros are skipped): mean 16/5.
    assert mean[1, 1] == pytest.approx(16.0 / 5.0)
    # A corner uses only the bins that exist.
    assert mean[0, 0] == pytest.approx(2.0)
