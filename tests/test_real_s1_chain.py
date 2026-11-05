"""Real IW1 VV windows from two Sentinel-1 SLC products, not a synthetic scene.

The products are the public AWS granules
S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B and
S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B.
Only the IW1 VV annotation, the IW1 VV measurement, and the precise orbit
are on disk. The other swaths and VH are not in this test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")

from ai_insar.conventions import los_from_phase
from ai_insar.pipeline.filtering import goldstein_filter
from ai_insar.pipeline.geometry import ecef_to_geodetic, ground_point_at_height
from ai_insar.pipeline.multilook import multilook_array
from ai_insar.pipeline.s1_annotation import read_eof_orbit, read_swath_annotation
from ai_insar.pipeline.snaphu_io import export_snaphu, import_unwrapped, run_snaphu
from ai_insar.pipeline.tops_ops import (
    RadarGrid,
    backgeocode,
    burst_overlap_rows,
    deburst_lines,
    deburst_window,
    enhanced_spectral_diversity,
    flat_earth_phase,
    overlap_doppler_separation_hz,
    radar_coordinates,
    square_pixel_azimuth_looks,
    subtract_phase,
    terrain_correct,
    write_kmz,
)
from ai_insar.sentinel import read_tiff_window

ROOT = Path("/tmp/safe-real")
EARLY_ANN = ROOT / "20210630/s1a-iw1-slc-vv.xml"
LATE_ANN = ROOT / "20210712/s1a-iw1-slc-vv.xml"
EARLY_TIFF = ROOT / "20210630/measurement.tiff"
LATE_TIFF = ROOT / "20210712/measurement.tiff"
EARLY_ORB = ROOT / "orbits/poeorb_20210630.EOF.zip"
LATE_ORB = ROOT / "orbits/poeorb_20210712.EOF.zip"
SNAPHU = Path("/usr/bin/snaphu")


def _need_real_data():
    needed = (EARLY_ANN, LATE_ANN, EARLY_TIFF, LATE_TIFF, EARLY_ORB, LATE_ORB, SNAPHU)
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        pytest.skip("real S1 files are not on disk: " + ", ".join(missing))


def test_precise_orbit_moves_the_annotation_state_and_stays_on_the_footprint():
    _need_real_data()
    early = read_swath_annotation(EARLY_ANN)
    precise = read_eof_orbit(EARLY_ORB)
    when = early.bursts[4].azimuth_time
    before, _ = early.orbit.interpolate(when, 3)
    after, velocity = precise.interpolate(when, 3)
    shift = float(np.linalg.norm(after - before))
    assert 0.01 < shift < 100.0
    assert 7000.0 < float(np.linalg.norm(velocity)) < 8000.0
    line = 4 * early.lines_per_burst + 700
    ground = ground_point_at_height(precise, early.line_time(line), early.slant_range_m(11000), 0.0)
    lat, lon, height = ecef_to_geodetic(ground)
    assert abs(float(height)) < 5.0
    assert 49.0 < float(np.rad2deg(lat)) < 52.0
    assert 9.0 < float(np.rad2deg(lon)) < 14.5


def test_real_burst_chain_deburst_flat_earth_coregistration_and_snaphu(tmp_path: Path):
    _need_real_data()
    early_model = read_swath_annotation(EARLY_ANN)
    late_model = read_swath_annotation(LATE_ANN)
    early_orbit = read_eof_orbit(EARLY_ORB)
    late_orbit = read_eof_orbit(LATE_ORB)
    kept = deburst_lines(early_model)
    assert kept[0][1] >= early_model.bursts[0].first_valid_line
    assert kept[1][1] > early_model.bursts[1].first_valid_line
    assert sum(stop - start + 1 for _index, start, stop in kept) < early_model.number_of_lines

    row = 4 * early_model.lines_per_burst + 700
    col = 11000
    height, width = 24, 24
    early = read_tiff_window(EARLY_TIFF, row, col, height, width)
    assert early.shape == (height, width) and np.isfinite(early).all()
    assert float(np.mean(np.abs(early))) > 1.0
    early_grid = RadarGrid(row, col, early, early_model)
    center = ground_point_at_height(
        early_orbit,
        early_model.line_time(row + height // 2),
        early_model.slant_range_m(col + width // 2),
        0.0,
    )
    late_line, late_sample = radar_coordinates(
        late_model,
        late_orbit,
        center,
        guess=late_model.line_time(row + height // 2),
    )
    late_row = int(np.floor(late_line)) - height // 2
    late_col = int(np.floor(late_sample)) - width // 2
    late = read_tiff_window(LATE_TIFF, late_row, late_col, height, width)
    late_grid = RadarGrid(late_row, late_col, late, late_model)

    same = flat_earth_phase(early_grid, early_grid, early_orbit=early_orbit, later_orbit=early_orbit)
    assert np.max(np.abs(same)) < 1e-3
    flat_grid = RadarGrid(early_grid.row, early_grid.col, early, late_model)
    flat = flat_earth_phase(early_grid, flat_grid, early_orbit=early_orbit, later_orbit=late_orbit)
    assert np.isfinite(flat).all()
    assert float(np.std(flat)) > 0.01

    coreg = backgeocode(
        early_grid,
        late_grid,
        reference_orbit=early_orbit,
        secondary_orbit=late_orbit,
    )
    assert coreg.shape == early.shape
    assert np.isfinite(coreg).all()
    # A pure copy of the late crop would match late, not a resampled grid.
    filled = np.abs(coreg) > 0
    assert int(filled.sum()) > coreg.size // 2
    copy_corr = abs(np.vdot(coreg[filled], late[filled])) / (
        np.linalg.norm(coreg[filled]) * np.linalg.norm(late[filled])
    )
    assert copy_corr < 0.999
    interferogram = coreg * np.conj(early)
    flattened = subtract_phase(interferogram, flat)
    assert np.isfinite(flattened).all()

    looks = square_pixel_azimuth_looks(early_model, 8, early_orbit)
    assert looks >= 2
    square = multilook_array(flattened, 8, looks)
    assert square.shape[0] < height and square.shape[1] < width
    # snaphu's wrapped-gradient box needs both dimensions above a few pixels.
    # The square-pixel cell of this small window is narrower than that.
    multi = multilook_array(flattened, 2, 2)
    early_multi = multilook_array(early, 2, 2)
    coreg_multi = multilook_array(coreg, 2, 2)
    coherence = np.abs(coreg_multi * np.conj(early_multi)) / (
        np.sqrt(np.abs(early_multi) ** 2 * np.abs(coreg_multi) ** 2) + 1e-12
    )
    coherence = np.clip(np.abs(coherence), 0.0, 1.0).astype(np.float64)
    assert coherence.shape == multi.shape
    assert float(np.nanmin(coherence)) >= 0.0

    filtered = goldstein_filter(multi if multi.ndim == 2 else multi, alpha=0.5, window=16, spectrum_smooth=3)
    phase = np.angle(filtered).astype(np.float64)
    export_snaphu(
        tmp_path / "export",
        phase,
        coherence,
        wavelength_m=early_model.wavelength_m,
        pair_count=1,
        tile_rows=1,
        tile_columns=1,
        processors=1,
    )
    run_snaphu(tmp_path / "export", SNAPHU)
    unwrapped = import_unwrapped(tmp_path / "export")
    assert unwrapped.shape == phase.shape
    assert np.isfinite(unwrapped).all()
    displacement = los_from_phase(unwrapped, early_model.wavelength_m)
    assert displacement.shape == unwrapped.shape
    assert np.isfinite(displacement).all()
    probe = los_from_phase(np.array(np.pi), early_model.wavelength_m)
    assert probe < 0.0
    assert abs(probe + early_model.wavelength_m / 4.0) < 1e-6

    geocoded, bounds = terrain_correct(early_grid, early_orbit, out_rows=6, out_cols=6)
    west, south, east, north = bounds
    assert 9.0 < west < east < 14.5
    assert 49.0 < south < north < 52.0
    # The radar window is a parallelogram. Its latitude/longitude box is larger,
    # so only the samples that fall back inside the window are filled.
    assert int(np.count_nonzero(np.abs(geocoded))) >= 8
    center = ground_point_at_height(
        early_orbit,
        early_model.line_time(row + height // 2),
        early_model.slant_range_m(col + width // 2),
        0.0,
    )
    center_lat, center_lon, _center_height = ecef_to_geodetic(center)
    center_lat = float(np.rad2deg(center_lat))
    center_lon = float(np.rad2deg(center_lon))
    assert south <= center_lat <= north
    assert west <= center_lon <= east
    out_row = int(np.clip(round((north - center_lat) / (north - south) * 5), 0, 5))
    out_col = int(np.clip(round((center_lon - west) / (east - west) * 5), 0, 5))
    assert abs(geocoded[out_row, out_col]) > 0.0
    write_kmz(tmp_path / "overlay.kmz", geocoded, bounds)
    import zipfile

    with zipfile.ZipFile(tmp_path / "overlay.kmz") as archive:
        kml = archive.read("doc.kml").decode("utf-8")
        png = archive.read("overlay.png")
    assert "<west>" in kml and "<GroundOverlay>" in kml
    assert png.startswith(b"\x89PNG")
    _assert_real_overlap_esd(early_model)
    deburst_check = deburst_window(early, early_model, row, col)
    assert deburst_check.data.shape[0] <= early.shape[0]
    assert deburst_check.data.shape[1] == width


def _assert_real_overlap_esd(model) -> None:
    """Burst-overlap double difference on the real IW1 VV measurement."""
    early_rows, late_rows = burst_overlap_rows(model, 4)
    count = early_rows[1] - early_rows[0]
    assert count >= 8
    separation = overlap_doppler_separation_hz(model, 4)
    assert 1000.0 < separation < 20000.0
    col = 11000
    width = 16
    master_early = read_tiff_window(EARLY_TIFF, early_rows[0], col, count, width)
    master_late = read_tiff_window(EARLY_TIFF, late_rows[0], col, count, width)
    shift = enhanced_spectral_diversity(
        master_early,
        master_late,
        master_early,
        master_late,
        separation_hz=separation,
    )
    assert abs(shift) < 1e-6
    injected = 2.0e-5
    phase = np.exp(1j * 2.0 * np.pi * separation * injected)
    shifted = enhanced_spectral_diversity(
        master_early,
        master_late,
        master_early * phase,
        master_late,
        separation_hz=separation,
    )
    assert abs(shifted - injected) / injected < 0.05
