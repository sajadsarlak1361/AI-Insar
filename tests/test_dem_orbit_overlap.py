"""DEM sampling, orbit selection, zip annotations, and a real burst overlap.

The SLC windows are IW1 VV from the two public 2021-06-30 and 2021-07-12
products. The DEM is the SRTM-1 tile N50E012 plus the EGM96 15-minute geoid.
This is not a nine-burst product.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")

from ai_insar.pipeline.dem import ensure_geoid, read_copernicus_tif, read_srtm_hgt
from ai_insar.pipeline.orbit_download import select_orbit_name, validate_orbit
from ai_insar.pipeline.s1_annotation import read_eof_orbit, read_swath_annotation, read_swath_from_product
from ai_insar.pipeline.snaphu_io import export_snaphu, import_unwrapped, run_snaphu
from ai_insar.pipeline.tops_ops import (
    RadarGrid,
    backgeocode,
    burst_overlap_rows,
    ellipsoid_geocode,
    enhanced_spectral_diversity,
    overlap_doppler_separation_hz,
    radar_coordinates,
    terrain_correct,
    write_kmz,
)
from ai_insar.safe import resolve_safe, select_measurement
from ai_insar.sentinel import read_tiff_window

ROOT = Path("/tmp/safe-real")
DEM = ROOT / "dem"
EARLY_ANN = ROOT / "20210630/s1a-iw1-slc-vv.xml"
LATE_ANN = ROOT / "20210712/s1a-iw1-slc-vv.xml"
EARLY_TIFF = ROOT / "20210630/measurement.tiff"
LATE_TIFF = ROOT / "20210712/measurement.tiff"
EARLY_ORB = ROOT / "orbits/poeorb_20210630.EOF.zip"
SNAPHU = Path("/usr/bin/snaphu")


def _need(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip("missing real inputs: " + ", ".join(missing))


def test_srtm_and_copernicus_heights_vary_and_skip_nodata():
    _need(DEM / "geoids/egm96-15.pgm", DEM / "N50E012.hgt.gz")
    geoid = ensure_geoid(DEM)
    srtm = read_srtm_hgt(DEM / "N50E012.hgt.gz", geoid)
    latitudes = np.linspace(50.30, 50.45, 5)
    longitudes = np.linspace(12.60, 12.85, 5)
    heights = []
    for lat in latitudes:
        for lon in longitudes:
            height, valid = srtm.ellipsoidal_height(float(lat), float(lon))
            assert valid
            heights.append(height)
    heights = np.array(heights)
    assert float(np.std(heights)) > 5.0
    assert float(np.mean(heights)) > 50.0
    broken = srtm
    broken.orthometric_m = np.array(broken.orthometric_m, copy=True)
    broken.orthometric_m[0, 0] = broken.nodata
    north = float(broken.lat_deg[0])
    west = float(broken.lon_deg[0])
    _value, valid = broken.ellipsoidal_height(north, west)
    assert not valid
    with pytest.raises(ValueError, match="EGM2008"):
        read_copernicus_tif(DEM / "copernicus_N50_E012.tif", geoid)
    egm2008 = ensure_geoid(DEM, "EGM2008")
    assert egm2008.model == "EGM2008"
    glo30 = read_copernicus_tif(DEM / "copernicus_N50_E012.tif", egm2008)
    other, other_valid = glo30.ellipsoidal_height(50.38, 12.74)
    reference, reference_valid = srtm.ellipsoidal_height(50.38, 12.74)
    assert other_valid and reference_valid
    assert glo30.orthometric_datum == "EGM2008"
    assert srtm.orthometric_datum == "EGM96"
    assert abs(other - reference) < 30.0


def test_orbit_name_matches_platform_and_validity_window():
    names = [
        "S1B_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF",
        "S1A_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF.zip",
        "S1A_OPER_AUX_RESORB_OPOD_20210701T000000_V20210629T225942_20210701T005942.EOF",
    ]
    from datetime import datetime, timezone

    when = datetime(2021, 6, 30, 5, 26, tzinfo=timezone.utc)
    chosen = select_orbit_name(names, when, "S1A")
    assert chosen is not None and chosen.startswith("S1A") and "POEORB" in chosen
    assert select_orbit_name(names, when, "S1C") is None
    _need(EARLY_ORB, EARLY_ANN)
    orbit = read_eof_orbit(EARLY_ORB)
    model = read_swath_annotation(EARLY_ANN)
    validate_orbit(orbit, model.bursts[4].azimuth_time, platform="S1A", name=EARLY_ORB.name)


def test_zip_annotation_matches_the_folder_annotation(tmp_path: Path):
    _need(EARLY_ANN, ROOT / "20210630/manifest.safe")
    safe = "S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
    archive_path = tmp_path / f"{safe[:-5]}.zip"
    measurement = f"{safe}/measurement/s1a-iw1-slc-vv-20210630t052604-20210630t052629-038565-048cfd-004.tiff"
    annotation = f"{safe}/annotation/s1a-iw1-slc-vv-20210630t052604-20210630t052629-038565-048cfd-004.xml"
    manifest = f"{safe}/manifest.safe"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(ROOT / "20210630/manifest.safe", manifest)
        archive.write(EARLY_ANN, annotation)
        archive.writestr(measurement, b"not-a-tiff")
    product = resolve_safe(archive_path)
    measurement_record = select_measurement(product, "IW1", "VV")
    zipped = read_swath_from_product(product, measurement_record)
    folder = read_swath_annotation(EARLY_ANN)
    assert zipped.lines_per_burst == folder.lines_per_burst
    assert abs(zipped.wavelength_m - folder.wavelength_m) < 1e-9
    assert len(zipped.bursts) == len(folder.bursts) == 9
    assert zipped.bursts[4].azimuth_time == folder.bursts[4].azimuth_time


def test_real_overlap_uses_a_varying_dem_and_rejects_weak_esd(tmp_path: Path):
    _need(EARLY_ANN, LATE_ANN, EARLY_TIFF, LATE_TIFF, EARLY_ORB, DEM / "N50E012.hgt.gz", DEM / "geoids/egm96-15.pgm", SNAPHU)
    before = EARLY_TIFF.stat().st_size
    geoid = ensure_geoid(DEM)
    dem = read_srtm_hgt(DEM / "N50E012.hgt.gz", geoid)
    early_model = read_swath_annotation(EARLY_ANN)
    late_model = read_swath_annotation(LATE_ANN)
    early_orbit = read_eof_orbit(EARLY_ORB)
    from ai_insar.pipeline.s1_annotation import read_eof_orbit as read_orbit

    late_orbit = read_orbit(ROOT / "orbits/poeorb_20210712.EOF.zip")
    (early_rows, late_rows) = burst_overlap_rows(early_model, 4)
    count = early_rows[1] - early_rows[0]
    assert count >= 100
    col = 11000
    width = 24
    looks = []
    for rows in (early_rows, late_rows):
        early = read_tiff_window(EARLY_TIFF, rows[0], col, count, width)
        center_line = rows[0] + count // 2
        center = __import__("ai_insar.pipeline.geometry", fromlist=["ground_point_at_height"]).ground_point_at_height(
            early_orbit,
            early_model.line_time(center_line),
            early_model.slant_range_m(col + width // 2),
            0.0,
        )
        late_line, late_sample = radar_coordinates(
            late_model,
            late_orbit,
            center,
            guess=late_model.line_time(center_line),
        )
        late_row = int(np.floor(late_line)) - count // 2
        late_col = int(np.floor(late_sample)) - width // 2
        late = read_tiff_window(LATE_TIFF, late_row, late_col, count, width)
        early_grid = RadarGrid(rows[0], col, early, early_model)
        late_grid = RadarGrid(late_row, late_col, late, late_model)
        coreg = backgeocode(
            early_grid,
            late_grid,
            reference_orbit=early_orbit,
            secondary_orbit=late_orbit,
            dem=dem,
            max_solves=2048,
            max_residual_px=0.05,
        )
        assert backgeocode.last_residual_px <= 0.05
        looks.append((early_grid, coreg))
    separation = overlap_doppler_separation_hz(early_model, 4)
    master_early, slave_early = looks[0][0].data, looks[0][1]
    master_late, slave_late = looks[1][0].data, looks[1][1]
    rng = np.random.default_rng(2)
    noise = np.exp(1j * rng.uniform(0, 2 * np.pi, master_early.shape)).astype(np.complex64)
    with pytest.raises(RuntimeError, match="coherence"):
        enhanced_spectral_diversity(
            master_early,
            master_late,
            master_early * noise,
            master_late,
            separation_hz=separation,
            estimator="Average",
            coherence_threshold=0.3,
        )
    try:
        real_shift, real_coherence = enhanced_spectral_diversity(
            master_early,
            master_late,
            slave_early,
            slave_late,
            separation_hz=separation,
            estimator="Periodogram",
            coherence_threshold=0.25,
        )
    except RuntimeError as exc:
        assert "coherence" in str(exc).lower()
        real_shift, real_coherence = float("nan"), 0.0
    else:
        assert np.isfinite(real_shift)
        assert real_coherence >= 0.25
    injected = 2.0e-5
    phase = np.exp(1j * 2.0 * np.pi * separation * injected)
    for estimator in ("Average", "Periodogram"):
        shift, coherence = enhanced_spectral_diversity(
            master_early,
            master_late,
            master_early * phase,
            master_late,
            separation_hz=separation,
            estimator=estimator,
            coherence_threshold=0.2,
        )
        assert abs(shift - injected) / injected < 0.05
        assert coherence > 0.5
    image, bounds, heights = terrain_correct(looks[0][0], early_orbit, dem, out_rows=10, out_cols=10)
    finite = heights[np.isfinite(heights)]
    assert finite.size >= 8
    assert float(np.std(finite)) > 1.0
    ellipsoid, ellipsoid_bounds = ellipsoid_geocode(looks[0][0], early_orbit, out_rows=10, out_cols=10)
    assert ellipsoid.shape == image.shape
    assert ellipsoid_bounds == bounds
    write_kmz(tmp_path / "overlap.kmz", image, bounds)
    with zipfile.ZipFile(tmp_path / "overlap.kmz") as archive:
        kml = archive.read("doc.kml").decode("utf-8")
    west, south, east, north = bounds
    assert f"{west:.8f}" in kml and f"{east:.8f}" in kml
    assert "<GroundOverlay>" in kml
    phase_ramp = np.linspace(0.0, 4.0, 64 * 64, dtype=np.float64).reshape(64, 64)
    wrapped = np.angle(np.exp(1j * phase_ramp))
    coherence = np.full(phase_ramp.shape, 0.9)
    export_snaphu(tmp_path / "snaphu", wrapped, coherence, tile_rows=1, tile_columns=1, processors=1)
    run_snaphu(tmp_path / "snaphu", SNAPHU)
    unwrapped = import_unwrapped(tmp_path / "snaphu")
    residual = np.angle(np.exp(1j * (unwrapped - phase_ramp)))
    assert float(np.max(np.abs(residual))) < 0.2
    assert EARLY_TIFF.stat().st_size == before
