"""Compare a two-burst IW1 VV window with SNAP 12 GPT output.

SNAP 12.0.0 is installed separately under ``/home/ubuntu/esa-snap``. The GUI
audit was SNAP 14. The comparison uses the pixels GPT wrote, not a copied
SNAP source tree. The raw measurement file is not modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")

from ai_insar.pipeline.dem import ensure_geoid, read_srtm_hgt
from ai_insar.pipeline.orbit_download import download_orbit
from ai_insar.pipeline.s1_annotation import read_eof_orbit, read_swath_annotation
from ai_insar.pipeline.snaphu_io import export_snaphu
from ai_insar.pipeline.tops_ops import (
    RadarGrid,
    burst_overlap_rows,
    deramp,
    enhanced_spectral_diversity,
    overlap_doppler_separation_hz,
)
from ai_insar.sentinel import read_tiff_window

SNAP_I = Path("/tmp/snap-ref/two_bursts.data/i_IW1_VV.img")
SNAP_Q = Path("/tmp/snap-ref/two_bursts.data/q_IW1_VV.img")
TIFF = Path("/tmp/safe-real/20210630/measurement.tiff")
ANN = Path("/tmp/safe-real/20210630/s1a-iw1-slc-vv.xml")
# linesPerBurst is 1508. SNAP burst indices are 1-based.
# 1-based bursts 5 and 6 are 0-based bursts 4 and 5, rows 6032:9048.
LINES_PER_BURST = 1508
ROW = 4 * LINES_PER_BURST
COL = 11000
HEIGHT = 2 * LINES_PER_BURST
WIDTH = 512


def test_burst_numbers_match_lines_per_burst():
    if not ANN.is_file():
        pytest.skip("annotation is not on disk")
    model = read_swath_annotation(ANN)
    assert model.lines_per_burst == LINES_PER_BURST
    # 0-based burst 4 starts at row 6032 and is SNAP burst 5.
    assert 4 * model.lines_per_burst == 6032
    assert 6 * model.lines_per_burst == 9048
    early, late = burst_overlap_rows(model, 4)
    assert early[0] == 7374
    assert early[1] == 7540 == late[0]
    assert 6032 <= early[0] < 7540
    assert 7540 <= late[0] < 9048


def test_gpt_reopens_beam_dimap_complex_values(tmp_path: Path):
    import subprocess

    from ai_insar.pipeline.dimap import write_beam_dimap

    values = (np.arange(12) + 1j * np.arange(12)).reshape(3, 4).astype(np.complex64)
    dim = write_beam_dimap(tmp_path, "sample", values, None, band_name="IW1_VV", unit="1")
    destination = tmp_path / "via_subset.dim"
    proc = subprocess.run(
        [
            "/home/ubuntu/esa-snap/bin/gpt",
            "Subset",
            f"-Ssource={dim}",
            "-Pregion=0,0,4,3",
            "-t",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(proc.stderr[-500:])
    header = (tmp_path / "via_subset.data" / "i_IW1_VV.hdr").read_text(encoding="utf-8")
    endian = ">i2" if "byte order = 1" in header else "<i2"
    real = np.fromfile(tmp_path / "via_subset.data" / "i_IW1_VV.img", dtype=endian)
    imag = np.fromfile(tmp_path / "via_subset.data" / "q_IW1_VV.img", dtype=endian)
    assert np.array_equal(real, np.arange(12, dtype=np.int16))
    assert np.array_equal(imag, np.arange(12, dtype=np.int16))


def test_snap_two_burst_subset_matches_the_slc_reader():
    if not SNAP_I.is_file() or not TIFF.is_file():
        pytest.skip("SNAP two-burst reference or the IW1 measurement is not on disk")
    before = TIFF.stat().st_size
    snap_i = np.fromfile(SNAP_I, dtype=">i2").reshape(HEIGHT, WIDTH)
    snap_q = np.fromfile(SNAP_Q, dtype=">i2").reshape(HEIGHT, WIDTH)
    snap = snap_i.astype(np.float32) + 1j * snap_q.astype(np.float32)
    ours = read_tiff_window(TIFF, ROW, COL, HEIGHT, WIDTH)
    assert ours.shape == (HEIGHT, WIDTH)
    assert float(np.max(np.abs(ours - snap))) == 0.0
    assert float(np.mean(np.abs(ours))) > 1.0
    assert TIFF.stat().st_size == before


def test_two_burst_deramp_roundtrip_is_exact():
    if not ANN.is_file() or not TIFF.is_file():
        pytest.skip("IW1 annotation or measurement is not on disk")
    model = read_swath_annotation(ANN)
    window = read_tiff_window(TIFF, ROW, COL, HEIGHT, WIDTH)
    grid = RadarGrid(ROW, COL, window, model)
    restored = deramp(RadarGrid(ROW, COL, deramp(grid, remove=True), model), remove=False)
    # complex64 multiply/divide of the chirp leaves a sub-milliradian residual.
    assert float(np.max(np.abs(restored - window))) < 1e-3
    early_rows, late_rows = burst_overlap_rows(model, 4)
    assert early_rows[0] >= ROW and late_rows[1] <= ROW + HEIGHT
    assert late_rows[0] - early_rows[0] >= 100


def test_twelve_day_overlap_is_not_forced_through_esd():
    """The real pair stays rejected. An injected shift is not this test."""
    late_ann = Path("/tmp/safe-real/20210712/s1a-iw1-slc-vv.xml")
    late_tiff = Path("/tmp/safe-real/20210712/measurement.tiff")
    dem_file = Path("/tmp/safe-real/dem/N50E012.hgt.gz")
    if not all(path.is_file() for path in (ANN, TIFF, late_ann, late_tiff, dem_file)):
        pytest.skip("two-date IW1 inputs or the SRTM tile are not on disk")
    from ai_insar.pipeline.geometry import ground_point_at_height
    from ai_insar.pipeline.tops_ops import backgeocode, radar_coordinates

    early_model = read_swath_annotation(ANN)
    late_model = read_swath_annotation(late_ann)
    early_orbit = read_eof_orbit("/tmp/safe-real/orbits/poeorb_20210630.EOF.zip")
    late_orbit = read_eof_orbit("/tmp/safe-real/orbits/poeorb_20210712.EOF.zip")
    dem = read_srtm_hgt(dem_file, ensure_geoid("/tmp/safe-real/dem", "EGM96"))
    early_rows, late_rows = burst_overlap_rows(early_model, 4)
    count = early_rows[1] - early_rows[0]
    width = 128
    looks = []
    for rows in (early_rows, late_rows):
        early = read_tiff_window(TIFF, rows[0], COL, count, width)
        center = ground_point_at_height(
            early_orbit,
            early_model.line_time(rows[0] + count // 2),
            early_model.slant_range_m(COL + width // 2),
            0.0,
        )
        late_line, late_sample = radar_coordinates(
            late_model, late_orbit, center, guess=late_model.line_time(rows[0] + count // 2)
        )
        late_row = int(np.floor(late_line)) - count // 2
        late_col = int(np.floor(late_sample)) - width // 2
        late = read_tiff_window(late_tiff, late_row, late_col, count, width)
        coreg = backgeocode(
            RadarGrid(rows[0], COL, early, early_model),
            RadarGrid(late_row, late_col, late, late_model),
            reference_orbit=early_orbit,
            secondary_orbit=late_orbit,
            dem=dem,
            max_solves=4096,
            max_residual_px=0.05,
        )
        assert backgeocode.last_residual_px <= 0.05
        looks.append((early, coreg))
    with pytest.raises(RuntimeError, match=r"coherence 0\.\d+ is below 0\.25"):
        enhanced_spectral_diversity(
            looks[0][0],
            looks[1][0],
            looks[0][1],
            looks[1][1],
            separation_hz=overlap_doppler_separation_hz(early_model, 4),
            estimator="Average",
            coherence_threshold=0.25,
        )


def test_precise_orbit_auto_download_covers_the_june_acquisition(tmp_path: Path):
    if not ANN.is_file():
        pytest.skip("IW1 annotation is not on disk")
    model = read_swath_annotation(ANN)
    orbit, path = download_orbit(model.bursts[4].azimuth_time, "S1A", tmp_path, kind="POEORB")
    assert path.is_file()
    assert "S1A" in path.name and "POEORB" in path.name
    assert orbit.time[0] <= model.bursts[4].azimuth_time <= orbit.time[-1]
    local = read_eof_orbit("/tmp/safe-real/orbits/poeorb_20210630.EOF.zip")
    got, _velocity = orbit.interpolate(model.bursts[4].azimuth_time, 3)
    reference, _velocity = local.interpolate(model.bursts[4].azimuth_time, 3)
    assert float(np.linalg.norm(got - reference)) < 1.0


def test_snaphu_configuration_names_float_data(tmp_path: Path):
    phase = np.zeros((16, 16))
    conf = export_snaphu(tmp_path, phase, np.full_like(phase, 0.9))
    text = conf.read_text(encoding="utf-8")
    assert "INFILEFORMAT FLOAT_DATA" in text
    assert "OUTFILEFORMAT FLOAT_DATA" in text
    assert "CORRFILEFORMAT FLOAT_DATA" in text


def test_dem_heights_on_the_two_burst_footprint_use_matching_geoids():
    dem_dir = Path("/tmp/safe-real/dem")
    if not (dem_dir / "N50E012.hgt.gz").is_file():
        pytest.skip("SRTM tile is not on disk")
    egm96 = ensure_geoid(dem_dir, "EGM96")
    egm2008 = ensure_geoid(dem_dir, "EGM2008")
    srtm = read_srtm_hgt(dem_dir / "N50E012.hgt.gz", egm96)
    assert abs(egm96.undulation(50.4, 12.7) - egm2008.undulation(50.4, 12.7)) > 0.05
    height, valid = srtm.ellipsoidal_height(50.4, 12.7)
    assert valid and height > 100.0
