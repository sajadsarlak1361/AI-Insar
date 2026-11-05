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
    deburst_output_rows,
    deburst_window,
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
    geocoded = write_beam_dimap(
        tmp_path,
        "geocoded",
        values,
        (12.6, 50.2, 12.9, 50.5),
        band_name="IW1_VV",
        unit="1",
    )
    geocoded_out = tmp_path / "geocoded_subset.dim"
    geocoded_proc = subprocess.run(
        [
            "/home/ubuntu/esa-snap/bin/gpt",
            "Subset",
            f"-Ssource={geocoded}",
            "-Pregion=0,0,4,3",
            "-t",
            str(geocoded_out),
        ],
        capture_output=True,
        text=True,
    )
    if geocoded_proc.returncode != 0:
        pytest.fail(geocoded_proc.stderr[-800:])
    opened = geocoded_out.read_text(encoding="utf-8", errors="replace")
    assert "TIE_POINT_GRID_NAME_LAT" in opened
    assert "latitude" in opened
    geo_header = (tmp_path / "geocoded_subset.data" / "i_IW1_VV.hdr").read_text(encoding="utf-8")
    geo_endian = ">i2" if "byte order = 1" in geo_header else "<i2"
    geo_real = np.fromfile(tmp_path / "geocoded_subset.data" / "i_IW1_VV.img", dtype=geo_endian)
    geo_imag = np.fromfile(tmp_path / "geocoded_subset.data" / "q_IW1_VV.img", dtype=geo_endian)
    assert np.array_equal(geo_real, np.arange(12, dtype=np.int16))
    assert np.array_equal(geo_imag, np.arange(12, dtype=np.int16))


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


def test_snap_deburst_of_bursts_5_and_6_matches_output_grid():
    """SNAP 12 TOPSAR-Deburst of annotation bursts 4 and 5, samples 11000:11520.

    Burst 4 (0-based) is the burst whose azimuth time is
    2021-06-30T05:26:15.752882. That is SNAP firstBurstIndex 5. The output
    has 2847 lines because (lastLineTime-firstLineTime)/dt truncates.
    """
    measurement = Path(
        "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
        "/measurement/s1a-iw1-slc-vv-20210630t052604-20210630t052629-038565-048cfd-004.tiff"
    )
    snap_i_path = Path("/tmp/snap-ref/stage/deburst_ref.data/i_IW1_VV.img")
    snap_q_path = Path("/tmp/snap-ref/stage/deburst_ref.data/q_IW1_VV.img")
    if not measurement.is_file() or not snap_i_path.is_file() or not ANN.is_file():
        pytest.skip("SNAP deburst reference is not on disk")
    model = read_swath_annotation(ANN)
    assert str(model.bursts[4].azimuth_time).startswith("2021-06-30T05:26:15.752882")
    assert str(model.bursts[5].azimuth_time).startswith("2021-06-30T05:26:18.509383")
    plan = deburst_output_rows(model, [4, 5])
    assert plan.shape == (2847,)
    assert int(np.sum(plan < 0)) == 36
    window = read_tiff_window(measurement, ROW, 11000, HEIGHT, 520)
    debursted = deburst_window(window, model, ROW, 11000)
    assert debursted.data.shape == (2847, 520)
    snap_i = np.fromfile(snap_i_path, dtype=">i2").reshape(2847, 520)
    snap_q = np.fromfile(snap_q_path, dtype=">i2").reshape(2847, 520)
    snap = snap_i.astype(np.float32) + 1j * snap_q.astype(np.float32)
    assert float(np.max(np.abs(debursted.data - snap))) == 0.0


def test_reramp_uses_the_source_phase_at_the_resampled_coordinate():
    """A half-pixel shift must restore the secondary chirp, not a different one.

    Deramp, bilinear resample, and reramp with the phase sampled by that same
    kernel returns the baseband times the secondary phase. Adding a different
    azimuth slope, which is what a reference-grid chirp does, leaves that slope
    in the residual. No coefficient is fitted.
    """
    from ai_insar.pipeline.tops_ops import resample_complex, resample_real

    rows, cols = 40, 6
    kt = 1733.0
    interval = 0.0020555563
    eta = np.arange(rows, dtype=np.float64) * interval
    phase_1d = -np.pi * kt * (eta - 0.031) ** 2 - 2.0 * np.pi * (-4.0) * eta
    phase = np.broadcast_to(phase_1d[:, None], (rows, cols)).copy()
    baseband = np.full((rows, cols), np.exp(1j * 0.35), dtype=np.complex128)
    slc = (baseband * np.exp(-1j * phase)).astype(np.complex64)
    deramped = (slc * np.exp(1j * phase)).astype(np.complex64)
    out_rows = rows - 2
    line = np.broadcast_to((np.arange(out_rows, dtype=np.float64) + 0.5)[:, None], (out_rows, cols)).copy()
    sample = np.broadcast_to(np.arange(cols, dtype=np.float64)[None, :], (out_rows, cols)).copy()
    sampled = resample_complex(deramped, line, sample, "BILINEAR_INTERPOLATION")
    phase_at = resample_real(phase, line, sample, "BILINEAR_INTERPOLATION")
    restored = sampled * np.exp(-1j * phase_at)
    expected = baseband[:out_rows] * np.exp(-1j * phase_at)
    residual = np.abs(np.angle(restored * np.conj(expected)))
    assert float(np.median(residual)) < 1e-3
    other = phase_at + 0.12 * np.arange(out_rows, dtype=np.float64)[:, None]
    wrong = np.abs(np.angle(sampled * np.exp(-1j * other) * np.conj(expected)))
    assert float(np.median(wrong)) > 0.2


def test_esd_blocks_keep_a_known_shift_and_reject_a_dark_block():
    """Block ESD is the weighted periodogram SNAP writes, in pixels.

    The phase is master·conj(slave), early·conj(late). A block whose early
    look has no power has weight 0 and is not a measurement. The 0.3 gate is
    that per-pixel weight, not a fit, and it is not lowered.
    """
    from ai_insar.pipeline.tops_ops import esd_applied_shift_px, overlap_block_shifts

    separation = 6400.0
    interval = 0.002
    pixels = -0.0015
    phase = 2.0 * np.pi * separation * interval * pixels
    rows, cols = 16, 20
    master = np.ones((rows, cols), dtype=np.complex64)
    slave_early = master.copy()
    # Zero early-look power in the second block. SNAP then stores weight 0.
    slave_early[:, cols // 2 :] = 0
    slave_late = np.exp(-1j * phase) * np.ones((rows, cols), dtype=np.complex64)
    reports = overlap_block_shifts(
        master,
        master,
        slave_early,
        slave_late,
        separation_hz=separation,
        azimuth_time_interval_s=interval,
        n_blocks=2,
        coherence_threshold=0.3,
    )
    # master·conj(slave) on this fixture is the opposite of the injected tone.
    assert reports[0]["accepted"] is True
    assert abs(float(reports[0]["shift_px"]) - (-pixels)) < 1.5e-4
    assert abs(float(reports[0]["shift_s"]) / interval - (-pixels)) < 1.5e-4
    assert reports[1]["accepted"] is False
    assert float(reports[1]["weight"]) == 0.0
    applied = esd_applied_shift_px(reports)
    assert abs(applied - (-float(reports[0]["shift_px"]))) < 1e-9


def test_overlap_azimuth_stays_on_the_requested_burst():
    """The same zero-Doppler instant in the overlap is two raster lines.

    The earlier burst is the default. Back-geocoding asks for the reference
    burst, which is the later copy and a different TOPS phase.
    """
    from ai_insar.pipeline.s1_annotation import BurstTiming, SwathModel
    from ai_insar.pipeline.tops_ops import _absolute_line

    interval = 0.002
    lines = 10
    start = np.datetime64("2021-06-30T00:00:00.000000000")
    later = start + np.timedelta64(int(round(8 * interval * 1e9)), "ns")
    bursts = (
        BurstTiming(start, 0, 9, 0, 10),
        BurstTiming(later, 0, 9, 0, 10),
    )
    model = SwathModel(
        wavelength_m=0.055,
        azimuth_time_interval_s=interval,
        range_pixel_spacing_m=2.3,
        slant_range_time_s=0.005,
        lines_per_burst=lines,
        samples_per_burst=20,
        number_of_lines=20,
        incidence_rad=0.6,
        fm_rate=np.array([-2000.0, 0.0, 0.0]),
        bursts=bursts,
        orbit=None,
    )
    early = _absolute_line(model, later)
    requested = _absolute_line(model, later, prefer_burst=1)
    assert early == pytest.approx(8.0)
    assert requested == pytest.approx(10.0)


def test_backgeo_secondary_phase_matches_snap_on_an_interior_patch():
    """Reramp uses the secondary phase. An interior patch must meet the declared bounds.

    Bounds are the ones fixed for the ROI: amplitude relative median 0.05 and
    circular phase median 0.20 rad. This patch is the same June/July bursts,
    SRTM 3Sec, and SNAP 12 slave as the full comparison.
    """
    from ai_insar.pipeline.dem import read_raster_dem
    from ai_insar.pipeline.s1_annotation import read_swath_from_product
    from ai_insar.pipeline.tops_ops import backgeocode
    from ai_insar.safe import resolve_safe, select_measurement

    june = Path(
        "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
    )
    july = Path(
        "/tmp/safe-full/S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B.SAFE"
    )
    snap_i = Path("/tmp/snap-ref/stage/backgeo_roi.data/i_IW1_VV_slv1_12Jul2021.img")
    snap_q = Path("/tmp/snap-ref/stage/backgeo_roi.data/q_IW1_VV_slv1_12Jul2021.img")
    dem_path = Path("/tmp/snap-ref/dem/srtm_39_02.tif")
    orbit_june = Path(
        "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/06/"
        "S1A_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF.zip"
    )
    orbit_july = Path(
        "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/07/"
        "S1A_OPER_AUX_POEORB_OPOD_20210801T121825_V20210711T225942_20210713T005942.EOF.zip"
    )
    if not all(path.is_file() for path in (snap_i, snap_q, dem_path, orbit_june, orbit_july)):
        pytest.skip("SNAP back-geocoding ROI or DEM/orbit is not on disk")
    if not (june / "manifest.safe").is_file() or not (july / "manifest.safe").is_file():
        pytest.skip("complete June/July SAFE is not on disk")
    june_product = resolve_safe(june / "manifest.safe")
    july_product = resolve_safe(july / "manifest.safe")
    june_meas = select_measurement(june_product, "IW1", "VV")
    july_meas = select_measurement(july_product, "IW1", "VV")
    june_model = read_swath_from_product(june_product, june_meas)
    july_model = read_swath_from_product(july_product, july_meas)
    secondary = read_tiff_window(july / july_meas.href, 5466, 10817, 3816, 840)
    dem = read_raster_dem(dem_path, ensure_geoid(Path("/tmp/safe-real/dem"), "EGM96"), "EGM96")
    june_orbit = read_eof_orbit(orbit_june)
    july_orbit = read_eof_orbit(orbit_july)
    snap = np.fromfile(snap_i, dtype=">f4").reshape(3016, 520)
    snap = snap + 1j * np.fromfile(snap_q, dtype=">f4").reshape(3016, 520)
    # Burst start, mid-burst, and the burst-5 overlap copy. The overlap used
    # to be read from the previous burst and the phase sat near pi/2.
    for r0, c0, height, width in ((8, 240, 24, 24), (200, 200, 48, 48), (1560, 240, 24, 24)):
        reference = read_tiff_window(june / june_meas.href, ROW + r0, COL + c0, height, width)
        coreg = backgeocode(
            RadarGrid(ROW + r0, COL + c0, reference, june_model),
            RadarGrid(5466, 10817, secondary, july_model),
            reference_orbit=june_orbit,
            secondary_orbit=july_orbit,
            dem=dem,
            max_solves=height * width,
            max_residual_px=0.05,
        )
        slave = snap[r0 : r0 + height, c0 : c0 + width]
        phase = np.abs(np.angle(slave * np.conj(coreg)))
        relative = np.abs(np.abs(slave) - np.abs(coreg)) / np.maximum(np.abs(slave), 1e-6)
        assert float(np.median(phase)) <= 0.20
        assert float(np.median(relative)) <= 0.05


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
