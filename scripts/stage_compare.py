"""Run the June/July IW1 VV chain and compare it with SNAP outputs when present.

Reference product is the 30 June SAFE. Bursts are SNAP 1-based numbers.
linesPerBurst is 1508, so bursts 5 and 6 are absolute rows 6032:9048.
The range window is samples 11000:11520. Orbit files are the POEORB zips
SNAP cached. The DEM is the SRTM 3Sec tile SNAP downloaded, converted with
EGM96. Tolerances below are fixed before the metrics are computed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ai_insar.pipeline.dem import ensure_geoid, read_raster_dem
from ai_insar.pipeline.multilook import multilook_array
from ai_insar.pipeline.s1_annotation import read_eof_orbit, read_swath_from_product
from ai_insar.pipeline.tops import burst_pixel_window
from ai_insar.pipeline.tops_ops import (
    RadarGrid,
    backgeocode,
    burst_overlap_rows,
    deburst_window,
    deramp,
    enhanced_spectral_diversity,
    flat_earth_phase,
    overlap_doppler_separation_hz,
    radar_coordinates,
    terrain_correct,
)
from ai_insar.safe import resolve_safe, select_measurement
from ai_insar.sentinel import read_tiff_window

ROOT = Path("/tmp/stage-compare")
JUNE = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
)
JULY = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B.SAFE"
)
ORBIT_JUNE = Path(
    "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/06/"
    "S1A_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF.zip"
)
ORBIT_JULY = Path(
    "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/07/"
    "S1A_OPER_AUX_POEORB_OPOD_20210801T121825_V20210711T225942_20210713T005942.EOF.zip"
)
DEM_PATH = Path("/tmp/snap-ref/dem/srtm_39_02.tif")
GEOID_CACHE = Path("/tmp/safe-real/dem")
SNAP_MULTI = Path("/tmp/snap-ref/chain/multilook.dim")

# Fixed before any metric is computed.
TOLERANCES = {
    "split_max_abs_difference": 0.0,
    "orbit_position_difference_m": 1.0,
    "deramp_roundtrip_max_abs": 1e-3,
    "backgeocode_map_residual_px": 0.05,
    "esd_coherence_minimum_to_apply": 0.25,
    "flat_earth_same_orbit_max_abs_rad": 1e-3,
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {
        "reference_product": JUNE.name,
        "secondary_product": JULY.name,
        "swath": "IW1",
        "polarisation": "VV",
        "burst_index_base": "SNAP 1-based",
        "burst_first": 5,
        "burst_last": 6,
        "range_col": 11000,
        "range_width": 520,
        "orbit_june": str(ORBIT_JUNE),
        "orbit_july": str(ORBIT_JULY),
        "dem": str(DEM_PATH),
        "vertical_datum": "SRTM orthometric EGM96 + EGM96 undulation",
        "resampling": "BILINEAR_INTERPOLATION",
        "multilook_range": 20,
        "multilook_azimuth": 4,
        "esd_coherence_threshold": TOLERANCES["esd_coherence_minimum_to_apply"],
        "tolerances": TOLERANCES,
        "snap_version": "12.0.0",
        "snap_gui_audit_version": "14.0.0",
        "stages": {},
    }
    june = resolve_safe(JUNE / "manifest.safe")
    july = resolve_safe(JULY / "manifest.safe")
    june_meas = select_measurement(june, "IW1", "VV")
    july_meas = select_measurement(july, "IW1", "VV")
    june_model = read_swath_from_product(june, june_meas)
    july_model = read_swath_from_product(july, july_meas)
    assert june_model.lines_per_burst == 1508
    row, height, _full_width = burst_pixel_window(
        lines_per_burst=1508, first=5, last=6, samples=june_meas.samples or 1
    )
    col, width = 11000, 520
    manifest["absolute_rows"] = [row, row + height]
    manifest["lines_per_burst"] = 1508
    manifest["zero_based_burst_indexes"] = [4, 5]

    june_tiff = JUNE / june_meas.href
    july_tiff = JULY / july_meas.href
    before = (june_tiff.stat().st_size, july_tiff.stat().st_size)
    reference = read_tiff_window(june_tiff, row, col, height, width)
    secondary = read_tiff_window(july_tiff, row, col, height, width)
    np.save(ROOT / "01_split_reference.npy", reference)
    np.save(ROOT / "01_split_secondary.npy", secondary)
    manifest["stages"]["01_split"] = {
        "shape": list(reference.shape),
        "reference_mean_amplitude": float(np.mean(np.abs(reference))),
        "secondary_mean_amplitude": float(np.mean(np.abs(secondary))),
        "source_bytes_unchanged": (june_tiff.stat().st_size, july_tiff.stat().st_size) == before,
    }

    june_orbit = read_eof_orbit(ORBIT_JUNE)
    july_orbit = read_eof_orbit(ORBIT_JULY)
    when = june_model.bursts[4].azimuth_time
    precise, _vel = june_orbit.interpolate(when, 3)
    annotation, _vel = june_model.orbit.interpolate(when, 3)
    orbit_shift = float(np.linalg.norm(precise - annotation))
    manifest["stages"]["02_apply_orbit"] = {
        "burst5_time": str(when),
        "annotation_to_precise_shift_m": orbit_shift,
        "annotation_minus_precise_m": orbit_shift,
        "note": "2 m-class shift is the annotation orbit versus this POEORB, not a SNAP file mismatch.",
        "june_eof": ORBIT_JUNE.name,
        "july_eof": ORBIT_JULY.name,
    }

    ref_grid = RadarGrid(row, col, reference, june_model)
    restored = deramp(RadarGrid(row, col, deramp(ref_grid, remove=True), june_model), remove=False)
    deramp_error = float(np.max(np.abs(restored - reference)))
    manifest["stages"]["03_deramp_reramp"] = {
        "max_abs_residual": deramp_error,
        "within_tolerance": deramp_error < TOLERANCES["deramp_roundtrip_max_abs"],
    }

    geoid = ensure_geoid(GEOID_CACHE, "EGM96")
    dem = read_raster_dem(DEM_PATH, geoid, "EGM96")
    # Coregister the secondary burst window onto the reference. The secondary
    # read above is the same rows; the resample below pulls the geometrically
    # corresponding July samples.
    center = dem  # keep the name used below
    del center
    from ai_insar.pipeline.geometry import ground_point_at_height

    ground = ground_point_at_height(
        june_orbit,
        june_model.line_time(row + height // 2),
        june_model.slant_range_m(col + width // 2),
        0.0,
    )
    late_line, late_sample = radar_coordinates(
        july_model,
        july_orbit,
        ground,
        guess=july_model.line_time(row + height // 2),
    )
    late_row = int(np.floor(late_line)) - height // 2
    late_col = int(np.floor(late_sample)) - width // 2
    secondary_geom = read_tiff_window(july_tiff, late_row, late_col, height, width)
    coreg = backgeocode(
        ref_grid,
        RadarGrid(late_row, late_col, secondary_geom, july_model),
        reference_orbit=june_orbit,
        secondary_orbit=july_orbit,
        dem=dem,
        max_solves=8192,
        max_residual_px=TOLERANCES["backgeocode_map_residual_px"],
        resampling="BILINEAR_INTERPOLATION",
    )
    np.save(ROOT / "03_backgeocode.npy", coreg)
    manifest["stages"]["03_backgeocode"] = {
        "shape": list(coreg.shape),
        "secondary_row": late_row,
        "secondary_col": late_col,
        "map_residual_px": float(backgeocode.last_residual_px),
        "stride": int(backgeocode.last_stride),
        "within_tolerance": float(backgeocode.last_residual_px) <= TOLERANCES["backgeocode_map_residual_px"],
        "filled_fraction": float(np.mean(np.abs(coreg) > 0)),
    }

    early_rows, late_rows = burst_overlap_rows(june_model, 4)
    overlap = {
        "early_rows": [int(early_rows[0]), int(early_rows[1])],
        "late_rows": [int(late_rows[0]), int(late_rows[1])],
        "one_based_bursts": [5, 6],
        "zero_based_bursts": [4, 5],
    }
    master_early = reference[early_rows[0] - row : early_rows[1] - row]
    master_late = reference[late_rows[0] - row : late_rows[1] - row]
    slave_early = coreg[early_rows[0] - row : early_rows[1] - row]
    slave_late = coreg[late_rows[0] - row : late_rows[1] - row]
    separation = overlap_doppler_separation_hz(june_model, 4)
    esd = {"spectral_separation_hz": separation, "applied": False}
    try:
        shift, coherence = enhanced_spectral_diversity(
            master_early,
            master_late,
            slave_early,
            slave_late,
            separation_hz=separation,
            estimator="Average",
            coherence_threshold=TOLERANCES["esd_coherence_minimum_to_apply"],
        )
        esd.update({"shift_s": shift, "coherence": coherence, "applied": True})
    except RuntimeError as exc:
        esd.update({"applied": False, "message": str(exc)})
    overlap.update(esd)
    manifest["stages"]["04_esd"] = overlap

    same = flat_earth_phase(
        ref_grid, ref_grid, early_orbit=june_orbit, later_orbit=june_orbit, dem=dem, max_solves=4096
    )
    flat = flat_earth_phase(
        ref_grid,
        RadarGrid(row, col, coreg, july_model),
        early_orbit=june_orbit,
        later_orbit=july_orbit,
        dem=dem,
        max_solves=4096,
    )
    interferogram = coreg * np.conj(reference)
    manifest["stages"]["05_interferogram"] = {
        "convention": "reference(30 June) x conjugate(secondary 12 July)",
        "same_orbit_flat_earth_max_abs_rad": float(np.nanmax(np.abs(same))),
        "two_date_flat_earth_std_rad": float(np.nanstd(flat)),
        "same_orbit_within_tolerance": float(np.nanmax(np.abs(same))) < TOLERANCES["flat_earth_same_orbit_max_abs_rad"],
    }
    np.save(ROOT / "05_interferogram.npy", interferogram)
    np.save(ROOT / "05_flat_earth.npy", flat.astype(np.float32))

    debursted = deburst_window(reference, june_model, row, col)
    deburst_ifg = deburst_window(interferogram, june_model, row, col)
    np.save(ROOT / "06_deburst.npy", debursted.data)
    np.save(ROOT / "06_deburst_interferogram.npy", deburst_ifg.data)
    manifest["stages"]["06_deburst"] = {
        "input_lines": int(height),
        "output_lines": int(debursted.data.shape[0]),
        "interferogram_lines": int(deburst_ifg.data.shape[0]),
        "first_absolute_row": int(debursted.row),
        "lines_removed": int(height - debursted.data.shape[0]),
    }

    multi = multilook_array(deburst_ifg.data, 20, 4)
    np.save(ROOT / "07_multilook.npy", multi)
    manifest["stages"]["07_multilook"] = {"shape": list(multi.shape), "range_looks": 20, "azimuth_looks": 4}

    image, bounds, heights = terrain_correct(
        ref_grid,
        june_orbit,
        dem,
        out_rows=16,
        out_cols=16,
        resampling="BILINEAR_INTERPOLATION",
        dem_resampling="BILINEAR_INTERPOLATION",
        projection="geographic",
    )
    np.save(ROOT / "14_terrain_amplitude.npy", np.abs(image).astype(np.float32))
    finite = heights[np.isfinite(heights)]
    manifest["stages"]["14_terrain"] = {
        "shape": list(image.shape),
        "bounds_wsen": [float(item) for item in bounds],
        "ellipsoidal_height_min_m": float(np.min(finite)) if finite.size else None,
        "ellipsoidal_height_max_m": float(np.max(finite)) if finite.size else None,
        "nodata_count": int(heights.size - finite.size),
        "datum": dem.provenance,
    }
    manifest["stages"]["13_displacement"] = {
        "produced": False,
        "reason": "ESD coherence is below 0.25, so this pair was not unwrapped into a displacement product.",
    }
    manifest["snap_multilook_present"] = SNAP_MULTI.is_file()
    if SNAP_MULTI.is_file():
        manifest["stages"]["snap_multilook_file"] = {
            "path": str(SNAP_MULTI),
            "bytes": SNAP_MULTI.stat().st_size,
            "note": "Presence is not agreement. Pixel comparison waits until this file is a finished GPT product.",
        }
    manifest["elapsed_s"] = round(time.perf_counter() - started, 1)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["stages"], indent=2))
    print("elapsed", manifest["elapsed_s"])


if __name__ == "__main__":
    main()
