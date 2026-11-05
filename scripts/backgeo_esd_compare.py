"""Back-Geocoding and ESD on the June/July IW1 VV burst pair.

Reference is 30 June. Annotation bursts 4 and 5 are SNAP bursts 5 and 6
(azimuth 2021-06-30T05:26:15.752882 and 2021-06-30T05:26:18.509383).
The range window is samples 11000:11520. Orbits are the POEORB files SNAP
cached. The DEM is SNAP's SRTM 3Sec tile plus EGM96.

Tolerances are written before any SNAP array is read.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ai_insar.pipeline.dem import ensure_geoid, read_raster_dem
from ai_insar.pipeline.geometry import ground_point_at_height
from ai_insar.pipeline.s1_annotation import read_eof_orbit, read_swath_from_product
from ai_insar.pipeline.tops_ops import (
    RadarGrid,
    backgeocode,
    burst_overlap_rows,
    deramp,
    enhanced_spectral_diversity,
    overlap_doppler_separation_hz,
    radar_coordinates,
    repeat_azimuth_time,
)
from ai_insar.safe import _read_href, resolve_safe, select_measurement
from ai_insar.sentinel import read_tiff_window

ROOT = Path("/tmp/stage-compare/backgeo")
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
SNAP_BACKGEO = Path("/tmp/snap-ref/stage/backgeo_roi.dim")
SNAP_ESD = Path("/tmp/snap-ref/stage/esd_roi.dim")

# Declared before SNAP pixels are read. A miss stays a miss.
TOLERANCES = {
    "interior_margin_px": 32,
    "minimum_amplitude": 1.0,
    "secondary_amplitude_relative_median": 0.05,
    "secondary_circular_phase_median_rad": 0.20,
    "reference_copy_max_abs": 0.0,
    "esd_coherence_threshold": 0.3,
    "esd_estimator": "Periodogram",
}


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _valid_line_mask(product, measurement, row: int, height: int, col: int, width: int) -> np.ndarray:
    """True where this crop's samples sit inside first/last valid sample."""
    payload = _read_href(product, measurement.annotation_href)
    root = ET.fromstring(payload)
    bursts = [element for element in root.iter() if _local(element.tag) == "burst"]
    mask = np.zeros((height, width), dtype=bool)
    lines = 1508
    for burst_index, burst in enumerate(bursts):
        first = last = None
        for child in burst:
            if _local(child.tag) == "firstValidSample":
                first = np.array((child.text or "").split(), dtype=np.int32)
            if _local(child.tag) == "lastValidSample":
                last = np.array((child.text or "").split(), dtype=np.int32)
        if first is None or last is None:
            continue
        burst_row = burst_index * lines
        for line, (sample0, sample1) in enumerate(zip(first, last)):
            absolute = burst_row + line
            local = absolute - row
            if local < 0 or local >= height or int(sample0) < 0:
                continue
            c0 = max(col, int(sample0))
            c1 = min(col + width - 1, int(sample1))
            if c0 <= c1:
                mask[local, c0 - col : c1 - col + 1] = True
    return mask


def _complex_from_envi(folder: Path, stem: str) -> np.ndarray | None:
    real = folder / f"i_{stem}.img"
    imag = folder / f"q_{stem}.img"
    if not real.is_file():
        return None
    header = (folder / f"i_{stem}.hdr").read_text(encoding="utf-8", errors="replace")
    samples = int(next(line.split("=")[1] for line in header.splitlines() if line.startswith("samples")))
    lines = int(next(line.split("=")[1] for line in header.splitlines() if line.startswith("lines")))
    endian = ">i2" if "byte order = 1" in header else "<i2"
    if "data type = 4" in header:
        endian = ">f4" if "byte order = 1" in header else "<f4"
    values_i = np.fromfile(real, dtype=endian).reshape(lines, samples)
    values_q = np.fromfile(imag, dtype=endian).reshape(lines, samples)
    return values_i.astype(np.float32) + 1j * values_q.astype(np.float32)


def _band_stems(dim: Path) -> list[str]:
    text = dim.read_text(encoding="utf-8", errors="replace")
    names = []
    for line in text.splitlines():
        if "<BAND_NAME>" in line and line.strip().startswith("<BAND_NAME>i_"):
            names.append(line.split(">", 1)[1].split("<", 1)[0][2:])
    return names


def _circular_stats(reference: np.ndarray, secondary: np.ndarray, valid: np.ndarray) -> dict:
    residual = np.angle(reference * np.conj(secondary))
    used = residual[valid]
    amplitude = np.abs(secondary)[valid]
    reference_amplitude = np.abs(reference)[valid]
    relative = np.abs(amplitude - reference_amplitude) / np.maximum(reference_amplitude, 1e-6)
    if used.size == 0:
        return {"count": 0}
    order = np.argsort(np.abs(used))
    return {
        "count": int(used.size),
        "circular_median_abs_rad": float(np.median(np.abs(used))),
        "circular_p95_abs_rad": float(np.percentile(np.abs(used), 95)),
        "amplitude_relative_median": float(np.median(relative)),
        "first_mismatch_row_col": _first_mismatch(reference, secondary, valid),
    }


def _first_mismatch(reference: np.ndarray, secondary: np.ndarray, valid: np.ndarray) -> list[int] | None:
    phase = np.abs(np.angle(reference * np.conj(secondary)))
    phase = np.where(valid, phase, 0.0)
    hits = np.argwhere(phase > TOLERANCES["secondary_circular_phase_median_rad"])
    if hits.size == 0:
        return None
    return [int(hits[0, 0]), int(hits[0, 1])]


def _esd(master: np.ndarray, slave: np.ndarray, model, row: int, separation: float) -> dict:
    early_rows, late_rows = burst_overlap_rows(model, 4)
    result = {
        "early_rows": [int(early_rows[0]), int(early_rows[1])],
        "late_rows": [int(late_rows[0]), int(late_rows[1])],
        "spectral_separation_hz": separation,
        "estimator": TOLERANCES["esd_estimator"],
        "coherence_threshold": TOLERANCES["esd_coherence_threshold"],
        "applied": False,
    }
    slices = []
    for start, stop in (early_rows, late_rows):
        slices.append((start - row, stop - row))
    try:
        shift, coherence = enhanced_spectral_diversity(
            master[slices[0][0] : slices[0][1]],
            master[slices[1][0] : slices[1][1]],
            slave[slices[0][0] : slices[0][1]],
            slave[slices[1][0] : slices[1][1]],
            separation_hz=separation,
            estimator=TOLERANCES["esd_estimator"],
            coherence_threshold=TOLERANCES["esd_coherence_threshold"],
        )
        result.update({"applied": True, "shift_s": shift, "coherence": coherence})
    except RuntimeError as exc:
        result["message"] = str(exc)
    return result


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "snap_version": "12.0.0",
        "reference_product": JUNE.name,
        "secondary_product": JULY.name,
        "annotation_bursts_0_based": [4, 5],
        "snap_bursts_1_based": [5, 6],
        "range_col": 11000,
        "range_width": 520,
        "dem": str(DEM_PATH),
        "vertical_datum": "SRTM 3Sec orthometric EGM96 + EGM96",
        "resampling": "BILINEAR_INTERPOLATION",
        "tolerances_declared_before_metrics": TOLERANCES,
        "note": "Reference-SLC deburst is not this comparison. Back-geocoding and ESD are steps 3 and 4.",
    }
    (ROOT / "tolerances.json").write_text(json.dumps(TOLERANCES, indent=2) + "\n", encoding="utf-8")

    june = resolve_safe(JUNE / "manifest.safe")
    july = resolve_safe(JULY / "manifest.safe")
    june_meas = select_measurement(june, "IW1", "VV")
    july_meas = select_measurement(july, "IW1", "VV")
    june_model = read_swath_from_product(june, june_meas)
    july_model = read_swath_from_product(july, july_meas)
    assert str(june_model.bursts[4].azimuth_time).startswith("2021-06-30T05:26:15.752882")
    row = 4 * june_model.lines_per_burst
    height = 2 * june_model.lines_per_burst
    col, width = 11000, 520
    reference = read_tiff_window(JUNE / june_meas.href, row, col, height, width)
    june_orbit = read_eof_orbit(ORBIT_JUNE)
    july_orbit = read_eof_orbit(ORBIT_JULY)
    dem = read_raster_dem(DEM_PATH, ensure_geoid(Path("/tmp/safe-real/dem"), "EGM96"), "EGM96")
    ref_grid = RadarGrid(row, col, reference, june_model)
    center_line = row + height // 2
    center_sample = col + width // 2
    ground = ground_point_at_height(
        june_orbit,
        june_model.line_time(center_line),
        june_model.slant_range_m(center_sample),
        0.0,
    )
    late_line, late_sample = radar_coordinates(
        july_model,
        july_orbit,
        ground,
        guess=repeat_azimuth_time(june_model, center_line, july_model),
    )
    margin_y, margin_x = 400, 160
    sec_row = int(np.floor(late_line)) - height // 2 - margin_y
    sec_col = int(np.floor(late_sample)) - width // 2 - margin_x
    sec_height = height + 2 * margin_y
    sec_width = width + 2 * margin_x
    secondary = read_tiff_window(JULY / july_meas.href, sec_row, sec_col, sec_height, sec_width)
    coreg = backgeocode(
        ref_grid,
        RadarGrid(sec_row, sec_col, secondary, july_model),
        reference_orbit=june_orbit,
        secondary_orbit=july_orbit,
        dem=dem,
        max_solves=8192,
        max_residual_px=0.05,
        resampling="BILINEAR_INTERPOLATION",
    )
    valid = _valid_line_mask(june, june_meas, row, height, col, width)
    np.save(ROOT / "reference.npy", reference)
    np.save(ROOT / "backgeo.npy", coreg)
    np.save(ROOT / "valid_mask.npy", valid)
    manifest["ai_insar"] = {
        "shape": list(coreg.shape),
        "secondary_read_row_col": [sec_row, sec_col],
        "secondary_read_shape": [sec_height, sec_width],
        "map_residual_px": float(backgeocode.last_residual_px),
        "stride": int(backgeocode.last_stride),
        "filled_fraction": float(np.mean(np.abs(coreg) > 0)),
        "valid_fraction": float(np.mean(valid)),
        "dem_provenance": dem.provenance,
    }
    master_deramp = deramp(ref_grid, remove=True)
    slave_deramp = deramp(RadarGrid(row, col, coreg, june_model), remove=True)
    separation = overlap_doppler_separation_hz(june_model, 4)
    manifest["ai_insar"]["esd_deramped_valid_masked"] = _esd(
        np.where(valid, master_deramp, 0),
        np.where(valid, slave_deramp, 0),
        june_model,
        row,
        separation,
    )
    manifest["ai_insar"]["esd_reramped_unmasked"] = _esd(reference, coreg, june_model, row, separation)
    manifest["snap_backgeo_present"] = SNAP_BACKGEO.is_file()
    if SNAP_BACKGEO.is_file():
        folder = SNAP_BACKGEO.with_suffix(".data")
        stems = _band_stems(SNAP_BACKGEO)
        manifest["snap_band_stems"] = stems
        cubes = {stem: _complex_from_envi(folder, stem) for stem in stems}
        manifest["snap_comparison"] = _compare_snap(reference, coreg, valid, cubes)
    else:
        manifest["snap_comparison"] = {"status": "SNAP backgeo_roi.dim is not written yet"}
    manifest["snap_esd_present"] = SNAP_ESD.is_file()
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["ai_insar"], indent=2))
    print(json.dumps(manifest.get("snap_comparison"), indent=2))


def _compare_snap(reference, coreg, valid, cubes: dict) -> dict:
    master = next((value for stem, value in cubes.items() if value is not None and "mst" in stem), None)
    slave = next((value for stem, value in cubes.items() if value is not None and "slv" in stem), None)
    if master is None or slave is None or master.shape != coreg.shape:
        return {"status": "SNAP bands were not both present at the ROI shape", "shapes": {key: None if value is None else list(value.shape) for key, value in cubes.items()}}
    margin = int(TOLERANCES["interior_margin_px"])
    interior = np.zeros(valid.shape, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    both = valid & interior & (np.abs(coreg) >= TOLERANCES["minimum_amplitude"]) & (np.abs(slave) >= TOLERANCES["minimum_amplitude"])
    master_diff = np.abs(master - reference)
    return {
        "reference_vs_snap_master_max_abs": float(np.max(master_diff)),
        "reference_vs_snap_master_mismatch_count": int(np.count_nonzero(master_diff)),
        "secondary_raw": _circular_stats(slave, coreg, both),
        "tolerances": TOLERANCES,
    }


if __name__ == "__main__":
    main()
