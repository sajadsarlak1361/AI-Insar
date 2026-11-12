"""Headless pair: track check, AOI, Split, annotation orbit, Back Geocoding, interferogram."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_insar.pipeline.aoi import footprint_polygon, prepare_aoi, refuse_if_outside
from ai_insar.pipeline.s1_annotation import read_swath_from_product, split_annotation
from ai_insar.pipeline.tops import split_swath
from ai_insar.pipeline.tops_ops import RadarGrid, backgeocode
from ai_insar.safe import resolve_safe
from ai_insar.safe_stack import inspect_safe_products


def run_safe_pair(
    reference: Path | str,
    secondary: Path | str,
    output: Path | str,
    *,
    aoi: str | None = None,
    swath: str = "IW1",
    pol: str = "VV",
) -> dict:
    """Run the first interferogram stages on two SAFE products.

    Apply Orbit File keeps the annotation state vectors. Nothing is downloaded.
    Back Geocoding uses the ellipsoid, so the run does not need a DEM tile.
    """
    reference_path = Path(reference)
    secondary_path = Path(secondary)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    records = inspect_safe_products(
        [reference_path, secondary_path],
        swath=swath,
        pol=pol,
        reference=reference_path,
    )
    reference_product = resolve_safe(reference_path)
    secondary_product = resolve_safe(secondary_path)
    reference_footprint = footprint_polygon(reference_product.footprint)
    secondary_footprint = footprint_polygon(secondary_product.footprint)
    region = aoi or _inset(reference_footprint, secondary_footprint)
    report = refuse_if_outside(
        prepare_aoi(
            region,
            reference=reference_footprint,
            secondary=secondary_footprint,
            subswath=swath,
            polarisation=pol,
        )
    )
    reference_data, _measurement, (row, height, width) = split_swath(
        reference_product, swath=swath, pol=pol, burst_first=1, burst_last=1
    )
    secondary_data, secondary_measurement, _window = split_swath(
        secondary_product, swath=swath, pol=pol, burst_first=1, burst_last=1
    )
    reference_model = split_annotation(
        read_swath_from_product(reference_product, _measurement), 1, 1
    )
    secondary_model = split_annotation(
        read_swath_from_product(secondary_product, secondary_measurement), 1, 1
    )
    reference_grid = RadarGrid(0, 0, reference_data, reference_model)
    secondary_grid = RadarGrid(0, 0, secondary_data, secondary_model)
    (folder / "orbit.txt").write_text(
        "Apply Orbit File kept the annotation state vectors. "
        f"Reference vectors {reference_model.orbit.time.size}. "
        f"Secondary vectors {secondary_model.orbit.time.size}. "
        "No orbit file was downloaded.\n",
        encoding="utf-8",
    )
    coregistered = backgeocode(
        reference_grid,
        secondary_grid,
        reference_orbit=reference_model.orbit,
        secondary_orbit=secondary_model.orbit,
        dem=None,
        resampling="BILINEAR_INTERPOLATION",
    )
    interferogram = coregistered * np.conj(reference_data)
    np.save(folder / "reference.npy", reference_data)
    np.save(folder / "secondary.npy", coregistered)
    np.save(folder / "interferogram.npy", interferogram)
    (folder / "aoi.wkt").write_text(report.clipped_wkt + "\n", encoding="utf-8")
    return {
        "records": len(records),
        "track": records[0].track,
        "split_shape": (int(height), int(width)),
        "row": int(row),
        "interferogram": str(folder / "interferogram.npy"),
        "aoi": report.clipped_wkt,
    }


def _inset(reference, secondary) -> str:
    from ai_insar.pipeline.aoi import format_wkt, geographic_bounds, overlap_polygon, polygon_from_bounds

    overlap = None
    if reference is not None and secondary is not None:
        overlap = overlap_polygon(reference, secondary)
    target = overlap or reference or secondary
    if target is None:
        raise ValueError("the SAFE products have no footprint for an analysis region")
    bounds = geographic_bounds(target)
    span_lat = max(bounds.north - bounds.south, 1.0e-6)
    span_lon = max(bounds.east - bounds.west, 1.0e-6)
    polygon = polygon_from_bounds(
        bounds.north - 0.2 * span_lat,
        bounds.south + 0.2 * span_lat,
        bounds.west + 0.2 * span_lon,
        bounds.east - 0.2 * span_lon,
    )
    return format_wkt(polygon)
