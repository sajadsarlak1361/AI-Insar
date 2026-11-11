"""AOI parsing, overlap clipping, and the 1 km burst-boundary square."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ai_insar.pipeline.aoi import (
    CHAIN_STAGES,
    SQUARE_100M_REFERENCE_WINDOW,
    SQUARE_100M_WKT,
    AoiError,
    BilinearLocator,
    burst_boundary_square,
    bursts_covering,
    chain_aoi,
    config_from_report,
    format_wkt,
    load_aoi_config,
    overlap_line_count,
    parse_wkt,
    pixel_window_for_polygon,
    polygon_from_bounds,
    prepare_aoi,
    save_aoi_config,
    square_100m_pixel_window,
    validation_square_covers_overlap,
)
from ai_insar.pipeline.aoi import _validation_burst_footprints

ROOT = Path(__file__).resolve().parents[1]
AOI_JSON = ROOT / "docs" / "parity" / "square1000m_burst56" / "AOI.json"


def test_wkt_and_bounds_round_trip():
    polygon = parse_wkt("POLYGON ((13 50, 14 50, 14 51, 13 51, 13 50))")
    assert polygon.ring[0] == (13.0, 50.0)
    box = polygon_from_bounds(51, 50, 13, 14)
    assert parse_wkt(format_wkt(box)).ring[0] == (13.0, 50.0)
    with pytest.raises(AoiError, match="South latitude"):
        polygon_from_bounds(50, 51, 13, 14)
    with pytest.raises(AoiError, match="POLYGON"):
        parse_wkt("LINESTRING (0 0, 1 1)")


def test_overlap_clip_refuses_a_region_that_misses_both_footprints():
    overlap = polygon_from_bounds(51, 50, 13, 14)
    outside = polygon_from_bounds(10, 9, 1, 2)
    report = prepare_aoi(outside, reference=overlap, secondary=overlap)
    assert report.outside
    assert report.area_km2 == 0.0
    with pytest.raises(AoiError, match="outside"):
        from ai_insar.pipeline.aoi import refuse_if_outside

        refuse_if_outside(report)
    inside = prepare_aoi(polygon_from_bounds(50.5, 50.2, 13.2, 13.6), reference=overlap, secondary=overlap)
    assert not inside.outside
    assert inside.area_km2 > 0.0
    assert "km²" in inside.message


def test_hundred_metre_square_still_maps_to_the_parity_window():
    geo = [
        (13.326887869665738, 50.37613874744699),
        (13.325508840648142, 50.37631330257908),
        (13.325235898826342, 50.375431422913316),
        (13.32661490328058, 50.37525687099057),
    ]
    pixels = [
        (2469.902178874556, 260.46713374857944),
        (2491.557741988094, 260.4671214177977),
        (2491.095673204336, 268.5329125140802),
        (2469.444445462294, 268.5329073446804),
    ]
    locator = BilinearLocator(geo, pixels)
    window = square_100m_pixel_window(locator)
    assert window.y == SQUARE_100M_REFERENCE_WINDOW["row0"]
    assert window.row_end == SQUARE_100M_REFERENCE_WINDOW["row1"]
    assert window.x == SQUARE_100M_REFERENCE_WINDOW["column0"]
    assert window.column_end == SQUARE_100M_REFERENCE_WINDOW["column1"]
    # The rounded WKT, not only the full-precision corners, stays on that window.
    rounded = pixel_window_for_polygon(parse_wkt(SQUARE_100M_WKT), locator)
    assert (rounded.y, rounded.row_end, rounded.x, rounded.column_end) == (
        window.y,
        window.row_end,
        window.x,
        window.column_end,
    )


def test_one_kilometre_square_selects_bursts_5_and_6_and_their_overlap():
    report = burst_boundary_square()
    assert report.subswath == "IW1"
    assert report.polarisation == "VV"
    assert (report.burst_first, report.burst_last) == (5, 6)
    assert report.burst_ids == (359478, 359479)
    assert report.esd_overlap_bursts == (5, 6)
    assert not report.outside
    assert abs(report.centre_latitude - 50.24663071127351) < 1e-9
    assert abs(report.centre_longitude - 13.286084846039746) < 1e-9
    burst_5, burst_6 = _validation_burst_footprints(overlap_line_count())
    square = parse_wkt(report.clipped_wkt)
    assert bursts_covering((burst_5, burst_6), square) == (1, 2)
    assert bursts_covering((burst_5, burst_6), parse_wkt(SQUARE_100M_WKT)) == (1, 1)
    cover = validation_square_covers_overlap()
    assert cover["square_straddles_boundary"]
    assert cover["overlap_inside_square"] > 40.0
    assert cover["square_line_start"] < cover["boundary_line"] < cover["square_line_stop"]
    stages = chain_aoi(report)
    assert list(stages) == list(CHAIN_STAGES)
    assert len(stages) == 15
    assert all(item["burst_first"] == 5 and item["burst_last"] == 6 for item in stages.values())
    assert all(item["esd_overlap_bursts"] == [5, 6] for item in stages.values())
    saved = json.loads(AOI_JSON.read_text(encoding="utf-8"))
    assert saved["clipped_wkt"] == report.clipped_wkt
    assert saved["centre_latitude"] == report.centre_latitude
    assert saved["burst_first"] == 5 and saved["burst_last"] == 6


def test_aoi_config_round_trip(tmp_path: Path):
    config = config_from_report(burst_boundary_square())
    path = save_aoi_config(tmp_path / "aoi.json", config)
    loaded = load_aoi_config(path)
    assert loaded.wkt == config.wkt
    assert loaded.burst_first == 5 and loaded.burst_last == 6
    assert loaded.to_dict()["geoRegion"] == config.wkt
    assert loaded.to_dict()["subSamplingX"] == 1
    assert loaded.to_dict()["copyMetadata"] is True


def test_deburst_stitches_the_two_bursts_whose_overlap_the_square_covers():
    from ai_insar.pipeline.geometry import Orbit
    from ai_insar.pipeline.s1_annotation import BurstTiming, SwathModel
    from ai_insar.pipeline.tops_ops import burst_overlap_rows, deburst_output_rows

    start = np.datetime64("2021-06-30T05:26:15.752882")
    following = start + np.timedelta64(int(round(2.756501 * 1e9)), "ns")
    orbit = Orbit(
        np.array([start, following]),
        np.array([[7_000_000.0, 0.0, 0.0], [7_000_000.0, 1_000.0, 0.0]]),
        np.array([[0.0, 7_000.0, 0.0], [0.0, 7_000.0, 0.0]]),
    )

    def burst(when: np.datetime64) -> BurstTiming:
        return BurstTiming(when, 0, 1507, 0, 20_000)

    model = SwathModel(
        wavelength_m=0.05546576,
        azimuth_time_interval_s=0.002055556299999998,
        range_pixel_spacing_m=2.329562,
        slant_range_time_s=0.005,
        lines_per_burst=1508,
        samples_per_burst=22_849,
        number_of_lines=3016,
        incidence_rad=0.6,
        fm_rate=np.zeros(3),
        bursts=(burst(start), burst(following)),
        orbit=orbit,
    )
    early, late = burst_overlap_rows(model, 0)
    assert early[1] - early[0] == overlap_line_count()
    assert late[0] == 1508
    stitched = deburst_output_rows(model, [0, 1])
    assert np.any((stitched >= 0) & (stitched < 1508))
    assert np.any(stitched >= 1508)
    cover = validation_square_covers_overlap()
    assert early[0] <= cover["boundary_line"] <= early[1]


def test_subset_dialog_exposes_snap_parameter_names():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ai_insar.gui.late_dialogs import SubsetDialog, TerrainCorrectionDialog

    QApplication.instance() or QApplication([])
    dialog = SubsetDialog(height=9, width=23, bands=["i_ifg", "q_ifg", "coh"])
    dialog.geo_region.setPlainText(SQUARE_100M_WKT)
    params = dialog.parameters()
    assert set(params) >= {"region", "geoRegion", "subSamplingX", "subSamplingY", "sourceBands", "copyMetadata"}
    assert params["region"] == {"x": 0, "y": 0, "width": 23, "height": 9}
    assert params["subSamplingX"] == 1 and params["subSamplingY"] == 1
    assert params["copyMetadata"] is True
    assert params["sourceBands"] == ["i_ifg", "q_ifg", "coh"]
    assert params["row_end"] == 9 and params["col_end"] == 23
    terrain = TerrainCorrectionDialog()
    assert terrain.findChild(type(terrain.geo_region), "tcGeoRegion") is not None
    terrain.geo_region.setPlainText(SQUARE_100M_WKT)
    assert terrain.parameters()["geoRegion"] == SQUARE_100M_WKT
    dialog.close()
    terrain.close()


def test_processing_area_dialog_draws_the_square_on_both_footprints(tmp_path: Path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ai_insar.gui.aoi_dialog import AoiDialog

    application = QApplication.instance() or QApplication([])
    report = burst_boundary_square()
    burst_5, burst_6 = _validation_burst_footprints(overlap_line_count())
    reference = tuple(burst_5.closed())
    secondary = tuple(burst_6.closed())
    dialog = AoiDialog(
        reference=reference,
        secondary=secondary,
        burst_footprints=(burst_5, burst_6),
        wkt=report.clipped_wkt,
        subswath="IW1",
        first_burst_number=5,
    )
    dialog.resize(900, 640)
    dialog.show()
    application.processEvents()
    assert "bursts 5–6" in dialog.status.text()
    assert "outside" not in dialog.status.text().lower()
    image = tmp_path / "aoi-dialog.png"
    dialog.grab().save(str(image))
    assert image.stat().st_size > 1000
    artifact = Path("/opt/cursor/artifacts")
    if artifact.is_dir():
        dialog.grab().save(str(artifact / "aoi-burst-boundary.png"))
    dialog.close()
