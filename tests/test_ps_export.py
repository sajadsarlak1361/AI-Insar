"""PS-linear products contain only the selected pixels and a real CRS."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from ai_insar.pipeline.ps_export import (
    export_ps_linear,
    geographic_edges,
    subsidence_rgba,
    symmetric_limits,
)


def _scene():
    # Two selected points. The other cells are a dense range-striped field.
    values = np.tile(np.linspace(-0.02, 0.02, 5), (4, 1))
    valid = np.zeros(values.shape, dtype=bool)
    valid[0, 0] = True
    valid[3, 4] = True
    values[0, 0] = -0.004
    values[3, 4] = 0.006
    latitude = np.linspace(50.23, 50.20, 4)[:, None] * np.ones((1, 5))
    longitude = np.linspace(13.20, 13.24, 5)[None, :] * np.ones((4, 1))
    return values, valid, latitude, longitude


def test_symmetric_limits_are_centred_on_zero():
    vmin, vmax = symmetric_limits(np.array([-4.0, -0.2, 0.1, 2.0]))
    assert vmin == -vmax
    assert vmax >= 2.0


def test_non_ps_pixels_are_transparent_and_the_striped_field_is_dropped():
    values, valid, _lat, _lon = _scene()
    rgba = subsidence_rgba(values, valid, -0.01, 0.01)
    assert rgba.shape[-1] == 4
    assert int(np.count_nonzero(rgba[..., 3])) == 2
    assert rgba[0, 0, 3] == 255
    assert rgba[1, 2, 3] == 0
    # Negative velocity is redder than it is blue.
    assert int(rgba[0, 0, 0]) > int(rgba[0, 0, 2])


def test_geotiff_is_north_up_wgs84_and_keeps_only_ps_pixels(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    values, valid, latitude, longitude = _scene()
    written = export_ps_linear(values, valid, latitude, longitude, tmp_path, unit="mm/yr", scale=1000.0)
    with rasterio.open(written["tif"]) as dataset:
        assert dataset.crs.to_epsg() == 4326
        assert dataset.transform.e < 0
        assert dataset.tags().get("AREA_OR_POINT") == "Area"
        field = dataset.read(1)
        assert int(np.count_nonzero(np.isfinite(field))) == 2
        west, south, east, north = geographic_edges(
            (float(longitude.min()), float(latitude.min()), float(longitude.max()), float(latitude.max())),
            dataset.height,
            dataset.width,
        )
        assert dataset.bounds.left == pytest.approx(west)
        assert dataset.bounds.top == pytest.approx(north)
        # The northern selected point is the first row's centre.
        centre_x, centre_y = dataset.xy(0, 0, offset="center")
        assert centre_y == pytest.approx(float(latitude.max()))
        assert centre_x == pytest.approx(float(longitude.min()))
    with rasterio.open(written["rgba"]) as coloured:
        assert coloured.count == 4
        assert coloured.crs.to_epsg() == 4326
        alpha = coloured.read(4)
        assert int(np.count_nonzero(alpha)) == 2
        assert alpha.max() == 255
    png = written["png"].read_bytes()
    assert png[25] == 6  # RGBA colour type
    assert written["legend"].read_bytes().startswith(b"\x89PNG")


def test_a_safe_tie_point_is_the_geotiff_pixel_centre(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from xml.etree import ElementTree as ET

    ann = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "safe"
        / "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4.SAFE"
        / "annotation"
        / "s1b-iw1-slc-vv-20210401t052624-20210401t052649-026269-032297-004.xml"
    )
    root = ET.parse(ann).getroot()
    point = next(element for element in root.iter() if element.tag.endswith("geolocationGridPoint"))

    def text(name: str) -> float:
        return float(next(child.text for child in point if child.tag.endswith(name)))

    lat = text("latitude")
    lon = text("longitude")
    values = np.array([[-0.001]], dtype=np.float64)
    valid = np.array([[True]])
    written = export_ps_linear(values, valid, np.array([[lat]]), np.array([[lon]]), tmp_path)
    with rasterio.open(written["tif"]) as dataset:
        centre_x, centre_y = dataset.xy(0, 0, offset="center")
        assert centre_y == pytest.approx(lat)
        assert centre_x == pytest.approx(lon)
        assert np.isfinite(dataset.read(1)[0, 0])


def test_kmz_has_placemarks_and_a_legend_and_no_ground_overlay(tmp_path: Path):
    from ai_insar.pipeline.ps_export import write_ps_kmz

    values, valid, latitude, longitude = _scene()
    display = np.where(valid, values * 1000.0, np.nan)
    path = tmp_path / "ps.kmz"
    write_ps_kmz(path, latitude, longitude, display, valid, vmin=-10.0, vmax=10.0, unit="mm/yr")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert "doc.kml" in names
        assert "legend.png" in names
        assert "dot.png" in names
        kml = archive.read("doc.kml").decode("utf-8")
    assert kml.count("<Placemark>") == 2
    assert "<ScreenOverlay>" in kml
    assert "<GroundOverlay>" not in kml
    assert f"{float(longitude[0, 0]):.8f},{float(latitude[0, 0]):.8f}" in kml
    assert "LOS velocity" in kml
    # KML colour is aabbggrr. The negative point's red byte is larger than its blue byte.
    negative = kml.split(f"{float(longitude[0, 0]):.8f},{float(latitude[0, 0]):.8f}", 1)[0]
    colour = negative.split("<color>")[-1][:8]
    red = int(colour[6:8], 16)
    blue = int(colour[2:4], 16)
    assert red > blue
