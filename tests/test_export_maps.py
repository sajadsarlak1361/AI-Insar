"""Velocity and displacement export: npy data, PNG preview, optional TIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_insar.export_maps import export_raster, export_velocity_map, map_preview, write_png


def test_negative_velocity_preview_is_blue():
    field = np.zeros((12, 12), dtype=np.float64)
    field[6, 6] = -1.0
    rgb = map_preview(field, cmap="velocity", vmin=-1.0, vmax=1.0)
    # Preview is the map, a 4-pixel gap, and a 12-pixel scale strip.
    assert rgb.shape == (12 + 4 + 12, 12, 3)
    pixel = rgb[6, 6]
    assert int(pixel[2]) > int(pixel[0])


def test_export_writes_npy_and_png(tmp_path: Path):
    velocity = np.linspace(-0.2, 0.1, 16, dtype=np.float64).reshape(4, 4)
    written = export_velocity_map(velocity, tmp_path)
    loaded = np.load(written["npy"])
    assert loaded.shape == (4, 4)
    assert loaded.dtype == np.float64
    assert np.allclose(loaded, velocity)
    png = written["png"].read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert written["png"].name == "velocity_m_yr.png"


def test_export_rejects_a_cube(tmp_path: Path):
    with pytest.raises(ValueError, match="2-D"):
        export_raster(
            np.zeros((2, 4, 4)),
            tmp_path,
            "los_m",
            cmap="velocity",
            vmin=-1.0,
            vmax=1.0,
            unit="m",
        )


def test_write_png_round_trip_header(tmp_path: Path):
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)
    path = tmp_path / "swatch.png"
    write_png(path, rgb)
    blob = path.read_bytes()
    assert blob[12:16] == b"IHDR"
    # Width and height are big-endian uint32 immediately after the IHDR tag.
    assert int.from_bytes(blob[16:20], "big") == 3
    assert int.from_bytes(blob[20:24], "big") == 2


def test_tiff_is_radar_grid_without_pretending_to_be_geocoded(tmp_path: Path):
    tifffile = pytest.importorskip("tifffile")
    velocity = np.array([[-0.2, 0.05], [0.0, np.nan]], dtype=np.float64)
    written = export_velocity_map(velocity, tmp_path)
    assert "tif" in written
    with tifffile.TiffFile(written["tif"]) as handle:
        page = handle.pages[0]
        assert "Not geocoded" in str(page.description)
        restored = page.asarray()
    assert restored.shape == (2, 2)
    assert restored.dtype == np.float32
    assert np.isclose(restored[0, 0], -0.2)
    assert np.isnan(restored[1, 1])
