"""A known 50 mm/yr PS rate stays 50 mm/yr in every product.

Subsidence is negative LOS (away from the radar), so the stored number is
-50 mm/yr. v0.2.1 labelled the on-screen colour scale in m/yr and printed
legend ticks with ``.3g``. A 1000 mm/yr endpoint became ``1e+03``, and the
legend font has no plus sign, so the scale read as 1.
"""

from __future__ import annotations

import os
import re
import zipfile
from io import BytesIO
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from ai_insar.export_maps import write_png
from ai_insar.pipeline.interferogram import date_pairs, form_interferograms
from ai_insar.pipeline.ps_export import (
    _tick,
    export_ps_linear,
    legend_image,
    symmetric_limits,
)
from ai_insar.pipeline.synthetic import generate_subsidence_stack
from ai_insar.pipeline.timeseries import persistent_scatterer_velocity

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from ai_insar.gui.main_window import MainWindow
from ai_insar.gui.ps_export_dialog import PsExportDialog


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - depends on the host plugins
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")
    return app


def _fifty_mm_per_year():
    """Noise-free stack whose every pixel subsides at 50 mm/yr."""
    stack = generate_subsidence_stack(
        n_dates=6,
        height=16,
        width=16,
        spacing_days=12,
        rate_m_yr=0.05,
        sigma_px=1.0e6,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=4,
    )
    pairs = date_pairs(stack.n_dates, 1)
    phase = np.angle(form_interferograms(stack.data, pairs).astype(np.complex128))
    coherence = np.ones(stack.data.shape[1:], dtype=np.float64)
    _los, velocity, mask = persistent_scatterer_velocity(
        stack.data,
        phase,
        pairs,
        stack.times_year(),
        stack.wavelength_m,
        np.ones(phase.shape, dtype=np.float64),
        coherence,
        da_max=1.0,
        min_coherence=0.0,
    )
    latitude = np.linspace(50.232, 50.228, stack.height)[:, None] * np.ones((1, stack.width))
    longitude = np.linspace(13.274, 13.278, stack.width)[None, :] * np.ones((stack.height, 1))
    return stack, velocity, mask, latitude, longitude


def test_legend_tick_does_not_turn_1000_mm_into_one():
    assert _tick(-50.0, "mm/yr") == "-50 mm/yr"
    assert _tick(50.0, "mm/yr") == "50 mm/yr"
    assert _tick(0.0, "mm/yr") == "0 mm/yr"
    assert _tick(1000.0, "mm/yr") == "1000 mm/yr"
    assert "e" not in _tick(1000.0, "mm/yr")


def test_known_50_mm_per_year_in_geotiff_legend_and_kmz(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    _stack, velocity, mask, latitude, longitude = _fifty_mm_per_year()
    assert int(mask.sum()) == velocity.size
    millimetres = velocity * 1000.0
    assert float(np.nanmedian(millimetres)) == pytest.approx(-50.0, abs=0.05)

    written = export_ps_linear(velocity, mask, latitude, longitude, tmp_path)
    with rasterio.open(written["tif"]) as dataset:
        assert dataset.tags().get("UNIT") == "mm/yr"
        field = dataset.read(1)
        finite = field[np.isfinite(field)]
        assert finite.size > 0
        assert float(np.median(finite)) == pytest.approx(-50.0, abs=0.05)

    display = np.where(mask, millimetres, np.nan)
    low, high = symmetric_limits(display)
    assert low == pytest.approx(-50.0, abs=0.05)
    assert high == pytest.approx(50.0, abs=0.05)
    buffer = BytesIO()
    write_png(buffer, legend_image(low, high, "mm/yr"))
    assert written["legend"].read_bytes() == buffer.getvalue()
    assert _tick(low, "mm/yr").startswith("-50")
    assert _tick(high, "mm/yr").startswith("50")

    with zipfile.ZipFile(written["kmz"]) as archive:
        kml = archive.read("doc.kml").decode("utf-8")
        legend = archive.read("legend.png")
    balloons = [float(item) for item in re.findall(r"LOS velocity ([+-]?\d+\.\d+) mm/yr", kml)]
    assert balloons
    assert all(abs(item + 50.0) < 0.05 for item in balloons)
    assert legend == buffer.getvalue()


def test_gui_shows_50_mm_per_year(qapp):
    _stack, velocity, mask, _latitude, _longitude = _fifty_mm_per_year()
    window = MainWindow()
    window.workspace.velocity_m_yr = velocity
    window.workspace.ps_mask = mask
    window._select_layer("LOS velocity", refresh=True)
    qapp.processEvents()
    assert window.colorbar._caption == "mm/yr"
    assert abs(float(window.colorbar._vmin)) == pytest.approx(50.0, abs=0.05)
    assert abs(float(window.colorbar._vmax)) == pytest.approx(50.0, abs=0.05)
    row, col = velocity.shape[0] // 2, velocity.shape[1] // 2
    window._on_pixel(row, col)
    readout = window.readout.text()
    assert "mm/yr" in readout
    assert "-50.00 mm/yr" in readout
    dialog = PsExportDialog(window)
    settings = dialog.settings()
    assert settings["quantity"] == "velocity"
    assert settings["unit"] == "mm/yr"
    assert settings["scale"] == 1000.0
    dialog.close()
    window.close()
