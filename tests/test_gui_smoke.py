"""Offscreen click-through of the numbered stage buttons."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

from ai_insar.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - depends on the host plugins
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")
    return app


def test_numbered_stage_buttons_run_the_pipeline(qapp, tmp_path):
    window = MainWindow()
    window.show()
    qapp.processEvents()
    assert len(window.stage_buttons) == 7
    labels = [button.text() for button in window.stage_buttons]
    assert labels[0].startswith("1")
    assert "Generate / load sample SLC" in labels[0]
    assert "Interferogram" in labels[1]
    assert "Goldstein" in labels[2]
    assert "Coherence" in labels[3]
    assert "Unwrap" in labels[4]
    assert "SBAS" in labels[5]
    assert "Displacement detection mask" in labels[6]

    for button in window.stage_buttons:
        button.click()
        qapp.processEvents()

    log = window.log.toPlainText()
    assert "ERROR" not in log
    assert "conjugate" in log.lower() or "conj" in log.lower()
    assert window.workspace.stack is not None
    assert window.workspace.stack.data.dtype.kind == "c"
    assert window.workspace.interferograms is not None
    assert window.workspace.filtered is not None
    assert window.workspace.coherence is not None
    assert window.workspace.unwrapped is not None
    assert window.workspace.velocity_m_yr is not None
    assert window.workspace.mask is not None
    assert int(window.workspace.mask.sum()) > 0
    assert float(np_min(window.workspace.velocity_m_yr)) < 0
    assert window.viewer.has_raster()

    window.layer_combo.setCurrentText("LOS velocity")
    qapp.processEvents()
    assert window.viewer.has_raster()
    assert "subsidence" in window.caption.text().lower() or "away from the radar" in window.caption.text().lower()

    center_row = window.workspace.stack.height // 2
    center_col = window.workspace.stack.width // 2
    window._on_pixel(center_row, center_col)
    assert "velocity" in window.readout.text()
    assert "LOS mm" in window.readout.text()

    # Menus expose the same stages and the load/save actions.
    titles = [action.text() for action in window.menuBar().actions()]
    assert titles[:3] == ["File", "Data", "Pipeline"]
    assert titles[-2:] == ["View", "Help"]
    for name in (
        "SBAS",
        "Stacking",
        "NSBAS",
        "PS linear",
        "Phase linking",
        "Seasonal",
        "Quality-guided",
        "Fast unwrap",
        "Goldstein",
        "Coherence",
    ):
        assert name in titles
    assert window.findChild(QAction, "runGoldstein") is not None
    assert window.findChild(QAction, "runCoherence") is not None
    assert window.findChild(QAction, "exportVelocityAction") is not None
    window.export_velocity_to(tmp_path)
    assert (tmp_path / "velocity_m_yr.npy").is_file()
    assert (tmp_path / "velocity_m_yr.png").is_file()
    window.layer_combo.setCurrentText("LOS displacement")
    window.slider.setValue(window.workspace.los_m.shape[0] - 1)
    qapp.processEvents()
    window.export_displacement_to(tmp_path)
    date = str(window.workspace.stack.dates[-1])
    assert (tmp_path / f"los_m_{date}.npy").is_file()
    assert (tmp_path / f"los_m_{date}.png").is_file()
    assert "not geocoded" in window.log.toPlainText()
    fetch = window.findChild(QAction, "fetchSentinelAction")
    assert fetch is not None
    assert fetch.text() == "Fetch Sentinel-1 SLC…"
    window.close()


def np_min(values):
    import numpy as np

    return float(np.min(values))
