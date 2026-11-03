"""Offscreen checks for the Sentinel-1 fetch dialog and Data menu."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

from ai_insar.gui.fetch_dialog import FetchSentinelDialog
from ai_insar.gui.main_window import MainWindow
from ai_insar.sentinel import SentinelGranule


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def test_data_menu_opens_the_same_fetch_action(qapp):
    window = MainWindow()
    fetch = window.findChild(QAction, "fetchSentinelAction")
    assert fetch is not None
    titles = [action.text() for action in window.menuBar().actions()]
    assert "Data" in titles
    data = next(action.menu() for action in window.menuBar().actions() if action.text() == "Data")
    assert any(item.text() == "Fetch Sentinel-1 SLC…" for item in data.actions())
    assert any(item.text() == "Import local SAFE / TIFF…" for item in data.actions())
    assert any(item.text() == "Open Sentinel-1 SAFE Product…" for item in data.actions())
    assert any(item.text() == "Open SAFE folder…" for item in data.actions())
    file_menu = next(action.menu() for action in window.menuBar().actions() if action.text() == "File")
    assert any(item.text() == "Open Sentinel-1 SAFE Product…" for item in file_menu.actions())
    window.close()


def test_dialog_search_and_import_without_network(qapp, tmp_path: Path):
    granule = SentinelGranule(
        granule_name="S1A_IW_SLC__1SDV_20240118T141632_20240118T141659_052162_064E27_074A",
        start_time="2024-01-18T14:16:32Z",
        platform="Sentinel-1A",
        flight_direction="DESCENDING",
        polarization="VV+VH",
        beam_mode="IW",
        size_mb=12.0,
        download_url="https://example.invalid/not-downloaded.zip",
    )

    def fake_search(**_kwargs):
        return [granule]

    dialog = FetchSentinelDialog(search_fn=fake_search)
    assert dialog.swath.currentText() == "IW1"
    assert dialog.pol.currentText() == "VV"
    dialog.show()
    qapp.processEvents()
    dialog.run_search()
    assert dialog.table.rowCount() == 1
    assert "S1A_IW_SLC" in dialog.table.item(0, 0).text()
    dialog.table.selectRow(0)
    assert dialog.selected_granules()[0].granule_name == granule.granule_name

    early = tmp_path / "s1a-iw1-slc-vv-20200108t000000.npy"
    late = tmp_path / "s1a-iw1-slc-vv-20200213t000000.npy"
    np.save(early, np.ones((8, 8), dtype=np.complex64))
    np.save(late, np.full((8, 8), 1j, dtype=np.complex64))
    dialog.row.setValue(0)
    dialog.col.setValue(0)
    dialog.height.setValue(8)
    dialog.width.setValue(8)
    dialog.load_paths([late, early])
    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.stack is not None
    assert dialog.stack.data.shape == (2, 8, 8)
    assert dialog.stack.data.dtype == np.complex64
    dialog.close()
