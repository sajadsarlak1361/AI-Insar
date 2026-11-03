"""Offscreen Product Explorer: open a SAFE product, preview, and crop."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("tifffile")

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

from ai_insar.gui.main_window import SAFE_OPEN_FILTER, MainWindow

ROOT = Path(__file__).resolve().parents[1]
SAFE_NAME = "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4"
SAFE = ROOT / "tests" / "fixtures" / "safe" / f"{SAFE_NAME}.SAFE"
ZIP = ROOT / "tests" / "fixtures" / "safe" / f"{SAFE_NAME}.zip"


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def test_file_menu_opens_safe_and_previews_a_band(qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()
    product_action = window.findChild(QAction, "openSafeProductAction")
    folder_action = window.findChild(QAction, "openSafeFolderAction")
    assert SAFE_OPEN_FILTER.startswith("Sentinel-1 SAFE zip (*.zip)")
    assert product_action is not None and product_action.text() == "Open Sentinel-1 SAFE Product…"
    assert folder_action is not None and folder_action.text() == "Open SAFE folder…"
    data = next(action.menu() for action in window.menuBar().actions() if action.text() == "Data")
    labels = [item.text() for item in data.actions()]
    assert "Open Sentinel-1 SAFE Product…" in labels
    assert "Open SAFE folder…" in labels

    window.open_safe_path(SAFE)
    qapp.processEvents()
    status = window.statusBar().currentMessage()
    assert "S1B IW SLC" in status
    assert "VV+VH" in status
    assert "IW1–3" in status
    assert "2021-04-01T05:26:22" in status
    assert "Orbit 26269" in status
    texts = _tree_texts(window.product_explorer.tree)
    assert "Metadata" in texts
    assert any(item.startswith("Intensity_IW1_VV") for item in texts)
    assert any(item.startswith("i_IW1_VH") for item in texts)
    assert "Quicklook" in texts
    assert any("IW1, IW2, IW3" in item for item in texts)
    assert any(item.startswith("Intensity_IW1_VV (13509×") for item in texts)

    band = _find_item(window.product_explorer.tree, "Intensity_IW1_VV")
    window.product_explorer.tree.setCurrentItem(band)
    window.product_explorer._on_double_click(band, 0)
    qapp.processEvents()
    assert window.product_explorer.preview.has_raster()
    assert "IW1 VV" in window.product_explorer.preview_caption.text()

    window.preview_safe_quicklook()
    qapp.processEvents()
    assert "Quick-look" in window.product_explorer.preview_caption.text()

    window.import_safe_crop(
        window.safe_product,
        swath="IW1",
        pol="VV",
        row=0,
        col=0,
        height=2,
        width=2,
    )
    qapp.processEvents()
    stack = window.workspace.stack
    assert stack is not None
    assert stack.data.shape == (1, 2, 2)
    assert stack.data.dtype == np.complex64
    assert stack.dates[0] == np.datetime64("2021-04-01", "D")
    assert "ERROR" not in window.log.toPlainText()

    window.open_safe_path(ZIP)
    qapp.processEvents()
    assert window.safe_product is not None
    assert window.safe_product.is_zip
    assert "S1B IW SLC" in window.statusBar().currentMessage()
    window.close()


def _tree_texts(tree) -> list[str]:
    texts: list[str] = []

    def walk(item):
        texts.append(item.text(0))
        for index in range(item.childCount()):
            walk(item.child(index))

    for index in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(index))
    return texts


def _find_item(tree, label: str):
    flags = Qt.MatchFlag.MatchStartsWith | Qt.MatchFlag.MatchRecursive
    found = tree.findItems(label, flags)
    assert found, label
    return found[0]
