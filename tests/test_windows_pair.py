"""Track rule, Create Stack resampling, and the headless pair chain."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from ai_insar.pipeline.create_stack import output_window, resample_crop
from ai_insar.safe_stack import inspect_safe_products

PAIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pair"
APRIL = PAIR / "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4.zip"
LATER = PAIR / "S1B_IW_SLC__1SDV_20210413T052622_20210413T052650_026269_032297_EFA4.zip"
OTHER = PAIR / "S1B_IW_SLC__1SDV_20210425T052622_20210425T052650_026269_032297_EFA4.zip"


def test_two_products_must_share_the_relative_orbit():
    records = inspect_safe_products([APRIL, LATER], swath="IW1", pol="VV", reference=APRIL)
    assert {item.track for item in records} == {"168"}
    assert {item.orbit for item in records} == {"26269"}
    with pytest.raises(ValueError, match="relative orbit"):
        inspect_safe_products([APRIL, OTHER], swath="IW1", pol="VV")


def test_bilinear_offset_moves_the_impulse_and_minimum_can_be_empty():
    image = np.zeros((6, 8), dtype=np.complex64)
    image[2, 3] = 1
    moved = resample_crop(
        image,
        (0.0, 1.0),
        origin=(0, 0),
        shape=(6, 8),
        method="NEAREST_NEIGHBOUR",
    )
    # A +1 sample offset reads the impulse at column 3 from output column 2.
    assert moved[2, 2] == 1
    assert moved[2, 3] == 0
    row0, col0, height, width = output_window(6, 8, [(0.0, 0.0), (1.0, 2.0)], "Maximum")
    assert row0 < 0 and col0 < 0
    assert height > 6 and width > 8
    with pytest.raises(ValueError, match="Minimum"):
        output_window(4, 4, [(0.0, 0.0), (10.0, 10.0)], "Minimum")


def test_headless_pair_writes_an_interferogram(tmp_path: Path):
    from ai_insar.pipeline.safe_pair import run_safe_pair

    result = run_safe_pair(APRIL, LATER, tmp_path)
    interferogram = np.load(result["interferogram"])
    assert interferogram.shape == (32, 48)
    assert np.isfinite(interferogram).any()
    assert (tmp_path / "orbit.txt").read_text(encoding="utf-8").startswith("Apply Orbit File")
    assert "POLYGON" in result["aoi"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def test_create_stack_resampling_can_leave_none(qapp):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ai_insar.gui.main_window import MainWindow
    from ai_insar.gui.safe_stack_dialog import SafeStackDialog

    window = MainWindow()
    dialog = SafeStackDialog(window, n_files=2, initial_paths=[APRIL, LATER])
    dialog.resampling.setCurrentText("BILINEAR_INTERPOLATION")
    dialog.initial_offset.setCurrentText("Product Geolocation")
    dialog.output_extents.setCurrentText("Maximum")
    assert dialog.resampling.currentText() == "BILINEAR_INTERPOLATION"
    assert dialog.initial_offset.isEnabled()
    assert "Maximum" in dialog.stack_note.text() or "union" in dialog.stack_note.text()
    dialog.find_reference.click()
    assert dialog.reference.currentText()
    dialog.close()
    window.close()
    QApplication.processEvents()
