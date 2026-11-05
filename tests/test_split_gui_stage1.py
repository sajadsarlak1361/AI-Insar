"""Open a SAFE folder stack, bind Split from that folder path, and save DIMAP.

The dialog is driven the way the menu does: source text, subswath, bursts, Save as.
This is not a direct call to the TIFF reader.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("tifffile")

from PySide6.QtWidgets import QApplication

from ai_insar.gui.chain_dialogs import TopsSplitDialog
from ai_insar.gui.main_window import MainWindow
from ai_insar.pipeline.dimap import read_beam_dimap_band
from ai_insar.safe import resolve_safe
from ai_insar.sentinel import read_tiff_window
from safe_factory import dated_safe_zip

JUNE = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
)
JULY = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B.SAFE"
)
SNAP_JUNE_DIM = Path(
    "/home/ubuntu/.cursor/projects/workspace/uploads/20210630_split_IW1_VV_b5-6_snap_bcd0.dim"
)


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def _commit_source(dialog: TopsSplitDialog, path: str) -> None:
    """Leave the source field the way a typed folder is committed."""
    dialog.source.setCurrentText(path)
    line = dialog.source.lineEdit()
    assert line is not None
    line.editingFinished.emit()


def _choose(dialog: TopsSplitDialog, swath: str, pol: str, first: int, last: int) -> None:
    dialog.subswath.setCurrentText(swath)
    for index in range(dialog.pol_list.count()):
        if dialog.pol_list.item(index).text() == pol:
            dialog.pol_list.setCurrentRow(index)
            break
    dialog.burst_first.setValue(first)
    dialog.burst_last.setValue(last)


def test_typing_a_safe_folder_fills_split_controls(qapp, tmp_path: Path):
    archive = dated_safe_zip(tmp_path, "20210630", 1 + 0j)
    folder = tmp_path / "unpacked"
    folder.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(folder)
    safe = next(folder.glob("*.SAFE"))
    dialog = TopsSplitDialog()
    assert dialog.subswath.count() == 0
    assert dialog.burst_text.text() == "1 to 9999 (max number of bursts: 9999)"
    assert "With no source" in dialog.notes.text()
    _commit_source(dialog, str(safe))
    qapp.processEvents()
    names = [dialog.subswath.itemText(i) for i in range(dialog.subswath.count())]
    assert names[0] == ""
    assert "IW1" in names
    assert "9999" not in dialog.burst_text.text()
    assert "is bound" in dialog.notes.text()
    assert "With no source" not in dialog.notes.text()
    _choose(dialog, "IW1", "VV", 5, 6)
    assert dialog.burst_text.text() == "1 to 9 (max number of bursts: 9)"
    assert dialog.burst_first.value() == 5
    assert dialog.burst_last.value() == 6
    assert dialog.validate() == ""
    assert dialog.parameters()["source"] == str(safe)
    _commit_source(dialog, str(tmp_path))
    qapp.processEvents()
    assert "not a Sentinel-1 SAFE" in dialog.error.text()
    _commit_source(dialog, str(safe))
    qapp.processEvents()
    assert dialog.error.text() == ""
    assert "is bound" in dialog.notes.text()
    assert dialog.subswath.findText("IW1") >= 0
    dialog.close()


def test_stack_split_saves_dimap_and_keeps_the_other_date(qapp, tmp_path: Path):
    if not JUNE.is_dir() or not JULY.is_dir():
        pytest.skip("full SAFE folders are not on disk")
    window = MainWindow()
    window.load_safe_stack(
        [JULY, JUNE],
        swath="IW1",
        pol="VV",
        row=0,
        col=0,
        height=8,
        width=8,
        reference=JUNE,
        form_interferograms=False,
    )
    qapp.processEvents()
    labels = [
        window.product_explorer.tree.topLevelItem(i).text(0)
        for i in range(window.product_explorer.tree.topLevelItemCount())
    ]
    assert any(label.endswith("(reference)") and "20210630" in label for label in labels)
    assert any(label.endswith("(secondary)") and "20210712" in label for label in labels)
    assert not any(label == "No Sentinel-1 product open" for label in labels)
    assert window.product_explorer.tree.topLevelItem(0).isExpanded() is False
    dialog = TopsSplitDialog(window)
    window._bind_split_dialog(dialog)
    assert dialog.source.count() == 2
    assert "20210630" in dialog.source.currentText()
    assert "is bound" in dialog.notes.text()
    assert "With no source" not in dialog.notes.text()
    _commit_source(dialog, str(JUNE))
    qapp.processEvents()
    _choose(dialog, "IW1", "VV", 5, 6)
    dialog.save_box.setChecked(True)
    dialog.format_box.setCurrentText("BEAM-DIMAP")
    dialog.directory.setText(str(tmp_path))
    dialog.target.setText("june_split")
    assert dialog.validate() == ""
    window.apply_split(dialog.parameters())
    saved = tmp_path / "june_split.dim"
    assert saved.is_file()
    real = read_beam_dimap_band(saved, 0)
    imag = read_beam_dimap_band(saved, 1)
    assert real.shape == (3016, 22849)
    written = real.astype(np.float32) + 1j * imag.astype(np.float32)
    measurement = JUNE / (
        "measurement/s1a-iw1-slc-vv-20210630t052604-20210630t052629-038565-048cfd-004.tiff"
    )
    source = read_tiff_window(measurement, 6032, 11000, 16, 16)
    assert np.allclose(written[0:16, 11000:11016], source)
    assert window.workspace.reference_grid is not None
    assert window.workspace.reference_grid.data.shape == (3016, 22849)
    assert window.workspace.safe_inputs and len(window.workspace.safe_inputs) == 2
    assert window.workspace.pair_stack is not None
    assert window.workspace.pair_stack.data.shape[0] == 2

    dialog.source.setCurrentText(str(JULY))
    dialog.source.lineEdit().editingFinished.emit()
    qapp.processEvents()
    _choose(dialog, "IW1", "VV", 5, 6)
    dialog.target.setText("july_split")
    window.apply_split(dialog.parameters())
    assert (tmp_path / "july_split.dim").is_file()
    assert window.workspace.reference_grid.data.shape == (3016, 22849)
    assert window.workspace.secondary_grid is not None
    assert window.workspace.secondary_grid.data.shape == (3016, 22849)
    assert len(window.workspace.safe_inputs) == 2
    log = window.log.toPlainText()
    assert "Stored the reference split (3016, 22849)" in log
    assert "Stored the secondary split (3016, 22849)" in log
    assert "20210630" in log.split("Stored the reference split")[1]
    assert "20210712" in log.split("Stored the secondary split")[1]
    labels = [
        window.product_explorer.tree.topLevelItem(i).text(0)
        for i in range(window.product_explorer.tree.topLevelItemCount())
    ]
    assert any("20210630" in label and label.endswith("(reference)") for label in labels)
    assert any("20210712" in label and label.endswith("(secondary)") for label in labels)
    assert window.product_explorer.tree.topLevelItemCount() == 2
    assert not any(label == "No Sentinel-1 product open" for label in labels)
    dialog.close()
    window.close()


def test_next_split_dialog_keeps_directory_and_the_unsplit_date(qapp, tmp_path: Path):
    """The menu builds a new dialog for every split. The second one must not
    return to the reference date or to /home/box, and the empty-source note
    must be gone once the product is bound.
    """
    if not JUNE.is_dir() or not JULY.is_dir():
        pytest.skip("full SAFE folders are not on disk")
    window = MainWindow()
    june = resolve_safe(JUNE)
    july = resolve_safe(JULY)
    window.safe_products = [june, july]
    window.safe_product = june
    window._show_safe_products()
    qapp.processEvents()
    labels = [
        window.product_explorer.tree.topLevelItem(i).text(0)
        for i in range(window.product_explorer.tree.topLevelItemCount())
    ]
    assert any(label.endswith("(reference)") and "20210630" in label for label in labels)
    assert any(label.endswith("(secondary)") and "20210712" in label for label in labels)

    first = TopsSplitDialog(window)
    window._bind_split_dialog(first)
    assert "20210630" in first.source.currentText()
    assert first.directory.text() == "/home/box"
    assert "is bound" in first.notes.text()
    assert "With no source" not in first.notes.text()
    assert "9999" not in first.burst_text.text()
    first.directory.setText(str(tmp_path))
    first.format_box.setCurrentText("BEAM-DIMAP")
    first.save_box.setChecked(True)
    window._split_io = {
        "directory": str(tmp_path),
        "format": "BEAM-DIMAP",
        "save": True,
        "open": True,
    }
    window._split_product_ids.add(june.product_id)
    first.close()

    second = TopsSplitDialog(window)
    window._bind_split_dialog(second)
    assert second.directory.text() == str(tmp_path)
    assert second.format_box.currentText() == "BEAM-DIMAP"
    assert second.save_box.isChecked()
    assert "20210712" in second.source.currentText()
    assert second.source.count() == 2
    assert "is bound" in second.notes.text()
    assert "With no source" not in second.notes.text()
    assert "9999" not in second.burst_text.text()
    labels = [
        window.product_explorer.tree.topLevelItem(i).text(0)
        for i in range(window.product_explorer.tree.topLevelItemCount())
    ]
    assert any(label.endswith("(reference)") and "20210630" in label for label in labels)
    assert any(label.endswith("(secondary)") and "20210712" in label for label in labels)
    second.close()
    window.close()


def test_snap14_scene_start_is_not_the_burst_line_clock():
    """DIMAP scene start is not raster line 0.

    Abstracted first_line_time is the zero-Doppler time of line 0 and matches
    the annotation burst azimuth time. The scene-start field is a different
    timestamp and is not an integer number of azimuth lines later.
    """
    if not SNAP_JUNE_DIM.is_file():
        pytest.skip("SNAP 14 split metadata was not uploaded")
    text = SNAP_JUNE_DIM.read_text(encoding="utf-8", errors="replace")
    assert "30-JUN-2021 05:26:15.752882" in text
    assert "30-JUN-2021 05:26:16.636158" in text
    assert "<BAND_RASTER_WIDTH>22849</BAND_RASTER_WIDTH>" in text
    assert "<BAND_RASTER_HEIGHT>3016</BAND_RASTER_HEIGHT>" in text
    assert 'name="subset_offset_y"' in text and ">6032</MDATTR>" in text
    start = 16.636158
    line0 = 15.752882
    interval = 0.002055556299999998
    lines = (start - line0) / interval
    assert abs(lines - round(lines)) > 0.2
