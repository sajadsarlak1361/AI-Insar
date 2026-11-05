"""Several SAFE folders are selected together, listed, and all kept in the stack."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("tifffile")

from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from ai_insar.gui.main_window import MainWindow
from ai_insar.gui.safe_chooser import SafeProductChooser
from ai_insar.gui.safe_stack_dialog import SafeStackDialog
from ai_insar.safe_stack import expand_safe_selection, inspect_safe_products
from safe_factory import dated_safe_zip


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def _extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(dest)
    return next(dest.glob("*.SAFE"))


def _wait_for_rows(chooser: SafeProductChooser, qapp) -> None:
    root = chooser.model.index(chooser.start_directory)
    for _ in range(200):
        qapp.processEvents()
        if chooser.model.canFetchMore(root):
            chooser.model.fetchMore(root)
        if chooser.model.rowCount(root) > 0:
            return
    raise AssertionError(f"chooser did not list {chooser.start_directory}")


def test_parent_folder_expands_to_every_safe_child(tmp_path: Path):
    early_zip = dated_safe_zip(tmp_path, "20210401", 1 + 0j)
    late_zip = dated_safe_zip(tmp_path, "20210507", 0 + 1j)
    library = tmp_path / "library"
    early = _extract(early_zip, library / "early")
    late = _extract(late_zip, library / "late")
    # Move the products up so the parent itself is not a SAFE folder.
    early_dest = library / early.name
    late_dest = library / late.name
    early.rename(early_dest)
    late.rename(late_dest)
    found = expand_safe_selection([library])
    names = {path.name for path in found}
    assert early_dest.name in names
    assert late_dest.name in names
    assert len(found) == 2


def test_chooser_selects_two_safe_folders(qapp, tmp_path: Path):
    early_zip = dated_safe_zip(tmp_path, "20210401", 2 + 0j)
    late_zip = dated_safe_zip(tmp_path, "20210507", 0 + 3j)
    library = tmp_path / "library"
    early = _extract(early_zip, library / "early")
    late = _extract(late_zip, library / "late")
    early_dest = library / early.name
    late_dest = library / late.name
    early.rename(early_dest)
    late.rename(late_dest)
    chooser = SafeProductChooser(None, start=str(library))
    chooser.show()
    _wait_for_rows(chooser, qapp)
    root = chooser.model.index(chooser.start_directory)
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    selected = []
    for row in range(chooser.model.rowCount(root)):
        index = chooser.model.index(row, 0, root)
        path = Path(chooser.model.filePath(index))
        if path.name in {early_dest.name, late_dest.name}:
            chooser.view.selectionModel().select(index, flags)
            selected.append(path.name)
    qapp.processEvents()
    assert set(selected) == {early_dest.name, late_dest.name}
    products = chooser.selected_products()
    assert {path.name for path in products} == {early_dest.name, late_dest.name}
    assert chooser.count.text() == "2 selected"
    assert len(products) == 2
    chooser.close()


def test_two_folders_stay_in_the_stack_with_a_chosen_reference(qapp, tmp_path: Path):
    early_zip = dated_safe_zip(tmp_path, "20210401", 2 + 0j)
    late_zip = dated_safe_zip(tmp_path, "20210507", 0 + 3j)
    early = _extract(early_zip, tmp_path / "early")
    late = _extract(late_zip, tmp_path / "late")
    window = MainWindow()
    dialog = SafeStackDialog(window, n_files=2, initial_paths=[early, late])
    assert dialog.product_set.count_label.text() == "2 Products"
    assert dialog.reference.count() == 2
    late_index = dialog.reference.findText(late.name)
    assert late_index >= 0
    dialog.reference.setCurrentIndex(late_index)
    dialog.form_ifg.setChecked(True)
    dialog.height.setValue(2)
    dialog.width.setValue(2)
    window.load_from_product_set(dialog)
    qapp.processEvents()
    stack = window.workspace.stack
    assert stack is not None
    assert stack.data.shape == (2, 2, 2)
    inputs = window.workspace.safe_inputs
    assert len(inputs) == 2
    assert {item.file_name for item in inputs} == {early.name, late.name}
    reference = [item for item in inputs if item.is_reference]
    assert len(reference) == 1
    assert reference[0].file_name == late.name
    assert all(item.product_type == "SLC" for item in inputs)
    assert all(item.orbit == "26269" for item in inputs)
    log = window.log.toPlainText()
    assert "2 Products" in log
    assert early.name in log
    assert late.name in log
    assert f"Reference: {late.name}" in log
    assert window.workspace.interferograms is not None
    assert window.workspace.interferograms.shape[0] == 1
    assert "ERROR" not in log
    window.close()
    dialog.close()


def test_incompatible_polarisation_names_the_product_and_loads_nothing(tmp_path: Path):
    archive = dated_safe_zip(tmp_path, "20210401", 1 + 0j)
    folder = _extract(archive, tmp_path / "only")
    with pytest.raises(ValueError, match="HH"):
        inspect_safe_products([folder], swath="IW1", pol="HH")
    window = MainWindow()
    dialog = SafeStackDialog(window, n_files=1, initial_paths=[folder])
    dialog.pol_box.setCurrentText("HH")
    dialog.height.setValue(2)
    dialog.width.setValue(2)
    with pytest.raises(ValueError, match="HH"):
        window.load_from_product_set(dialog)
    assert window.workspace.stack is None
    assert window.workspace.safe_inputs == []
    window.close()
    dialog.close()
