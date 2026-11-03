"""File → Open SAFE stack reads many zips and can run the interferogram stage."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("tifffile")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QLabel

from ai_insar.gui.main_window import MainWindow
from ai_insar.gui.safe_stack_dialog import SafeStackDialog
from safe_factory import dated_safe_zip


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def test_file_menu_loads_a_safe_stack_and_forms_interferograms(qapp, tmp_path: Path):
    early = dated_safe_zip(tmp_path, "20210401", 1 + 0j)
    late = dated_safe_zip(tmp_path, "20210507", 0 + 1j)
    window = MainWindow()
    window.show()
    qapp.processEvents()
    action = window.findChild(QAction, "openSafeStackAction")
    assert action is not None
    assert action.text() == "Open SAFE stack…"
    data = next(item.menu() for item in window.menuBar().actions() if item.text() == "Data")
    assert any(item.text() == "Open SAFE stack…" for item in data.actions())

    dialog = SafeStackDialog(window, n_files=2)
    note = dialog.findChild(QLabel, "safeStackNote")
    assert note is not None
    assert "crop-window alignment" in note.text()
    assert "no coregistration" in note.text()
    assert dialog.form_ifg.isChecked()

    window.load_safe_stack(
        [late, early],
        swath="IW1",
        pol="VV",
        row=0,
        col=0,
        height=2,
        width=2,
        form_interferograms=True,
    )
    qapp.processEvents()
    stack = window.workspace.stack
    assert stack is not None
    assert stack.data.shape == (2, 2, 2)
    assert stack.data.dtype == np.complex64
    assert stack.dates[0] == np.datetime64("2021-04-01", "D")
    ifg = window.workspace.interferograms
    assert ifg is not None
    assert ifg.shape == (1, 2, 2)
    assert np.allclose(ifg[0], stack.data[1] * np.conj(stack.data[0]))
    log = window.log.toPlainText()
    assert "SAFE stack (2, 2, 2)" in log
    assert "later × conjugate(earlier)" in log
    window.run_filter()
    window.run_coherence()
    qapp.processEvents()
    assert window.workspace.filtered is not None
    assert window.workspace.filtered.shape == ifg.shape
    assert window.workspace.coherence is not None
    assert window.workspace.coherence.shape == ifg.shape
    assert "ERROR" not in window.log.toPlainText()
    window.close()
    dialog.close()
