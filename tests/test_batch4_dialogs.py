"""Batch-4 dialogs and the processing they are allowed to run."""

from __future__ import annotations

import os
import stat
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from ai_insar.gui.batch4_dialogs import BatchSnaphuUnwrapDialog, MultilookDialog, SnaphuExportDialog, SnaphuImportDialog
from ai_insar.gui.main_window import MainWindow
from ai_insar.pipeline.filtering import goldstein_filter
from ai_insar.pipeline.multilook import multilook_array
from ai_insar.pipeline.snaphu_io import export_snaphu, import_unwrapped, run_snaphu


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def test_multilook_dialog_matches_the_empty_product_state(qapp):
    dialog = MultilookDialog()
    dialog.save_box.setChecked(False)
    dialog.source.setCurrentText("Workspace SLC stack")
    assert dialog.bands.count() == 0
    assert dialog.gr_square.isChecked()
    assert not dialog.independent.isChecked()
    assert dialog.range_looks.text() == "1"
    assert dialog.range_looks.isEnabled()
    assert dialog.azimuth_looks.text() == ""
    assert not dialog.azimuth_looks.isEnabled()
    assert dialog.mean_square.text() == ""
    assert not dialog.mean_square.isEnabled()
    assert not dialog.output_intensity.isChecked()
    assert "without resampling" in dialog.findChild(QLabel, "multilookNote").text()
    dialog.bands.addItem("Interferogram")
    dialog.bands.item(0).setSelected(True)
    assert "GR Square Pixel" in dialog.validate()
    dialog.independent.setChecked(True)
    assert not dialog.gr_square.isChecked()
    assert dialog.azimuth_looks.isEnabled()
    dialog.independent.setChecked(False)
    assert dialog.gr_square.isChecked()
    assert not dialog.azimuth_looks.isEnabled()
    dialog.close()


def test_independent_multilook_averages_cells_and_square_pixel_does_not():
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    reduced = multilook_array(values, 2, 2)
    assert reduced.shape == (2, 2)
    assert np.isclose(reduced[0, 0], np.mean(values[:2, :2]))
    complex_values = (values + 1j * values).astype(np.complex64)
    complex_reduced = multilook_array(complex_values, 2, 2, output_intensity=False)
    assert complex_reduced.dtype == np.complex64
    assert complex_reduced.shape == (2, 2)
    window = MainWindow()
    window.workspace.interferograms = complex_values[None, ...]
    with pytest.raises(RuntimeError, match="spacing"):
        window.apply_multilook(
            {
                "gr_square": True,
                "bands": ["Interferogram"],
                "range_looks": 1,
                "azimuth_looks": 0,
                "output_intensity": False,
            }
        )
    assert window.workspace.interferograms.shape == (1, 4, 4)
    window.apply_multilook(
        {
            "gr_square": False,
            "bands": ["Interferogram"],
            "range_looks": 2,
            "azimuth_looks": 2,
            "output_intensity": False,
        }
    )
    assert window.workspace.interferograms.shape == (1, 2, 2)
    assert "not a resample" in window.log.toPlainText()
    window.close()


def test_adaptive_goldstein_uses_coherence_and_constant_alpha_does_not():
    rng = np.random.default_rng(0)
    interferogram = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))).astype(np.complex64)
    coherence = np.ones((32, 32), dtype=np.float32)
    adaptive = goldstein_filter(
        interferogram,
        alpha=1.0,
        window=32,
        overlap=0.5,
        spectrum_smooth=1,
        coherence=coherence,
        alpha_min=0.2,
        alpha_max=1.0,
    )
    strong = goldstein_filter(interferogram, alpha=1.0, window=32, overlap=0.5, spectrum_smooth=1)
    assert adaptive.shape == interferogram.shape
    assert not np.allclose(adaptive, strong)
    with pytest.raises(ValueError, match="shape"):
        goldstein_filter(interferogram, coherence=np.ones((8, 8), dtype=np.float32))


def test_snaphu_export_runs_the_external_binary_and_import_reads_it(tmp_path: Path):
    phase = np.array([[0.1, -0.2], [1.2, -1.1]], dtype=np.float64)
    coherence = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float64)
    conf = export_snaphu(tmp_path / "export", phase, coherence, wavelength_m=0.055, pair_count=2)
    text = conf.read_text(encoding="utf-8")
    assert "STATCOSTMODE DEFO" in text
    assert "INITMETHOD MCF" in text
    assert "LINELENGTH 2" in text
    assert "NTILEROW 10" in text
    header = (tmp_path / "export" / "phase.hdr").read_text(encoding="utf-8")
    assert "samples = 2" in header
    assert "lines = 2" in header
    assert "data type = 4" in header
    install = tmp_path / "snaphu-install"
    install.mkdir()
    binary = install / "snaphu"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "phase = np.fromfile('phase.img', dtype='<f4')\n"
        "phase.tofile('unwrap.img')\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    completed = run_snaphu(tmp_path / "export", install)
    assert completed.returncode == 0
    imported = import_unwrapped(tmp_path / "export")
    assert imported.shape == (2, 2)
    assert np.allclose(imported, phase.astype(np.float32))
    with pytest.raises(RuntimeError, match="fast unwrap was not used"):
        run_snaphu(tmp_path / "export", tmp_path / "missing-install")


def test_batch4_dialogs_and_disabled_subset(qapp, tmp_path: Path):
    window = MainWindow()
    window.workspace.interferograms = np.ones((1, 4, 4), dtype=np.complex64)
    window.workspace.coherence = np.full((1, 4, 4), 0.5, dtype=np.float32)
    export = SnaphuExportDialog(window, has_interferogram=True)
    assert export.source.currentText() == ""
    assert export.data_format.count() == 0
    assert not export.findChild(QPushButton, "snaphuExportAdvanced").isEnabled()
    assert export.status.text() == "Source product not selected"
    assert export.stat_mode.currentText() == "DEFO"
    assert export.init_method.currentText() == "MCF"
    assert export.tile_rows.text() == "10"
    assert export.processors.text() == "4"
    assert export.tile_cost.text() == "500"
    export.source.setCurrentText("Workspace interferogram")
    export.target_folder.setText(str(tmp_path / "out"))
    window.apply_snaphu_export(export.parameters())
    assert (tmp_path / "out" / "snaphu.conf").is_file()
    assert "Not unwrapped" in window.log.toPlainText()
    assert window.workspace.unwrapped is None

    unwrap = BatchSnaphuUnwrapDialog(window)
    unwrap.save_box.setChecked(False)
    assert unwrap.export_folder.text() == ""
    assert unwrap.install_location.text() == ""
    assert "required" in unwrap.validate().lower()
    with pytest.raises(RuntimeError, match="fast unwrap was not used"):
        window.apply_batch_snaphu(
            {"export_folder": str(tmp_path / "out"), "install_location": str(tmp_path / "absent")}
        )
    assert window.workspace.unwrapped is None
    assert window.workspace.config.unwrap_method == "separable"

    importer = SnaphuImportDialog(window)
    assert [importer.tabs.tabText(index) for index in range(importer.tabs.count())] == [
        "1-Read-Phase",
        "2-Read-Unwrapped-Phase",
        "3-SnaphuImport",
        "4-Write",
    ]
    assert not importer.skip_wrapped.isChecked()
    assert importer.target.text() == "target"
    assert importer.format_box.currentText() == "BEAM-DIMAP"
    assert importer.directory.text() == "/home/box"
    assert importer.status.text() == "Source product not selected"
    subset = window.findChild(QAction, "radarSubset")
    save_as = window.findChild(QAction, "radarSaveProductAs")
    assert subset is not None and not subset.isEnabled()
    assert save_as is not None and not save_as.isEnabled()
    assert window.findChild(QAction, "radarMultilook") is not None
    assert window.findChild(QAction, "radarSnaphuExport") is not None
    assert window.findChild(QAction, "radarBatchSnaphu") is not None
    window.close()
    export.close()
    unwrap.close()
    importer.close()
