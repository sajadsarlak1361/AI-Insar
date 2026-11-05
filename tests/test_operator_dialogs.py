"""Offscreen SNAP-style dialogs and the multi-SAFE product list."""

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
from ai_insar.gui.operator_dialog import (
    ApplyOrbitDialog,
    CoherenceDialog,
    GoldsteinDialog,
    InterferogramDialog,
    TopoPhaseDialog,
)
from ai_insar.gui.product_set import ProductSetWidget
from ai_insar.gui.safe_stack_dialog import SafeStackDialog
from ai_insar.pipeline.coherence import coherence_map
from ai_insar.pipeline.synthetic import generate_subsidence_stack
from ai_insar.snap_products import OPERATOR_HELP, PRODUCT_SUFFIX, suffixed_stem
from safe_factory import FIXTURE_ZIP, dated_safe_zip


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def _prepare(dialog) -> None:
    dialog.save_box.setChecked(False)


def test_snap_suffixes_and_help_lines():
    assert PRODUCT_SUFFIX["interferogram"] == "_Ifg"
    assert PRODUCT_SUFFIX["coherence"] == "_Coh"
    assert PRODUCT_SUFFIX["topo"] == "_DInSAR"
    assert PRODUCT_SUFFIX["goldstein"] == "_Flt"
    assert PRODUCT_SUFFIX["apply_orbit"] == "_Orb"
    assert PRODUCT_SUFFIX["stack"] == "_Stack"
    assert PRODUCT_SUFFIX["displacement"] == "_Disp"
    assert suffixed_stem("target_Ifg", "_Ifg") == "target_Ifg"
    assert suffixed_stem("target", "_Coh") == "target_Coh"
    assert "conj" in OPERATOR_HELP["interferogram"]
    assert "γ" in OPERATOR_HELP["coherence"]
    assert "B_perp" in OPERATOR_HELP["topo"]
    assert "Z_filt" in OPERATOR_HELP["goldstein"]
    assert "_Orb" in OPERATOR_HELP["apply_orbit"]
    assert "d = " in OPERATOR_HELP["displacement"]


def test_rectangular_coherence_window_is_accepted():
    early = np.ones((6, 8), dtype=np.complex64)
    late = np.ones((6, 8), dtype=np.complex64)
    gamma = coherence_map(early, late, 4, azimuth_window=2)
    assert gamma.shape == (6, 8)
    assert np.allclose(gamma, 1.0)


def test_product_set_counts_and_reads_manifest_fields(qapp, tmp_path: Path):
    other = dated_safe_zip(tmp_path, "20210507", 0 + 1j)
    widget = ProductSetWidget()
    widget.add_paths([FIXTURE_ZIP, other, tmp_path / "missing.zip"])
    assert widget.count_label.text() == "3 Products"
    assert widget.table.columnCount() == 5
    headers = [widget.table.horizontalHeaderItem(column).text() for column in range(5)]
    assert headers == ["File Name", "Type", "Acquisition", "Track", "Orbit"]
    assert widget.table.item(0, 1).text() == "SLC"
    assert widget.table.item(0, 2).text().startswith("2021-04-01")
    assert widget.table.item(0, 3).text() == "168"
    assert widget.table.item(0, 4).text() == "26269"
    assert widget.table.item(2, 1).text() == ""
    assert widget.table.item(2, 4).text() == ""
    for name, tip in (
        ("productSetAdd", "Add"),
        ("productSetAddOpened", "Add Opened"),
        ("productSetRemove", "Remove"),
        ("productSetMoveTop", "Move Top"),
        ("productSetMoveUp", "Move Up"),
        ("productSetMoveDown", "Move Down"),
        ("productSetMoveBottom", "Move Bottom"),
        ("productSetRefresh", "Refresh"),
        ("productSetClear", "Clear"),
    ):
        button = widget.findChild(type(widget.add_button), name)
        assert button is not None
        assert button.toolTip() == tip
    widget.table.selectRow(2)
    widget.move_top_button.click()
    qapp.processEvents()
    assert widget.table.item(0, 0).text() == "missing.zip"
    assert widget.count_label.text() == "3 Products"
    widget.remove_selected()
    assert widget.count_label.text() == "2 Products"
    widget.clear()
    assert widget.count_label.text() == "0 Products"
    assert "Please add a source product" in widget.status.text()
    widget.close()


def test_operator_dialogs_expose_snap_controls(qapp):
    window = MainWindow()
    ifg = InterferogramDialog(window, sources=["Workspace SLC stack"])
    _prepare(ifg)
    help_line = ifg.findChild(QLabel, "operatorHelp")
    assert help_line is not None and "conj" in help_line.text() and "_Ifg" in help_line.text()
    assert ifg.findChild(type(ifg.subtract_flat), "ifgSubtractFlatEarth") is not None
    assert ifg.subtract_flat.isChecked() and not ifg.subtract_flat.isEnabled()
    assert ifg.flat_degree.currentText() == "5"
    assert ifg.flat_points.currentText() == "501"
    assert ifg.orbit_degree.currentText() == "3"
    assert not ifg.subtract_topo.isChecked()
    assert ifg.output_coherence.isChecked()
    assert ifg.square_pixel.isChecked()
    assert not ifg.coherence_azimuth.isEnabled()
    assert ifg.coherence_range.value() == 10
    assert ifg.coherence_azimuth.value() == 2
    assert ifg.findChild(type(ifg.output_latlon), "ifgOutputLatLon") is not None
    assert "not implemented" in ifg.subtract_flat.toolTip().lower() or "not applied" in ifg.subtract_flat.toolTip().lower()
    assert ifg.validate() == ""
    ifg.save_box.setChecked(True)
    assert "directory" in ifg.validate().lower()
    ifg.close()

    coh = CoherenceDialog(window, sources=["Workspace SLC stack"])
    _prepare(coh)
    assert "γ" in coh.findChild(QLabel, "operatorHelp").text()
    assert coh.single_reference.isChecked()
    assert coh.subtract_flat.isChecked() and not coh.subtract_flat.isEnabled()
    assert coh.subtract_topo.isChecked() and not coh.subtract_topo.isEnabled()
    assert coh.correct_bias.isChecked() and not coh.correct_bias.isEnabled()
    assert not coh.remove_ramp.isEnabled()
    assert coh.coherence_range.value() == 10
    assert not coh.coherence_azimuth.isEnabled()
    assert coh.findChild(type(coh.dem), "cohDem") is not None
    assert coh.validate() == ""
    coh.close()

    topo = TopoPhaseDialog(window, sources=["Workspace SLC stack"], baseline_m=12.0)
    _prepare(topo)
    assert "B_perp" in topo.findChild(QLabel, "operatorHelp").text()
    assert topo.orbit_degree.text() == "3" and not topo.orbit_degree.isEnabled()
    assert topo.dem.currentText().startswith("Workspace")
    assert topo.tile.currentText() == "100" and not topo.tile.isEnabled()
    assert not topo.output_phase.isEnabled()
    assert not topo.output_elevation.isEnabled()
    assert not topo.output_latlon.isEnabled()
    assert topo.baseline.value() == 12.0
    topo.close()

    gold = GoldsteinDialog(window, sources=["Workspace SLC stack"])
    _prepare(gold)
    assert "Z_filt" in gold.findChild(QLabel, "operatorHelp").text()
    assert gold.alpha.value() == 1.0
    assert not gold.adaptive.isEnabled()
    assert "not implemented" in gold.adaptive.toolTip().lower()
    assert gold.alpha_min.value() == 0.2 and not gold.alpha_min.isEnabled()
    assert gold.alpha_max.value() == 1.0
    assert gold.fft_size.currentText() == "64"
    assert not gold.block_extent.isEnabled()
    assert gold.window.currentText() == "3"
    assert not gold.coherence_mask.isEnabled()
    assert gold.coherence_threshold.value() == 0.2 and not gold.coherence_threshold.isEnabled()
    gold.close()

    orbit = ApplyOrbitDialog(window, sources=["Workspace SLC stack"])
    _prepare(orbit)
    assert "_Orb" in orbit.findChild(QLabel, "operatorHelp").text()
    assert orbit.orbit_type.currentText() == "Sentinel Precise (Auto Download)"
    assert orbit.degree.text() == "3"
    assert not orbit.do_not_fail.isChecked()
    assert orbit.validate() == ""
    orbit.degree.setText("nope")
    assert "integer" in orbit.validate().lower()
    orbit.degree.setText("0")
    assert "at least 1" in orbit.validate().lower()
    orbit.degree.setText("3")
    orbit.source.clear()
    assert "source" in orbit.validate().lower()
    orbit.close()

    stack = SafeStackDialog(window, n_files=0)
    assert "NONE" in stack.findChild(QLabel, "operatorHelp").text()
    assert stack.resampling.currentText() == "NONE"
    assert stack.initial_offset.currentText() == "Orbit"
    assert stack.output_extents.currentText() == "Master"
    assert not stack.find_reference.isEnabled()
    assert not stack.initial_offset.isEnabled()
    assert "not applied" in stack.initial_offset.toolTip().lower()
    bilinear = stack.resampling.model().item(stack.resampling.findText("BILINEAR_INTERPOLATION"))
    assert bilinear is not None and not bilinear.isEnabled()
    assert window.findChild(QAction, "applyOrbitAction") is not None
    assert window.findChild(QAction, "interferogramFormationAction") is not None
    assert window.findChild(QAction, "runGoldstein") is not None
    assert window.findChild(QAction, "runCoherence") is not None
    window.close()
    stack.close()


def test_dialogs_run_existing_pipeline_stages(qapp, tmp_path: Path):
    window = MainWindow()
    window.workspace.stack = generate_subsidence_stack(
        n_dates=4,
        height=24,
        width=24,
        spacing_days=12,
        rate_m_yr=0.05,
        sigma_px=4,
        temporal_coherence=1.0,
        ramp_rad=0.0,
        seed=2,
    )
    ifg = InterferogramDialog(window, sources=window._source_names())
    _prepare(ifg)
    ifg.output_coherence.setChecked(False)
    ifg.subtract_topo.setChecked(False)
    window.apply_interferogram_params(ifg.parameters())
    assert window.workspace.interferograms is not None
    assert window.workspace.interferograms.shape[0] >= 1
    assert "later × conjugate(earlier)" in window.log.toPlainText()
    assert "not applied" in window.log.toPlainText().lower()

    coh = CoherenceDialog(window, sources=window._source_names())
    _prepare(coh)
    coh.single_reference.setChecked(False)
    coh.square_pixel.setChecked(False)
    coh.independent.setChecked(True)
    coh.coherence_range.setValue(4)
    coh.coherence_azimuth.setValue(2)
    window.apply_coherence_params(coh.parameters())
    assert window.workspace.config.coherence_window == 4
    assert window.workspace.config.coherence_azimuth_window == 2
    assert window.workspace.coherence.shape[1:] == (24, 24)

    gold = GoldsteinDialog(window, sources=window._source_names())
    _prepare(gold)
    gold.alpha.setValue(0.4)
    gold.fft_size.setCurrentText("32")
    gold.window.setCurrentText("3")
    gold.save_box.setChecked(True)
    gold.directory.setText(str(tmp_path))
    gold.target.setText("scene")
    window.apply_goldstein_params(gold.parameters())
    assert window.workspace.config.goldstein_alpha == 0.4
    assert window.workspace.config.goldstein_window == 32
    assert window.workspace.config.goldstein_spectrum_smooth == 3
    assert window.workspace.filtered is not None
    assert (tmp_path / "scene_Flt.npy").is_file()

    dem = np.zeros((24, 24), dtype=np.float64)
    dem[8:16, 8:16] = 40.0
    window.workspace.dem_m = dem
    topo = TopoPhaseDialog(window, sources=window._source_names(), baseline_m=80.0)
    _prepare(topo)
    before = window.workspace.interferograms.copy()
    window.apply_topo_params(topo.parameters())
    assert not np.allclose(window.workspace.interferograms, before)
    assert "_DInSAR" in window.log.toPlainText()

    orbit = ApplyOrbitDialog(window, sources=window._source_names())
    _prepare(orbit)
    orbit.save_box.setChecked(True)
    orbit.directory.setText(str(tmp_path))
    orbit.target.setText("scene")
    window.apply_orbit_params(orbit.parameters())
    stub = tmp_path / "scene_Orb.json"
    assert stub.is_file()
    assert "not changed" in window.log.toPlainText()
    assert "ERROR" not in window.log.toPlainText()
    window.close()
    ifg.close()
    coh.close()
    gold.close()
    topo.close()
    orbit.close()


def test_product_set_dialog_loads_the_listed_safes(qapp, tmp_path: Path):
    early = dated_safe_zip(tmp_path, "20210401", 2 + 0j)
    late = dated_safe_zip(tmp_path, "20210507", 0 + 3j)
    window = MainWindow()
    dialog = SafeStackDialog(window, n_files=2)
    dialog.product_set.add_paths([late, early])
    assert dialog.product_set.count_label.text() == "2 Products"
    assert dialog.product_set.table.item(0, 1).text() == "SLC"
    dialog.form_ifg.setChecked(True)
    dialog.height.setValue(2)
    dialog.width.setValue(2)
    window.load_from_product_set(dialog)
    qapp.processEvents()
    log = window.log.toPlainText()
    assert "2 Products" in log
    assert early.name in log
    assert late.name in log
    assert "SLC" in log
    stack = window.workspace.stack
    assert stack is not None and stack.data.shape == (2, 2, 2)
    assert window.workspace.interferograms is not None
    assert "ERROR" not in log
    window.close()
    dialog.close()
