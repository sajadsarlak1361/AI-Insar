"""Displacement, subset, save, and the precise-orbit file path."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel

from ai_insar.gui.late_dialogs import PhaseToDisplacementDialog, TerrainCorrectionDialog
from ai_insar.gui.main_window import MainWindow
from ai_insar.gui.operator_dialog import OperatorRequest
from ai_insar.pipeline.synthetic import SLCStack


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def _stack() -> SLCStack:
    data = np.ones((1, 8, 8), dtype=np.complex64)
    return SLCStack(
        data=data,
        dates=np.array(["2021-06-30"], dtype="datetime64[D]"),
        wavelength_m=0.0554658,
        sensor="Sentinel-1 SLC",
    )


def test_phase_to_displacement_subset_and_save(qapp, tmp_path: Path):
    window = MainWindow()
    window.workspace.stack = _stack()
    window.workspace.unwrapped = np.full((1, 8, 8), np.pi, dtype=np.float64)
    dialog = PhaseToDisplacementDialog(window, sources=["Workspace SLC stack"])
    dialog.save_box.setChecked(False)
    blank = dialog.findChild(QLabel, "phaseProcessingBlank")
    assert blank is not None
    assert blank.text() == ""
    window.apply_phase_to_displacement(dialog.parameters())
    assert window.workspace.los_m.shape == (1, 8, 8)
    assert float(window.workspace.los_m[0, 0, 0]) < 0.0
    source = tmp_path / "source.npy"
    np.save(source, window.workspace.stack.data)
    before = source.read_bytes()
    window.apply_subset({"row_start": 1, "row_end": 5, "col_start": 2, "col_end": 6})
    assert window.workspace.stack.data.shape == (1, 4, 4)
    assert window.workspace.los_m.shape == (1, 4, 4)
    assert source.read_bytes() == before
    window.apply_save_product(
        {
            "target": "disp",
            "format_name": "NumPy (.npy)",
            "directory": str(tmp_path),
            "band": "Displacement",
        }
    )
    assert (tmp_path / "disp.npy").is_file()
    window.apply_save_product(
        {
            "target": "disp",
            "format_name": "BEAM-DIMAP",
            "directory": str(tmp_path),
            "band": "Displacement",
        }
    )
    from ai_insar.pipeline.dimap import read_beam_dimap_band

    assert (tmp_path / "disp.dim").is_file()
    restored = read_beam_dimap_band(tmp_path / "disp.dim")
    assert restored.shape == window.workspace.los_m.shape[-2:]
    window.close()
    dialog.close()


def test_selected_reference_is_not_replaced_by_the_later_date(qapp):
    from ai_insar.pipeline.engine import compute_interferograms

    window = MainWindow()
    data = np.zeros((2, 4, 4), dtype=np.complex64)
    data[0] = 2
    data[1] = 1j
    window.workspace.stack = SLCStack(
        data=data,
        dates=np.array(["2021-06-30", "2021-07-12"], dtype="datetime64[D]"),
        wavelength_m=0.055,
    )
    window.workspace.reference_index = 0
    compute_interferograms(window.workspace, window._log)
    assert np.allclose(window.workspace.interferograms[0], data[0] * np.conj(data[1]))
    assert "not later × conjugate(earlier)" in window.log.toPlainText()
    window.close()


def test_three_distinct_products_keep_time_order_and_the_middle_reference(qapp, tmp_path: Path):
    from safe_factory import dated_safe_zip

    early = dated_safe_zip(tmp_path, "20210401", 1 + 0j)
    middle = dated_safe_zip(tmp_path, "20210513", 4 + 0j)
    late = dated_safe_zip(tmp_path, "20210624", 0 + 3j)
    window = MainWindow()
    window.load_safe_stack(
        [late, early, middle],
        swath="IW1",
        pol="VV",
        row=0,
        col=0,
        height=2,
        width=2,
        form_interferograms=True,
        reference=middle,
    )
    stack = window.workspace.stack
    assert stack is not None
    assert list(stack.dates) == [
        np.datetime64("2021-04-01"),
        np.datetime64("2021-05-13"),
        np.datetime64("2021-06-24"),
    ]
    assert window.workspace.reference_index == 1
    assert sum(item.is_reference for item in window.workspace.safe_inputs) == 1
    assert window.workspace.interferograms.shape[0] == 2
    assert np.allclose(window.workspace.interferograms[0], stack.data[1] * np.conj(stack.data[0]))
    assert np.allclose(window.workspace.interferograms[1], stack.data[1] * np.conj(stack.data[2]))
    window.close()


def test_terrain_dialog_lists_match_the_empty_product_audit(qapp):
    dialog = TerrainCorrectionDialog()
    dialog.save_box.setChecked(False)
    assert dialog.dem.currentText() == "Copernicus 30m Global DEM (Auto Download)"
    assert dialog.dem.count() == 12
    assert dialog.dem.findText("WGS84 ellipsoid (not terrain correction)") >= 0
    assert dialog.dem_resampling.currentText() == "BILINEAR_INTERPOLATION"
    assert dialog.dem_resampling.findText("DELAUNAY_INTERPOLATION") >= 0
    assert dialog.image_resampling.findText("DELAUNAY_INTERPOLATION") < 0
    assert dialog.projection.text() == "WGS84(DD)"
    assert dialog.mask_elevation.isChecked()
    assert not dialog.output_complex.isChecked()
    assert dialog.validate() == ""
    dialog.image_resampling.setCurrentText("NEAREST_NEIGHBOUR")
    assert dialog.validate() == ""
    dialog.close()


def test_local_orbit_file_replaces_annotation_vectors(qapp):
    annotation = Path("/tmp/safe-real/20210630/s1a-iw1-slc-vv.xml")
    orbit = Path("/tmp/safe-real/orbits/poeorb_20210630.EOF.zip")
    if not annotation.is_file() or not orbit.is_file():
        pytest.skip("real annotation and precise orbit are not on disk")
    from ai_insar.pipeline.s1_annotation import read_swath_annotation

    window = MainWindow()
    model = read_swath_annotation(annotation)
    window.workspace.swath_model = model
    request = OperatorRequest("scene", False, "", False)
    window._install_precise_orbit(str(orbit), 3, {"request": request})
    text = window.log.toPlainText()
    assert "replaced state vectors" in text
    assert "Shift" in text
    assert window.workspace.precise_orbit is not model.orbit
    window.close()
