"""SNAP 14 batch-3 dialogs: observed controls, and dispatch that does not fake science."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("tifffile")

from PySide6.QtWidgets import QApplication

from ai_insar.gui.chain_dialogs import BackGeocodingDialog, EsdDialog, TopsDeburstDialog, TopsSplitDialog
from ai_insar.gui.main_window import MainWindow
from ai_insar.pipeline.tops import burst_pixel_window, split_swath
from ai_insar.safe import resolve_safe
from safe_factory import FIXTURE_ZIP


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def _ready(dialog) -> None:
    dialog.save_box.setChecked(False)


def test_burst_window_math_and_fixture_split_refuses_a_short_raster():
    assert burst_pixel_window(lines_per_burst=4, first=2, last=3, samples=5) == (4, 8, 5)
    product = resolve_safe(FIXTURE_ZIP)
    measurement = next(item for item in product.measurements if item.swath == "IW1" and item.polarisation == "VV")
    assert measurement.lines_per_burst == 1501
    assert len(measurement.bursts) == 9
    with pytest.raises(ValueError, match="not a TOPS split"):
        split_swath(product, swath="IW1", pol="VV", burst_first=1, burst_last=1)


def test_split_dialog_stays_empty_until_a_product_is_bound(qapp):
    dialog = TopsSplitDialog()
    _ready(dialog)
    assert dialog.subswath.count() == 0
    assert dialog.pol_list.count() == 0
    assert dialog.burst_text.text() == "1 to 9999 (max number of bursts: 9999)"
    assert "source:" in dialog._source_label
    product = resolve_safe(FIXTURE_ZIP)
    dialog.bind_product(product)
    names = [dialog.subswath.itemText(i) for i in range(dialog.subswath.count())]
    assert names[0] == ""
    assert names[1:] == list(product.swaths)
    assert "IW1" in names
    assert dialog.subswath.currentText() == ""
    assert dialog.pol_list.count() == len(product.polarisations)
    assert "9999" not in dialog.burst_text.text()
    assert "is bound" in dialog.notes.text()
    assert "With no source" not in dialog.notes.text()
    assert dialog.target.text().endswith("_split")
    assert "empty" in dialog.validate().lower()
    dialog.subswath.setCurrentText("IW1")
    dialog.pol_list.setCurrentRow(0)
    assert dialog.burst_text.text() == "1 to 9 (max number of bursts: 9)"
    assert dialog.validate() == ""
    window = MainWindow()
    with pytest.raises(ValueError, match="not a TOPS split"):
        window.apply_split(dialog.parameters())
    assert window.workspace.stack is None
    from PySide6.QtGui import QAction

    assert window.findChild(QAction, "radarTopsSplit") is not None
    assert window.findChild(QAction, "radarBackGeocoding") is not None
    assert window.findChild(QAction, "radarEsd") is not None
    assert window.findChild(QAction, "radarTopsDeburst") is not None
    window.close()
    dialog.close()


def test_esd_has_eighteen_controls_and_does_not_run(qapp):
    dialog = EsdDialog(sources=["Workspace SLC stack"])
    _ready(dialog)
    assert dialog.processing_control_count() == 18
    assert dialog.findChild(type(dialog.source), "esdWinWidth").currentText() == "512"
    assert dialog.findChild(type(dialog.source), "esdWinHeight").currentText() == "512"
    assert dialog.findChild(type(dialog.source), "esdEstimator").currentText() == "Periodogram"
    assert dialog.findChild(type(dialog.source), "esdWeight").currentText() == "Inv Quadratic"
    assert dialog.findChild(type(dialog.source), "esdIntegration").currentText() == "L1 and L2"
    assert dialog.findChild(type(dialog.target), "esdXcorr").text() == "0.1"
    assert dialog.findChild(type(dialog.target), "esdCohThreshold").text() == "0.3"
    assert dialog.findChild(type(dialog.target), "esdWindows").text() == "10"
    assert dialog.findChild(type(dialog.target), "esdMaxBaseline").text() == "4"
    assert dialog.findChild(type(dialog.target), "esdRangeShift").text() == "0.0"
    assert dialog.findChild(type(dialog.target), "esdAzimuthShift").text() == "0.0"
    assert not dialog.findChild(type(dialog.save_box), "esdUserRange").isChecked()
    assert not dialog.findChild(type(dialog.save_box), "esdUserAzimuth").isChecked()
    assert not dialog.findChild(type(dialog.save_box), "esdNoWrite").isChecked()
    assert "not implemented" in dialog.block_reason().lower() or "not run" in dialog.block_reason().lower()
    assert dialog.block_reason() in dialog.validate()
    dialog.close()


def test_back_geocoding_matches_the_graph_dialog_and_does_not_coregister(qapp, tmp_path):
    dialog = BackGeocodingDialog()
    assert dialog.status.text() == "Error: [NodeId: ProductSet-Reader] Please add a source product"
    assert dialog.dem.currentText() == "Copernicus 30m Global DEM (Auto Download)"
    assert dialog.dem.count() == 11
    assert dialog.dem_resampling.count() == 7
    assert dialog.dem_resampling.currentText() == "BILINEAR_INTERPOLATION"
    assert dialog.resampling.currentText() == "BISINC_5_POINT_INTERPOLATION"
    assert dialog.mask_elevation.isChecked()
    assert not dialog.output_deramp.isChecked()
    assert not dialog.disable_reramp.isChecked()
    assert dialog.target.text() == "target"
    assert dialog.format_box.currentText() == "BEAM-DIMAP"
    assert dialog.directory.text() == "/home/box"
    assert dialog.findChild(type(dialog.target), "ioOpen") is None
    assert "coregistration" in dialog.block_reason().lower()
    dialog.product_set.add_paths([tmp_path / "not-a-product.zip"])
    dialog._validate_clicked()
    assert "Please add a source product" not in dialog.status.text()
    assert "not run" in dialog.status.text().lower() or "not this operator" in dialog.status.text().lower()
    window = MainWindow()
    dialog._run()
    assert window.workspace.stack is None
    window.close()
    dialog.close()


def test_deburst_has_only_an_empty_polarisation_list(qapp):
    dialog = TopsDeburstDialog()
    _ready(dialog)
    assert dialog.pol_list.count() == 0
    assert dialog.pol_label.text() == "Polarisations:"
    dialog.source.setCurrentText("unset")
    assert "not a deburst" in dialog.validate().lower() or "not run" in dialog.validate().lower()
    dialog.close()
