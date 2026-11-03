"""Many Sentinel-1 SAFE zips become one SLC stack, then interferograms."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("tifffile")
import tifffile

from ai_insar.conventions import C_BAND_WAVELENGTH_M
from ai_insar.pipeline.engine import Workspace, compute_interferograms
from ai_insar.pipeline.interferogram import date_pairs, form_interferograms
from ai_insar.sentinel import expand_safe_inputs, load_safe_slc_stack

RADAR_HZ = 5.405e9


def _token(iso: str) -> str:
    return iso[:4] + iso[5:7] + iso[8:10] + "T" + iso[11:13] + iso[14:16] + iso[17:19]


def _product_id(platform: str, start: str, stop: str, orbit: int, datatake: int, suffix: str) -> str:
    return (
        f"S1{platform}_IW_SLC__1SDV_{_token(start)}_{_token(stop)}_"
        f"{orbit:06d}_{datatake:06X}_{suffix}"
    )


def _write_product(
    directory: Path,
    *,
    platform: str,
    start: str,
    stop: str,
    orbit: int,
    datatake: int,
    suffix: str,
    value: complex,
    bands: list[tuple[str, str]] | None = None,
    shape: tuple[int, int] = (8, 8),
    as_zip: bool = True,
) -> Path:
    """Miniature SAFE zip (ZIP_STORED) or folder. Not a copy of SNAP."""
    product_id = _product_id(platform, start, stop, orbit, datatake, suffix)
    folder = directory / f"{product_id}.SAFE"
    measurement = folder / "measurement"
    annotation = folder / "annotation"
    measurement.mkdir(parents=True)
    annotation.mkdir()
    used = bands or [("IW1", "VV")]
    swath_tags = "\n".join(f"<swath>{swath}</swath>" for swath, _pol in used)
    pol_tags = "\n".join(
        f"<transmitterReceiverPolarisation>{pol}</transmitterReceiverPolarisation>" for _swath, pol in used
    )
    stamp = _token(start).lower()
    for index, (swath, pol) in enumerate(used, start=1):
        href = f"s1{platform.lower()}-{swath.lower()}-slc-{pol.lower()}-{stamp}-{index:03d}"
        array = np.full(shape, value, dtype=np.complex64)
        tifffile.imwrite(measurement / f"{href}.tiff", array)
        (annotation / f"{href}.xml").write_text(
            f"""<product>
              <adsHeader><swath>{swath}</swath><polarisation>{pol}</polarisation></adsHeader>
              <generalAnnotation><productInformation>
                <radarFrequency>{RADAR_HZ}</radarFrequency>
              </productInformation></generalAnnotation>
              <imageAnnotation><imageInformation>
                <numberOfSamples>{shape[1]}</numberOfSamples>
                <numberOfLines>{shape[0]}</numberOfLines>
              </imageInformation></imageAnnotation>
            </product>""",
            encoding="utf-8",
        )
    (folder / "manifest.safe").write_text(
        f"""<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:safe="http://www.esa.int/safe/sentinel-1.0">
          <metadataSection>
            <metadataObject ID="platform"><metadataWrap><xmlData>
              <safe:platform>
                <safe:familyName>SENTINEL-1</safe:familyName>
                <safe:number>{platform}</safe:number>
                <safe:instrument><safe:extension><instrumentMode>
                  <mode>IW</mode>
                  {swath_tags}
                </instrumentMode></safe:extension></safe:instrument>
              </safe:platform>
            </xmlData></metadataWrap></metadataObject>
            <metadataObject ID="generalProductInformation"><metadataWrap><xmlData>
              <standAloneProductInformation>
                <productType>SLC</productType>
                {pol_tags}
                <missionDataTakeID>{datatake}</missionDataTakeID>
              </standAloneProductInformation>
            </xmlData></metadataWrap></metadataObject>
            <metadataObject ID="acquisitionPeriod"><metadataWrap><xmlData>
              <acquisitionPeriod>
                <startTime>{start}</startTime>
                <stopTime>{stop}</stopTime>
              </acquisitionPeriod>
            </xmlData></metadataWrap></metadataObject>
            <metadataObject ID="measurementOrbitReference"><metadataWrap><xmlData>
              <orbitReference>
                <orbitNumber type="start">{orbit}</orbitNumber>
                <relativeOrbitNumber type="start">7</relativeOrbitNumber>
                <pass>ASCENDING</pass>
              </orbitReference>
            </xmlData></metadataWrap></metadataObject>
          </metadataSection>
        </xfdu:XFDU>""",
        encoding="utf-8",
    )
    if not as_zip:
        return folder
    dest = directory / f"{product_id}.zip"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in folder.rglob("*"):
            if path.is_file():
                archive.write(path, f"{product_id}.SAFE/{path.relative_to(folder).as_posix()}")
    shutil.rmtree(folder)
    return dest


def _three_dates(directory: Path) -> tuple[Path, Path, Path]:
    early = _write_product(
        directory,
        platform="A",
        start="2020-03-01T12:00:00.000000",
        stop="2020-03-01T12:00:30.000000",
        orbit=32100,
        datatake=20,
        suffix="A1B2",
        value=1 + 0j,
    )
    middle = _write_product(
        directory,
        platform="A",
        start="2020-03-13T12:00:00.000000",
        stop="2020-03-13T12:00:30.000000",
        orbit=32113,
        datatake=21,
        suffix="B2C3",
        value=0 + 1j,
    )
    late = _write_product(
        directory,
        platform="B",
        start="2020-04-06T12:00:00.000000",
        stop="2020-04-06T12:00:30.000000",
        orbit=26200,
        datatake=22,
        suffix="C3D4",
        value=2 + 3j,
    )
    return early, middle, late


def test_three_safe_zips_stack_and_interferograms(tmp_path: Path):
    early, middle, late = _three_dates(tmp_path)
    progress: list[tuple[int, int, str]] = []
    loaded = load_safe_slc_stack(
        [late, early, middle],
        row=0,
        col=0,
        height=8,
        width=8,
        on_progress=lambda index, total, message: progress.append((index, total, message)),
    )
    stack = loaded.stack
    assert loaded.skipped == ()
    assert stack.data.shape == (3, 8, 8)
    assert stack.data.dtype == np.complex64
    assert list(stack.dates) == [
        np.datetime64("2020-03-01", "D"),
        np.datetime64("2020-03-13", "D"),
        np.datetime64("2020-04-06", "D"),
    ]
    assert stack.wavelength_m == pytest.approx(C_BAND_WAVELENGTH_M)
    assert "Not coregistered" in stack.description
    assert "IW1 VV" in stack.description
    assert np.allclose(stack.data[0], 1 + 0j)
    assert np.allclose(stack.data[1], 1j)
    assert np.allclose(stack.data[2], 2 + 3j)
    assert progress[0][0] == 1 and progress[0][1] == 3
    assert any(message.startswith("Reading ") for _index, _total, message in progress)

    pairs = date_pairs(stack.n_dates, 3)
    assert pairs == [(0, 1), (0, 2), (1, 2)]
    formed = form_interferograms(stack.data, pairs)
    assert formed.shape == (3, 8, 8)
    assert formed.dtype == np.complex64
    assert np.allclose(formed[0], stack.data[1] * np.conj(stack.data[0]))
    assert np.allclose(formed[1], 2 + 3j)
    assert np.allclose(formed[2], 3 - 2j)

    workspace = Workspace()
    workspace.stack = stack
    workspace.config.max_date_span = 1
    compute_interferograms(workspace)
    assert workspace.pairs == [(0, 1), (1, 2)]
    assert workspace.interferograms is not None
    assert workspace.interferograms.shape == (2, 8, 8)
    assert np.allclose(workspace.interferograms[0], 1j)


def test_folder_of_zips_skips_unsuitable_products(tmp_path: Path):
    folder = tmp_path / "downloads"
    folder.mkdir()
    _three_dates(folder)
    _write_product(
        folder,
        platform="A",
        start="2020-03-01T12:00:00.000000",
        stop="2020-03-01T12:00:30.000000",
        orbit=32150,
        datatake=23,
        suffix="D4E5",
        value=9 + 0j,
    )
    _write_product(
        folder,
        platform="A",
        start="2020-05-01T12:00:00.000000",
        stop="2020-05-01T12:00:30.000000",
        orbit=32200,
        datatake=24,
        suffix="E5F6",
        value=5 + 0j,
        bands=[("IW2", "VH")],
    )
    _write_product(
        folder,
        platform="A",
        start="2020-05-13T12:00:00.000000",
        stop="2020-05-13T12:00:30.000000",
        orbit=32212,
        datatake=25,
        suffix="F6A7",
        value=4 + 0j,
        shape=(4, 4),
    )
    (folder / "notes.txt").write_text("not a product", encoding="utf-8")
    found = expand_safe_inputs([folder])
    assert len(found) == 6
    assert all(path.suffix.lower() == ".zip" and path.name.lower().startswith("s1") for path in found)

    loaded = load_safe_slc_stack([folder], row=0, col=0, height=256, width=256)
    stack = loaded.stack
    assert stack.data.shape == (3, 8, 8)
    assert list(stack.dates) == [
        np.datetime64("2020-03-01", "D"),
        np.datetime64("2020-03-13", "D"),
        np.datetime64("2020-04-06", "D"),
    ]
    assert np.allclose(stack.data[0], 1 + 0j)
    reasons = " ".join(reason for _label, reason in loaded.skipped)
    assert "IW1 VV" in reasons
    assert "already in the stack" in reasons
    assert "crop is 4×4" in reasons
    assert len(loaded.skipped) == 3


def test_safe_folder_is_optional_and_empty_selection_fails(tmp_path: Path):
    early, _middle, _late = _three_dates(tmp_path)
    folder = _write_product(
        tmp_path,
        platform="A",
        start="2020-06-01T12:00:00.000000",
        stop="2020-06-01T12:00:30.000000",
        orbit=33001,
        datatake=26,
        suffix="A7B8",
        value=3 + 1j,
        as_zip=False,
    )
    loaded = load_safe_slc_stack([folder, early], row=1, col=2, height=2, width=3)
    assert loaded.stack.data.shape == (2, 2, 3)
    assert loaded.stack.dates[0] == np.datetime64("2020-03-01", "D")
    assert loaded.stack.dates[1] == np.datetime64("2020-06-01", "D")
    assert np.allclose(loaded.stack.data[1], 3 + 1j)
    with pytest.raises(ValueError, match="no Sentinel-1 SAFE"):
        load_safe_slc_stack([])
    only_vh = _write_product(
        tmp_path,
        platform="A",
        start="2020-07-01T12:00:00.000000",
        stop="2020-07-01T12:00:30.000000",
        orbit=33020,
        datatake=27,
        suffix="B8C9",
        value=1 + 0j,
        bands=[("IW2", "VH")],
    )
    with pytest.raises(ValueError, match="IW1 VV"):
        load_safe_slc_stack([only_vh])


def test_stored_zip_stack_does_not_extract(tmp_path: Path, monkeypatch):
    early, middle, _late = _three_dates(tmp_path)

    def fail_extract(*_args, **_kwargs):
        raise AssertionError("STORED zip entry was extracted")

    monkeypatch.setattr("ai_insar.safe._cache_zip_member", fail_extract)
    loaded = load_safe_slc_stack([middle, early], row=0, col=0, height=2, width=2)
    assert loaded.stack.data.shape == (2, 2, 2)
    assert loaded.stack.dates[0] == np.datetime64("2020-03-01", "D")
    assert np.allclose(loaded.stack.data[1], 1j)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def test_gui_open_safe_stack_runs_interferograms(qapp, tmp_path: Path):
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QAction

    from ai_insar.gui.main_window import MainWindow
    from ai_insar.gui.safe_stack_dialog import SafeStackDialog

    window = MainWindow()
    action = window.findChild(QAction, "openSafeStackAction")
    assert action is not None and action.text() == "Open SAFE stack…"
    for menu_name in ("File", "Data"):
        menu = next(item.menu() for item in window.menuBar().actions() if item.text() == menu_name)
        assert any(item.text() == "Open SAFE stack…" for item in menu.actions())

    early, middle, late = _three_dates(tmp_path)
    dialog = SafeStackDialog()
    dialog.add_paths([late, early, middle])
    assert dialog.files.count() == 3
    dialog.height.setValue(8)
    dialog.width.setValue(8)
    dialog.build_stack()
    assert dialog.load_result is not None
    assert dialog.load_result.stack.data.shape == (3, 8, 8)
    window.apply_safe_stack(dialog.load_result)
    qapp.processEvents()
    assert "3 dates" in window.statusBar().currentMessage()

    window.load_safe_products([late, middle, early], row=0, col=0, height=8, width=8)
    qapp.processEvents()
    log = window.log.toPlainText()
    assert "SAFE stack 1/3" in log
    assert "Reading " in log
    assert window.workspace.stack is not None
    assert window.workspace.stack.data.shape == (3, 8, 8)

    window.run_interferograms()
    qapp.processEvents()
    interferograms = window.workspace.interferograms
    assert interferograms is not None
    assert interferograms.shape == (3, 8, 8)
    assert np.allclose(interferograms[0], 1j)
    assert "later × conjugate(earlier)" in window.log.toPlainText()

    window.run_all_stages()
    qapp.processEvents()
    assert window.workspace.interferograms is not None
    assert window.workspace.interferograms.shape == (3, 8, 8)
    assert window.workspace.velocity_m_yr is not None
    assert window.workspace.velocity_m_yr.shape == (8, 8)
    assert "ERROR" not in window.log.toPlainText()
    window.close()
