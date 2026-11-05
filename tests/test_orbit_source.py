"""Apply Orbit File follows the Source product and writes a BEAM-DIMAP.

The June workspace reference must not consume a July EOF. Each date keeps
its own precise orbit, and the saved product carries those state vectors.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("tifffile")

from PySide6.QtWidgets import QApplication

from ai_insar.gui.main_window import MainWindow
from ai_insar.gui.operator_dialog import OperatorRequest
from ai_insar.pipeline.dimap import write_beam_dimap
from ai_insar.pipeline.dimap_s1 import dimap_abstracted_time, parse_snap_utc, slc_split_source
from ai_insar.pipeline.s1_annotation import read_eof_orbit
from ai_insar.safe import resolve_safe, select_measurement
from ai_insar.sentinel import read_tiff_window

JUNE = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
)
JULY = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B.SAFE"
)
JUNE_EOF = Path(
    "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/06/"
    "S1A_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF.zip"
)
JULY_EOF = Path(
    "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/07/"
    "S1A_OPER_AUX_POEORB_OPOD_20210801T121825_V20210711T225942_20210713T005942.EOF.zip"
)


@pytest.fixture(scope="module")
def qapp():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt offscreen platform is unavailable: {exc}")


def _request(directory: Path, target: str) -> OperatorRequest:
    return OperatorRequest(
        target=target,
        save=True,
        directory=str(directory),
        open_in_app=False,
        format_name="BEAM-DIMAP",
    )


def _split_dimap(directory: Path, safe: Path, stem: str) -> tuple[object, object, Path]:
    product = resolve_safe(safe / "manifest.safe")
    measurement = select_measurement(product, "IW1", "VV")
    data = read_tiff_window(safe / measurement.href, 6032, 0, 8, 16)
    source = slc_split_source(product, measurement, row=6032, height=8, width=16)
    path = write_beam_dimap(
        directory,
        stem,
        data,
        None,
        band_name="IW1_VV",
        source=source,
    )
    return product, measurement, path


def _apply(window: MainWindow, directory: Path, source: Path, orbit: Path, target: str) -> None:
    window.apply_orbit_params(
        {
            "request": _request(directory, target),
            "orbit_type": "Sentinel Precise (Auto Download)",
            "polynomial_degree": 3,
            "do_not_fail": False,
            "source": str(source),
            "orbit_file": str(orbit),
        }
    )


def test_july_eof_is_not_applied_to_the_june_reference(qapp, tmp_path: Path):
    if not JUNE.is_dir() or not JULY.is_dir() or not JUNE_EOF.is_file() or not JULY_EOF.is_file():
        pytest.skip("SAFE products or POEORB files are not on disk")
    june_product, june_measurement, june_dim = _split_dimap(tmp_path, JUNE, "june_split")
    july_product, july_measurement, july_dim = _split_dimap(tmp_path, JULY, "july_split")
    window = MainWindow()
    june_data = read_tiff_window(JUNE / june_measurement.href, 6032, 0, 8, 16)
    july_data = read_tiff_window(JULY / july_measurement.href, 6032, 0, 8, 16)
    window._store_split_grid(june_product, june_measurement, june_data, 6032)
    window._store_split_grid(july_product, july_measurement, july_data, 6032)
    window._split_bindings[june_product.product_id]["dimap"] = june_dim
    window._split_bindings[july_product.product_id]["dimap"] = july_dim
    assert window._split_bindings[june_product.product_id]["role"] == "reference"
    assert window._split_bindings[july_product.product_id]["role"] == "secondary"

    with pytest.raises(RuntimeError, match="2021-06-30"):
        _apply(window, tmp_path, june_dim, JULY_EOF, "wrong_june")
    assert window.workspace.precise_orbit is None
    assert window.workspace.secondary_orbit is None

    _apply(window, tmp_path, june_dim, JUNE_EOF, "june_orbit")
    june_orbit = window.workspace.precise_orbit
    assert june_orbit is not None
    assert window.workspace.secondary_orbit is None
    june_out = tmp_path / "june_orbit.dim"
    assert june_out.is_file()
    assert (tmp_path / "june_orbit_Orb.json").is_file()
    june_when = parse_snap_utc(dimap_abstracted_time(june_out, "first_line_time"))
    assert str(june_when).startswith("2021-06-30")
    june_precise = read_eof_orbit(JUNE_EOF)
    june_vector = next(when for when in june_precise.time if abs(int((when - june_when).astype(np.int64))) < 120_000_000_000)
    index = int(np.where(june_precise.time == june_vector)[0][0])
    assert f"{float(june_precise.position_m[index, 0]):.6f}" in june_out.read_text(encoding="utf-8")

    _apply(window, tmp_path, july_dim, JULY_EOF, "july_orbit")
    assert window.workspace.precise_orbit is june_orbit
    assert window.workspace.secondary_orbit is not None
    assert window.workspace.secondary_orbit.time[0] != june_orbit.time[0]
    july_out = tmp_path / "july_orbit.dim"
    assert july_out.is_file()
    july_when = parse_snap_utc(dimap_abstracted_time(july_out, "first_line_time"))
    assert str(july_when).startswith("2021-07-12")
    july_precise = read_eof_orbit(JULY_EOF)
    july_vector = next(when for when in july_precise.time if abs(int((when - july_when).astype(np.int64))) < 120_000_000_000)
    index = int(np.where(july_precise.time == july_vector)[0][0])
    july_text = july_out.read_text(encoding="utf-8")
    assert f"{float(july_precise.position_m[index, 0]):.6f}" in july_text
    assert "2021-06-30T05:26:15" not in july_text.split("Orbit_State_Vectors", 1)[1].split("</MDElem>", 1)[0]
    log = window.log.toPlainText()
    assert "20210712" in log
    assert "20210630" in log
    window.close()
