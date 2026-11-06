"""Full-size split DIMAP, then Apply Orbit File in a new process.

The second process has no in-memory split. It must read the BEAM-DIMAP from
disk, copy i/q unchanged, and write precise-orbit state vectors.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")

JUNE = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
)
JUNE_EOF = Path(
    "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/06/"
    "S1A_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF.zip"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_full_split_dimap_orbit_in_a_new_process(tmp_path: Path):
    if not JUNE.is_dir() or not JUNE_EOF.is_file():
        pytest.skip("June SAFE or POEORB is not on disk")
    from ai_insar.pipeline.dimap import write_beam_dimap
    from ai_insar.pipeline.dimap_s1 import dimap_abstracted_time, slc_split_source
    from ai_insar.safe import resolve_safe, select_measurement
    from ai_insar.sentinel import read_tiff_window

    product = resolve_safe(JUNE / "manifest.safe")
    measurement = select_measurement(product, "IW1", "VV")
    data = read_tiff_window(JUNE / measurement.href, 6032, 0, 3016, 22849)
    assert data.shape == (3016, 22849)
    source = slc_split_source(product, measurement, row=6032, height=3016, width=22849)
    split = write_beam_dimap(tmp_path, "june_full_split", data, None, band_name="IW1_VV", source=source)
    del data
    assert dimap_abstracted_time(split, "first_line_time") == "30-JUN-2021 05:26:15.752882"
    split_i = tmp_path / "june_full_split.data" / "i_IW1_VV.img"
    split_q = tmp_path / "june_full_split.data" / "q_IW1_VV.img"
    assert split_i.stat().st_size == 3016 * 22849 * 2
    i_hash = _sha256(split_i)
    q_hash = _sha256(split_q)

    out_dir = tmp_path / "orbit"
    out_dir.mkdir()
    script = textwrap.dedent(
        f"""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        application = QApplication([])
        from ai_insar.gui.main_window import MainWindow
        from ai_insar.gui.operator_dialog import OperatorRequest
        window = MainWindow()
        before = window.product_explorer.tree.topLevelItem(0).text(0)
        if before != "No Sentinel-1 product open":
            raise SystemExit("explorer was not empty at start: " + before)
        window.apply_orbit_params({{
            "request": OperatorRequest(
                target="june_full_orbit",
                save=True,
                directory={str(out_dir)!r},
                open_in_app=False,
                format_name="BEAM-DIMAP",
            ),
            "orbit_type": "Sentinel Precise (Auto Download)",
            "polynomial_degree": 3,
            "do_not_fail": False,
            "source": {str(split)!r},
            "orbit_file": {str(JUNE_EOF)!r},
        }})
        titles = [
            window.product_explorer.tree.topLevelItem(i).text(0)
            for i in range(window.product_explorer.tree.topLevelItemCount())
        ]
        if "No Sentinel-1 product open" in titles:
            raise SystemExit("DIMAP was not added to Product Explorer: " + repr(titles))
        if "june_full_split.dim" not in titles or "june_full_orbit.dim" not in titles:
            raise SystemExit("explorer titles: " + repr(titles))
        log = window.log.toPlainText()
        if "Shift not measured" in log:
            raise SystemExit(log)
        if "Saved june_full_orbit.dim" not in log:
            raise SystemExit(log)
        print(log)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + "\n" + completed.stderr)
    orbit_dim = out_dir / "june_full_orbit.dim"
    assert orbit_dim.is_file()
    assert (out_dir / "june_full_orbit_Orb.json").is_file()
    assert _sha256(out_dir / "june_full_orbit.data" / "i_IW1_VV.img") == i_hash
    assert _sha256(out_dir / "june_full_orbit.data" / "q_IW1_VV.img") == q_hash
    text = orbit_dim.read_text(encoding="utf-8", errors="replace")
    assert "30-JUN-2021 05:26:15.752882" in text
    assert "<NCOLS>22849</NCOLS>" in text
    assert "<NROWS>3016</NROWS>" in text
    assert "30-JUN-2021 05:26:05.752882" in text
    assert "30-JUN-2021 05:26:29.752882" in text
    assert "Sentinel Precise  S1A_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF" in text
    assert "Apply-Orbit-File" in text
    assert "Shift " in completed.stdout
    assert "not measured" not in completed.stdout
