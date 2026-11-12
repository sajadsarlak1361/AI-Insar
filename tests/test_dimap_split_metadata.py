"""Split DIMAP carries the burst-window clock, not the scene start."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from ai_insar.pipeline.dimap import read_beam_dimap_band, write_beam_dimap
from ai_insar.pipeline.dimap_s1 import slc_split_source
from ai_insar.safe import resolve_safe, select_measurement

JUNE = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
)
JULY = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B.SAFE"
)
GPT = Path("/home/ubuntu/esa-snap/bin/gpt")


def _source(folder: Path, *, row: int, height: int, width: int, col: int = 0):
    product = resolve_safe(folder / "manifest.safe")
    measurement = select_measurement(product, "IW1", "VV")
    return slc_split_source(product, measurement, row=row, height=height, width=width, col=col)


def test_full_burst_window_clock_matches_snap14_and_is_not_the_scene_start():
    if not JUNE.is_dir() or not JULY.is_dir():
        pytest.skip("complete SAFE folders are not on disk")
    june = _source(JUNE, row=6032, height=3016, width=22849)
    july = _source(JULY, row=6032, height=3016, width=22849)
    assert june.first_line_snap == "30-JUN-2021 05:26:15.752882"
    assert june.last_line_snap == "30-JUN-2021 05:26:21.607106"
    assert june.scene_first_snap == "30-JUN-2021 05:26:04.720712"
    assert june.subset_offset_y == 6032
    assert june.subset_offset_x == 0
    assert june.product_id.startswith("S1A_IW_SLC__1SDV_20210630")
    assert july.first_line_snap == "12-JUL-2021 05:26:16.426464"
    assert july.last_line_snap == "12-JUL-2021 05:26:22.280688"
    assert july.scene_first_snap == "12-JUL-2021 05:26:05.392238"
    assert june.annotation_md.count('<MDElem name="burst">') == 2
    assert june.tie_grids[0][2].shape == (4, 21)
    offset_x, offset_y, step_x, step_y, cyclic = june.tie_grids[0][3:]
    assert offset_x == pytest.approx(0.5)
    assert offset_y == pytest.approx(-1507.5)
    assert step_x == pytest.approx(1142.45)
    assert step_y == pytest.approx(1508.0)
    assert cyclic is False


def test_saved_dimap_samples_are_unchanged_and_snap_can_backgeocode(tmp_path: Path):
    if not JUNE.is_dir() or not JULY.is_dir():
        pytest.skip("complete SAFE folders are not on disk")
    values = (np.arange(8 * 16).reshape(8, 16) + 1j * np.arange(8 * 16).reshape(8, 16)).astype(np.complex64)
    june = _source(JUNE, row=6032, height=8, width=16, col=11000)
    dim = write_beam_dimap(tmp_path, "june_meta", values, None, band_name="IW1_VV", source=june)
    text = dim.read_text(encoding="utf-8")
    abstracted = text.split('name="Abstracted_Metadata"', 1)[1].split('name="Band_IW1_VV"', 1)[0]
    assert ">30-JUN-2021 05:26:15.752882<" in abstracted
    assert "05:26:04.720712" not in abstracted
    assert ">6032<" in abstracted
    assert "Derived from (SLC)" in text
    assert "<WKT>" not in text
    assert "vector_data" not in text
    assert (tmp_path / "june_meta.data" / "tie_point_grids" / "latitude.img").is_file()
    restored = read_beam_dimap_band(dim, 0) + 1j * read_beam_dimap_band(dim, 1)
    assert np.array_equal(restored, values)
    hdr = (tmp_path / "june_meta.data" / "i_IW1_VV.hdr").read_text(encoding="utf-8")
    assert "Unit: real" in hdr
    if not GPT.is_file():
        pytest.skip("SNAP GPT is not installed")
    july_values = np.conjugate(values)
    july = _source(JULY, row=6032, height=8, width=16, col=11000)
    write_beam_dimap(tmp_path, "july_meta", july_values, None, band_name="IW1_VV", source=july)
    graph = tmp_path / "backgeo.xml"
    graph.write_text(
        f"""<graph id="backgeo-dimap">
  <version>1.0</version>
  <node id="Read-ref"><operator>Read</operator><parameters><file>{tmp_path / "june_meta.dim"}</file></parameters></node>
  <node id="Read-sec"><operator>Read</operator><parameters><file>{tmp_path / "july_meta.dim"}</file></parameters></node>
  <node id="Orbit-ref"><operator>Apply-Orbit-File</operator><sources><source>Read-ref</source></sources>
    <parameters><orbitType>Sentinel Precise (Auto Download)</orbitType><polyDegree>3</polyDegree><continueOnFail>false</continueOnFail></parameters></node>
  <node id="Orbit-sec"><operator>Apply-Orbit-File</operator><sources><source>Read-sec</source></sources>
    <parameters><orbitType>Sentinel Precise (Auto Download)</orbitType><polyDegree>3</polyDegree><continueOnFail>false</continueOnFail></parameters></node>
  <node id="Back-Geocoding"><operator>Back-Geocoding</operator>
    <sources><sourceProduct refid="Orbit-ref"/><sourceProduct.1 refid="Orbit-sec"/></sources>
    <parameters><demName>SRTM 3Sec</demName><demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <resamplingType>BILINEAR_INTERPOLATION</resamplingType><maskOutAreaWithoutElevation>false</maskOutAreaWithoutElevation></parameters></node>
  <node id="Write"><operator>Write</operator><sources><sourceProduct refid="Back-Geocoding"/></sources>
    <parameters><file>{tmp_path / "backgeo.dim"}</file><formatName>BEAM-DIMAP</formatName></parameters></node>
</graph>
""",
        encoding="utf-8",
    )
    completed = subprocess.run([str(GPT), str(graph)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    written = (tmp_path / "backgeo.dim").read_text(encoding="utf-8", errors="replace")
    assert "30-JUN-2021 05:26:15.752882" in written
    assert "S1A_OPER_AUX_POEORB" in written
    master = read_beam_dimap_band(tmp_path / "backgeo.dim", 0) + 1j * read_beam_dimap_band(tmp_path / "backgeo.dim", 1)
    assert np.array_equal(master, values)
