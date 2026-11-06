"""Split tie points and the one-second orbit table against SNAP 12 products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_insar.pipeline.dimap_s1 import precise_orbit_table, slc_split_source
from ai_insar.pipeline.s1_annotation import read_eof_orbit
from ai_insar.safe import resolve_safe, select_measurement

JUNE = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
)
SNAP_SPLIT = Path("/tmp/snap-stage2/june_split.dim")
SNAP_ORBIT = Path("/tmp/snap-stage2/june_orbit.dim")
JUNE_EOF = Path(
    "/home/ubuntu/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2021/06/"
    "S1A_OPER_AUX_POEORB_OPOD_20210720T121856_V20210629T225942_20210701T005942.EOF.zip"
)


def test_split_tie_points_match_snap_bytes():
    if not JUNE.is_dir() or not SNAP_SPLIT.is_file():
        pytest.skip("June SAFE or the SNAP split DIMAP is not on disk")
    product = resolve_safe(JUNE / "manifest.safe")
    measurement = select_measurement(product, "IW1", "VV")
    source = slc_split_source(product, measurement, row=6032, height=3016, width=22849)
    folder = SNAP_SPLIT.with_suffix(".data") / "tie_point_grids"
    for name, _unit, values, offset_x, offset_y, step_x, step_y, _cyclic in source.tie_grids:
        reference = np.fromfile(folder / f"{name}.img", dtype=">f4").reshape(values.shape)
        assert offset_x == 0.5
        assert offset_y == -1507.5
        assert step_x == pytest.approx(1142.45)
        assert step_y == pytest.approx(1508.0)
        assert np.array_equal(np.asarray(values, dtype=np.float32), reference)


def test_one_second_orbit_table_matches_snap():
    if not SNAP_ORBIT.is_file() or not JUNE_EOF.is_file():
        pytest.skip("SNAP orbit DIMAP or the June POEORB is not on disk")
    import xml.etree.ElementTree as ET

    orbit = read_eof_orbit(JUNE_EOF)
    table = precise_orbit_table(
        orbit,
        np.datetime64("2021-06-30T05:26:15.752882"),
        np.datetime64("2021-06-30T05:26:21.607106"),
        3,
    )
    root = ET.parse(SNAP_ORBIT).getroot()

    def local(tag):
        return tag.split("}")[-1]

    found = None
    for element in root.iter():
        if local(element.tag) == "MDElem" and element.get("name") == "Orbit_State_Vectors":
            found = element
            break
    rows = []
    for child in list(found):
        if not str(child.get("name", "")).startswith("orbit_vector"):
            continue
        attrs = {
            item.get("name"): (item.text or "").strip()
            for item in list(child)
            if local(item.tag) == "MDATTR"
        }
        rows.append(attrs)
    assert len(table) == len(rows) == 25
    assert table[0][0] == rows[0]["time"] == "30-JUN-2021 05:26:05.752882"
    assert table[-1][0] == rows[-1]["time"] == "30-JUN-2021 05:26:29.752882"
    max_pos = 0.0
    max_vel = 0.0
    for ours, snap in zip(table, rows):
        pos = np.array([float(ours[1]), float(ours[2]), float(ours[3])])
        vel = np.array([float(ours[4]), float(ours[5]), float(ours[6])])
        spos = np.array([float(snap["x_pos"]), float(snap["y_pos"]), float(snap["z_pos"])])
        svel = np.array([float(snap["x_vel"]), float(snap["y_vel"]), float(snap["z_vel"])])
        max_pos = max(max_pos, float(np.linalg.norm(pos - spos)))
        max_vel = max(max_vel, float(np.linalg.norm(vel - svel)))
    assert max_pos < 0.001
    assert max_vel < 1e-6
