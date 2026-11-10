"""Scene centre, raw-storage size, and Java-style doubles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_insar.pipeline.dimap_s1 import (
    _num,
    _product_storage_bytes,
    _scene_centre,
    _spherical_azimuths,
)
from ai_insar.safe import resolve_safe

FIXTURE = Path(
    "/workspace/tests/fixtures/safe/"
    "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4.SAFE"
)
JUNE = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210630T052604_20210630T052631_038565_048CFD_632B.SAFE"
)
JULY = Path(
    "/tmp/safe-full/S1A_IW_SLC__1SDV_20210712T052605_20210712T052632_038740_049246_C00B.SAFE"
)


def test_float32_incidence_text_keeps_the_binary_value():
    value = float(np.float32(np.uint32(0x41F1FAE8).view(np.float32)))
    assert _num(value) == "30.247512817382812"


def test_spherical_azimuths_are_not_a_flat_180_degree_pair():
    forward, reverse = _spherical_azimuths(51.04, 13.70, 49.54, 13.22)
    assert forward == np.float64(forward)
    assert abs((forward - reverse) - 180.0) > 0.001


def test_fixture_scene_centre_is_finite():
    if not (FIXTURE / "manifest.safe").is_file():
        return
    product = resolve_safe(FIXTURE / "manifest.safe")
    lat, lon, heading, heading2 = _scene_centre(product)
    assert np.isfinite([lat, lon, heading, heading2]).all()
    assert 40.0 < lat < 55.0
    assert 5.0 < lon < 20.0


def test_june_and_july_match_the_snap14_centre_and_size():
    if not JUNE.is_dir() or not JULY.is_dir():
        return
    from ai_insar.pipeline.dimap_s1 import slc_split_source
    from ai_insar.safe import select_measurement

    expected = {
        JUNE: {
            "centre_lat": "50.43669367015973",
            "centre_lon": "11.646378560769534",
            "centre_heading": "190.13355836716534",
            "centre_heading2": "10.131184741104054",
            "total_size": "294567749",
        },
        JULY: {
            "centre_lat": "50.43710089779335",
            "centre_lon": "11.645129057609031",
            "centre_heading": "190.13224928914647",
            "centre_heading2": "10.129875536029664",
            "total_size": "294565746",
        },
    }
    for folder, want in expected.items():
        product = resolve_safe(folder / "manifest.safe")
        measurement = select_measurement(product, "IW1", "VV")
        source = slc_split_source(product, measurement, row=6032, height=3016, width=22849)
        text = source.dataset_sources_xml(str(folder / "out.dim"))
        for name, value in want.items():
            assert f'name="{name}"' in text
            assert f">{value}<" in text, name
        size = _product_storage_bytes(product, 3016, 22849, source.tie_grids)
        assert str(size) == want["total_size"]
