"""Parsed IW annotation fields against a real SAFE annotation file."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ai_insar.pipeline.s1_annotation import read_swath_annotation

ANN = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "safe"
    / "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4.SAFE"
    / "annotation"
    / "s1b-iw1-slc-vv-20210401t052624-20210401t052649-026269-032297-004.xml"
)


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _texts(root: ET.Element, name: str) -> list[str]:
    return [(element.text or "").strip() for element in root.iter() if _local(element.tag) == name and (element.text or "").strip()]


def test_annotation_fields_match_the_safe_file():
    root = ET.parse(ANN).getroot()
    model = read_swath_annotation(ANN)
    assert model.range_sampling_rate_hz == float(_texts(root, "rangeSamplingRate")[0])
    assert model.wavelength_m == 299_792_458.0 / float(_texts(root, "radarFrequency")[0])
    assert abs(np.rad2deg(model.azimuth_steering_rate_rad_s) - float(_texts(root, "azimuthSteeringRate")[0])) < 1e-12
    assert model.azimuth_time_interval_s == float(_texts(root, "azimuthTimeInterval")[0])
    assert model.slant_range_time_s == float(_texts(root, "slantRangeTime")[0])
    assert model.range_pixel_spacing_m == float(_texts(root, "rangePixelSpacing")[0])
    assert model.azimuth_pixel_spacing_m == float(_texts(root, "azimuthPixelSpacing")[0])
    assert model.azimuth_ground_spacing_m() == model.azimuth_pixel_spacing_m
    assert model.lines_per_burst == int(float(_texts(root, "linesPerBurst")[0]))
    assert model.samples_per_burst == int(float(_texts(root, "samplesPerBurst")[0]))
    assert abs(np.rad2deg(model.incidence_rad) - float(_texts(root, "incidenceAngleMidSwath")[0])) < 1e-12
    assert str(model.product_first_line_utc)[:19] == "2021-04-01T05:26:24"
    heights = [float(item) for item in _texts(root, "value") if item]
    # terrainHeight/value is the first group of those <value> tags.
    terrain = []
    for element in root.iter():
        if _local(element.tag) != "terrainHeight":
            continue
        for child in list(element):
            if _local(child.tag) == "value":
                terrain.append(float(child.text))
    assert abs(model.terrain_height_m - sum(terrain) / len(terrain)) < 1e-6
    assert model.range_window_type == "Hamming"
    assert model.range_window_coefficient == 0.75
    assert model.azimuth_window_type == "Hamming"
    assert model.azimuth_window_coefficient == 0.7
    assert model.range_bandwidth_hz == 5.65e7
    assert model.azimuth_bandwidth_hz == 327.0
    assert len(model.fm_rates) == 10
    assert model.fm_rates[0].coefficients[0] == float(_texts(root, "azimuthFmRatePolynomial")[0].split()[0])
    assert model.fm_rates[0].t0_s == float(_texts(root, "t0")[0])
    assert len(model.dc_estimates) >= 1
    assert model.bursts[0].first_valid_sample >= 0
    assert model.orbit.time.size >= 3
    assert model.geolocation_height_m.size == 210
    assert abs(model.geolocation_height_m[0] - 2322.000320347026) < 1e-6
    assert int(model.geolocation_line[0]) == 0
    assert int(model.geolocation_pixel[0]) == 0
    del heights
