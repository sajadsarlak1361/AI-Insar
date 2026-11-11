"""Parsed IW annotation fields against a real SAFE annotation file."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ai_insar.pipeline.geometry import Orbit
from ai_insar.pipeline.s1_annotation import BurstTiming, SwathModel, read_swath_annotation, split_annotation
from ai_insar.pipeline.tops_ops import _speed, _with_orbit

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


def test_split_restarts_the_burst_index_and_rewrites_the_line_times():
    model = read_swath_annotation(ANN)
    assert len(model.bursts) >= 6
    split = split_annotation(model, 5, 6)
    assert len(split.bursts) == 2
    assert split.bursts[0].azimuth_time == model.bursts[4].azimuth_time
    assert split.bursts[1].azimuth_time == model.bursts[5].azimuth_time
    assert len(split.fm_rates) == len(model.fm_rates)
    assert len(split.dc_estimates) == len(model.dc_estimates)
    assert np.array_equal(split.fm_rates[0].coefficients, model.fm_rates[0].coefficients)
    assert split.product_first_line_utc == np.datetime64(model.bursts[4].azimuth_time, "ns")
    ticks = int(round((model.lines_per_burst - 1) * model.azimuth_time_interval_s * 1e9))
    end = np.datetime64(model.bursts[5].azimuth_time, "ns") + np.timedelta64(ticks, "ns")
    assert split.product_last_line_utc == end
    assert split.number_of_lines == 2 * model.lines_per_burst
    assert split.geolocation_line[0] == model.geolocation_line[0] - 4 * model.lines_per_burst
    assert split.range_sampling_rate_hz == model.range_sampling_rate_hz
    assert split.terrain_height_m == model.terrain_height_m


def test_orbit_replacement_keeps_every_parsed_annotation_field():
    model = read_swath_annotation(ANN)
    other = Orbit(model.orbit.time, model.orbit.position_m + 10.0, model.orbit.velocity_m_s)
    replaced = _with_orbit(model, other)
    assert replaced.orbit is other
    names = (
        "range_sampling_rate_hz",
        "azimuth_pixel_spacing_m",
        "product_first_line_utc",
        "product_last_line_utc",
        "terrain_height_m",
        "range_bandwidth_hz",
        "azimuth_bandwidth_hz",
        "range_window_type",
        "range_window_coefficient",
        "azimuth_window_type",
        "azimuth_window_coefficient",
        "geolocation_line",
        "geolocation_pixel",
        "geolocation_height_m",
        "geolocation_incidence_rad",
    )
    for name in names:
        left = getattr(replaced, name)
        right = getattr(model, name)
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        else:
            assert left == right
    assert len(names) == 15


def test_split_speed_is_the_retained_midpoint_not_the_full_scene():
    epoch = np.datetime64("2021-06-30T05:26:00")
    times = epoch + np.arange(8) * np.timedelta64(10, "s")
    seconds = np.arange(8, dtype=np.float64) * 10.0
    position = np.zeros((8, 3))
    position[:, 0] = 7_000_000.0
    velocity = np.zeros((8, 3))
    velocity[:, 0] = seconds
    orbit = Orbit(times, position, velocity)
    interval = 0.1
    lines = 10
    bursts = tuple(
        BurstTiming(epoch + np.timedelta64(int(index * 3e9), "ns"), 0, lines - 1, 0, 20) for index in range(9)
    )
    last_tick = int(round((lines - 1) * interval * 1e9))
    model = SwathModel(
        wavelength_m=0.055,
        azimuth_time_interval_s=interval,
        range_pixel_spacing_m=2.3,
        slant_range_time_s=0.006,
        lines_per_burst=lines,
        samples_per_burst=20,
        number_of_lines=lines * 9,
        incidence_rad=0.6,
        fm_rate=np.array([-2000.0]),
        bursts=bursts,
        orbit=orbit,
        azimuth_steering_rate_rad_s=0.02,
        product_first_line_utc=np.datetime64(bursts[0].azimuth_time, "ns"),
        product_last_line_utc=np.datetime64(bursts[-1].azimuth_time, "ns") + np.timedelta64(last_tick, "ns"),
    )
    split = split_annotation(model, 5, 6)
    # Burst 5 starts at 12 s and burst 6's last line is at 15.9 s.
    assert abs(_speed(split, orbit) - 13.95) < 1e-6
    assert abs(_speed(model, orbit) - _speed(split, orbit)) > 1.0
    kept = _with_orbit(split, orbit)
    assert kept.product_first_line_utc == split.product_first_line_utc
    assert abs(_speed(kept, orbit) - 13.95) < 1e-6
