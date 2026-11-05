"""Read the Sentinel-1 IW annotation fields the TOPS chain uses.

This is the public product annotation, not a copy of SNAP. Only the swath
timing, burst valid-sample rows, FM-rate polynomial, and annotation orbit
are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ai_insar.pipeline.geometry import SPEED_OF_LIGHT_M_S, Orbit, parse_utc


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if child is element:
            continue
        if _local(child.tag) == name and (child.text or "").strip():
            return (child.text or "").strip()
    raise ValueError(f"annotation has no {name}")


def _first(element: ET.Element, name: str) -> ET.Element:
    for child in element.iter():
        if _local(child.tag) == name:
            return child
    raise ValueError(f"annotation has no {name}")


@dataclass
class BurstTiming:
    azimuth_time: np.datetime64
    first_valid_line: int
    last_valid_line: int
    first_valid_sample: int
    last_valid_sample: int


@dataclass
class SwathModel:
    """One subswath and polarisation, enough for split, deburst, and geometry."""

    wavelength_m: float
    azimuth_time_interval_s: float
    range_pixel_spacing_m: float
    slant_range_time_s: float
    lines_per_burst: int
    samples_per_burst: int
    number_of_lines: int
    incidence_rad: float
    fm_rate: np.ndarray
    bursts: tuple[BurstTiming, ...]
    orbit: Orbit

    def line_time(self, absolute_line: int) -> np.datetime64:
        burst_index = int(absolute_line) // self.lines_per_burst
        line_in_burst = int(absolute_line) % self.lines_per_burst
        if burst_index < 0 or burst_index >= len(self.bursts):
            raise ValueError(f"line {absolute_line} is outside the swath")
        start = self.bursts[burst_index].azimuth_time
        ticks = int(round(line_in_burst * self.azimuth_time_interval_s * 1e9))
        return start + np.timedelta64(ticks, "ns")

    def slant_range_m(self, sample: float) -> float:
        delay = self.slant_range_time_s + 2.0 * float(sample) * self.range_pixel_spacing_m / SPEED_OF_LIGHT_M_S
        return SPEED_OF_LIGHT_M_S * delay / 2.0

    def azimuth_ground_spacing_m(self, degree: int = 3) -> float:
        """Ground spacing from the satellite speed and the line time."""
        when = self.bursts[len(self.bursts) // 2].azimuth_time
        _pos, vel = self.orbit.interpolate(when, degree)
        speed = float(np.linalg.norm(vel))
        radius = float(np.linalg.norm(_pos))
        ground_speed = speed * (WGS84_MEAN_RADIUS_M / radius)
        return ground_speed * self.azimuth_time_interval_s

    def ground_range_spacing_m(self) -> float:
        return self.range_pixel_spacing_m / float(np.sin(self.incidence_rad))


WGS84_MEAN_RADIUS_M = 6371000.0


def read_swath_annotation(path: Path | str) -> SwathModel:
    """Parse one measurement annotation XML into a :class:`SwathModel`."""
    root = ET.parse(Path(path)).getroot()
    frequency = float(_text(root, "radarFrequency"))
    wavelength = SPEED_OF_LIGHT_M_S / frequency
    image = _first(root, "imageInformation")
    azimuth_dt = float(_text(image, "azimuthTimeInterval"))
    range_spacing = float(_text(image, "rangePixelSpacing"))
    slant_time = float(_text(image, "slantRangeTime"))
    incidence = float(np.deg2rad(float(_text(image, "incidenceAngleMidSwath"))))
    lines_per_burst = int(float(_text(root, "linesPerBurst")))
    samples_per_burst = int(float(_text(root, "samplesPerBurst")))
    number_of_lines = int(float(_text(image, "numberOfLines")))
    fm = np.array([float(item) for item in _text(root, "azimuthFmRatePolynomial").split()], dtype=np.float64)
    times = []
    position = []
    velocity = []
    for orbit in root.iter():
        if _local(orbit.tag) != "orbit":
            continue
        times.append(parse_utc(_text(orbit, "time")))
        coords = [float(item) for item in _direct_triplet(orbit, "position")]
        speeds = [float(item) for item in _direct_triplet(orbit, "velocity")]
        position.append(coords)
        velocity.append(speeds)
    bursts: list[BurstTiming] = []
    for burst in root.iter():
        if _local(burst.tag) != "burst":
            continue
        first = np.array(_child_text(burst, "firstValidSample").split(), dtype=np.int32)
        last = np.array(_child_text(burst, "lastValidSample").split(), dtype=np.int32)
        valid = np.flatnonzero(first >= 0)
        if valid.size == 0:
            raise ValueError("burst has no valid lines")
        sample = int(first[valid[0]])
        sample_last = int(last[valid[0]])
        bursts.append(
            BurstTiming(
                azimuth_time=parse_utc(_child_text(burst, "azimuthTime")),
                first_valid_line=int(valid[0]),
                last_valid_line=int(valid[-1]),
                first_valid_sample=sample,
                last_valid_sample=sample_last,
            )
        )
    return SwathModel(
        wavelength_m=wavelength,
        azimuth_time_interval_s=azimuth_dt,
        range_pixel_spacing_m=range_spacing,
        slant_range_time_s=slant_time,
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
        number_of_lines=number_of_lines,
        incidence_rad=incidence,
        fm_rate=fm,
        bursts=tuple(bursts),
        orbit=Orbit(np.array(times), np.array(position), np.array(velocity)),
    )


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in list(element):
        if _local(child.tag) == name and (child.text or "").strip():
            return (child.text or "").strip()
    raise ValueError(f"element has no {name}")


def _direct_triplet(element: ET.Element, name: str) -> list[str]:
    for child in list(element):
        if _local(child.tag) == name:
            values = []
            for item in list(child):
                if (item.text or "").strip():
                    values.append((item.text or "").strip())
            return values
    raise ValueError(f"element has no {name}")


def read_eof_orbit(path: Path | str) -> Orbit:
    """Read a Sentinel precise-orbit EOF (or a zip that contains one)."""
    source = Path(path)
    if source.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(source) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".eof") and "var/www" not in name]
            if not names:
                names = [name for name in archive.namelist() if name.lower().endswith(".eof")]
            payload = archive.read(names[0])
        root = ET.fromstring(payload)
    else:
        root = ET.parse(source).getroot()
    times = []
    position = []
    velocity = []
    for osv in root.iter():
        if _local(osv.tag) != "OSV":
            continue
        times.append(parse_utc(_child_text(osv, "UTC")))
        position.append([float(_child_text(osv, axis)) for axis in ("X", "Y", "Z")])
        velocity.append([float(_child_text(osv, axis)) for axis in ("VX", "VY", "VZ")])
    if len(times) < 4:
        raise ValueError(f"{source.name} does not contain orbit state vectors")
    return Orbit(np.array(times), np.array(position), np.array(velocity))
