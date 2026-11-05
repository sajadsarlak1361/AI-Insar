"""Read the Sentinel-1 IW annotation fields the TOPS chain uses.

This is the public product annotation, not a copy of SNAP. Only the swath
timing, burst valid-sample rows, FM-rate polynomial, and annotation orbit
are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class RangePolynomial:
    """A slant-range polynomial dated by one azimuth time.

    ``coefficients[k]`` multiplies ``(tau - t0_s)`` to the power ``k``.
    ``tau`` is two-way slant-range time in seconds.
    """

    azimuth_time: np.datetime64
    t0_s: float
    coefficients: np.ndarray


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
    fm_rates: tuple[RangePolynomial, ...] = ()
    dc_estimates: tuple[RangePolynomial, ...] = ()
    azimuth_steering_rate_rad_s: float = 0.0
    first_valid_pixel: int = 0

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
# SNAP ProductData.UTC.getMJD is days since 2000-01-01, then burst times are
# getMJD()*86400. 1/86400 in float64 is not a dyadic rational, so a burst
# interval is not an integer number of azimuth lines.
_SNAP_UTC_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
_SNAP_DAY_SCALE = 1.0 / 86400.0


def snap_utc_seconds(when: np.datetime64) -> float:
    """Seconds since 2000-01-01 on SNAP's UTC scale.

    This is the clock ``Sentinel1Utils`` stores in ``burstFirstLineTime``.
    Subtracting two values and dividing by the azimuth interval must use
    these doubles: a nanosecond datetime difference rounds the other way
    and changes the deburst line count.
    """
    text = str(when)
    head, fraction = text.split(".")
    micros = int((fraction + "000000")[:6])
    stamp = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    days = (datetime(stamp.year, stamp.month, stamp.day, tzinfo=timezone.utc) - _SNAP_UTC_EPOCH).days
    seconds = stamp.hour * 3600 + stamp.minute * 60 + stamp.second
    mjd = days + (seconds + micros * 1e-6) * _SNAP_DAY_SCALE
    return float(mjd * 86400.0)


def read_swath_annotation(path: Path | str) -> SwathModel:
    """Parse one measurement annotation XML file into a :class:`SwathModel`."""
    return read_swath_annotation_bytes(Path(path).read_bytes())


def read_swath_annotation_bytes(payload: bytes) -> SwathModel:
    """Parse annotation XML bytes. A zip member does not have to be extracted first."""
    root = ET.fromstring(payload)
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
    fm_rates = _range_polynomials(root, "azimuthFmRate", "azimuthFmRatePolynomial")
    dc_tag = "dataDcPolynomial" if "data analysis" in _optional_text(root, "dcMethod").lower() else "geometryDcPolynomial"
    dc_estimates = _range_polynomials(root, "dcEstimate", dc_tag)
    steering = _optional_float(root, "azimuthSteeringRate")
    first_valid_pixel = _first_valid_pixel(root)
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
        fm_rates=tuple(fm_rates),
        dc_estimates=tuple(dc_estimates),
        azimuth_steering_rate_rad_s=float(np.deg2rad(steering)),
        first_valid_pixel=first_valid_pixel,
    )


def _optional_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if _local(element.tag) == name and (element.text or "").strip():
            return (element.text or "").strip()
    return ""


def _optional_float(root: ET.Element, name: str) -> float:
    text = _optional_text(root, name)
    return float(text) if text else 0.0


def _range_polynomials(root: ET.Element, record: str, coefficient: str) -> list[RangePolynomial]:
    found: list[RangePolynomial] = []
    for element in root.iter():
        if _local(element.tag) != record:
            continue
        try:
            when = parse_utc(_child_text(element, "azimuthTime"))
            origin = float(_child_text(element, "t0"))
            coeffs = np.array(_child_text(element, coefficient).split(), dtype=np.float64)
        except ValueError:
            continue
        found.append(RangePolynomial(azimuth_time=when, t0_s=origin, coefficients=coeffs))
    return found


def _first_valid_pixel(root: ET.Element) -> int:
    """Largest annotated first-valid range sample. Invalid lines are ignored."""
    widest = 0
    for element in root.iter():
        if _local(element.tag) != "firstValidSample":
            continue
        values = np.array((element.text or "").split(), dtype=np.int32)
        valid = values[values >= 0]
        if valid.size:
            widest = max(widest, int(np.max(valid)))
    return widest


def read_swath_from_product(product, measurement) -> SwathModel:
    """Read the annotation for one measurement from a SAFE folder or zip."""
    from ai_insar.safe import _read_href

    if not measurement.annotation_href:
        raise ValueError(f"{product.product_id} has no annotation for {measurement.href}")
    return read_swath_annotation_bytes(_read_href(product, measurement.annotation_href))


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
