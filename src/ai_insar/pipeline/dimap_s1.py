"""Sentinel-1 fields a BEAM-DIMAP split must carry for SNAP orbit and back-geocoding.

The raster clock is the selected burst's azimuth time. The annotation scene
start stays in the source band record and in the annotation header. It is not
written as the product ``first_line_time``. Tie-point grids and vector layers
are not invented: the values below are read from the SAFE annotation and manifest.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from datetime import datetime, timedelta
from xml.sax.saxutils import escape

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class SlcDimapSource:
    """Identity and burst-window clock for one saved SLC raster."""

    product_id: str
    mission: str
    mode: str
    swath: str
    polarisation: str
    orbit_pass: str
    absolute_orbit: int | None
    relative_orbit: int | None
    datatake_id: str | None
    annotation_name: str
    first_line_snap: str
    last_line_snap: str
    scene_first_snap: str
    scene_last_snap: str
    scene_lines: int
    scene_samples: int
    line_time_interval_s: float
    subset_offset_x: int
    subset_offset_y: int
    lines: int
    samples: int
    range_spacing_m: float
    azimuth_spacing_m: float
    slant_range_m: float
    radar_frequency_mhz: float
    range_sampling_rate_mhz: float
    pulse_repetition_frequency_hz: float
    orbit_vectors: tuple[tuple[str, str, str, str, str, str, str], ...]
    annotation_md: str
    tie_grids: tuple[object, ...] = ()

    def dataset_sources_xml(self) -> str:
        product = escape(self.product_id)
        vectors = "\n".join(_orbit_vector_xml(index + 1, item) for index, item in enumerate(self.orbit_vectors))
        state_time = self.orbit_vectors[0][0] if self.orbit_vectors else self.first_line_snap
        orbit_abs = "" if self.absolute_orbit is None else str(self.absolute_orbit)
        orbit_rel = "" if self.relative_orbit is None else str(self.relative_orbit)
        datatake = self.datatake_id or ""
        band = f"Band_{self.swath}_{self.polarisation}"
        names = f"i_{self.swath}_{self.polarisation} q_{self.swath}_{self.polarisation}"
        return f"""  <Dataset_Sources>
    <MDElem name="metadata">
      <MDElem name="Abstracted_Metadata">
        <MDATTR name="PRODUCT" desc="Product name" unit="" type="ascii" mode="rw">{product}</MDATTR>
        <MDATTR name="PRODUCT_TYPE" desc="Product type" unit="" type="ascii" mode="rw">SLC</MDATTR>
        <MDATTR name="SPH_DESCRIPTOR" desc="Description" unit="" type="ascii" mode="rw">Sentinel-1 IW Level-1 SLC Product</MDATTR>
        <MDATTR name="MISSION" desc="Satellite mission" unit="" type="ascii" mode="rw">{escape(self.mission)}</MDATTR>
        <MDATTR name="ACQUISITION_MODE" desc="Acquisition mode" unit="" type="ascii" mode="rw">{escape(self.mode)}</MDATTR>
        <MDATTR name="antenna_pointing" desc="Right or left facing" unit="" type="ascii" mode="rw">right</MDATTR>
        <MDATTR name="SWATH" desc="Swath name" unit="" type="ascii" mode="rw">{escape(self.swath)}</MDATTR>
        <MDATTR name="PASS" desc="ASCENDING or DESCENDING" unit="" type="ascii" mode="rw">{escape(self.orbit_pass)}</MDATTR>
        <MDATTR name="ABS_ORBIT" desc="Orbit" unit="" type="int32" mode="rw">{orbit_abs}</MDATTR>
        <MDATTR name="REL_ORBIT" desc="Track" unit="" type="int32" mode="rw">{orbit_rel}</MDATTR>
        <MDATTR name="data_take_id" desc="Data take identifier" unit="" type="int32" mode="rw">{escape(datatake)}</MDATTR>
        <MDATTR name="STATE_VECTOR_TIME" desc="Time of orbit state vector" unit="utc" type="utc" mode="rw">{escape(state_time)}</MDATTR>
        <MDATTR name="first_line_time" desc="First zero doppler azimuth time" unit="utc" type="utc" mode="rw">{escape(self.first_line_snap)}</MDATTR>
        <MDATTR name="last_line_time" desc="Last zero doppler azimuth time" unit="utc" type="utc" mode="rw">{escape(self.last_line_snap)}</MDATTR>
        <MDATTR name="SAMPLE_TYPE" desc="DETECTED or COMPLEX" unit="" type="ascii" mode="rw">COMPLEX</MDATTR>
        <MDATTR name="mds1_tx_rx_polar" desc="Polarization" unit="" type="ascii" mode="rw">{escape(self.polarisation)}</MDATTR>
        <MDATTR name="azimuth_looks" unit="" type="float64" mode="rw">1.0</MDATTR>
        <MDATTR name="range_looks" unit="" type="float64" mode="rw">1.0</MDATTR>
        <MDATTR name="range_spacing" desc="Range sample spacing" unit="m" type="float64" mode="rw">{_num(self.range_spacing_m)}</MDATTR>
        <MDATTR name="azimuth_spacing" desc="Azimuth sample spacing" unit="m" type="float64" mode="rw">{_num(self.azimuth_spacing_m)}</MDATTR>
        <MDATTR name="radar_frequency" desc="Radar frequency" unit="MHz" type="float64" mode="rw">{_num(self.radar_frequency_mhz)}</MDATTR>
        <MDATTR name="line_time_interval" unit="s" type="float64" mode="rw">{_num(self.line_time_interval_s)}</MDATTR>
        <MDATTR name="num_output_lines" desc="Raster height" unit="lines" type="uint32" mode="rw">{self.lines}</MDATTR>
        <MDATTR name="num_samples_per_line" desc="Raster width" unit="samples" type="uint32" mode="rw">{self.samples}</MDATTR>
        <MDATTR name="subset_offset_x" desc="X coordinate of UL corner of subset in original image" unit="samples" type="uint32" mode="rw">{self.subset_offset_x}</MDATTR>
        <MDATTR name="subset_offset_y" desc="Y coordinate of UL corner of subset in original image" unit="samples" type="uint32" mode="rw">{self.subset_offset_y}</MDATTR>
        <MDATTR name="srgr_flag" desc="SRGR applied" unit="flag" type="uint8" mode="rw">0</MDATTR>
        <MDATTR name="slant_range_to_first_pixel" desc="Slant range to 1st data sample" unit="m" type="float64" mode="rw">{_num(self.slant_range_m)}</MDATTR>
        <MDATTR name="range_sampling_rate" desc="Range Sampling Rate" unit="MHz" type="float64" mode="rw">{_num(self.range_sampling_rate_mhz)}</MDATTR>
        <MDATTR name="pulse_repetition_frequency" desc="PRF" unit="Hz" type="float64" mode="rw">{_num(self.pulse_repetition_frequency_hz)}</MDATTR>
        <MDATTR name="coregistered_stack" desc="Coregistration applied" unit="flag" type="uint8" mode="rw">0</MDATTR>
        <MDATTR name="metadata_version" desc="AbsMetadata version" unit="" type="ascii" mode="rw">7.0</MDATTR>
        <MDElem name="Orbit_State_Vectors">
{vectors}
        </MDElem>
        <MDElem name="{escape(band)}">
          <MDATTR name="swath" desc="Swath name" unit="" type="ascii" mode="rw">{escape(self.swath)}</MDATTR>
          <MDATTR name="polarization" desc="Polarization" unit="" type="ascii" mode="rw">{escape(self.polarisation)}</MDATTR>
          <MDATTR name="annotation" desc="metadata file" unit="" type="ascii" mode="rw">{escape(self.annotation_name)}</MDATTR>
          <MDATTR name="band_names" desc="corresponding bands" unit="" type="ascii" mode="rw">{escape(names)}</MDATTR>
          <MDATTR name="first_line_time" desc="First zero doppler azimuth time" unit="utc" type="utc" mode="rw">{escape(self.scene_first_snap)}</MDATTR>
          <MDATTR name="last_line_time" desc="Last zero doppler azimuth time" unit="utc" type="utc" mode="rw">{escape(self.scene_last_snap)}</MDATTR>
          <MDATTR name="line_time_interval" desc="Time per line" unit="s" type="float64" mode="rw">{_num(self.line_time_interval_s)}</MDATTR>
          <MDATTR name="num_output_lines" desc="Raster height" unit="lines" type="uint32" mode="rw">{self.scene_lines}</MDATTR>
          <MDATTR name="num_samples_per_line" desc="Raster width" unit="samples" type="uint32" mode="rw">{self.scene_samples}</MDATTR>
          <MDATTR name="sample_type" desc="DETECTED or COMPLEX" unit="" type="ascii" mode="rw">COMPLEX</MDATTR>
        </MDElem>
      </MDElem>
      <MDElem name="Original_Product_Metadata">
        <MDElem name="annotation">
          <MDElem name="{escape(self.annotation_name)}">
{self.annotation_md}
          </MDElem>
        </MDElem>
      </MDElem>
    </MDElem>
  </Dataset_Sources>"""


def slc_split_source(product, measurement, *, row: int, height: int, width: int, col: int = 0) -> SlcDimapSource:
    """Metadata for the burst window ``split_swath`` just wrote.

    ``row`` is the absolute annotation line of raster line 0. ``height`` and
    ``width`` are the array that was written. The scene-start time in the
    annotation header is left unchanged and is not used as ``first_line_time``.
    """
    from ai_insar.safe import _read_href

    if not measurement.annotation_href:
        raise ValueError(f"{product.product_id} has no annotation for this split")
    payload = _read_href(product, measurement.annotation_href)
    root = ET.fromstring(payload)
    image = _first(root, "imageInformation")
    interval = float(_text(image, "azimuthTimeInterval"))
    lines_per_burst = int(float(_text(root, "linesPerBurst")))
    scene_first = _text(image, "productFirstLineUtcTime")
    scene_last = _text(image, "productLastLineUtcTime")
    scene_lines = int(float(_text(image, "numberOfLines")))
    scene_samples = int(float(_text(image, "numberOfSamples")))
    burst_list = _first(root, "burstList")
    bursts = [element for element in list(burst_list) if _local(element.tag) == "burst"]
    first_index = int(row) // lines_per_burst
    last_row = int(row) + int(height) - 1
    last_index = last_row // lines_per_burst
    if first_index < 0 or last_index >= len(bursts) or first_index > last_index:
        raise ValueError(f"rows {row}:{row + height} do not fall inside the annotation bursts")
    kept = bursts[first_index : last_index + 1]
    for element in bursts:
        burst_list.remove(element)
    for element in kept:
        burst_list.append(element)
    if "count" in burst_list.attrib:
        burst_list.attrib["count"] = str(len(kept))
    first_iso = _absolute_line_iso(bursts, lines_per_burst, interval, int(row))
    last_iso = _absolute_line_iso(bursts, lines_per_burst, interval, last_row)
    _set_text(image, "productFirstLineUtcTime", first_iso)
    _set_text(image, "productLastLineUtcTime", last_iso)
    _set_text(image, "numberOfLines", str(int(height)))
    if int(width) != scene_samples:
        _set_text(image, "numberOfSamples", str(int(width)))
    vectors = tuple(_orbit_record(element) for element in root.iter() if _local(element.tag) == "orbit")
    if not vectors:
        raise ValueError(f"{product.product_id} annotation has no orbit state vectors")
    frequency_hz = float(_text(root, "radarFrequency"))
    sampling_hz = float(_text(root, "rangeSamplingRate"))
    slant_time = float(_text(image, "slantRangeTime"))
    mission = "SENTINEL-1" + product.platform[-1]
    name = measurement.annotation_href.replace("\\", "/").split("/")[-1]
    return SlcDimapSource(
        product_id=product.product_id[:-5] if product.product_id.upper().endswith(".SAFE") else product.product_id,
        mission=mission,
        mode=product.mode,
        swath=measurement.swath,
        polarisation=measurement.polarisation,
        orbit_pass=product.orbit_pass,
        absolute_orbit=product.absolute_orbit,
        relative_orbit=product.relative_orbit,
        datatake_id=product.datatake_id,
        annotation_name=name,
        first_line_snap=snap_utc(first_iso),
        last_line_snap=snap_utc(last_iso),
        scene_first_snap=snap_utc(scene_first),
        scene_last_snap=snap_utc(scene_last),
        scene_lines=scene_lines,
        scene_samples=scene_samples,
        line_time_interval_s=interval,
        subset_offset_x=int(col),
        subset_offset_y=int(row),
        lines=int(height),
        samples=int(width),
        range_spacing_m=float(_text(image, "rangePixelSpacing")),
        azimuth_spacing_m=float(_text(image, "azimuthPixelSpacing")),
        slant_range_m=slant_time * _SPEED_OF_LIGHT_M_S / 2.0,
        radar_frequency_mhz=frequency_hz / 1e6,
        range_sampling_rate_mhz=sampling_hz / 1e6,
        pulse_repetition_frequency_hz=float(_text(root, "prf")),
        orbit_vectors=vectors,
        annotation_md=_element_to_md(root),
        tie_grids=_tie_grids(root, row=int(row), height=int(height), col=int(col), scene_samples=scene_samples, lines_per_burst=lines_per_burst),
    )


def _tie_grids(root: ET.Element, *, row: int, height: int, col: int, scene_samples: int, lines_per_burst: int) -> tuple[object, ...]:
    """Lat/lon and angle grids from the annotation geolocation points.

    The points are the SAFE records. Their raster position is the annotation
    line and pixel minus this window's origin, with SNAP's half-pixel tie
    offset. No corner is invented.
    """
    points = [element for element in root.iter() if _local(element.tag) == "geolocationGridPoint"]
    if not points:
        return ()
    by_line: dict[int, list[ET.Element]] = {}
    for element in points:
        by_line.setdefault(int(float(_text(element, "line"))), []).append(element)
    for line in by_line:
        by_line[line].sort(key=lambda item: int(float(_text(item, "pixel"))))
    ordered = sorted(by_line)
    stop = row + height
    chosen = [line for line in ordered if row <= line <= stop]
    if not chosen:
        return ()
    predecessor = [line for line in ordered if line < chosen[0]]
    if predecessor:
        chosen = [predecessor[-1], *chosen]
    ncols = len(by_line[chosen[0]])
    if any(len(by_line[line]) != ncols for line in chosen):
        return ()
    step_y = float(chosen[1] - chosen[0]) if len(chosen) > 1 else float(lines_per_burst)
    if any(chosen[i + 1] - chosen[i] != chosen[1] - chosen[0] for i in range(len(chosen) - 1)):
        step_y = float(lines_per_burst)
    offset_y = float(chosen[0] - row) + 0.5
    offset_x = 0.5 - float(col)
    step_x = float(scene_samples) / float(ncols - 1) if ncols > 1 else 1.0
    fields = (
        ("latitude", "latitude", "deg", False, 1.0),
        ("longitude", "longitude", "deg", True, 1.0),
        ("incident_angle", "incidenceAngle", "deg", False, 1.0),
        ("elevation_angle", "elevationAngle", "deg", False, 1.0),
        ("slant_range_time", "slantRangeTime", "ns", False, 1e9),
    )
    grids = []
    for name, tag, unit, cyclic, scale in fields:
        values = np.empty((len(chosen), ncols), dtype=np.float32)
        for i, line in enumerate(chosen):
            for j, element in enumerate(by_line[line]):
                values[i, j] = float(_text(element, tag)) * scale
        grids.append((name, unit, values, offset_x, offset_y, step_x, step_y, cyclic))
    return tuple(grids)


def snap_utc(iso_time: str) -> str:
    """``2021-06-30T05:26:15.752882`` to ``30-JUN-2021 05:26:15.752882``."""
    stamp = _parse_iso(iso_time)
    month = _MONTHS[stamp.month - 1]
    return (
        f"{stamp.day:02d}-{month}-{stamp.year:04d} "
        f"{stamp.hour:02d}:{stamp.minute:02d}:{stamp.second:02d}.{stamp.microsecond:06d}"
    )


def _absolute_line_iso(bursts: list[ET.Element], lines_per_burst: int, interval: float, absolute_row: int) -> str:
    index = int(absolute_row) // int(lines_per_burst)
    local = int(absolute_row) % int(lines_per_burst)
    return _line_iso(bursts, lines_per_burst, interval, local, origin_burst=index)


def _line_iso(
    bursts: list[ET.Element],
    lines_per_burst: int,
    interval: float,
    local_line: int,
    *,
    origin_burst: int,
) -> str:
    base = _parse_iso(_text(bursts[origin_burst], "azimuthTime"))
    shifted = base + timedelta(microseconds=int(round(float(local_line) * float(interval) * 1_000_000.0)))
    return shifted.strftime("%Y-%m-%dT%H:%M:%S.%f")


def _parse_iso(text: str) -> datetime:
    head, _, fraction = text.strip().partition(".")
    micros = int((fraction + "000000")[:6]) if fraction else 0
    stamp = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    return stamp.replace(microsecond=micros)


def _orbit_record(element: ET.Element) -> tuple[str, str, str, str, str, str, str]:
    when = snap_utc(_text(element, "time"))
    position = _triplet(element, "position")
    velocity = _triplet(element, "velocity")
    return (when, *position, *velocity)


def _triplet(element: ET.Element, name: str) -> tuple[str, str, str]:
    node = _first(element, name)
    values = { _local(child.tag): (child.text or "").strip() for child in list(node) }
    return values["x"], values["y"], values["z"]


def _orbit_vector_xml(index: int, record: tuple[str, ...]) -> str:
    when, x, y, z, vx, vy, vz = record
    return "\n".join(
        (
            f'          <MDElem name="orbit_vector{index}">',
            f'            <MDATTR name="time" type="utc" mode="rw">{escape(when)}</MDATTR>',
            f'            <MDATTR name="x_pos" type="float64" mode="rw">{escape(x)}</MDATTR>',
            f'            <MDATTR name="y_pos" type="float64" mode="rw">{escape(y)}</MDATTR>',
            f'            <MDATTR name="z_pos" type="float64" mode="rw">{escape(z)}</MDATTR>',
            f'            <MDATTR name="x_vel" type="float64" mode="rw">{escape(vx)}</MDATTR>',
            f'            <MDATTR name="y_vel" type="float64" mode="rw">{escape(vy)}</MDATTR>',
            f'            <MDATTR name="z_vel" type="float64" mode="rw">{escape(vz)}</MDATTR>',
            "          </MDElem>",
        )
    )


def _element_to_md(element: ET.Element) -> str:
    name = _local(element.tag)
    children = list(element)
    text = (element.text or "").strip()
    if not children and not element.attrib:
        return f'<MDATTR name="{escape(name)}" type="ascii" mode="rw">{escape(text)}</MDATTR>'
    lines = [f'<MDElem name="{escape(name)}">']
    for key, value in element.attrib.items():
        lines.append(f'<MDATTR name="{escape(_local(key))}" type="ascii" mode="rw">{escape(value)}</MDATTR>')
    # A leaf such as firstValidSample count="1508">-1 -1 ... stores the text
    # under an attribute of the same name. SNAP reads that attribute.
    if text:
        lines.append(f'<MDATTR name="{escape(name)}" type="ascii" mode="rw">{escape(text)}</MDATTR>')
    for child in children:
        lines.append(_element_to_md(child))
    lines.append("</MDElem>")
    return "".join(lines)


def _set_text(parent: ET.Element, name: str, value: str) -> None:
    for child in parent:
        if _local(child.tag) == name:
            child.text = value
            return
    raise ValueError(f"annotation element {name} is missing")


def _text(parent: ET.Element, name: str) -> str:
    for child in parent.iter() if parent is not None else ():
        if _local(child.tag) == name and (child.text or "").strip():
            return (child.text or "").strip()
    raise ValueError(f"annotation element {name} is missing")


def _first(parent: ET.Element, name: str) -> ET.Element:
    for element in parent.iter():
        if _local(element.tag) == name:
            return element
    raise ValueError(f"annotation element {name} is missing")


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _num(value: float) -> str:
    return format(float(value), ".16g")

