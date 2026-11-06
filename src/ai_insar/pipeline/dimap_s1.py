"""Sentinel-1 fields a BEAM-DIMAP split must carry for SNAP orbit and back-geocoding.

The raster clock is the selected burst's azimuth time. The annotation scene
start stays in the source band record and in the annotation header. It is not
written as the product ``first_line_time``. Tie-point grids and vector layers
are not invented: the values below are read from the SAFE annotation and manifest.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

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
    extra_abstracted_xml: str = ""
    split_parameters: tuple[str, str, int, int] | None = None
    image_number: int = 0

    def dataset_sources_xml(self, output_file: str = "") -> str:
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
{self.extra_abstracted_xml}        <MDElem name="Orbit_State_Vectors">
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
          <MDATTR name="image_number" type="int32" mode="rw">{self.image_number}</MDATTR>
        </MDElem>
      </MDElem>
      <MDElem name="Original_Product_Metadata">
        <MDElem name="annotation">
          <MDElem name="{escape(self.annotation_name)}">
{self.annotation_md}
          </MDElem>
        </MDElem>
      </MDElem>
{self._processing_graph_xml(output_file)}    </MDElem>
  </Dataset_Sources>"""

    def _processing_graph_xml(self, output_file: str) -> str:
        if self.split_parameters is None:
            return ""
        swath, polarisation, first_burst, last_burst = self.split_parameters
        target = escape(output_file) if output_file else ""
        return f"""      <MDElem name="Processing_Graph">
        <MDElem name="node.0">
          <MDATTR name="operator" type="ascii" mode="rw">TOPSAR-Split</MDATTR>
          <MDElem name="parameters">
            <MDATTR name="subswath" type="ascii">{escape(swath)}</MDATTR>
            <MDATTR name="selectedPolarisations" type="ascii">{escape(polarisation)}</MDATTR>
            <MDATTR name="firstBurstIndex" type="int32">{first_burst}</MDATTR>
            <MDATTR name="lastBurstIndex" type="int32">{last_burst}</MDATTR>
          </MDElem>
        </MDElem>
        <MDElem name="node.1">
          <MDATTR name="operator" type="ascii" mode="rw">Write</MDATTR>
          <MDElem name="parameters">
            <MDATTR name="file" type="ascii">{target}</MDATTR>
            <MDATTR name="formatName" type="ascii">BEAM-DIMAP</MDATTR>
          </MDElem>
        </MDElem>
      </MDElem>
"""


def orbit_records_around(orbit, start, stop, *, margin_s: float = 120.0) -> tuple[tuple[str, str, str, str, str, str, str], ...]:
    """Precise-orbit state vectors that cover one acquisition, in SNAP metadata order.

    ``start`` and ``stop`` are zero-Doppler times. Vectors outside
    ``margin_s`` of that window are left out. Positions are metres and
    velocities are metres per second, which is what abstracted metadata stores.
    """
    begin = np.datetime64(start) - np.timedelta64(int(margin_s * 1e9), "ns")
    end = np.datetime64(stop) + np.timedelta64(int(margin_s * 1e9), "ns")
    records = []
    for index, when in enumerate(orbit.time):
        if when < begin or when > end:
            continue
        position = orbit.position_m[index]
        velocity = orbit.velocity_m_s[index]
        records.append(
            (
                snap_utc(_iso_from_datetime64(when)),
                f"{float(position[0]):.6f}",
                f"{float(position[1]):.6f}",
                f"{float(position[2]):.6f}",
                f"{float(velocity[0]):.8f}",
                f"{float(velocity[1]):.8f}",
                f"{float(velocity[2]):.8f}",
            )
        )
    if len(records) < 4:
        raise ValueError("the precise orbit does not contain enough state vectors for this acquisition")
    return tuple(records)


def parse_snap_utc(text: str):
    """``30-JUN-2021 05:26:15.752882`` to a numpy datetime64 in microseconds."""
    date, clock = text.strip().split()
    day, month, year = date.split("-")
    hour, minute, second = clock.split(":")
    whole, _, fraction = second.partition(".")
    micros = int((fraction + "000000")[:6]) if fraction else 0
    stamp = datetime(int(year), _MONTHS.index(month.upper()) + 1, int(day), int(hour), int(minute), int(whole), micros)
    return np.datetime64(stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{micros:06d}")


def inspect_beam_dimap(path) -> dict[str, object]:
    """Read product identity from a BEAM-DIMAP header. Rasters are not loaded."""
    source = Path(path)
    root = ET.parse(source).getroot()
    bands = [((item.text or "").strip()) for item in root.iter() if _local(item.tag) == "BAND_NAME" and (item.text or "").strip()]
    rows = _dim_text(root, "NROWS")
    cols = _dim_text(root, "NCOLS")
    name = _dim_text(root, "DATASET_NAME") or source.stem
    vectors = _abstracted_orbit_vectors(root)
    return {
        "path": source,
        "name": name,
        "first_line": dimap_abstracted_time(source, "first_line_time"),
        "last_line": dimap_abstracted_time(source, "last_line_time"),
        "lines": int(rows or "0"),
        "samples": int(cols or "0"),
        "bands": tuple(bands),
        "orbit_vectors": len(vectors),
    }


def dimap_orbit_shift(path, precise, degree: int = 3) -> tuple[float, np.ndarray]:
    """Metres between the DIMAP state vectors and ``precise`` at the first line.

    The comparison time is abstracted ``first_line_time``. Both orbits are
    interpolated with the same polynomial degree.
    """
    root = ET.parse(path).getroot()
    current = _orbit_from_vectors(_abstracted_orbit_vectors(root))
    when = parse_snap_utc(dimap_abstracted_time(path, "first_line_time"))
    before, _velocity = current.interpolate(when, degree)
    after, velocity = precise.interpolate(when, degree)
    before = np.atleast_2d(before)[0]
    after = np.atleast_2d(after)[0]
    return float(np.linalg.norm(after - before)), np.atleast_2d(velocity)[0]


def precise_orbit_table(orbit, first, last, degree: int = 3) -> tuple[tuple[str, str, str, str, str, str, str], ...]:
    """State vectors on the one-second grid Apply Orbit File stores.

    The first time is ten seconds before the product's first line and the
    last is the last whole second that still falls inside ten seconds after
    the last line. Position and velocity are the degree-3 polynomial through
    the surrounding orbit-file records.
    """
    start = np.datetime64(first, "us") - np.timedelta64(10, "s")
    stop = np.datetime64(last, "us") + np.timedelta64(10, "s")
    span_us = int((stop - start) / np.timedelta64(1, "us"))
    count = span_us // 1_000_000
    if count < 4:
        raise ValueError("the acquisition is too short for a one-second orbit table")
    records = []
    for index in range(count):
        when = start + np.timedelta64(index, "s")
        position, velocity = orbit.interpolate(when, degree)
        records.append(
            (
                snap_utc(_iso_from_datetime64(when)),
                _num(position[0]),
                _num(position[1]),
                _num(position[2]),
                _num(velocity[0]),
                _num(velocity[1]),
                _num(velocity[2]),
            )
        )
    return tuple(records)


def orbit_state_vector_file_name(orbit_type: str, orbit_file) -> str:
    """``Sentinel Precise  <eof name>``, the abstracted-metadata text."""
    label = str(orbit_type or "Sentinel Precise")
    if "(" in label:
        label = label[: label.index("(")]
    name = Path(orbit_file).name
    if name.lower().endswith(".zip"):
        name = name[:-4]
    return f"{label} {name}"


def apply_precise_orbit_dimap(
    source_dim,
    directory,
    target: str,
    precise,
    *,
    orbit_file=None,
    orbit_type: str = "Sentinel Precise (Auto Download)",
    degree: int = 3,
) -> Path:
    """Copy a BEAM-DIMAP and replace its orbit state vectors.

    The ``.data`` rasters, including i/q and tie-point grids, are copied
    byte for byte. Only the header orbit list and the file hrefs change.
    """
    import shutil

    source = Path(source_dim)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stem = (target or "product").strip() or "product"
    dest = folder / f"{stem}.dim"
    dest_data = folder / f"{stem}.data"
    source_data = source.with_suffix(".data")
    if not source_data.is_dir():
        raise ValueError(f"{source} has no {source_data.name} raster directory")
    if dest_data.exists():
        shutil.rmtree(dest_data)
    shutil.copytree(source_data, dest_data)
    root = ET.parse(source).getroot()
    old = source.stem + ".data/"
    new = stem + ".data/"
    for element in root.iter():
        href = element.get("href")
        if href and href.startswith(old):
            element.set("href", new + href[len(old) :])
        if _local(element.tag) == "DATASET_NAME" and (element.text or "").strip():
            element.text = stem
    vectors = _vectors_for_header(precise, root, degree)
    _replace_abstracted_orbit_vectors(root, vectors)
    if orbit_file is not None:
        _set_abstracted_attr(root, "orbit_state_vector_file", orbit_state_vector_file_name(orbit_type, orbit_file))
    _append_orbit_graph(root, dest, orbit_type, degree)
    dest.write_bytes(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + ET.tostring(root, encoding="utf-8"))
    return dest


def _vectors_for_header(precise, root: ET.Element, degree: int = 3) -> tuple[tuple[str, str, str, str, str, str, str], ...]:
    first = parse_snap_utc(dimap_abstracted_time_from_root(root, "first_line_time"))
    last = parse_snap_utc(dimap_abstracted_time_from_root(root, "last_line_time"))
    return precise_orbit_table(precise, first, last, degree)


def _set_abstracted_attr(root: ET.Element, name: str, value: str) -> None:
    abstracted = None
    for element in root.iter():
        if _local(element.tag) == "MDElem" and element.get("name") == "Abstracted_Metadata":
            abstracted = element
            break
    if abstracted is None:
        return
    for child in list(abstracted):
        if _local(child.tag) == "MDATTR" and child.get("name") == name:
            child.text = value
            return
    tag = "MDATTR"
    for child in list(abstracted):
        if _local(child.tag) == "MDATTR":
            tag = child.tag
            break
    attr = ET.Element(tag)
    attr.set("name", name)
    attr.set("type", "ascii")
    attr.set("mode", "rw")
    attr.text = value
    abstracted.append(attr)


def _append_orbit_graph(root: ET.Element, dest: Path, orbit_type: str, degree: int) -> None:
    metadata = None
    for element in root.iter():
        if _local(element.tag) == "MDElem" and element.get("name") == "metadata":
            metadata = element
            break
    if metadata is None:
        return
    graph = None
    for child in list(metadata):
        if _local(child.tag) == "MDElem" and child.get("name") == "Processing_Graph":
            graph = child
            break
    elem_tag = metadata.tag
    attr_tag = "MDATTR"
    for child in list(metadata):
        if _local(child.tag) == "MDATTR":
            attr_tag = child.tag
            break
    if graph is None:
        graph = ET.SubElement(metadata, elem_tag)
        graph.set("name", "Processing_Graph")
        _append_split_node_from_header(graph, elem_tag, attr_tag, root)
    existing = [
        child
        for child in list(graph)
        if _local(child.tag) == "MDElem" and str(child.get("name", "")).startswith("node.")
    ]
    next_index = len(existing)
    _append_operator_node(
        graph,
        elem_tag,
        attr_tag,
        next_index,
        "Apply-Orbit-File",
        (
            ("orbitType", orbit_type),
            ("polyDegree", str(int(degree))),
            ("continueOnFail", "false"),
        ),
    )
    _append_operator_node(
        graph,
        elem_tag,
        attr_tag,
        next_index + 1,
        "Write",
        (("file", str(dest)), ("formatName", "BEAM-DIMAP")),
    )


def _append_split_node_from_header(graph: ET.Element, elem_tag: str, attr_tag: str, root: ET.Element) -> None:
    swath = dimap_abstracted_time_from_root(root, "SWATH") if _has_attr(root, "SWATH") else "IW1"
    polarisation = dimap_abstracted_time_from_root(root, "mds1_tx_rx_polar") if _has_attr(root, "mds1_tx_rx_polar") else "VV"
    offset = int(float(dimap_abstracted_time_from_root(root, "subset_offset_y"))) if _has_attr(root, "subset_offset_y") else 0
    lines = int(float(dimap_abstracted_time_from_root(root, "num_output_lines"))) if _has_attr(root, "num_output_lines") else 0
    per_burst = 1508
    first = offset // per_burst + 1
    last = (offset + max(lines, 1) - 1) // per_burst + 1
    _append_operator_node(
        graph,
        elem_tag,
        attr_tag,
        0,
        "TOPSAR-Split",
        (
            ("subswath", swath),
            ("selectedPolarisations", polarisation),
            ("firstBurstIndex", str(first)),
            ("lastBurstIndex", str(last)),
        ),
    )
    _append_operator_node(graph, elem_tag, attr_tag, 1, "Write", (("formatName", "BEAM-DIMAP"),))


def _has_attr(root: ET.Element, name: str) -> bool:
    for element in root.iter():
        if _local(element.tag) == "MDATTR" and element.get("name") == name and (element.text or "").strip():
            return True
    return False


def _append_operator_node(graph, elem_tag, attr_tag, index: int, operator: str, parameters: tuple[tuple[str, str], ...]) -> None:
    node = ET.SubElement(graph, elem_tag)
    node.set("name", f"node.{index}")
    op = ET.SubElement(node, attr_tag)
    op.set("name", "operator")
    op.set("type", "ascii")
    op.set("mode", "rw")
    op.text = operator
    params = ET.SubElement(node, elem_tag)
    params.set("name", "parameters")
    for name, value in parameters:
        attr = ET.SubElement(params, attr_tag)
        attr.set("name", name)
        attr.set("type", "ascii")
        attr.text = value


def dimap_abstracted_time_from_root(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if _local(element.tag) == "MDATTR" and element.get("name") == name and (element.text or "").strip():
            return (element.text or "").strip()
    raise ValueError(f"DIMAP has no {name}")


def _abstracted_orbit_vectors(root: ET.Element) -> list[tuple[str, float, float, float, float, float, float]]:
    found = None
    for element in root.iter():
        if _local(element.tag) == "MDElem" and element.get("name") == "Orbit_State_Vectors":
            found = element
            break
    if found is None:
        raise ValueError("DIMAP has no Orbit_State_Vectors")
    records = []
    for child in list(found):
        if _local(child.tag) != "MDElem" or not str(child.get("name", "")).startswith("orbit_vector"):
            continue
        attrs = {
            item.get("name"): (item.text or "").strip()
            for item in list(child)
            if _local(item.tag) == "MDATTR"
        }
        records.append(
            (
                attrs["time"],
                float(attrs["x_pos"]),
                float(attrs["y_pos"]),
                float(attrs["z_pos"]),
                float(attrs["x_vel"]),
                float(attrs["y_vel"]),
                float(attrs["z_vel"]),
            )
        )
    if len(records) < 4:
        raise ValueError("DIMAP orbit list has fewer than 4 state vectors")
    return records


def _orbit_from_vectors(records: list[tuple[str, float, float, float, float, float, float]]):
    from ai_insar.pipeline.geometry import Orbit

    times = [parse_snap_utc(item[0]) for item in records]
    position = [[item[1], item[2], item[3]] for item in records]
    velocity = [[item[4], item[5], item[6]] for item in records]
    return Orbit(times, position, velocity)


def _replace_abstracted_orbit_vectors(root: ET.Element, records: tuple[tuple[str, ...], ...]) -> None:
    found = None
    for element in root.iter():
        if _local(element.tag) == "MDElem" and element.get("name") == "Orbit_State_Vectors":
            found = element
            break
    if found is None:
        raise ValueError("DIMAP has no Orbit_State_Vectors")
    child_tag = found.tag
    attr_tag = "MDATTR"
    for child in list(found):
        if _local(child.tag) == "MDATTR":
            attr_tag = child.tag
        found.remove(child)
    for index, record in enumerate(records, start=1):
        node = ET.Element(child_tag)
        node.set("name", f"orbit_vector{index}")
        when, x, y, z, vx, vy, vz = record
        for name, value, kind in (
            ("time", when, "utc"),
            ("x_pos", x, "float64"),
            ("y_pos", y, "float64"),
            ("z_pos", z, "float64"),
            ("x_vel", vx, "float64"),
            ("y_vel", vy, "float64"),
            ("z_vel", vz, "float64"),
        ):
            attr = ET.SubElement(node, attr_tag)
            attr.set("name", name)
            attr.set("type", kind)
            attr.set("mode", "rw")
            attr.text = value
        found.append(node)


def _dim_text(root: ET.Element, tag: str) -> str:
    for element in root.iter():
        if _local(element.tag) == tag and (element.text or "").strip():
            return (element.text or "").strip()
    return ""


def dimap_abstracted_time(path, name: str) -> str:
    """Return one abstracted-metadata UTC string from a BEAM-DIMAP header."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    marker = f'name="{name}"'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"{path} has no {name}")
    open_end = text.find(">", start)
    close = text.find("</MDATTR>", open_end)
    return text[open_end + 1 : close].strip()


def slc_split_source(
    product,
    measurement,
    *,
    row: int,
    height: int,
    width: int,
    col: int = 0,
    precise_orbit=None,
) -> SlcDimapSource:
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
    if precise_orbit is not None:
        _replace_annotation_orbits(root, precise_orbit, first_iso, last_iso)
    vectors = tuple(_orbit_record(element) for element in root.iter() if _local(element.tag) == "orbit")
    if not vectors:
        raise ValueError(f"{product.product_id} annotation has no orbit state vectors")
    frequency_hz = float(_text(root, "radarFrequency"))
    sampling_hz = float(_text(root, "rangeSamplingRate"))
    slant_time = float(_text(image, "slantRangeTime"))
    mission = "SENTINEL-1" + product.platform[-1]
    name = measurement.annotation_href.replace("\\", "/").split("/")[-1]
    tie_grids = _tie_grids(
        root,
        row=int(row),
        height=int(height),
        col=int(col),
        scene_samples=scene_samples,
        scene_lines=scene_lines,
    )
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
        tie_grids=tie_grids,
        extra_abstracted_xml=_extra_abstracted_xml(
            root,
            product,
            tie_grids=tie_grids,
            row=int(row),
            height=int(height),
            width=int(width),
            lines_per_burst=lines_per_burst,
            polarisation=measurement.polarisation,
        ),
        split_parameters=(
            measurement.swath,
            measurement.polarisation,
            int(row) // lines_per_burst + 1,
            (int(row) + int(height) - 1) // lines_per_burst + 1,
        ),
        image_number=int(float(_text(root, "imageNumber") or "0")),
    )


def _extra_abstracted_xml(
    root: ET.Element,
    product,
    *,
    tie_grids: tuple,
    row: int,
    height: int,
    width: int,
    lines_per_burst: int,
    polarisation: str,
) -> str:
    """Abstracted attributes SNAP writes and this header did not yet carry."""
    manifest = _manifest_fields(product)
    first_burst = int(row) // int(lines_per_burst) + 1
    last_burst = (int(row) + int(height) - 1) // int(lines_per_burst) + 1
    corners = _burst_corners(root, first_burst, last_burst, lines_per_burst)
    incidence = _incidence_from_grids(tie_grids, height, width)
    window = _processing_window(root)
    bandwidth = _bandwidth_mhz_hz(root)
    polars = _ordered_polarisations(product)
    mds = ["-", "-", "-", "-"]
    for index, value in enumerate(polars[:4]):
        mds[index] = value
    if polarisation and mds[0] not in ("-", polarisation):
        mds[0] = polarisation
    centre = _scene_centre(product, polarisation)
    lines = [
        _attr("BEAMS", "-"),
        _attr("PROC_TIME", manifest["proc_time"], kind="utc", unit="utc", desc="Processing time"),
        _attr("Processing_system_identifier", manifest["processor"]),
        _attr("orbit_cycle", manifest["cycle"], kind="int32", desc="Orbit cycle"),
        _attr("VECTOR_SOURCE", "-"),
        _attr("incidence_near", _num(incidence[0]), kind="float64", unit="deg"),
        _attr("incidence_far", _num(incidence[1]), kind="float64", unit="deg"),
        _attr("slice_num", window["slice"], kind="int32"),
        _attr("first_near_lat", _num(corners[0]), kind="float64", unit="deg"),
        _attr("first_near_long", _num(corners[1]), kind="float64", unit="deg"),
        _attr("first_far_lat", _num(corners[2]), kind="float64", unit="deg"),
        _attr("first_far_long", _num(corners[3]), kind="float64", unit="deg"),
        _attr("last_near_lat", _num(corners[4]), kind="float64", unit="deg"),
        _attr("last_near_long", _num(corners[5]), kind="float64", unit="deg"),
        _attr("last_far_lat", _num(corners[6]), kind="float64", unit="deg"),
        _attr("last_far_long", _num(corners[7]), kind="float64", unit="deg"),
        _attr("mds2_tx_rx_polar", mds[1]),
        _attr("mds3_tx_rx_polar", mds[2]),
        _attr("mds4_tx_rx_polar", mds[3]),
        _attr("polsar_data", "0", kind="uint8", unit="flag"),
        _attr("algorithm", "-"),
        _attr("range_window_type", window["range_window"]),
        _attr("range_window_coefficient", _num(window["range_coefficient"]), kind="float64"),
        _attr("total_size", str(_product_storage_bytes(product, height, width)), kind="uint32", unit="MB", desc="Total product size"),
        _attr("avg_scene_height", _num(_mean_terrain_height(product)), kind="float64", unit="m"),
        _attr("map_projection", "-"),
        _attr("is_terrain_corrected", "0", kind="uint8", unit="flag"),
        _attr("DEM", "-"),
        _attr("geo_ref_system", "-"),
        _attr("lat_pixel_res", "99999.0", kind="float64", unit="deg"),
        _attr("lon_pixel_res", "99999.0", kind="float64", unit="deg"),
        _attr("ant_elev_corr_flag", "0", kind="uint8", unit="flag"),
        _attr("range_spread_comp_flag", "0", kind="uint8", unit="flag"),
        _attr("replica_power_corr_flag", "0", kind="uint8", unit="flag"),
        _attr("abs_calibration_flag", "0", kind="uint8", unit="flag"),
        _attr("calibration_factor", "99999.0", kind="float64", unit="dB"),
        _attr("chirp_power", "99999.0", kind="float64"),
        _attr("inc_angle_comp_flag", "0", kind="uint8", unit="flag"),
        _attr("ref_inc_angle", "99999.0", kind="float64"),
        _attr("ref_slant_range", "99999.0", kind="float64"),
        _attr("ref_slant_range_exp", "99999.0", kind="float64"),
        _attr("rescaling_factor", "99999.0", kind="float64"),
        _attr("bistatic_correction_applied", "1", kind="uint8", unit="flag"),
        _attr("range_bandwidth", _num(bandwidth[0]), kind="float64", unit="MHz"),
        _attr("azimuth_bandwidth", _num(bandwidth[1]), kind="float64", unit="Hz"),
        _attr("multilook_flag", "0", kind="uint8", unit="flag"),
        _attr("external_calibration_file", "-"),
        _attr("orbit_state_vector_file", "-"),
        _attr("centre_lat", _num(centre[0]), kind="float64"),
        _attr("centre_lon", _num(centre[1]), kind="float64"),
        _attr("centre_heading", _num(centre[2]), kind="float64"),
        _attr("centre_heading2", _num(centre[3]), kind="float64"),
    ]
    return "        " + "\n        ".join(lines) + "\n" + _doppler_xml(root) + "        <MDElem name=\"SRGR_Coefficients\"/>\n"


def _attr(name: str, value: str, *, kind: str = "ascii", unit: str = "", desc: str = "") -> str:
    unit_text = f' unit="{escape(unit)}"' if unit else ' unit=""'
    desc_text = f' desc="{escape(desc)}"' if desc else ""
    return f'<MDATTR name="{name}"{desc_text}{unit_text} type="{kind}" mode="rw">{escape(str(value))}</MDATTR>'


def _manifest_fields(product) -> dict[str, str]:
    text = _manifest_text(product)
    cycle = _xml_text(text, "cycleNumber") or ""
    start = _xml_text(text, "processing") 
    # The first processing start attribute, which is the SLC post-processing clock.
    marker = 'name="SLC Post Processing"'
    proc = ""
    at = text.find("processing")
    import re
    match = re.search(r'<[^>]*processing[^>]*start="([^"]+)"', text)
    if match:
        proc = snap_utc(match.group(1).replace("Z", ""))
    software = re.search(r'<[^>]*software[^>]*name="([^"]*)"[^>]*version="([^"]*)"', text)
    processor = "ESA Sentinel-1 IPF"
    if software and software.group(1):
        processor = f"ESA {software.group(1)} {software.group(2)}".strip()
    return {"cycle": cycle, "proc_time": proc, "processor": processor}


def _manifest_text(product) -> str:
    root = Path(getattr(product, "root", ""))
    manifest = root / "manifest.safe"
    if manifest.is_file():
        return manifest.read_text(encoding="utf-8", errors="replace")
    return ""


def _xml_text(text: str, tag: str) -> str:
    import re
    match = re.search(rf"<(?:[^:>]+:)?{tag}[^>]*>([^<]+)", text)
    return match.group(1).strip() if match else ""


def _burst_corners(root: ET.Element, first_burst: int, last_burst: int, lines_per_burst: int) -> tuple[float, ...]:
    """Near and far geolocation at the first burst and at the line after the last."""
    by_line: dict[int, list[tuple[int, float, float]]] = {}
    for element in root.iter():
        if _local(element.tag) != "geolocationGridPoint":
            continue
        line = int(float(_text(element, "line")))
        by_line.setdefault(line, []).append(
            (int(float(_text(element, "pixel"))), float(_text(element, "latitude")), float(_text(element, "longitude")))
        )
    for line in by_line:
        by_line[line].sort()
    first_line = (int(first_burst) - 1) * int(lines_per_burst)
    last_line = int(last_burst) * int(lines_per_burst)
    if first_line not in by_line:
        first_line = min(by_line)
    if last_line not in by_line:
        last_line = max(line for line in by_line if line >= first_line)
    def ends(line: int) -> tuple[float, float, float, float]:
        row = by_line[line]
        return row[0][1], row[0][2], row[-1][1], row[-1][2]
    near_lat, near_lon, far_lat, far_lon = ends(first_line)
    last_near_lat, last_near_lon, last_far_lat, last_far_lon = ends(last_line)
    return (near_lat, near_lon, far_lat, far_lon, last_near_lat, last_near_lon, last_far_lat, last_far_lon)


def _incidence_from_grids(grids, height: int, width: int) -> tuple[float, float]:
    grid = next((item for item in grids if item[0] == "incident_angle"), None)
    if grid is None:
        return (0.0, 0.0)
    _name, _unit, values, offset_x, offset_y, step_x, step_y, _cyclic = grid
    mid = int(height) // 2
    return (
        _tie_pixel(values, offset_x, offset_y, step_x, step_y, 0, mid),
        _tie_pixel(values, offset_x, offset_y, step_x, step_y, int(width) - 1, mid),
    )


def _tie_pixel(values, offset_x, offset_y, step_x, step_y, x: int, y: int) -> float:
    rows, cols = values.shape
    fi = (float(x) + 0.5 - float(offset_x)) / float(step_x)
    fj = (float(y) + 0.5 - float(offset_y)) / float(step_y)
    i = min(max(int(np.floor(fi)), 0), cols - 2)
    j = min(max(int(np.floor(fj)), 0), rows - 2)
    wx = fi - i
    wy = fj - j
    v00 = float(values[j, i])
    v10 = float(values[j, i + 1])
    v01 = float(values[j + 1, i])
    v11 = float(values[j + 1, i + 1])
    return v00 + wx * (v10 - v00) + wy * (v01 - v00) + wx * wy * (v11 - v10 - v01 + v00)


def _processing_window(root: ET.Element) -> dict[str, object]:
    slice_num = "0"
    range_window = "Hamming"
    coefficient = 0.75
    for element in root.iter():
        if _local(element.tag) == "sliceNumber" and slice_num == "0":
            slice_num = (element.text or "0").strip()
        if _local(element.tag) == "rangeProcessing":
            for child in element.iter():
                if _local(child.tag) == "windowType" and (child.text or "").strip():
                    range_window = (child.text or "").strip()
                if _local(child.tag) == "windowCoefficient" and (child.text or "").strip():
                    coefficient = float(child.text)
                    break
    return {"slice": slice_num, "range_window": range_window, "range_coefficient": coefficient}


def _bandwidth_mhz_hz(root: ET.Element) -> tuple[float, float]:
    range_hz = 0.0
    azimuth_hz = 0.0
    for element in root.iter():
        if _local(element.tag) == "rangeProcessing":
            for child in element.iter():
                if _local(child.tag) == "processingBandwidth" and (child.text or "").strip():
                    range_hz = float(child.text)
                    break
        if _local(element.tag) == "azimuthProcessing":
            for child in element.iter():
                if _local(child.tag) == "processingBandwidth" and (child.text or "").strip():
                    azimuth_hz = float(child.text)
                    break
    return range_hz / 1e6, azimuth_hz


def _ordered_polarisations(product) -> list[str]:
    root = Path(getattr(product, "root", ""))
    folder = root / "annotation"
    found: list[str] = []
    if not folder.is_dir():
        return list(getattr(product, "polarisations", ()) or [])
    for path in sorted(folder.glob("s1*-iw*.xml")):
        text = path.name.lower()
        pol = ""
        for candidate in ("vh", "vv", "hh", "hv"):
            if f"-{candidate}-" in text:
                pol = candidate.upper()
                break
        if pol and pol not in found:
            found.append(pol)
    return found or list(getattr(product, "polarisations", ()) or [])


def _mean_terrain_height(product) -> float:
    root = Path(getattr(product, "root", ""))
    folder = root / "annotation"
    values: list[float] = []
    paths = sorted(folder.glob("s1*-iw*.xml")) if folder.is_dir() else []
    for path in paths:
        tree = ET.parse(path)
        for element in tree.getroot().iter():
            if _local(element.tag) == "terrainHeight":
                for child in list(element):
                    if _local(child.tag) == "value" and (child.text or "").strip():
                        values.append(float(child.text))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _product_storage_bytes(product, height: int, width: int) -> int:
    """Byte count SNAP stores in ``total_size``.

    The unit label is MB. The number is the subset product's raw storage:
    separate int16 i and q rasters, a 256-byte node header on each band, the
    five tie-point grids, and the SAFE metadata files SNAP keeps on the product.
    """
    bands = 2 * (256 + 2 * int(height) * int(width))
    ties = 5 * (256 + 4 * 4 * 21)
    root = Path(getattr(product, "root", ""))
    meta = 0
    if root.is_dir():
        meta = sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".xml", ".safe", ".png"})
    return int(bands + ties + meta)


def _scene_centre(product, polarisation: str) -> tuple[float, float, float, float]:
    """Latitude, longitude, and the two headings at the scene centre.

    The centre is the mean of the near and far geolocation corners of the
    first and last subswath. Heading is the azimuth from the centre toward
    the next burst along the near-range side, and the second heading is that
    azimuth turned through 180 degrees.
    """
    root = Path(getattr(product, "root", ""))
    folder = root / "annotation"
    paths = sorted(folder.glob("s1*-iw*.xml")) if folder.is_dir() else []
    pol = polarisation.lower()
    chosen = [path for path in paths if f"-{pol}-" in path.name.lower()] or paths
    if not chosen:
        return (0.0, 0.0, 0.0, 0.0)
    first = _corner_pair(chosen[0])
    last = _corner_pair(chosen[-1] if len(chosen) > 2 else chosen[min(1, len(chosen) - 1)])
    # first/last swath in range order, not file order. IW1 is the near swath.
    ordered = sorted(chosen, key=lambda path: path.name)
    iw = [path for path in ordered if "-iw1-" in path.name or "-iw2-" in path.name or "-iw3-" in path.name]
    if iw:
        first = _corner_pair(iw[0])
        last = _corner_pair(iw[-1])
    lats = (first[0], last[2], first[4], last[6])
    lons = (first[1], last[3], first[5], last[7])
    lat = float(sum(lats) / 4.0)
    lon = float(sum(lons) / 4.0)
    heading = _centre_heading(iw or ordered)
    other = heading - 180.0 if heading >= 180.0 else heading + 180.0
    return lat, lon, heading, other


def _centre_heading(paths: list[Path]) -> float:
    """Azimuth from the scene-centre pixel to the pixel 100 lines later.

    Scene width is the sum of one polarisation's subswaths and the centre
    falls in the middle subswath. That is the direction SNAP stores.
    """
    loaded = [_swath_grid(path) for path in paths]
    if not loaded:
        return 0.0
    scene_w = sum(item[0] for item in loaded)
    scene_h = max(item[1] for item in loaded)
    x = scene_w // 2
    y = scene_h // 2
    origin = 0
    for width, _height, lat, lon, step_x, step_y in loaded:
        if origin <= x < origin + width:
            local_x = x - origin
            return _azimuth_deg(
                _sample_grid(lat, local_x, y, step_x, step_y),
                _sample_grid(lon, local_x, y, step_x, step_y),
                _sample_grid(lat, local_x, y + 100, step_x, step_y),
                _sample_grid(lon, local_x, y + 100, step_x, step_y),
            )
        origin += width
    return 0.0


def _swath_grid(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    width = int(float(_text(root, "numberOfSamples")))
    height = int(float(_text(root, "numberOfLines")))
    by_line: dict[int, list[tuple[int, float, float]]] = {}
    for element in root.iter():
        if _local(element.tag) != "geolocationGridPoint":
            continue
        line = int(float(_text(element, "line")))
        by_line.setdefault(line, []).append(
            (int(float(_text(element, "pixel"))), float(_text(element, "latitude")), float(_text(element, "longitude")))
        )
    lines = sorted(by_line)
    for line in lines:
        by_line[line].sort()
    ncols = len(by_line[lines[0]])
    lat = np.empty((len(lines), ncols))
    lon = np.empty_like(lat)
    pixels = np.array([item[0] for item in by_line[lines[0]]], dtype=np.float64)
    for j, line in enumerate(lines):
        for i, (_pix, latitude, longitude) in enumerate(by_line[line]):
            lat[j, i] = latitude
            lon[j, i] = longitude
    step_x = width / (ncols - 1)
    step_y = height / (len(lines) - 1)
    even_lat = _even_range_grid(lat, pixels, step_x, width).astype(np.float64)
    even_lon = _even_range_grid(lon, pixels, step_x, width).astype(np.float64)
    return width, height, even_lat, even_lon, step_x, step_y


def _sample_grid(grid, x, y, step_x, step_y) -> float:
    rows, cols = grid.shape
    fi = (float(x) - 0.5) / float(step_x)
    fj = (float(y) - 0.5) / float(step_y)
    i = min(max(int(np.floor(fi)), 0), cols - 2)
    j = min(max(int(np.floor(fj)), 0), rows - 2)
    wx = fi - i
    wy = fj - j
    return float(
        grid[j, i]
        + wx * (grid[j, i + 1] - grid[j, i])
        + wy * (grid[j + 1, i] - grid[j, i])
        + wx * wy * (grid[j + 1, i + 1] - grid[j, i + 1] - grid[j + 1, i] + grid[j, i])
    )


def _corner_pair(path: Path) -> tuple[float, ...]:
    tree = ET.parse(path)
    by_line: dict[int, list[tuple[int, float, float]]] = {}
    for element in tree.getroot().iter():
        if _local(element.tag) != "geolocationGridPoint":
            continue
        line = int(float(_text(element, "line")))
        by_line.setdefault(line, []).append(
            (int(float(_text(element, "pixel"))), float(_text(element, "latitude")), float(_text(element, "longitude")))
        )
    lines = sorted(by_line)
    for line in lines:
        by_line[line].sort()
    def ends(line: int) -> tuple[float, float, float, float]:
        row = by_line[line]
        return row[0][1], row[0][2], row[-1][1], row[-1][2]
    a = ends(lines[0])
    b = ends(lines[-1])
    return (*a, *b)


def _azimuth_deg(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    lat0_r = np.deg2rad(lat0)
    lat1_r = np.deg2rad(lat1)
    dlon = np.deg2rad(lon1 - lon0)
    y = np.sin(dlon) * np.cos(lat1_r)
    x = np.cos(lat0_r) * np.sin(lat1_r) - np.sin(lat0_r) * np.cos(lat1_r) * np.cos(dlon)
    heading = float(np.rad2deg(np.arctan2(y, x)))
    if heading < 0.0:
        heading += 360.0
    return heading


def _doppler_xml(root: ET.Element) -> str:
    estimates = [element for element in root.iter() if _local(element.tag) == "dcEstimate"]
    if not estimates:
        return ""
    blocks = []
    for index, element in enumerate(estimates, start=1):
        when = snap_utc(_text(element, "azimuthTime"))
        slant = float(_text(element, "t0")) * 1e9
        poly = ""
        for child in element.iter():
            if _local(child.tag) == "geometryDcPolynomial" and (child.text or "").strip():
                poly = (child.text or "").strip()
                break
        coefficients = [float(item) for item in poly.split()]
        coef_xml = []
        for coef_index, value in enumerate(coefficients, start=1):
            coef_xml.append(
                "            <MDElem name=\"coefficient.%d\">\n"
                "              <MDATTR name=\"dop_coef\" desc=\"Doppler Centroid Coefficient\" unit=\"\" type=\"float64\" mode=\"rw\">%s</MDATTR>\n"
                "            </MDElem>" % (coef_index, _num(value))
            )
        blocks.append(
            "          <MDElem name=\"dop_coef_list.%d\">\n"
            "            <MDATTR name=\"zero_doppler_time\" type=\"utc\" mode=\"rw\">%s</MDATTR>\n"
            "            <MDATTR name=\"slant_range_time\" desc=\"Slant Range Time\" unit=\"ns\" type=\"float64\" mode=\"rw\">%s</MDATTR>\n"
            "%s\n"
            "          </MDElem>" % (index, escape(when), _num(slant), "\n".join(coef_xml))
        )
    return (
        "        <MDElem name=\"Doppler_Centroid_Coefficients\">\n"
        + "\n".join(blocks)
        + "\n        </MDElem>\n"
    )


def _tie_grids(root: ET.Element, *, row: int, height: int, col: int, scene_samples: int, scene_lines: int) -> tuple[object, ...]:
    """Tie points on the even pixel grid the Sentinel-1 reader builds.

    Annotation geolocation samples are bilinearly placed onto
    ``x = i * width / (ncols - 1)`` and ``y = j * height / (nrows - 1)``,
    then the rows that cover this burst window are kept. The step and the
    half-pixel offset stay those of the full swath.
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
    ncols = len(by_line[ordered[0]])
    if ncols < 2 or len(ordered) < 2 or any(len(by_line[line]) != ncols for line in ordered):
        return ()
    step_x = float(scene_samples) / float(ncols - 1)
    step_y = float(scene_lines) / float(len(ordered) - 1)
    pixels = np.array([int(float(_text(element, "pixel"))) for element in by_line[ordered[0]]], dtype=np.float64)
    fields = (
        ("latitude", "latitude", "deg", False, 1.0),
        ("longitude", "longitude", "deg", True, 1.0),
        ("incident_angle", "incidenceAngle", "deg", False, 1.0),
        ("elevation_angle", "elevationAngle", "deg", False, 1.0),
        ("slant_range_time", "slantRangeTime", "ns", False, 1e9),
    )
    stop = int(row) + int(height)
    chosen = [index for index, line in enumerate(ordered) if int(row) <= line <= stop]
    if not chosen:
        return ()
    if chosen[0] > 0:
        chosen = [chosen[0] - 1, *chosen]
    offset_y = float(ordered[chosen[0]] - int(row)) + 0.5
    offset_x = 0.5 - float(col)
    grids = []
    for name, tag, unit, cyclic, scale in fields:
        source = np.empty((len(ordered), ncols), dtype=np.float64)
        for i, line in enumerate(ordered):
            for j, element in enumerate(by_line[line]):
                source[i, j] = float(_text(element, tag)) * scale
        if cyclic and np.any(np.abs(np.diff(source, axis=1)) > 180.0):
            source = np.where(source < 0.0, source + 360.0, source)
        even = _even_range_grid(source, pixels, step_x, float(scene_samples))
        grids.append((name, unit, even[chosen], offset_x, offset_y, step_x, step_y, cyclic))
    return tuple(grids)


def _even_range_grid(source: np.ndarray, pixels: np.ndarray, step_x: float, scene_samples: float) -> np.ndarray:
    """Bilinear in range onto ``i * step_x``, then cast to float32 once."""
    out = np.empty_like(source)
    last_x = float(scene_samples) - 1.0
    for j in range(source.shape[0]):
        for i in range(source.shape[1]):
            x = min(i * step_x, last_x)
            bracket = 1
            while bracket < pixels.shape[0] - 1 and pixels[bracket] <= x:
                bracket += 1
            if pixels[bracket] <= x:
                bracket = pixels.shape[0] - 1
            left = bracket - 1
            span = float(pixels[bracket] - pixels[left])
            weight = 0.0 if span == 0.0 else (x - float(pixels[left])) / span
            out[j, i] = source[j, left] + weight * (source[j, bracket] - source[j, left])
    return out.astype(np.float32)


def _iso_from_datetime64(when) -> str:
    text = str(np.datetime_as_string(np.datetime64(when, "us")))
    if text.endswith("Z"):
        text = text[:-1]
    if "." not in text:
        text += ".000000"
    return text


def _replace_annotation_orbits(root: ET.Element, orbit, first_iso: str, last_iso: str) -> None:
    """Replace annotation ``<orbit>`` records with the precise-orbit file."""
    parent = None
    for element in root.iter():
        if any(_local(child.tag) == "orbit" for child in list(element)):
            parent = element
            break
    if parent is None:
        raise ValueError("annotation has no orbit list to replace")
    for child in list(parent):
        if _local(child.tag) == "orbit":
            parent.remove(child)
    start = _parse_iso(first_iso)
    stop = _parse_iso(last_iso)
    begin = np.datetime64(start.strftime("%Y-%m-%dT%H:%M:%S.") + f"{start.microsecond:06d}") - np.timedelta64(120, "s")
    end = np.datetime64(stop.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stop.microsecond:06d}") + np.timedelta64(120, "s")
    written = 0
    for index, when in enumerate(orbit.time):
        if when < begin or when > end:
            continue
        position = orbit.position_m[index]
        velocity = orbit.velocity_m_s[index]
        node = ET.Element("orbit")
        ET.SubElement(node, "time").text = _iso_from_datetime64(when)
        ET.SubElement(node, "frame").text = "EARTH_FIXED"
        pos = ET.SubElement(node, "position")
        vel = ET.SubElement(node, "velocity")
        for axis, value in zip(("x", "y", "z"), position):
            ET.SubElement(pos, axis).text = f"{float(value):.6f}"
        for axis, value in zip(("x", "y", "z"), velocity):
            ET.SubElement(vel, axis).text = f"{float(value):.8f}"
        parent.append(node)
        written += 1
    if "count" in parent.attrib:
        parent.attrib["count"] = str(written)
    if written < 4:
        raise ValueError("the precise orbit does not cover this burst window")


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
    position = tuple(_num(float(item)) for item in _triplet(element, "position"))
    velocity = tuple(_num(float(item)) for item in _triplet(element, "velocity"))
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
    text = format(float(value), ".16g")
    if "e" not in text.lower() and "." not in text:
        return text + ".0"
    return text

