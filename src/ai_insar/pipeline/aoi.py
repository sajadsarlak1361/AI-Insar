"""Processing area (AOI) shared by TOPS Split, Subset, and terrain correction.

A region is a WKT polygon or a north/south/west/east box, the same two
ways SNAP asks for one. The polygon is clipped to the reference/secondary
overlap. Pixels outside that clip are not a processing area. Burst
selection is the contiguous burst range whose footprints meet the clip,
which is what TOPS Split does with ``wktAoi``.

The validation square is 200 m on a side. It replaces the earlier 1 km
square and keeps that square's centre. The centre sits on the burst
boundary between product bursts 5 and 6 of IW1 (Copernicus burst ids
359478 and 359479) for the 2021-06-30 / 2021-07-12 pair, relative orbit 168.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from pathlib import Path

import numpy as np

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Fifteen names in the order the Sentinel-1 chain runs. Every stage receives
# the same clipped AOI; none of them may grow back outside it.
CHAIN_STAGES: tuple[str, ...] = (
    "TOPS Split",
    "Apply Orbit File",
    "Back Geocoding",
    "Enhanced Spectral Diversity",
    "Interferogram Formation",
    "TOPS Deburst",
    "Multilook",
    "Goldstein Phase Filtering",
    "Snaphu Export",
    "Snaphu Unwrap",
    "Snaphu Import",
    "Phase to Displacement",
    "Range-Doppler Terrain Correction",
    "Subset",
    "Export map",
)

# The previous 100 m square, kept so an AOI entered as that polygon still
# maps to the radar window used by docs/parity/square100m_9e0457d.
SQUARE_100M_WKT = (
    "POLYGON ((13.326887869666 50.376138747447, "
    "13.325508840648 50.376313302579, "
    "13.325235898826 50.375431422913, "
    "13.326614903281 50.375256870991, "
    "13.326887869666 50.376138747447))"
)
SQUARE_100M_REFERENCE_WINDOW = {"row0": 260, "row1": 269, "column0": 2469, "column1": 2492}

# Radar placement of the 200 m square. The 100 m square centre is 1,098
# lines from the burst-overlap zone (measured on the real split). The next
# burst starts 2.756501 s later and each burst has 1,508 lines, so the
# overlap is 166 lines. The boundary used by deburst is the middle of that
# overlap. Ground spacing is the value measured on the same square.
_OLD_LAT = 50.37578508812358
_OLD_LON = 13.326061878105238
_LINES_TO_OVERLAP = 1098.0
_BURST_GAP_S = 2.756501
_AZIMUTH_INTERVAL_S = 0.002055556299999998
_LINES_PER_BURST = 1508
_GROUND_M_PER_LINE = 12.400850123770615
_SQUARE_EDGE_M = 200.0
# Corners of the 100 m square, (lat, lon), low-row/low-col then low-row/high-col
# then high-row/low-col. They fix the range and azimuth axes.
_CORNER_AZ_START = (50.37613874744699, 13.326887869665738)
_CORNER_RANGE = (50.37631330257908, 13.325508840648142)
_CORNER_AZ_END = (50.37525687099057, 13.32661490328058)


class AoiError(ValueError):
    """The region cannot be parsed or lies outside the overlap."""


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PAIR = re.compile(rf"({_NUMBER})\s+({_NUMBER})")


@dataclass(frozen=True)
class LonLatPolygon:
    """Outer ring in longitude, latitude order. It need not repeat the first vertex."""

    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.ring) < 3:
            raise AoiError("A region needs at least three corners.")

    @property
    def ring(self) -> tuple[tuple[float, float], ...]:
        vertices = tuple((float(lon), float(lat)) for lon, lat in self.vertices)
        if len(vertices) >= 2 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        return vertices

    def closed(self) -> tuple[tuple[float, float], ...]:
        ring = self.ring
        return ring + (ring[0],)


@dataclass(frozen=True)
class GeographicBounds:
    north: float
    south: float
    west: float
    east: float


@dataclass(frozen=True)
class PixelWindow:
    """SNAP ``region`` plus subsampling. Ends are exclusive, so they match NumPy slices."""

    x: int
    y: int
    width: int
    height: int
    sub_sampling_x: int = 1
    sub_sampling_y: int = 1

    @property
    def row_start(self) -> int:
        return int(self.y)

    @property
    def row_end(self) -> int:
        return int(self.y) + int(self.height)

    @property
    def column_start(self) -> int:
        return int(self.x)

    @property
    def column_end(self) -> int:
        return int(self.x) + int(self.width)

    def as_snap_region(self) -> dict[str, int]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass
class AoiReport:
    """The clipped processing area and the bursts that cover it."""

    requested_wkt: str
    clipped_wkt: str
    area_km2: float
    outside: bool
    message: str
    subswath: str = "IW1"
    polarisation: str = "VV"
    burst_first: int | None = None
    burst_last: int | None = None
    burst_ids: tuple[int, ...] = ()
    centre_latitude: float | None = None
    centre_longitude: float | None = None
    north: float | None = None
    south: float | None = None
    west: float | None = None
    east: float | None = None
    pixel_region: dict | None = None
    esd_overlap_bursts: tuple[int, int] | None = None
    copy_metadata: bool = True
    source_bands: tuple[str, ...] = ()
    sub_sampling_x: int = 1
    sub_sampling_y: int = 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.esd_overlap_bursts is None:
            payload["esd_overlap_bursts"] = None
        else:
            payload["esd_overlap_bursts"] = [int(self.esd_overlap_bursts[0]), int(self.esd_overlap_bursts[1])]
        payload["burst_ids"] = [int(item) for item in self.burst_ids]
        payload["source_bands"] = list(self.source_bands)
        return payload


@dataclass
class AoiConfig:
    """Saved AOI. Empty WKT means the chain has no geographic subset."""

    wkt: str = ""
    subswath: str = "IW1"
    polarisation: str = "VV"
    burst_first: int | None = None
    burst_last: int | None = None
    burst_ids: tuple[int, ...] = ()
    north: float | None = None
    south: float | None = None
    west: float | None = None
    east: float | None = None
    region: dict | None = None
    sub_sampling_x: int = 1
    sub_sampling_y: int = 1
    source_bands: tuple[str, ...] = field(default_factory=tuple)
    copy_metadata: bool = True
    area_km2: float | None = None
    centre_latitude: float | None = None
    centre_longitude: float | None = None
    esd_overlap_bursts: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "wkt": self.wkt,
            "geoRegion": self.wkt,
            "subswath": self.subswath,
            "polarisation": self.polarisation,
            "burst_first": self.burst_first,
            "burst_last": self.burst_last,
            "burst_ids": list(self.burst_ids),
            "north": self.north,
            "south": self.south,
            "west": self.west,
            "east": self.east,
            "region": self.region,
            "subSamplingX": int(self.sub_sampling_x),
            "subSamplingY": int(self.sub_sampling_y),
            "sourceBands": list(self.source_bands),
            "copyMetadata": bool(self.copy_metadata),
            "area_km2": self.area_km2,
            "centre_latitude": self.centre_latitude,
            "centre_longitude": self.centre_longitude,
            "esd_overlap_bursts": None
            if self.esd_overlap_bursts is None
            else [int(self.esd_overlap_bursts[0]), int(self.esd_overlap_bursts[1])],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> AoiConfig:
        wkt = str(payload.get("wkt") or payload.get("geoRegion") or payload.get("wktAoi") or "").strip()
        bursts = payload.get("esd_overlap_bursts")
        pair = None
        if isinstance(bursts, (list, tuple)) and len(bursts) == 2:
            pair = (int(bursts[0]), int(bursts[1]))
        ids = tuple(int(item) for item in payload.get("burst_ids") or ())
        bands = tuple(str(item) for item in payload.get("sourceBands") or payload.get("source_bands") or ())
        return cls(
            wkt=wkt,
            subswath=str(payload.get("subswath") or "IW1"),
            polarisation=str(payload.get("polarisation") or "VV"),
            burst_first=None if payload.get("burst_first") in (None, "") else int(payload["burst_first"]),
            burst_last=None if payload.get("burst_last") in (None, "") else int(payload["burst_last"]),
            burst_ids=ids,
            north=_optional_float(payload.get("north")),
            south=_optional_float(payload.get("south")),
            west=_optional_float(payload.get("west")),
            east=_optional_float(payload.get("east")),
            region=payload.get("region"),
            sub_sampling_x=int(payload.get("subSamplingX") or payload.get("sub_sampling_x") or 1),
            sub_sampling_y=int(payload.get("subSamplingY") or payload.get("sub_sampling_y") or 1),
            source_bands=bands,
            copy_metadata=bool(payload.get("copyMetadata", payload.get("copy_metadata", True))),
            area_km2=_optional_float(payload.get("area_km2")),
            centre_latitude=_optional_float(payload.get("centre_latitude")),
            centre_longitude=_optional_float(payload.get("centre_longitude")),
            esd_overlap_bursts=pair,
        )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _metres_per_degree(latitude_deg: float) -> tuple[float, float]:
    lat = math.radians(float(latitude_deg))
    sine = math.sin(lat)
    normal = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sine * sine)
    meridian = WGS84_A_M * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sine * sine) ** 1.5
    return normal * math.cos(lat), meridian


def local_xy(lon: float, lat: float, *, lon0: float, lat0: float) -> tuple[float, float]:
    """Local easting and northing in metres. Accurate over a few tens of kilometres."""
    east_scale, north_scale = _metres_per_degree(lat0)
    return math.radians(lon - lon0) * east_scale, math.radians(lat - lat0) * north_scale


def local_lonlat(x_m: float, y_m: float, *, lon0: float, lat0: float) -> tuple[float, float]:
    east_scale, north_scale = _metres_per_degree(lat0)
    return lon0 + math.degrees(x_m / east_scale), lat0 + math.degrees(y_m / north_scale)


def parse_wkt(text: str) -> LonLatPolygon:
    """Parse a SNAP geographic region.

    ``POLYGON ((lon lat, ...))`` is the form Subset and TOPS Split store.
    A ``MULTIPOLYGON`` uses its first outer ring.
    """
    raw = " ".join(str(text).strip().split())
    if not raw:
        raise AoiError("The geographic region is empty.")
    upper = raw.upper()
    if upper.startswith("MULTIPOLYGON"):
        raw = raw[raw.upper().find("((") :]
    elif upper.startswith("POLYGON"):
        raw = raw[len("POLYGON") :].strip()
    else:
        raise AoiError("The geographic region must be a POLYGON WKT, for example POLYGON ((lon lat, ...)).")
    if "((" not in raw:
        raise AoiError("POLYGON WKT is missing its coordinate ring.")
    body = raw[raw.find("((") + 2 : raw.find("))")]
    # An inner ring follows the outer ring. SNAP's AOI is the outer ring.
    body = body.split("),")[0]
    pairs = [(float(lon), float(lat)) for lon, lat in _PAIR.findall(body)]
    if len(pairs) < 3:
        raise AoiError("POLYGON WKT does not contain a closed ring.")
    for lon, lat in pairs:
        if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
            raise AoiError(f"Coordinate {lon}, {lat} is outside longitude/latitude range.")
    return LonLatPolygon(tuple(pairs))


def format_wkt(polygon: LonLatPolygon, *, decimals: int = 12) -> str:
    places = max(int(decimals), 0)
    parts = [f"{lon:.{places}f} {lat:.{places}f}" for lon, lat in polygon.closed()]
    return "POLYGON ((" + ", ".join(parts) + "))"


def polygon_from_bounds(north: float, south: float, west: float, east: float) -> LonLatPolygon:
    """Axis-aligned geographic box, the Geo Coordinates fields of Subset."""
    north_f, south_f = float(north), float(south)
    west_f, east_f = float(west), float(east)
    if not south_f < north_f:
        raise AoiError("South latitude must be less than north latitude.")
    if not west_f < east_f:
        raise AoiError("West longitude must be less than east longitude.")
    if not -90.0 <= south_f <= 90.0 or not -90.0 <= north_f <= 90.0:
        raise AoiError("Latitude is outside -90 to 90 degrees.")
    if not -180.0 <= west_f <= 180.0 or not -180.0 <= east_f <= 180.0:
        raise AoiError("Longitude is outside -180 to 180 degrees.")
    return LonLatPolygon(
        (
            (west_f, south_f),
            (east_f, south_f),
            (east_f, north_f),
            (west_f, north_f),
        )
    )


def geographic_bounds(polygon: LonLatPolygon) -> GeographicBounds:
    ring = polygon.ring
    lats = [lat for _lon, lat in ring]
    lons = [lon for lon, _lat in ring]
    return GeographicBounds(north=max(lats), south=min(lats), west=min(lons), east=max(lons))


def polygon_area_km2(polygon: LonLatPolygon) -> float:
    """Geodesic-approx area of a small ring, in square kilometres."""
    ring = polygon.ring
    lat0 = sum(lat for _lon, lat in ring) / len(ring)
    lon0 = sum(lon for lon, _lat in ring) / len(ring)
    points = [local_xy(lon, lat, lon0=lon0, lat0=lat0) for lon, lat in ring]
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5 / 1.0e6


def _signed_area(ring: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(index + 1) % len(ring)]
        area += x1 * y2 - x2 * y1
    return area * 0.5


def _ccw(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return ring if _signed_area(ring) >= 0.0 else list(reversed(ring))


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _left(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> bool:
    return _cross(end[0] - start[0], end[1] - start[1], point[0] - start[0], point[1] - start[1]) >= -1.0e-12


def _segment_intersection(
    start: tuple[float, float],
    end: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> tuple[float, float]:
    direction = (end[0] - start[0], end[1] - start[1])
    edge = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
    denominator = _cross(direction[0], direction[1], edge[0], edge[1])
    if abs(denominator) < 1.0e-18:
        return ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
    factor = _cross(edge_start[0] - start[0], edge_start[1] - start[1], edge[0], edge[1]) / denominator
    return (start[0] + factor * direction[0], start[1] + factor * direction[1])


def clip_polygon(subject: LonLatPolygon, clip: LonLatPolygon) -> LonLatPolygon | None:
    """Sutherland–Hodgman clip. The clip ring is treated as convex."""
    output = _ccw(list(subject.ring))
    clip_ring = _ccw(list(clip.ring))
    for index, edge_start in enumerate(clip_ring):
        edge_end = clip_ring[(index + 1) % len(clip_ring)]
        incoming = output
        output = []
        if not incoming:
            return None
        previous = incoming[-1]
        for current in incoming:
            current_inside = _left(current, edge_start, edge_end)
            previous_inside = _left(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_segment_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_segment_intersection(previous, current, edge_start, edge_end))
            previous = current
    if len(output) < 3:
        return None
    return LonLatPolygon(tuple(output))


def point_in_polygon(lon: float, lat: float, polygon: LonLatPolygon) -> bool:
    ring = polygon.ring
    inside = False
    previous = len(ring) - 1
    for index, (xi, yi) in enumerate(ring):
        xj, yj = ring[previous]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            edge = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < edge:
                inside = not inside
        previous = index
    return inside


def polygons_intersect(left: LonLatPolygon, right: LonLatPolygon) -> bool:
    clipped = clip_polygon(left, right)
    if clipped is not None:
        return True
    return any(point_in_polygon(lon, lat, right) for lon, lat in left.ring) or any(
        point_in_polygon(lon, lat, left) for lon, lat in right.ring
    )


def points_in_polygon(lon: np.ndarray, lat: np.ndarray, polygon: LonLatPolygon) -> np.ndarray:
    """Ray-cast mask. ``lon`` and ``lat`` share a shape. The border counts as inside."""
    x = np.asarray(lon, dtype=np.float64)
    y = np.asarray(lat, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    ring = polygon.ring
    previous = len(ring) - 1
    for index, (xi, yi) in enumerate(ring):
        xj, yj = ring[previous]
        denominator = yj - yi
        safe = denominator if denominator != 0.0 else 1.0
        crosses = (yi > y) != (yj > y)
        edge = (xj - xi) * (y - yi) / safe + xi
        inside ^= crosses & (x < edge)
        previous = index
    return inside


def footprint_polygon(lat_lon: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> LonLatPolygon | None:
    """A SAFE footprint is stored as latitude, longitude. Return longitude, latitude."""
    if len(lat_lon) < 3:
        return None
    return LonLatPolygon(tuple((float(lon), float(lat)) for lat, lon in lat_lon))


def azimuth_burst_footprints(scene: LonLatPolygon, count: int) -> tuple[LonLatPolygon, ...]:
    """Split a scene quadrilateral into ``count`` along-track burst footprints.

    The ring is divided into two opposite sides. Each burst is the quad
    between matching fractions of those sides. This is the estimate used
    when a product has a frame footprint but no per-burst polygon yet.
    """
    count = int(count)
    if count < 1:
        raise AoiError("Burst count must be positive.")
    ring = list(scene.ring)
    if len(ring) < 4:
        raise AoiError("A burst estimate needs a scene footprint with at least four corners.")
    half = len(ring) // 2
    side_a = ring[: half + 1]
    side_b = list(reversed(ring[half:])) + [ring[0]]

    def _resample(side: list[tuple[float, float]], samples: int) -> list[tuple[float, float]]:
        if len(side) == 1:
            return [side[0]] * samples
        lengths = [0.0]
        for start, end in zip(side, side[1:]):
            lengths.append(lengths[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
        total = lengths[-1] or 1.0
        points = []
        for index in range(samples):
            distance = total * index / (samples - 1)
            segment = max(i for i, value in enumerate(lengths) if value <= distance + 1.0e-15)
            if segment >= len(side) - 1:
                points.append(side[-1])
                continue
            span = lengths[segment + 1] - lengths[segment]
            fraction = 0.0 if span == 0.0 else (distance - lengths[segment]) / span
            lon = side[segment][0] + fraction * (side[segment + 1][0] - side[segment][0])
            lat = side[segment][1] + fraction * (side[segment + 1][1] - side[segment][1])
            points.append((lon, lat))
        return points

    left = _resample(side_a, count + 1)
    right = _resample(side_b, count + 1)
    bursts = []
    for index in range(count):
        bursts.append(
            LonLatPolygon(
                (
                    left[index],
                    left[index + 1],
                    right[index + 1],
                    right[index],
                )
            )
        )
    return tuple(bursts)


def bursts_covering(footprints: list[LonLatPolygon] | tuple[LonLatPolygon, ...], area: LonLatPolygon) -> tuple[int, int]:
    """1-based inclusive burst indexes whose footprints meet ``area``.

    SNAP writes one contiguous range, from the first hit to the last hit,
    even when a middle burst is only touched by the overlap.
    """
    hits = [index for index, footprint in enumerate(footprints, start=1) if polygons_intersect(footprint, area)]
    if not hits:
        raise AoiError("The region does not cover any burst of this subswath.")
    return hits[0], hits[-1]


def overlap_polygon(reference: LonLatPolygon, secondary: LonLatPolygon) -> LonLatPolygon | None:
    return clip_polygon(reference, secondary)


def prepare_aoi(
    region: str | LonLatPolygon,
    *,
    reference: LonLatPolygon | None = None,
    secondary: LonLatPolygon | None = None,
    burst_footprints: list[LonLatPolygon] | tuple[LonLatPolygon, ...] | None = None,
    subswath: str = "IW1",
    polarisation: str = "VV",
    burst_ids: tuple[int, ...] = (),
    first_burst_number: int = 1,
) -> AoiReport:
    """Clip a region to the common overlap and name the bursts it uses.

    With one footprint, the clip is that footprint and the message says the
    second date is not loaded yet. With both, an empty clip is refused.
    """
    polygon = region if isinstance(region, LonLatPolygon) else parse_wkt(region)
    requested = format_wkt(polygon)
    clip_target = None
    if reference is not None and secondary is not None:
        clip_target = overlap_polygon(reference, secondary)
        if clip_target is None:
            return AoiReport(
                requested_wkt=requested,
                clipped_wkt="",
                area_km2=0.0,
                outside=True,
                message="The reference and secondary footprints do not overlap, so the AOI was refused.",
                subswath=subswath,
                polarisation=polarisation,
            )
    elif reference is not None:
        clip_target = reference
    elif secondary is not None:
        clip_target = secondary
    clipped = polygon if clip_target is None else clip_polygon(polygon, clip_target)
    if clipped is None:
        return AoiReport(
            requested_wkt=requested,
            clipped_wkt="",
            area_km2=0.0,
            outside=True,
            message=(
                "The AOI is fully outside the reference/secondary overlap. "
                "Nothing is computed outside that overlap."
            ),
            subswath=subswath,
            polarisation=polarisation,
        )
    area = polygon_area_km2(clipped)
    bounds = geographic_bounds(clipped)
    centre_lon = sum(lon for lon, _lat in clipped.ring) / len(clipped.ring)
    centre_lat = sum(lat for _lon, lat in clipped.ring) / len(clipped.ring)
    burst_first = burst_last = None
    esd_pair = None
    if burst_footprints:
        burst_first, burst_last = bursts_covering(burst_footprints, clipped)
        burst_first += int(first_burst_number) - 1
        burst_last += int(first_burst_number) - 1
        if burst_last == burst_first + 1:
            esd_pair = (int(burst_first), int(burst_last))
    waiting = reference is None or secondary is None
    if waiting:
        message = (
            f"AOI clipped to the loaded footprint, area {area:.6f} km². "
            "The second date is not loaded, so the common-overlap clip is still open."
        )
    else:
        burst_text = "no burst footprints" if burst_first is None else f"bursts {burst_first}–{burst_last}"
        message = (
            f"AOI clipped to the common overlap, area {area:.6f} km², {burst_text}, subswath {subswath}."
        )
    return AoiReport(
        requested_wkt=requested,
        clipped_wkt=format_wkt(clipped),
        area_km2=area,
        outside=False,
        message=message,
        subswath=subswath,
        polarisation=polarisation,
        burst_first=burst_first,
        burst_last=burst_last,
        burst_ids=tuple(burst_ids),
        centre_latitude=centre_lat,
        centre_longitude=centre_lon,
        north=bounds.north,
        south=bounds.south,
        west=bounds.west,
        east=bounds.east,
        esd_overlap_bursts=esd_pair,
    )


def refuse_if_outside(report: AoiReport) -> AoiReport:
    if report.outside:
        raise AoiError(report.message)
    return report


class BilinearLocator:
    """Map lon/lat through a 4-corner bilinear fit.

    Four geographic corners and the radar coordinates of those same corners
    fix the mapping SNAP's tie-point grid uses on a small crop. The fit is
    exact on the corners.
    """

    def __init__(
        self,
        geo_corners: list[tuple[float, float]] | tuple[tuple[float, float], ...],
        pixel_corners: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    ) -> None:
        if len(geo_corners) != 4 or len(pixel_corners) != 4:
            raise AoiError("A bilinear locator needs four corners.")
        design = []
        columns = []
        rows = []
        for (lon, lat), (column, row) in zip(geo_corners, pixel_corners):
            design.append([1.0, lon, lat, lon * lat])
            columns.append(column)
            rows.append(row)
        matrix = np.asarray(design, dtype=np.float64)
        self._column = np.linalg.solve(matrix, np.asarray(columns, dtype=np.float64))
        self._row = np.linalg.solve(matrix, np.asarray(rows, dtype=np.float64))
        self._geo = tuple((float(lon), float(lat)) for lon, lat in geo_corners)
        self._pixels = tuple((float(column), float(row)) for column, row in pixel_corners)

    def to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        basis = np.array([1.0, lon, lat, lon * lat], dtype=np.float64)
        return float(basis @ self._column), float(basis @ self._row)

    def to_geo(self, column: float, row: float) -> tuple[float, float]:
        """Invert the bilinear map with a few Newton steps from the nearest corner."""
        guess = min(self._geo, key=lambda item: abs(item[0] - self._geo[0][0]) + abs(item[1] - self._geo[0][1]))
        lon, lat = guess
        for _step in range(12):
            column_hat, row_hat = self.to_pixel(lon, lat)
            error_c = column_hat - column
            error_r = row_hat - row
            if abs(error_c) < 1.0e-9 and abs(error_r) < 1.0e-9:
                break
            step = 1.0e-6
            c1, r1 = self.to_pixel(lon + step, lat)
            c2, r2 = self.to_pixel(lon, lat + step)
            jacobian = np.array([[(c1 - column_hat) / step, (c2 - column_hat) / step], [(r1 - row_hat) / step, (r2 - row_hat) / step]])
            try:
                delta = np.linalg.solve(jacobian, np.array([-error_c, -error_r]))
            except np.linalg.LinAlgError:
                break
            lon += float(delta[0])
            lat += float(delta[1])
        return float(lon), float(lat)


def pixel_window_for_polygon(
    polygon: LonLatPolygon,
    locator: BilinearLocator,
    *,
    sub_sampling_x: int = 1,
    sub_sampling_y: int = 1,
    height: int | None = None,
    width: int | None = None,
) -> PixelWindow:
    """Half-open pixel window covering every corner of ``polygon``.

    ``floor`` of the minimum and ``ceil`` of the maximum match the window
    recorded for the 100 m square.
    """
    rows = []
    columns = []
    for lon, lat in polygon.ring:
        column, row = locator.to_pixel(lon, lat)
        rows.append(row)
        columns.append(column)
    x0 = int(math.floor(min(columns) + 1.0e-9))
    y0 = int(math.floor(min(rows) + 1.0e-9))
    x1 = int(math.ceil(max(columns) - 1.0e-9))
    y1 = int(math.ceil(max(rows) - 1.0e-9))
    if height is not None:
        y0 = max(y0, 0)
        y1 = min(y1, int(height))
    if width is not None:
        x0 = max(x0, 0)
        x1 = min(x1, int(width))
    if x1 <= x0 or y1 <= y0:
        raise AoiError("The region does not cover any pixel of this raster.")
    step_x = max(int(sub_sampling_x), 1)
    step_y = max(int(sub_sampling_y), 1)
    return PixelWindow(x=x0, y=y0, width=x1 - x0, height=y1 - y0, sub_sampling_x=step_x, sub_sampling_y=step_y)


def subset_from_parameters(params: dict, *, height: int | None = None, width: int | None = None) -> PixelWindow:
    """Accept either the old exclusive line/sample keys or SNAP's region.

    SNAP's dialog shows inclusive scene end coordinates. ``region`` is
    ``x, y, width, height`` with ``width = endX - startX + 1``.
    """
    step_x = int(params.get("subSamplingX") or params.get("sub_sampling_x") or 1)
    step_y = int(params.get("subSamplingY") or params.get("sub_sampling_y") or 1)
    if step_x < 1 or step_y < 1:
        raise AoiError("subSamplingX and subSamplingY must be at least 1.")
    region = params.get("region")
    if isinstance(region, dict) and {"x", "y", "width", "height"} <= set(region):
        x = int(region["x"])
        y = int(region["y"])
        window_w = int(region["width"])
        window_h = int(region["height"])
    elif "scene_start_x" in params:
        x = int(params["scene_start_x"])
        y = int(params["scene_start_y"])
        window_w = int(params["scene_end_x"]) - x + 1
        window_h = int(params["scene_end_y"]) - y + 1
    else:
        y = int(params["row_start"])
        x = int(params["col_start"])
        window_h = int(params["row_end"]) - y
        window_w = int(params["col_end"]) - x
    if window_w < 1 or window_h < 1:
        raise AoiError("The subset is empty.")
    if height is not None and (y < 0 or y + window_h > int(height)):
        raise AoiError("The subset lines are outside the raster.")
    if width is not None and (x < 0 or x + window_w > int(width)):
        raise AoiError("The subset samples are outside the raster.")
    return PixelWindow(x=x, y=y, width=window_w, height=window_h, sub_sampling_x=step_x, sub_sampling_y=step_y)


def limit_geographic_bounds(
    west: float,
    south: float,
    east: float,
    north: float,
    geo_region: str,
) -> tuple[float, float, float, float, LonLatPolygon]:
    """Intersect an output frame with a SNAP ``geoRegion``. Empty means refuse."""
    polygon = parse_wkt(geo_region)
    bounds = geographic_bounds(polygon)
    west_out = max(float(west), bounds.west)
    east_out = min(float(east), bounds.east)
    south_out = max(float(south), bounds.south)
    north_out = min(float(north), bounds.north)
    if not (west_out < east_out and south_out < north_out):
        raise AoiError(
            "Output geoRegion is outside the terrain-correction footprint. No output pixels were written."
        )
    return west_out, south_out, east_out, north_out, polygon


def chain_aoi(report: AoiReport) -> dict[str, dict]:
    """The same clipped AOI, once per stage of the 15-stage chain."""
    payload = report.to_dict()
    return {stage: dict(payload) for stage in CHAIN_STAGES}


def config_from_report(report: AoiReport) -> AoiConfig:
    return AoiConfig(
        wkt=report.clipped_wkt or report.requested_wkt,
        subswath=report.subswath,
        polarisation=report.polarisation,
        burst_first=report.burst_first,
        burst_last=report.burst_last,
        burst_ids=tuple(report.burst_ids),
        north=report.north,
        south=report.south,
        west=report.west,
        east=report.east,
        region=report.pixel_region,
        sub_sampling_x=report.sub_sampling_x,
        sub_sampling_y=report.sub_sampling_y,
        source_bands=tuple(report.source_bands),
        copy_metadata=report.copy_metadata,
        area_km2=report.area_km2,
        centre_latitude=report.centre_latitude,
        centre_longitude=report.centre_longitude,
        esd_overlap_bursts=report.esd_overlap_bursts,
    )


def save_aoi_config(path: Path | str, config: AoiConfig) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    return destination


def load_aoi_config(path: Path | str) -> AoiConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AoiError("AOI config must be a JSON object.")
    return AoiConfig.from_dict(payload)


def overlap_line_count(gap_s: float = _BURST_GAP_S, lines_per_burst: int = _LINES_PER_BURST) -> int:
    """Lines shared by two IW bursts, from the burst-start gap and the line count."""
    first = int(math.ceil(float(gap_s) / _AZIMUTH_INTERVAL_S - 1.0e-9))
    count = int(lines_per_burst) - first
    if count < 8:
        raise AoiError("The burst overlap is shorter than 8 lines.")
    return count


def validation_square_centre_lines() -> float:
    """Lines from the 100 m square to the centre of the burst-5/6 boundary.

    The overlap is clamped to the end of the burst. The unclamped midpoint
    sits about 127 m past that boundary, and a 200 m square centred there
    would fall entirely in burst 6. The square is centred on the boundary.
    """
    overlap_start = _LINES_TO_OVERLAP
    overlap_span = min(
        float(overlap_line_count()),
        float(_LINES_PER_BURST) - (264.5 + overlap_start),
    )
    return overlap_start + overlap_span / 2.0


def radar_axes() -> tuple[tuple[float, float], tuple[float, float]]:
    """Unit east/north vectors of increasing azimuth line and increasing range sample."""
    lon0, lat0 = _OLD_LON, _OLD_LAT
    start = local_xy(_CORNER_AZ_START[1], _CORNER_AZ_START[0], lon0=lon0, lat0=lat0)
    along = local_xy(_CORNER_AZ_END[1], _CORNER_AZ_END[0], lon0=lon0, lat0=lat0)
    across = local_xy(_CORNER_RANGE[1], _CORNER_RANGE[0], lon0=lon0, lat0=lat0)
    azimuth = (along[0] - start[0], along[1] - start[1])
    rng = (across[0] - start[0], across[1] - start[1])

    def _unit(vector: tuple[float, float]) -> tuple[float, float]:
        length = math.hypot(*vector)
        return (vector[0] / length, vector[1] / length)

    return _unit(azimuth), _unit(rng)


def burst_boundary_square() -> AoiReport:
    """The 200 m validation square centred on the burst-5/6 boundary.

    Azimuth runs from the 100 m square toward increasing line, which is the
    direction of the overlap zone 1,098 lines away. The square is 100 m
    either side of the burst boundary and 100 m either side in range, so
    it straddles bursts 5 and 6 and contains their overlap.
    """
    overlap_lines = overlap_line_count()
    distance_m = validation_square_centre_lines() * _GROUND_M_PER_LINE
    azimuth, rng = radar_axes()
    centre_x = distance_m * azimuth[0]
    centre_y = distance_m * azimuth[1]
    half = _SQUARE_EDGE_M / 2.0
    corners = []
    for azimuth_sign, range_sign in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        x = centre_x + azimuth_sign * half * azimuth[0] + range_sign * half * rng[0]
        y = centre_y + azimuth_sign * half * azimuth[1] + range_sign * half * rng[1]
        corners.append(local_lonlat(x, y, lon0=_OLD_LON, lat0=_OLD_LAT))
    polygon = LonLatPolygon(tuple(corners))
    centre = local_lonlat(centre_x, centre_y, lon0=_OLD_LON, lat0=_OLD_LAT)
    bounds = geographic_bounds(polygon)
    # Footprints in the same local frame: burst 5 owns the lines up through
    # the overlap, burst 6 owns the overlap and the following lines. Their
    # intersection is the overlap the square is centred on.
    burst_5, burst_6 = _validation_burst_footprints(overlap_lines)
    report = prepare_aoi(
        polygon,
        reference=_validation_scene(burst_5, burst_6),
        secondary=_validation_scene(burst_5, burst_6),
        burst_footprints=(burst_5, burst_6),
        subswath="IW1",
        polarisation="VV",
        burst_ids=(359478, 359479),
    )
    # prepare_aoi numbers the two footprints as bursts 1 and 2. The product
    # indexes of this pair are 5 and 6.
    report.burst_first = 5
    report.burst_last = 6
    report.burst_ids = (359478, 359479)
    report.esd_overlap_bursts = (5, 6)
    report.centre_latitude = centre[1]
    report.centre_longitude = centre[0]
    report.north = bounds.north
    report.south = bounds.south
    report.west = bounds.west
    report.east = bounds.east
    report.area_km2 = polygon_area_km2(polygon)
    report.clipped_wkt = format_wkt(polygon)
    report.requested_wkt = format_wkt(polygon)
    report.message = (
        f"200 m square on the IW1 burst 5/6 boundary, area {report.area_km2:.6f} km². "
        f"Centre latitude {centre[1]:.8f}, longitude {centre[0]:.8f}. "
        "ESD uses the overlap between bursts 5 and 6. "
        "The stage 5–15 comparison skips ESD until that estimator is revised."
    )
    return report


def _span_polygon(near_m: float, far_m: float) -> LonLatPolygon:
    """A range-wide quad between two azimuth distances from the old square."""
    azimuth, rng = radar_axes()
    # Stay near the old square's range so the quad contains the validation square
    # and does not pretend to be the whole 80 km subswath.
    range_near = -2_000.0
    range_far = 4_000.0
    corners = []
    for distance, range_m in (
        (near_m, range_near),
        (near_m, range_far),
        (far_m, range_far),
        (far_m, range_near),
    ):
        x = distance * azimuth[0] + range_m * rng[0]
        y = distance * azimuth[1] + range_m * rng[1]
        corners.append(local_lonlat(x, y, lon0=_OLD_LON, lat0=_OLD_LAT))
    return LonLatPolygon(tuple(corners))


def _validation_burst_footprints(overlap_lines: int) -> tuple[LonLatPolygon, LonLatPolygon]:
    spacing = _GROUND_M_PER_LINE
    # Line 0 of burst 5 is 264.5 lines before the old square centre.
    burst_start = -264.5 * spacing
    overlap_start = _LINES_TO_OVERLAP * spacing
    overlap_end = overlap_start + overlap_lines * spacing
    burst_5_end = burst_start + _LINES_PER_BURST * spacing
    burst_6_end = burst_5_end + (_LINES_PER_BURST - overlap_lines) * spacing
    # Burst 5 includes its unique lines and the overlap. Burst 6 starts at
    # the overlap and continues past it.
    return _span_polygon(burst_start, max(burst_5_end, overlap_end)), _span_polygon(overlap_start, burst_6_end)


def _validation_scene(burst_5: LonLatPolygon, burst_6: LonLatPolygon) -> LonLatPolygon:
    """A scene footprint that contains both bursts, identical on both dates."""
    points = list(burst_5.ring) + list(burst_6.ring)
    lons = [lon for lon, _lat in points]
    lats = [lat for _lon, lat in points]
    pad_lat = 0.01
    pad_lon = 0.01
    return polygon_from_bounds(max(lats) + pad_lat, min(lats) - pad_lat, min(lons) - pad_lon, max(lons) + pad_lon)


def square_100m_pixel_window(locator: BilinearLocator) -> PixelWindow:
    return pixel_window_for_polygon(parse_wkt(SQUARE_100M_WKT), locator)


def validation_square_covers_overlap() -> dict:
    """Lines of the burst-5/6 overlap that fall inside the 200 m square.

    Rows are relative to the first line of burst 5, which is where TOPS
    Split starts when bursts 5 and 6 are the selected range and the raster
    origin is that burst.
    """
    overlap_lines = overlap_line_count()
    square_centre_line = 264.5 + validation_square_centre_lines()
    half_lines = (_SQUARE_EDGE_M / 2.0) / _GROUND_M_PER_LINE
    square_start = square_centre_line - half_lines
    square_stop = square_centre_line + half_lines
    overlap_start = 264.5 + _LINES_TO_OVERLAP
    # The burst ends at ``lines_per_burst``. A timing overlap that would run
    # past that line is the same boundary, clamped to the burst.
    overlap_stop = min(overlap_start + overlap_lines, float(_LINES_PER_BURST))
    boundary = overlap_start + max(overlap_stop - overlap_start, 0.0) / 2.0
    inside_start = max(square_start, overlap_start)
    inside_stop = min(square_stop, overlap_stop)
    return {
        "burst_first": 5,
        "burst_last": 6,
        "overlap_lines": overlap_lines,
        "square_line_start": square_start,
        "square_line_stop": square_stop,
        "overlap_line_start": overlap_start,
        "overlap_line_stop": overlap_stop,
        "boundary_line": boundary,
        "overlap_inside_square": inside_stop - inside_start,
        "square_straddles_boundary": square_start < boundary < square_stop,
    }
