"""Elevation grids for range-Doppler terrain correction.

Heights are sampled at each latitude and longitude. Orthometric DEMs are
converted to WGS84 ellipsoidal height with a geoid grid. No-data stays
missing. A single mean height is not a terrain correction, and ellipsoid
height 0 is not a DEM.
"""

from __future__ import annotations

import gzip
import math
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{folder}/{name}.hgt.gz"
_COPERNICUS_URL = (
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    "Copernicus_DSM_COG_10_{lat}_{lon}_DEM/Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif"
)
_GEOID_URL = "https://downloads.sourceforge.net/project/geographiclib/geoids-distrib/egm96-15.tar.bz2"


@dataclass
class GeoidGrid:
    """EGM96 undulation in metres. Bilinear in latitude and longitude."""

    lat_deg: np.ndarray
    lon_deg: np.ndarray
    undulation_m: np.ndarray
    source: str

    def undulation(self, lat_deg: float, lon_deg: float) -> float:
        lon = float(lon_deg) % 360.0
        lat = float(lat_deg)
        lats = self.lat_deg
        lons = self.lon_deg
        row = _bracket(lats, lat)
        col = _bracket(lons, lon)
        return float(_bilinear_value(self.undulation_m, lats, lons, row, col, lat, lon))


@dataclass
class ElevationModel:
    """Orthometric heights on a latitude/longitude grid, plus the geoid."""

    lat_deg: np.ndarray
    lon_deg: np.ndarray
    orthometric_m: np.ndarray
    nodata: float
    orthometric_datum: str
    geoid: GeoidGrid
    provenance: str

    def ellipsoidal_height(self, lat_deg: float, lon_deg: float) -> tuple[float, bool]:
        """Return ``(h_ellipsoid, valid)``. Invalid samples are no-data, not zero."""
        if not self.covers(lat_deg, lon_deg):
            return float("nan"), False
        row = _bracket(self.lat_deg, float(lat_deg))
        col = _bracket(self.lon_deg, float(lon_deg))
        samples = _corner_samples(self.orthometric_m, row, col)
        if not np.isfinite(samples).all() or np.any(np.abs(samples - float(self.nodata)) < 1e-3):
            return float("nan"), False
        height = _bilinear_value(self.orthometric_m, self.lat_deg, self.lon_deg, row, col, float(lat_deg), float(lon_deg))
        if not np.isfinite(height):
            return float("nan"), False
        undulation = self.geoid.undulation(lat_deg, lon_deg)
        return float(height) + undulation, True

    def covers(self, lat_deg: float, lon_deg: float) -> bool:
        lat = float(lat_deg)
        lon = float(lon_deg)
        return (
            float(self.lat_deg.min()) <= lat <= float(self.lat_deg.max())
            and float(self.lon_deg.min()) <= lon <= float(self.lon_deg.max())
        )


def read_egm96_pgm(path: Path | str) -> GeoidGrid:
    """Read a GeographicLib EGM96 PGM (offset and scale in the header)."""
    payload = Path(path).read_bytes()
    header, pixel_offset = _pgm_header(payload)
    width = int(header["width"])
    height = int(header["height"])
    count = width * height
    values = np.frombuffer(payload, dtype=">u2", count=count, offset=pixel_offset).reshape(height, width)
    undulation = float(header["offset"]) + float(header["scale"]) * values.astype(np.float64)
    # Origin is the centre of the first post: 90N, 0E. Longitude increases east.
    lat = 90.0 - np.arange(height, dtype=np.float64) * (180.0 / (height - 1))
    lon = np.arange(width, dtype=np.float64) * (360.0 / width)
    return GeoidGrid(lat_deg=lat, lon_deg=lon, undulation_m=undulation, source=str(path))


def read_srtm_hgt(path: Path | str, geoid: GeoidGrid) -> ElevationModel:
    """Read one SRTM 1-arc-second ``.hgt`` or ``.hgt.gz`` tile. Datum is EGM96."""
    source = Path(path)
    raw = source.read_bytes()
    if source.suffix.lower() == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    side = int(round(math.sqrt(len(raw) // 2)))
    if side * side * 2 != len(raw):
        raise ValueError(f"{source.name} is not a square SRTM HGT")
    heights = np.frombuffer(raw, dtype=">i2").reshape(side, side).astype(np.float64)
    lat0, lon0 = _hgt_origin(source.name)
    step = 1.0 / (side - 1)
    lat = lat0 + 1.0 - np.arange(side, dtype=np.float64) * step
    lon = lon0 + np.arange(side, dtype=np.float64) * step
    return ElevationModel(
        lat_deg=lat,
        lon_deg=lon,
        orthometric_m=heights,
        nodata=-32768.0,
        orthometric_datum="EGM96",
        geoid=geoid,
        provenance=f"SRTM1 {source.name} + {geoid.source}",
    )


def read_copernicus_tif(path: Path | str, geoid: GeoidGrid) -> ElevationModel:
    """Read one Copernicus 30 m GeoTIFF. Heights are EGM2008 orthometric.

    The geoid applied here is EGM96. The EGM2008−EGM96 difference is not
    removed and is recorded in ``provenance``.
    """
    import tifffile

    source = Path(path)
    with tifffile.TiffFile(source) as tiff:
        page = tiff.pages[0]
        try:
            heights = np.asarray(page.asarray(), dtype=np.float64)
        except ValueError as exc:
            if "imagecodecs" in str(exc):
                raise RuntimeError(
                    "This Copernicus GeoTIFF uses a floating-point predictor. "
                    "Install the imagecodecs package to read it."
                ) from exc
            raise
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
    rows, cols = heights.shape
    lon = float(tie[3]) + (np.arange(cols, dtype=np.float64) + 0.5) * float(scale[0])
    lat = float(tie[4]) - (np.arange(rows, dtype=np.float64) + 0.5) * float(scale[1])
    return ElevationModel(
        lat_deg=lat,
        lon_deg=lon,
        orthometric_m=heights,
        nodata=float("nan"),
        orthometric_datum="EGM2008",
        geoid=geoid,
        provenance=(
            f"Copernicus GLO-30 {source.name} (EGM2008 orthometric) + EGM96 undulation "
            f"from {geoid.source}. EGM2008−EGM96 is not modelled."
        ),
    )


def ensure_geoid(cache: Path | str) -> GeoidGrid:
    """Return the cached EGM96 15-minute grid, downloading the public archive once."""
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    pgm = folder / "geoids" / "egm96-15.pgm"
    if not pgm.is_file():
        archive = folder / "egm96-15.tar.bz2"
        if not archive.is_file():
            _download(_GEOID_URL, archive)
        with tarfile.open(archive, "r:bz2") as bundle:
            bundle.extractall(folder, filter="data")
    return read_egm96_pgm(pgm)


def ensure_srtm(lat_deg: float, lon_deg: float, cache: Path | str, geoid: GeoidGrid) -> ElevationModel:
    """Download the SRTM-1 tile that contains the point, then open it."""
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    name = _srtm_name(lat_deg, lon_deg)
    compressed = folder / f"{name}.hgt.gz"
    if not compressed.is_file() and not (folder / f"{name}.hgt").is_file():
        url = _SRTM_URL.format(folder=name[:3], name=name)
        _download(url, compressed)
    path = compressed if compressed.is_file() else folder / f"{name}.hgt"
    return read_srtm_hgt(path, geoid)


def ensure_copernicus(lat_deg: float, lon_deg: float, cache: Path | str, geoid: GeoidGrid) -> ElevationModel:
    """Download the Copernicus 30 m tile that contains the point, then open it."""
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    lat_tag, lon_tag = _copernicus_tags(lat_deg, lon_deg)
    destination = folder / f"Copernicus_DSM_COG_10_{lat_tag}_{lon_tag}_DEM.tif"
    if not destination.is_file():
        url = _COPERNICUS_URL.format(lat=lat_tag, lon=lon_tag)
        _download(url, destination)
    return read_copernicus_tif(destination, geoid)


def wgs84_to_utm(lat_deg: float, lon_deg: float) -> tuple[float, float, int]:
    """Return easting, northing, and the UTM zone for a WGS84 geographic point."""
    zone = int(math.floor((float(lon_deg) + 180.0) / 6.0)) + 1
    central = math.radians(zone * 6.0 - 183.0)
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = ep2 * math.cos(lat) ** 2
    aa = math.cos(lat) * (lon - central)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat)
        - (35 * e2**3 / 3072) * math.sin(6 * lat)
    )
    easting = 500_000.0 + 0.9996 * n * (
        aa + (1 - t + c) * aa**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    )
    northing = 0.9996 * (
        m
        + n
        * math.tan(lat)
        * (aa**2 / 2 + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24 + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720)
    )
    if lat_deg < 0:
        northing += 10_000_000.0
    return easting, northing, zone


def utm_to_wgs84(easting: float, northing: float, zone: int, *, northern: bool = True) -> tuple[float, float]:
    """Invert :func:`wgs84_to_utm` by a few Newton steps."""
    lon = zone * 6.0 - 183.0
    lat = 0.0 if northern else -45.0
    north = float(northing) if northern else float(northing) - 10_000_000.0
    for _ in range(6):
        e0, n0, _zone = wgs84_to_utm(lat, lon)
        if not northern:
            n0 -= 10_000_000.0
        e1, n1, _zone = wgs84_to_utm(lat + 1e-5, lon)
        e2, n2, _zone = wgs84_to_utm(lat, lon + 1e-5)
        if not northern:
            n1 -= 10_000_000.0
            n2 -= 10_000_000.0
        de_dlat = (e1 - e0) / 1e-5
        dn_dlat = (n1 - n0) / 1e-5
        de_dlon = (e2 - e0) / 1e-5
        dn_dlon = (n2 - n0) / 1e-5
        det = de_dlat * dn_dlon - de_dlon * dn_dlat
        if abs(det) < 1e-12:
            break
        de = float(easting) - e0
        dn = north - n0
        lat += (dn_dlon * de - de_dlon * dn) / det
        lon += (-dn_dlat * de + de_dlat * dn) / det
    return lat, lon


def _srtm_name(lat_deg: float, lon_deg: float) -> str:
    lat_i = math.floor(float(lat_deg))
    lon_i = math.floor(float(lon_deg))
    ns = "N" if lat_i >= 0 else "S"
    ew = "E" if lon_i >= 0 else "W"
    return f"{ns}{abs(lat_i):02d}{ew}{abs(lon_i):03d}"


def _copernicus_tags(lat_deg: float, lon_deg: float) -> tuple[str, str]:
    lat_i = math.floor(float(lat_deg))
    lon_i = math.floor(float(lon_deg))
    ns = "N" if lat_i >= 0 else "S"
    ew = "E" if lon_i >= 0 else "W"
    return f"{ns}{abs(lat_i):02d}_00", f"{ew}{abs(lon_i):03d}_00"


def _hgt_origin(name: str) -> tuple[int, int]:
    stem = Path(name).name
    for suffix in (".hgt.gz", ".HGT.GZ", ".hgt", ".HGT"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if len(stem) < 7:
        raise ValueError(f"cannot read an SRTM tile name from {name}")
    lat = int(stem[1:3]) * (1 if stem[0].upper() == "N" else -1)
    lon = int(stem[4:7]) * (1 if stem[3].upper() == "E" else -1)
    return lat, lon


def _pgm_header(payload: bytes) -> tuple[dict, int]:
    if not payload.startswith(b"P5"):
        raise ValueError("geoid file is not a binary PGM")
    header: dict[str, float | int] = {}
    cursor = payload.find(b"\n") + 1
    tokens: list[bytes] = []
    while len(tokens) < 3:
        line_end = payload.find(b"\n", cursor)
        line = payload[cursor:line_end]
        cursor = line_end + 1
        if line.startswith(b"#"):
            text = line.decode("ascii", errors="replace")
            if "Offset" in text:
                header["offset"] = float(text.split()[-1])
            elif "Scale" in text:
                header["scale"] = float(text.split()[-1])
            continue
        tokens.extend(line.split())
    header["width"] = int(tokens[0])
    header["height"] = int(tokens[1])
    if "offset" not in header or "scale" not in header:
        raise ValueError("PGM geoid header has no Offset and Scale")
    return header, cursor


def _bracket(axis: np.ndarray, value: float) -> int:
    """Index of the sample at or just before ``value`` along a monotonic axis."""
    increasing = float(axis[0]) <= float(axis[-1])
    if increasing:
        index = int(np.searchsorted(axis, value, side="right") - 1)
    else:
        index = int(np.searchsorted(-axis, -value, side="right") - 1)
    return int(np.clip(index, 0, axis.shape[0] - 2))


def _corner_samples(grid: np.ndarray, row: int, col: int) -> np.ndarray:
    row1 = min(row + 1, grid.shape[0] - 1)
    col1 = min(col + 1, grid.shape[1] - 1)
    return np.array([grid[row, col], grid[row, col1], grid[row1, col], grid[row1, col1]], dtype=np.float64)


def _bilinear_value(
    grid: np.ndarray,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    row: int,
    col: int,
    lat: float,
    lon: float,
) -> float:
    row1 = min(row + 1, grid.shape[0] - 1)
    col1 = min(col + 1, grid.shape[1] - 1)
    lat0 = float(lat_axis[row])
    lat1 = float(lat_axis[row1])
    lon0 = float(lon_axis[col])
    lon1 = float(lon_axis[col1])
    wy = 0.0 if lat1 == lat0 else (lat - lat0) / (lat1 - lat0)
    wx = 0.0 if lon1 == lon0 else (lon - lon0) / (lon1 - lon0)
    samples = np.array(
        [grid[row, col], grid[row, col1], grid[row1, col], grid[row1, col1]],
        dtype=np.float64,
    )
    if not np.isfinite(samples).all():
        return float("nan")
    return float(
        (1 - wy) * (1 - wx) * samples[0]
        + (1 - wy) * wx * samples[1]
        + wy * (1 - wx) * samples[2]
        + wy * wx * samples[3]
    )


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    temporary.replace(destination)
