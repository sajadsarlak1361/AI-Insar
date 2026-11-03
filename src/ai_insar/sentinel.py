"""Search ASF for Sentinel-1 SLC granules and import a crop into an SLC stack.

Metadata search uses the public ASF Search API and does not need a login.
Downloading the zip from the ASF datapool needs Earthdata credentials and the
optional ``asf_search`` package. A measurement GeoTIFF or complex ``.npy`` that
is already on disk can be cropped into the same ``complex64`` stack the rest
of the workbench uses. Passwords are read from the environment or ``~/.netrc``
and are never written into the stack or the log messages this module raises.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_insar.conventions import C_BAND_WAVELENGTH_M
from ai_insar.io_stack import default_dates
from ai_insar.pipeline.synthetic import SLCStack

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"
_USER_AGENT = "ai-insar/0.1.0 (SLC workbench; metadata search)"
_DATE_IN_NAME = re.compile(r"(20\d{2})(\d{2})(\d{2})")


@dataclass(frozen=True)
class SentinelGranule:
    """One Sentinel-1 SLC product from an ASF search hit."""

    granule_name: str
    start_time: str
    platform: str
    flight_direction: str
    polarization: str
    beam_mode: str
    size_mb: float | None
    download_url: str

    @property
    def date(self) -> np.datetime64 | None:
        return date_from_text(self.start_time) or date_from_text(self.granule_name)


def search_sentinel1_slc(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    start: str,
    end: str,
    max_results: int = 10,
    urlopen=urllib.request.urlopen,
) -> list[SentinelGranule]:
    """Search the public ASF catalog for Sentinel-1 SLC products.

    ``bbox`` is west, south, east, north in degrees. ``start`` and ``end`` are
    ``YYYY-MM-DD``. No Earthdata credentials are sent.
    """
    _validate_bbox(west, south, east, north)
    _validate_dates(start, end)
    if not 1 <= int(max_results) <= 50:
        raise ValueError("max_results must be between 1 and 50")
    query = urllib.parse.urlencode(
        {
            "platform": "Sentinel-1",
            "processingLevel": "SLC",
            "bbox": f"{west},{south},{east},{north}",
            "start": start,
            "end": end,
            "maxResults": str(int(max_results)),
            "output": "json",
        }
    )
    request = urllib.request.Request(
        f"{ASF_SEARCH_URL}?{query}",
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ASF search failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ASF search could not connect ({exc.reason}).") from exc
    granules = parse_asf_payload(payload)
    granules.sort(key=lambda item: (item.start_time, item.granule_name))
    return granules


def parse_asf_payload(payload: object) -> list[SentinelGranule]:
    """Turn an ASF JSON list or GeoJSON feature collection into granules."""
    records = _records(payload)
    granules: list[SentinelGranule] = []
    for record in records:
        level = str(record.get("processingLevel") or "SLC").upper()
        if level and level != "SLC":
            continue
        name = str(record.get("granuleName") or record.get("sceneName") or record.get("fileName") or "")
        url = str(record.get("downloadUrl") or record.get("url") or "")
        if not name or not url:
            continue
        size = record.get("sizeMB")
        if size is None and record.get("bytes") not in (None, ""):
            size = float(record["bytes"]) / 1_000_000.0
        granules.append(
            SentinelGranule(
                granule_name=name.removesuffix(".zip"),
                start_time=str(record.get("startTime") or record.get("sceneDate") or ""),
                platform=str(record.get("platform") or ""),
                flight_direction=str(record.get("flightDirection") or ""),
                polarization=str(record.get("polarization") or ""),
                beam_mode=str(record.get("beamModeType") or record.get("beamMode") or ""),
                size_mb=None if size in (None, "") else float(size),
                download_url=url,
            )
        )
    return granules


def earthdata_credentials() -> tuple[str, str, str] | None:
    """Return ``(username, password, source)`` or ``None``.

    ``source`` is ``environment`` or ``netrc``. The password is only returned
    to the caller that passes it to ASF; this function does not log it.
    """
    username = os.environ.get("EARTHDATA_USERNAME", "").strip()
    password = os.environ.get("EARTHDATA_PASSWORD", "")
    if username and password:
        return username, password, "environment"
    try:
        import netrc

        authenticators = netrc.netrc().authenticators(EARTHDATA_HOST)
    except (FileNotFoundError, OSError, netrc.NetrcParseError):
        return None
    if not authenticators or not authenticators[0] or not authenticators[2]:
        return None
    return authenticators[0], authenticators[2], "netrc"


def credentials_summary() -> str:
    """One line for the dialog. The password is not included."""
    found = earthdata_credentials()
    if found is None:
        return (
            "No Earthdata credentials. Search and local import still work. "
            "To download, set EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or add "
            f"a ~/.netrc machine {EARTHDATA_HOST}. Do not commit the password."
        )
    username, _password, source = found
    where = "environment variables" if source == "environment" else "~/.netrc"
    extra = ""
    if _load_asf_search() is None:
        extra = ' Download also needs pip install "ai-insar[slc]" (asf_search).'
    return f"Earthdata login found for {username} ({where}).{extra}"


def download_sentinel1_slc(
    granules: list[SentinelGranule],
    directory: Path,
    *,
    username: str,
    password: str,
) -> list[Path]:
    """Download SLC zip files into ``directory`` using ``asf_search``.

    Raises ``RuntimeError`` with a message that does not contain the password
    when credentials are missing, ``asf_search`` is not installed, or ASF
    rejects the login.
    """
    if not granules:
        raise ValueError("no granules selected")
    if not username or not password:
        raise RuntimeError(
            "Earthdata username and password are required to download. "
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or ~/.netrc for "
            f"{EARTHDATA_HOST}."
        )
    asf = _load_asf_search()
    if asf is None:
        raise RuntimeError(
            'ASF download needs the optional asf_search package: pip install "ai-insar[slc]". '
            "Search and importing a local SAFE, TIFF, or .npy do not need it."
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        session = asf.ASFSession()
        session.auth_with_creds(username, password)
        results = asf.granule_search([item.granule_name for item in granules])
        if len(results) == 0:
            raise RuntimeError("ASF returned no products for the selected granules.")
        results.download(path=str(directory), session=session)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "ASF download failed ("
            + type(exc).__name__
            + "). Check the Earthdata login. AI Insar does not store the password."
        ) from exc
    return sorted(path for path in directory.glob("*.zip") if path.is_file())


def import_slc_paths(
    paths: list[Path],
    *,
    row: int = 0,
    col: int = 0,
    height: int = 256,
    width: int = 256,
    wavelength_m: float = C_BAND_WAVELENGTH_M,
) -> SLCStack:
    """Build a ``complex64`` stack from rasters or per-date ``.npy`` planes.

    One 3-D complex ``.npy`` is loaded as a finished stack (the crop is
    applied). Otherwise each file or SAFE folder contributes one date. A SAFE
    folder contributes a single measurement raster: IW1 VV when several
    subswaths are present. The same crop window is read from every date.
    """
    if not paths:
        raise ValueError("no SLC files were selected")
    rasters = discover_slc_rasters([Path(path) for path in paths])
    if len(rasters) == 1 and rasters[0].suffix.lower() == ".npy":
        array = np.load(rasters[0])
        if array.ndim == 3 and np.iscomplexobj(array):
            cropped = _crop_numpy(array, row, col, height, width)
            dates = _dates_for_paths(rasters, count=cropped.shape[0])
            return SLCStack(
                data=cropped,
                dates=dates,
                wavelength_m=wavelength_m,
                sensor="Sentinel-1 SLC" if _looks_like_sentinel(rasters[0].name) else "imported SLC",
                description=_import_description(rasters, row, col, cropped.shape[1], cropped.shape[2]),
            )
    rasters = _order_by_date(rasters)
    planes: list[np.ndarray] = []
    for path in rasters:
        planes.append(_read_plane(path, row, col, height, width))
    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        raise ValueError(f"cropped SLC dates do not share a shape: {sorted(shapes)}")
    data = np.stack(planes, axis=0).astype(np.complex64)
    dates = _dates_for_paths(rasters, count=data.shape[0])
    return SLCStack(
        data=data,
        dates=dates,
        wavelength_m=wavelength_m,
        sensor="Sentinel-1 SLC" if any(_looks_like_sentinel(path.name) for path in rasters) else "imported SLC",
        description=_import_description(rasters, row, col, data.shape[1], data.shape[2]),
    )


def discover_slc_rasters(paths: list[Path]) -> list[Path]:
    """Expand files and SAFE folders into one raster path per date."""
    found: list[Path] = []
    for path in paths:
        if path.is_file():
            if path.suffix.lower() not in {".npy", ".tif", ".tiff"}:
                raise ValueError(f"{path.name} is not a .npy, .tif, or .tiff SLC")
            found.append(path)
            continue
        if not path.is_dir():
            raise ValueError(f"{path} does not exist")
        measurements = [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".tif", ".tiff"} and "measurement" in item.parts
        ]
        if not measurements:
            measurements = [
                item
                for item in path.glob("*")
                if item.is_file() and item.suffix.lower() in {".tif", ".tiff", ".npy"}
            ]
        if not measurements:
            raise ValueError(f"no measurement TIFF or .npy SLC found under {path}")
        found.append(_prefer_measurement(measurements))
    if not found:
        raise ValueError("no SLC rasters found")
    return found


def date_from_text(text: str) -> np.datetime64 | None:
    """Parse a day from an ISO timestamp or a Sentinel-1 ``YYYYMMDD`` name."""
    source = text or ""
    iso = re.search(r"(20\d{2})-(\d{2})-(\d{2})", source)
    match = iso or _DATE_IN_NAME.search(source)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return np.datetime64(f"{year}-{month}-{day}", "D")
    except ValueError:
        return None


def _records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        features = payload.get("features")
        if isinstance(features, list):
            records = []
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                properties = feature.get("properties")
                records.append(properties if isinstance(properties, dict) else feature)
            return records
    raise ValueError("ASF search returned an unexpected JSON shape")


def _validate_bbox(west: float, south: float, east: float, north: float) -> None:
    if not -180.0 <= west < east <= 180.0:
        raise ValueError("longitude must satisfy -180 <= west < east <= 180")
    if not -90.0 <= south < north <= 90.0:
        raise ValueError("latitude must satisfy -90 <= south < north <= 90")


def _validate_dates(start: str, end: str) -> None:
    try:
        start_day = np.datetime64(start, "D")
        end_day = np.datetime64(end, "D")
    except ValueError as exc:
        raise ValueError("start and end must be YYYY-MM-DD") from exc
    if start_day > end_day:
        raise ValueError("start date must be on or before the end date")


def _load_asf_search():
    try:
        import asf_search as asf
    except ImportError:
        return None
    return asf


def _prefer_measurement(paths: list[Path]) -> Path:
    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        rank = 0
        if "-vv-" in name or "vv" in name:
            rank += 20
        if "iw1" in name:
            rank += 10
        elif "iw2" in name:
            rank += 5
        return rank, name

    return sorted(paths, key=score, reverse=True)[0]


def _read_plane(path: Path, row: int, col: int, height: int, width: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.ndim != 2 or not np.iscomplexobj(array):
            raise ValueError(f"{path.name} must be a 2-D complex array (or a single 3-D stack)")
        cropped = _crop_numpy(array, row, col, height, width)
        return np.asarray(cropped, dtype=np.complex64)
    return read_tiff_window(path, row, col, height, width)


def read_tiff_window(path: Path, row: int, col: int, height: int, width: int) -> np.ndarray:
    """Read a complex crop from a measurement GeoTIFF.

    When ``zarr`` is installed the crop is read through tifffile's chunk store
    so a full Sentinel-1 raster is not loaded. Without ``zarr``, only a raster
    under 512 MiB is read, and then cropped.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            'Reading a measurement TIFF needs tifffile: pip install "ai-insar[slc]".'
        ) from exc
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        spatial = _spatial_shape(page.shape)
        row, col, height, width = _fit_window(spatial, row, col, height, width)
        selector = _window_index(page.shape, row, col, height, width)
        try:
            import zarr

            store = tiff.aszarr()
            array = zarr.open(store, mode="r")
            raw = np.asarray(array[selector])
        except Exception:
            nbytes = int(np.prod(page.shape)) * max(np.dtype(page.dtype).itemsize, 1)
            if nbytes > 512 * 1024 * 1024:
                raise RuntimeError(
                    f"{path.name} is about {nbytes / 1e9:.1f} GB. "
                    'Install the slc extra (tifffile and zarr) so the crop is read in tiles: '
                    'pip install "ai-insar[slc]".'
                )
            raw = np.asarray(page.asarray())[selector]
    return to_complex64(raw)


def to_complex64(array: np.ndarray) -> np.ndarray:
    """Interpret a raster as complex64 SLC samples."""
    if np.iscomplexobj(array):
        return np.asarray(array, dtype=np.complex64)
    if array.dtype.names and {"real", "imag"} <= set(array.dtype.names):
        return (array["real"].astype(np.float32) + 1j * array["imag"].astype(np.float32)).astype(np.complex64)
    if array.ndim == 3 and array.shape[-1] == 2:
        return (array[..., 0].astype(np.float32) + 1j * array[..., 1].astype(np.float32)).astype(np.complex64)
    if array.ndim == 3 and array.shape[0] == 2:
        return (array[0].astype(np.float32) + 1j * array[1].astype(np.float32)).astype(np.complex64)
    raise ValueError(
        "TIFF is not complex SLC data. Expected a complex raster or integer I/Q bands, "
        f"got shape {array.shape} dtype {array.dtype}."
    )


def _spatial_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[-1] == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] == 2:
        return int(shape[1]), int(shape[2])
    raise ValueError(f"unsupported TIFF shape {shape}")


def _fit_window(
    spatial: tuple[int, int],
    row: int,
    col: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    image_h, image_w = spatial
    if row == 0 and col == 0 and (height > image_h or width > image_w):
        height, width = image_h, image_w
    if row < 0 or col < 0 or height < 1 or width < 1:
        raise ValueError("crop row, column, height, and width must be positive")
    if row + height > image_h or col + width > image_w:
        raise ValueError(
            f"crop rows {row}:{row + height}, cols {col}:{col + width} "
            f"is outside the raster ({image_h}×{image_w})"
        )
    return row, col, height, width


def _window_index(shape: tuple[int, ...], row: int, col: int, height: int, width: int):
    if len(shape) == 2 or (len(shape) == 3 and shape[-1] == 2):
        return (slice(row, row + height), slice(col, col + width))
    if len(shape) == 3 and shape[0] == 2:
        return (slice(None), slice(row, row + height), slice(col, col + width))
    raise ValueError(f"unsupported TIFF shape {shape}")


def _crop_numpy(array: np.ndarray, row: int, col: int, height: int, width: int) -> np.ndarray:
    if array.ndim == 2:
        row, col, height, width = _fit_window(array.shape, row, col, height, width)
        return np.asarray(array[row : row + height, col : col + width])
    if array.ndim == 3:
        row, col, height, width = _fit_window(array.shape[1:], row, col, height, width)
        return np.asarray(array[:, row : row + height, col : col + width])
    raise ValueError(f"complex .npy must be 2-D or 3-D, got ndim {array.ndim}")


def _order_by_date(paths: list[Path]) -> list[Path]:
    parsed = [date_from_text(path.name) for path in paths]
    if any(item is None for item in parsed):
        return list(paths)
    order = np.argsort(np.array(parsed, dtype="datetime64[D]"), kind="mergesort")
    return [paths[int(index)] for index in order]


def _dates_for_paths(paths: list[Path], *, count: int) -> np.ndarray:
    if count != len(paths):
        # A single 3-D cube has one path and many dates.
        parsed = date_from_text(paths[0].name) if len(paths) == 1 else None
        if parsed is not None and count > 1:
            return parsed + np.arange(count, dtype=np.int64) * np.timedelta64(12, "D")
        return default_dates(count)
    parsed = [date_from_text(path.name) for path in paths]
    if any(item is None for item in parsed):
        return default_dates(count)
    dates = np.array(parsed, dtype="datetime64[D]")
    order = np.argsort(dates, kind="mergesort")
    dates = dates[order]
    if np.any(np.diff(dates) <= np.timedelta64(0, "D")):
        raise ValueError("imported SLC dates are not strictly increasing; rename files so each date differs")
    return dates


def _looks_like_sentinel(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("s1") or "slc" in lowered or "safe" in lowered


def _import_description(paths: list[Path], row: int, col: int, height: int, width: int) -> str:
    names = ", ".join(path.name for path in paths[:4])
    if len(paths) > 4:
        names += f", +{len(paths) - 4} more"
    return (
        f"Imported SLC crop rows {row}:{row + height}, cols {col}:{col + width} "
        f"from {names}. Not geocoded."
    )
