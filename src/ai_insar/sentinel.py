"""Search ASF for Sentinel-1 SLC granules and import a crop into an SLC stack.

Metadata search uses the public ASF Search API and does not need a login.
Downloading the zip from the ASF datapool needs Earthdata credentials and the
optional ``asf_search`` package. A SAFE folder, ``manifest.safe``, product
``.zip``, measurement GeoTIFF, or complex ``.npy`` that is already on disk
can be cropped into the same ``complex64`` stack the rest of the workbench
uses. Several SAFE zips (or a folder of them) become one stack: the same
swath, polarisation, and pixel window on every date. SAFE metadata comes
from :mod:`ai_insar.safe`. Passwords are read from the environment or
``~/.netrc`` and are never written into the stack or the log messages this
module raises.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_insar.conventions import C_BAND_WAVELENGTH_M, SPEED_OF_LIGHT_M_S
from ai_insar.io_stack import default_dates
from ai_insar.safe import (
    DEFAULT_POLARISATION,
    DEFAULT_SWATH,
    SafeMeasurement,
    SafeProduct,
    band_from_measurement_name,
    read_annotation,
    read_window,
    resolve_safe,
    select_measurement,
)

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


@dataclass(frozen=True)
class _SlcSource:
    """One date to crop into the stack."""

    path: Path
    kind: str
    date: np.datetime64 | None
    wavelength_m: float | None
    label: str
    band_note: str | None = None
    product: SafeProduct | None = None
    measurement: SafeMeasurement | None = None


def import_slc_paths(
    paths: list[Path],
    *,
    row: int = 0,
    col: int = 0,
    height: int = 256,
    width: int = 256,
    wavelength_m: float | None = None,
    swath: str = DEFAULT_SWATH,
    pol: str = DEFAULT_POLARISATION,
) -> SLCStack:
    """Build a ``complex64`` stack from rasters, SAFE products, or ``.npy`` planes.

    One 3-D complex ``.npy`` is loaded as a finished stack (the crop is
    applied). Otherwise each file, SAFE folder, ``manifest.safe``, or product
    zip contributes one date. A SAFE product contributes one measurement:
    ``swath`` / ``pol``, default IW1 and VV. The same crop window is read
    from every date. When the annotation has ``radarFrequency`` and
    ``wavelength_m`` is omitted, that frequency sets the stack wavelength.
    Manifest ``startTime`` supplies the date when it is present.
    """
    from ai_insar.pipeline.synthetic import SLCStack

    if not paths:
        raise ValueError("no SLC files were selected")
    sources = _expand_sources([Path(path) for path in paths], swath=swath, pol=pol)
    if len(sources) == 1 and sources[0].path.suffix.lower() == ".npy" and sources[0].kind == "raster":
        array = np.load(sources[0].path)
        if array.ndim == 3 and np.iscomplexobj(array):
            cropped = _crop_numpy(array, row, col, height, width)
            dates = _dates_for_sources([sources[0]], count=cropped.shape[0])
            return SLCStack(
                data=cropped,
                dates=dates,
                wavelength_m=_stack_wavelength(sources, wavelength_m),
                sensor="Sentinel-1 SLC" if _looks_like_sentinel(sources[0].label) else "imported SLC",
                description=_import_description(
                    [sources[0].label],
                    row,
                    col,
                    cropped.shape[1],
                    cropped.shape[2],
                    band=sources[0].band_note,
                ),
            )
    sources = _order_sources(sources)
    planes = [_read_source(source, row, col, height, width) for source in sources]
    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        raise ValueError(f"cropped SLC dates do not share a shape: {sorted(shapes)}")
    data = np.stack(planes, axis=0).astype(np.complex64)
    return SLCStack(
        data=data,
        dates=_dates_for_sources(sources, count=data.shape[0]),
        wavelength_m=_stack_wavelength(sources, wavelength_m),
        sensor="Sentinel-1 SLC" if any(_looks_like_sentinel(source.label) for source in sources) else "imported SLC",
        description=_import_description(
            [source.label for source in sources],
            row,
            col,
            data.shape[1],
            data.shape[2],
            band=next((source.band_note for source in sources if source.band_note), None),
        ),
    )


@dataclass(frozen=True)
class SafeStackLoad:
    """One SLC stack built from several SAFE products, and what was left out."""

    stack: SLCStack
    skipped: tuple[tuple[str, str], ...]


def expand_safe_inputs(paths: list[Path]) -> list[Path]:
    """Resolve SAFE zips, ``.SAFE`` folders, manifests, and a folder of those.

    A directory that is not itself a SAFE product contributes the Sentinel-1
    ``.zip`` files and ``.SAFE`` folders directly inside it. Nested folders
    are not searched. The zip is the primary input and is not extracted.
    """
    if not paths:
        raise ValueError("no Sentinel-1 SAFE products were selected")
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        if _is_safe_container(path):
            found.append(path)
            continue
        if path.is_dir():
            children = _direct_safe_products(path)
            if not children:
                raise ValueError(
                    f"{path.name} has no Sentinel-1 SAFE zip or .SAFE folder. "
                    "Choose the product zips, or a folder that contains them."
                )
            found.extend(children)
            continue
        raise ValueError(
            f"{path.name} is not a Sentinel-1 SAFE zip, .SAFE folder, or manifest.safe"
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_safe_slc_stack(
    paths: list[Path],
    *,
    row: int = 0,
    col: int = 0,
    height: int = 256,
    width: int = 256,
    wavelength_m: float | None = None,
    swath: str = DEFAULT_SWATH,
    pol: str = DEFAULT_POLARISATION,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> SafeStackLoad:
    """Build a ``complex64`` stack from many Sentinel-1 SAFE products.

    Every product is read at the same ``(row, col, height, width)`` window
    and the same swath and polarisation (default IW1 VV). The date is the
    manifest start time. The stack is sorted so the date index increases.
    A product that cannot supply that band, or whose crop does not match
    the others, is skipped. Zips stay zipped: an uncompressed measurement
    TIFF is read by seeking to the zip entry.

    Products are not coregistered. The shared window is a pixel crop for
    teaching, not a TOPS interferogram.
    """
    products = expand_safe_inputs([Path(path) for path in paths])
    band = f"{swath.strip().upper()} {pol.strip().upper()}"
    skipped: list[tuple[str, str]] = []
    prepared: list[_SlcSource] = []
    total = len(products)
    for index, path in enumerate(products, start=1):
        _notify_progress(on_progress, index, total, f"Opening {path.name}")
        try:
            prepared.append(_safe_source(path, swath=swath, pol=pol))
        except Exception as exc:
            skipped.append((path.name, str(exc) or exc.__class__.__name__))
    dated: list[_SlcSource] = []
    for source in prepared:
        if source.date is None:
            skipped.append((source.label, "manifest start time has no calendar date"))
            continue
        dated.append(source)
    ordered = _order_sources(dated)
    unique_dates: list[_SlcSource] = []
    for source in ordered:
        if any(source.date == earlier.date for earlier in unique_dates):
            skipped.append((source.label, f"start date {source.date} is already in the stack"))
            continue
        unique_dates.append(source)
    read: list[tuple[_SlcSource, np.ndarray]] = []
    readable = len(unique_dates)
    for index, source in enumerate(unique_dates, start=1):
        _notify_progress(on_progress, index, readable, f"Reading {source.label}")
        try:
            plane = _read_source(source, row, col, height, width)
        except Exception as exc:
            skipped.append((source.label, str(exc) or exc.__class__.__name__))
            continue
        read.append((source, np.asarray(plane, dtype=np.complex64)))
    if not read:
        detail = "; ".join(f"{name}: {reason}" for name, reason in skipped[:8]) or "no products"
        raise ValueError(f"no SAFE product contributed {band}. {detail}")
    chosen = _majority_shape([plane.shape for _source, plane in read])
    loaded: list[_SlcSource] = []
    planes: list[np.ndarray] = []
    for source, plane in read:
        if plane.shape != chosen:
            skipped.append(
                (
                    source.label,
                    f"crop is {plane.shape[0]}×{plane.shape[1]}, stack is {chosen[0]}×{chosen[1]}",
                )
            )
            continue
        loaded.append(source)
        planes.append(plane)
    data = np.stack(planes, axis=0).astype(np.complex64)
    stack = SLCStack(
        data=data,
        dates=_dates_for_sources(loaded, count=data.shape[0]),
        wavelength_m=_stack_wavelength(loaded, wavelength_m),
        sensor="Sentinel-1 SLC",
        description=_safe_stack_description(
            loaded,
            skipped_count=len(skipped),
            band=band,
            row=row,
            col=col,
            height=int(data.shape[1]),
            width=int(data.shape[2]),
        ),
    )
    return SafeStackLoad(stack=stack, skipped=tuple(skipped))


def discover_slc_rasters(
    paths: list[Path],
    *,
    swath: str = DEFAULT_SWATH,
    pol: str = DEFAULT_POLARISATION,
) -> list[Path]:
    """Expand files and SAFE products into one path per date.

    A SAFE folder or ``manifest.safe`` contributes the selected measurement
    GeoTIFF. A zip contributes the zip path; the measurement is chosen again
    when the crop is read. ``swath`` and ``pol`` default to IW1 and VV.
    """
    sources = _expand_sources([Path(path) for path in paths], swath=swath, pol=pol)
    found: list[Path] = []
    for source in sources:
        if source.kind == "safe" and source.product is not None and not source.product.is_zip and source.measurement:
            found.append(source.product.root / source.measurement.href)
        else:
            found.append(source.path)
    if not found:
        raise ValueError("no SLC rasters found")
    return found


def _expand_sources(paths: list[Path], *, swath: str, pol: str) -> list[_SlcSource]:
    found: list[_SlcSource] = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        if _is_safe_container(path):
            found.append(_safe_source(path, swath=swath, pol=pol))
            continue
        if path.is_file() and path.suffix.lower() in {".npy", ".tif", ".tiff"}:
            found.append(
                _SlcSource(
                    path=path,
                    kind="raster",
                    date=date_from_text(path.name),
                    wavelength_m=None,
                    label=path.name,
                )
            )
            continue
        if path.is_dir():
            found.append(_legacy_folder_source(path, swath=swath, pol=pol))
            continue
        raise ValueError(f"{path.name} is not a .npy, .tif, .tiff, SAFE folder, manifest.safe, or .zip SLC")
    if not found:
        raise ValueError("no SLC rasters found")
    return found


def _is_safe_container(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() == ".zip" or path.name.lower() == "manifest.safe"
    return path.is_dir() and (path / "manifest.safe").is_file()


def _direct_safe_products(directory: Path) -> list[Path]:
    """Sentinel-1 zips and SAFE folders that sit directly in ``directory``."""
    children: list[Path] = []
    for item in sorted(directory.iterdir(), key=lambda path: path.name.lower()):
        name = item.name.lower()
        if item.is_file() and item.suffix.lower() == ".zip" and name.startswith("s1"):
            children.append(item)
        elif item.is_dir() and (name.endswith(".safe") or (item / "manifest.safe").is_file()):
            children.append(item)
        elif item.is_file() and name == "manifest.safe":
            children.append(item)
    return children


def _notify_progress(
    on_progress: Callable[[int, int, str], None] | None,
    index: int,
    total: int,
    message: str,
) -> None:
    if on_progress is not None and total > 0:
        on_progress(index, total, message)


def _majority_shape(shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Most common crop shape. A tie keeps the earliest one in the list."""
    counts = Counter(shapes)
    best = max(counts.values())
    for shape in shapes:
        if counts[shape] == best:
            return shape
    return shapes[0]


def _safe_stack_description(
    sources: list[_SlcSource],
    *,
    skipped_count: int,
    band: str,
    row: int,
    col: int,
    height: int,
    width: int,
) -> str:
    labels = [source.label for source in sources]
    names = ", ".join(labels[:4])
    if len(labels) > 4:
        names += f", +{len(labels) - 4} more"
    skipped = f" Skipped {skipped_count} product(s) with no matching {band} window." if skipped_count else ""
    return (
        f"SAFE SLC stack, {band}, rows {row}:{row + height}, cols {col}:{col + width}, "
        f"{len(sources)} dates ({names}).{skipped} "
        "Same pixel window on each product. Not coregistered, deburst, or calibrated."
    )


def _safe_source(path: Path, *, swath: str, pol: str) -> _SlcSource:
    product = resolve_safe(path)
    measurement = select_measurement(product, swath, pol)
    wavelength = None
    try:
        read_annotation(product, measurement)
    except ValueError:
        wavelength = None
    else:
        if measurement.radar_frequency:
            wavelength = SPEED_OF_LIGHT_M_S / float(measurement.radar_frequency)
    date = date_from_text(product.start_utc) or date_from_text(product.product_id)
    return _SlcSource(
        path=path,
        kind="safe",
        date=date,
        wavelength_m=wavelength,
        label=product.product_id,
        band_note=f"{measurement.swath} {measurement.polarisation}",
        product=product,
        measurement=measurement,
    )


def _legacy_folder_source(path: Path, *, swath: str, pol: str) -> _SlcSource:
    """A folder of measurement TIFFs that has no manifest.safe."""
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
    chosen = _select_named_measurement(measurements, swath=swath, pol=pol, folder=path)
    return _SlcSource(
        path=chosen,
        kind="raster",
        date=date_from_text(chosen.name),
        wavelength_m=None,
        label=chosen.name,
        band_note=f"{swath.strip().upper()} {pol.strip().upper()}" if chosen.suffix.lower() != ".npy" else None,
    )


def _select_named_measurement(paths: list[Path], *, swath: str, pol: str, folder: Path) -> Path:
    """Choose IW1/VV (or the requested band) by filename tokens, not a substring."""
    swath_key = swath.strip().upper()
    pol_key = pol.strip().upper()
    parsed = [(item, band_from_measurement_name(item.name)) for item in paths]
    known = [(item, band) for item, band in parsed if band is not None]
    if known:
        hits = [item for item, band in known if band == (swath_key, pol_key)]
        if not hits:
            available = ", ".join(sorted({f"{band[0]}/{band[1]}" for _, band in known}))
            raise ValueError(f"no {swath_key} {pol_key} measurement under {folder.name}. Available: {available}")
        return sorted(hits, key=lambda item: item.name)[0]
    if len(paths) == 1:
        return paths[0]
    raise ValueError(
        f"no {swath_key} {pol_key} measurement under {folder.name}. "
        "Name measurement files as s1x-iw1-slc-vv-… or pass swath= and pol=."
    )


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


def _read_source(source: _SlcSource, row: int, col: int, height: int, width: int) -> np.ndarray:
    if source.kind == "safe" and source.product is not None and source.measurement is not None:
        return read_window(source.product, source.measurement, row, col, height, width)
    return _read_plane(source.path, row, col, height, width)


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
            "Reading a measurement TIFF needs tifffile. "
            "Install the project with pip install -e . or run run_windows.bat."
        ) from exc
    with tifffile.TiffFile(path) as tiff:
        return read_open_tiff_window(tiff, row, col, height, width, name=path.name)


def read_open_tiff_window(
    tiff,
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    name: str = "TIFF",
    y_step: int = 1,
    x_step: int = 1,
) -> np.ndarray:
    """Read a complex crop from an already open TIFF, including a zip offset.

    ``y_step`` and ``x_step`` subsample the source region. A Sentinel-1 SLC
    sample packed in one int32 uses the low 16 bits as I and ``>> 16`` as Q.
    """
    if y_step < 1 or x_step < 1:
        raise ValueError("subsampling steps must be at least 1")
    page = tiff.pages[0]
    spatial = _spatial_shape(page.shape)
    row, col, height, width = _fit_window(spatial, row, col, height, width)
    selector = _window_index(page.shape, row, col, height, width, y_step=y_step, x_step=x_step)
    try:
        import zarr

        store = tiff.aszarr()
        array = zarr.open(store, mode="r")
        raw = np.asarray(array[selector])
    except Exception:
        nbytes = int(np.prod(page.shape)) * max(np.dtype(page.dtype).itemsize, 1)
        if nbytes > 512 * 1024 * 1024:
            raise RuntimeError(
                f"{name} is about {nbytes / 1e9:.1f} GB. "
                "Install the project (tifffile and zarr) so the crop is read in tiles: "
                "pip install -e . or run run_windows.bat."
            ) from None
        raw = np.asarray(page.asarray())[selector]
    return to_complex64(raw)


def to_complex64(array: np.ndarray) -> np.ndarray:
    """Interpret a raster as complex64 SLC samples.

    A 2-D int32 (or uint32) raster is Sentinel-1 SLC packing: I is the low 16
    bits and Q is ``value >> 16``, both signed.
    """
    if np.iscomplexobj(array):
        return np.asarray(array, dtype=np.complex64)
    if array.dtype.names and {"real", "imag"} <= set(array.dtype.names):
        return (array["real"].astype(np.float32) + 1j * array["imag"].astype(np.float32)).astype(np.complex64)
    if array.ndim == 3 and array.shape[-1] == 2:
        return (array[..., 0].astype(np.float32) + 1j * array[..., 1].astype(np.float32)).astype(np.complex64)
    if array.ndim == 3 and array.shape[0] == 2:
        return (array[0].astype(np.float32) + 1j * array[1].astype(np.float32)).astype(np.complex64)
    if array.ndim == 2 and array.dtype.kind in "iu" and array.dtype.itemsize == 4:
        return _complex_from_packed_slc(array)
    raise ValueError(
        "TIFF is not complex SLC data. Expected a complex raster, packed int32 I/Q, "
        f"or integer I/Q bands, got shape {array.shape} dtype {array.dtype}."
    )


def _complex_from_packed_slc(array: np.ndarray) -> np.ndarray:
    """Unpack SLC samples stored as int32 (I in the low 16 bits, Q in ``>> 16``)."""
    packed = np.asarray(array)
    if packed.dtype.kind == "u":
        packed = packed.view(np.int32)
    real = packed.astype(np.int16)
    imag = (packed >> np.int32(16)).astype(np.int16)
    return (real.astype(np.float32) + 1j * imag.astype(np.float32)).astype(np.complex64)


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


def _window_index(
    shape: tuple[int, ...],
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    y_step: int = 1,
    x_step: int = 1,
):
    rows = slice(row, row + height, y_step)
    cols = slice(col, col + width, x_step)
    if len(shape) == 2 or (len(shape) == 3 and shape[-1] == 2):
        if len(shape) == 2:
            return (rows, cols)
        return (rows, cols, slice(None))
    if len(shape) == 3 and shape[0] == 2:
        return (slice(None), rows, cols)
    raise ValueError(f"unsupported TIFF shape {shape}")


def _crop_numpy(array: np.ndarray, row: int, col: int, height: int, width: int) -> np.ndarray:
    if array.ndim == 2:
        row, col, height, width = _fit_window(array.shape, row, col, height, width)
        return np.asarray(array[row : row + height, col : col + width])
    if array.ndim == 3:
        row, col, height, width = _fit_window(array.shape[1:], row, col, height, width)
        return np.asarray(array[:, row : row + height, col : col + width])
    raise ValueError(f"complex .npy must be 2-D or 3-D, got ndim {array.ndim}")


def _order_sources(sources: list[_SlcSource]) -> list[_SlcSource]:
    if any(source.date is None for source in sources):
        return list(sources)
    order = np.argsort(np.array([source.date for source in sources], dtype="datetime64[D]"), kind="mergesort")
    return [sources[int(index)] for index in order]


def _dates_for_sources(sources: list[_SlcSource], *, count: int) -> np.ndarray:
    if count != len(sources):
        # A single 3-D cube has one path and many dates.
        parsed = sources[0].date if len(sources) == 1 else None
        if parsed is not None and count > 1:
            return parsed + np.arange(count, dtype=np.int64) * np.timedelta64(12, "D")
        return default_dates(count)
    if any(source.date is None for source in sources):
        return default_dates(count)
    dates = np.array([source.date for source in sources], dtype="datetime64[D]")
    if np.any(np.diff(dates) <= np.timedelta64(0, "D")):
        raise ValueError("imported SLC dates are not strictly increasing; rename files so each date differs")
    return dates


def _stack_wavelength(sources: list[_SlcSource], wavelength_m: float | None) -> float:
    if wavelength_m is not None:
        return float(wavelength_m)
    for source in sources:
        if source.wavelength_m:
            return float(source.wavelength_m)
    return C_BAND_WAVELENGTH_M


def _looks_like_sentinel(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("s1") or "slc" in lowered or "safe" in lowered


def _import_description(
    labels: list[str],
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    band: str | None = None,
) -> str:
    names = ", ".join(labels[:4])
    if len(labels) > 4:
        names += f", +{len(labels) - 4} more"
    band_text = f" Band {band}." if band else ""
    return (
        f"Imported SLC crop rows {row}:{row + height}, cols {col}:{col + width} "
        f"from {names}.{band_text} Not geocoded."
    )
