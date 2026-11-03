"""Open a Sentinel-1 SAFE zip for inspection.

The primary input is the product ``.zip``. The user does not unzip it. A
``.SAFE`` folder or ``manifest.safe`` is accepted as an extra. The zip name
must start with ``s1``. ETAD (``_eta_``) and COG products are refused.

Open reads ``manifest.safe`` for product metadata and the XML files directly
under ``annotation/``. Measurement files are the GeoTIFFs directly under
``measurement/``, paired with annotations by basename. The manifest
``dataObjectSection`` is not used to find those files. Band height and width
come from annotation ``numberOfLines`` and ``numberOfSamples``. TIFF headers
are not read at open.

Pixels are read later, in a window with optional subsampling. Inside a zip,
uncompressed (``ZIP_STORED``) measurement TIFFs are read by seeking to the
entry's data offset. They are not extracted. A compressed entry is the slow
path: that one TIFF is extracted. The quick-look PNG is the only other zip
member read, and it is read into memory.

This is an independent reader of the public SAFE layout. It is not a copy of
ESA SNAP. Calibration, orbits, and TOPS deburst are out of scope.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

DEFAULT_SWATH = "IW1"
DEFAULT_POLARISATION = "VV"

_PRODUCT_RE = re.compile(
    r"^(S1[ABCD])_"
    r"(IW|EW|WV|SM|S[1-6])_"
    r"(SLC|GRD|OCN|RAW)([FHM_])_"
    r"([12][A-Z](?:SH|SV|DH|DV|HH|HV|VV|VH))_"
    r"(\d{8}T\d{6})_"
    r"(\d{8}T\d{6})_"
    r"(\d{6})_"
    r"([0-9A-F]{6})_"
    r"([0-9A-F]{4})$",
    re.IGNORECASE,
)
_MEAS_BAND = re.compile(
    r"s1[a-d]-([a-z0-9]+)-[a-z0-9]+-([vh]h|[vh]v)-",
    re.IGNORECASE,
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class SafeBurst:
    """One TOPS burst from an annotation ``burstList``."""

    index: int
    azimuth_time: str
    sensing_time: str | None = None
    byte_offset: int | None = None


@dataclass
class SafeMeasurement:
    """One measurement TIFF and the annotation that describes it."""

    href: str
    annotation_href: str | None
    swath: str
    polarisation: str
    lines: int | None = None
    samples: int | None = None
    bursts: list[SafeBurst] = field(default_factory=list)
    azimuth_time_interval: float | None = None
    range_pixel_spacing: float | None = None
    radar_frequency: float | None = None
    slant_range_time: float | None = None
    incidence: float | None = None
    annotation_ready: bool = False


@dataclass
class SafeProduct:
    """Manifest metadata for one Sentinel-1 SAFE product."""

    root: Path
    is_zip: bool
    product_id: str
    platform: str
    mode: str
    product_type: str
    level: str
    polarisations: tuple[str, ...]
    swaths: tuple[str, ...]
    start_utc: str
    stop_utc: str
    absolute_orbit: int | None
    relative_orbit: int | None
    orbit_pass: str
    datatake_id: str | None
    footprint: tuple[tuple[float, float], ...]
    measurements: tuple[SafeMeasurement, ...]
    quicklook_href: str | None
    archive_prefix: str = ""


@dataclass
class _Located:
    root: Path
    is_zip: bool
    manifest_bytes: bytes
    archive_prefix: str
    product_id: str


class MeasurementHandle:
    """Open measurement TIFF. Use as a context manager."""

    def __init__(self, tiff, name: str) -> None:
        self.tiff = tiff
        self.name = name

    def __enter__(self) -> MeasurementHandle:
        return self

    def __exit__(self, *args) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self.tiff.close()

    @property
    def shape(self) -> tuple[int, int]:
        from ai_insar.sentinel import _spatial_shape

        return _spatial_shape(tuple(self.tiff.pages[0].shape))

    def read_window(
        self,
        row: int,
        col: int,
        height: int,
        width: int,
        *,
        y_step: int = 1,
        x_step: int = 1,
    ) -> np.ndarray:
        from ai_insar.sentinel import read_open_tiff_window

        return read_open_tiff_window(
            self.tiff,
            row,
            col,
            height,
            width,
            name=self.name,
            y_step=y_step,
            x_step=x_step,
        )


def resolve_safe(path: Path | str) -> SafeProduct:
    """Open a Sentinel-1 SAFE ``.zip``, or a ``.SAFE`` folder / ``manifest.safe``."""
    return read_manifest(path)


def read_manifest(src: Path | str | _Located) -> SafeProduct:
    """Parse ``manifest.safe`` into a :class:`SafeProduct`.

    ``src`` is a product path or the location already resolved from one.
    Tag matching uses local names, ignoring namespace URIs.
    """
    located = src if isinstance(src, _Located) else _locate(Path(src))
    root = ET.fromstring(located.manifest_bytes)
    platform_letter, mode, swaths = _platform_fields(root)
    product_type, polarisations, datatake = _product_fields(root)
    absolute, relative, orbit_pass = _orbit_fields(root)
    start, stop = _acquisition_times(root)
    footprint = _footprint(root)
    product_id = located.product_id
    _cross_check(
        product_id,
        platform_letter=platform_letter,
        mode=mode,
        product_type=product_type,
        absolute_orbit=absolute,
        datatake_id=datatake,
    )
    names = _entry_names(located)
    level = _detect_level(names, is_zip=located.is_zip)
    if level == "L2":
        raise ValueError(
            "Level-2 Sentinel-1 products (.nc) are not SLC measurements. Open the SLC SAFE zip."
        )
    if level == "L0":
        raise ValueError(
            "Level-0 Sentinel-1 products (.dat) are not SLC measurements. Open the SLC SAFE zip."
        )
    measurements, quicklook = _discover_measurements(located, names)
    if level != "L1" or not measurements:
        raise ValueError(
            f"incomplete SAFE product {product_id}: no measurement TIFF in measurement/"
        )
    product = SafeProduct(
        root=located.root,
        is_zip=located.is_zip,
        product_id=product_id,
        platform=f"S1{platform_letter}",
        mode=mode,
        product_type=product_type,
        level=level,
        polarisations=polarisations,
        swaths=swaths,
        start_utc=start,
        stop_utc=stop,
        absolute_orbit=absolute,
        relative_orbit=relative,
        orbit_pass=orbit_pass,
        datatake_id=datatake,
        footprint=footprint,
        measurements=tuple(measurements),
        quicklook_href=quicklook,
        archive_prefix=located.archive_prefix,
    )
    for measurement in product.measurements:
        if measurement.annotation_href and _href_exists(product, measurement.annotation_href):
            read_annotation(product, measurement)
    return product


def read_annotation(src: SafeProduct | Path | str, meas: SafeMeasurement) -> SafeMeasurement:
    """Fill image size, bursts, spacing, and radar frequency on ``meas``."""
    if meas.annotation_ready:
        return meas
    product = src if isinstance(src, SafeProduct) else resolve_safe(src)
    if not meas.annotation_href:
        raise ValueError(f"{meas.href} has no annotation XML in the manifest")
    if not _href_exists(product, meas.annotation_href):
        raise ValueError(f"annotation {meas.annotation_href} is not in {product.product_id}")
    root = ET.fromstring(_read_href(product, meas.annotation_href))
    header = _first(root, "adsHeader")
    if header is not None:
        swath = _child_text(header, "swath")
        polarisation = _child_text(header, "polarisation")
        if swath:
            meas.swath = swath.strip().upper()
        if polarisation:
            meas.polarisation = polarisation.strip().upper()
    image = _first(root, "imageInformation")
    if image is not None:
        meas.lines = _optional_int(_child_text(image, "numberOfLines"))
        meas.samples = _optional_int(_child_text(image, "numberOfSamples"))
        meas.azimuth_time_interval = _optional_float(_child_text(image, "azimuthTimeInterval"))
        meas.range_pixel_spacing = _optional_float(_child_text(image, "rangePixelSpacing"))
        meas.slant_range_time = _optional_float(_child_text(image, "slantRangeTime"))
        meas.incidence = _optional_float(_child_text(image, "incidenceAngleMidSwath"))
    information = _first(root, "productInformation")
    if information is not None:
        meas.radar_frequency = _optional_float(_child_text(information, "radarFrequency"))
    meas.bursts = _bursts(root)
    meas.annotation_ready = True
    return meas


def open_measurement(product: SafeProduct, meas: SafeMeasurement) -> MeasurementHandle:
    """Open one measurement TIFF, using a zip offset when the entry is stored."""
    _require_tifffile()
    import tifffile

    path, offset, size = measurement_storage(product, meas)
    if product.is_zip and offset is None:
        path = _cache_zip_member(product.root, _member_name(product, meas.href))
        tiff = tifffile.TiffFile(path)
    elif offset is None:
        tiff = tifffile.TiffFile(path)
    else:
        try:
            tiff = tifffile.TiffFile(path, offset=offset, size=size)
        except Exception:
            path = _cache_zip_member(product.root, _member_name(product, meas.href))
            tiff = tifffile.TiffFile(path)
    return MeasurementHandle(tiff, Path(meas.href).name)


def read_window(
    product: SafeProduct,
    meas: SafeMeasurement,
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    y_step: int = 1,
    x_step: int = 1,
) -> np.ndarray:
    """Read a ``complex64`` window. ``y_step`` and ``x_step`` subsample the region."""
    with open_measurement(product, meas) as handle:
        return handle.read_window(row, col, height, width, y_step=y_step, x_step=x_step)


def preview_amplitude(
    product: SafeProduct,
    measurement: SafeMeasurement | None = None,
    *,
    max_side: int = 512,
) -> np.ndarray:
    """Decimated amplitude. The quick-look PNG is used when no band is named."""
    if measurement is None and product.quicklook_href and _href_exists(product, product.quicklook_href):
        decoded = _decode_png(_read_href(product, product.quicklook_href))
        if decoded.ndim == 3:
            decoded = decoded.astype(np.float32).mean(axis=2)
        return np.asarray(decoded, dtype=np.float32)
    meas = measurement if measurement is not None else select_measurement(product)
    with open_measurement(product, meas) as handle:
        height, width = handle.shape
        step = max(1, int(math.ceil(max(height, width) / max(1, int(max_side)))))
        complex_image = handle.read_window(0, 0, height, width, y_step=step, x_step=step)
    return np.abs(complex_image).astype(np.float32)


def select_measurement(
    product: SafeProduct,
    swath: str = DEFAULT_SWATH,
    pol: str = DEFAULT_POLARISATION,
    *,
    present_only: bool = True,
) -> SafeMeasurement:
    """Pick one swath and polarisation. Defaults are IW1 and VV, exactly."""
    swath_key = swath.strip().upper()
    pol_key = pol.strip().upper()
    matches = [
        item
        for item in product.measurements
        if item.swath.upper() == swath_key and item.polarisation.upper() == pol_key
    ]
    if present_only:
        present = [item for item in matches if measurement_is_present(product, item)]
        if matches and not present:
            raise ValueError(
                f"{swath_key} {pol_key} is listed in {product.product_id} "
                "but the measurement TIFF is not in the product"
            )
        matches = present
    if not matches:
        available = ", ".join(_measurement_label(product, item) for item in product.measurements) or "none"
        raise ValueError(f"no {swath_key} {pol_key} measurement in {product.product_id}. Available: {available}")
    return matches[0]


def measurement_is_present(product: SafeProduct, meas: SafeMeasurement) -> bool:
    """True when the measurement TIFF named by the manifest is in the product."""
    return _href_exists(product, meas.href)


def measurement_storage(product: SafeProduct, meas: SafeMeasurement) -> tuple[Path, int | None, int | None]:
    """Return ``(path, offset, size)`` for the TIFF.

    ``offset`` is set for an uncompressed zip entry so the TIFF can be read
    without extracting it. A deflated entry returns the zip path and an empty
    offset; :func:`open_measurement` then extracts that one file.
    """
    if not product.is_zip:
        path = product.root / _normalize_href(meas.href)
        if not path.is_file():
            raise ValueError(f"measurement {meas.href} is not in {product.product_id}")
        return path, None, None
    member = _member_name(product, meas.href)
    located = _stored_offset(product.root, member)
    if located is None:
        if not _zip_has(product.root, member):
            raise ValueError(f"measurement {meas.href} is not in {product.product_id}")
        return product.root, None, None
    offset, size = located
    return product.root, offset, size


def format_product_status(product: SafeProduct) -> str:
    """One status-bar line, in the spirit of a SNAP product header."""
    pols = "+".join(product.polarisations) if product.polarisations else "none"
    start = product.start_utc or "unknown time"
    if product.absolute_orbit is None:
        orbit = "Orbit unknown"
    else:
        orbit = f"Orbit {product.absolute_orbit}"
        if product.relative_orbit is not None:
            orbit += f" ({product.relative_orbit})"
    return (
        f"{product.platform} {product.mode} {product.product_type} | {pols} | "
        f"{_format_swaths(product.swaths)} | {start} UTC | {orbit}"
    )


def band_from_measurement_name(name: str) -> tuple[str, str] | None:
    """Return ``(swath, polarisation)`` from a measurement filename, or ``None``."""
    match = _MEAS_BAND.search(Path(name).name)
    if not match:
        return None
    return match.group(1).upper(), match.group(2).upper()


def _locate(path: Path) -> _Located:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    _reject_etad(path.name)
    _reject_cog(path.name)
    if path.is_file() and path.name.lower() == "manifest.safe":
        root = path.parent.resolve()
        product_id = _require_product_id(root.name)
        return _Located(root, False, path.read_bytes(), "", product_id)
    if path.is_dir():
        manifest = path / "manifest.safe"
        if not manifest.is_file():
            raise ValueError(f"{path} is not a Sentinel-1 SAFE product (no manifest.safe)")
        root = path.resolve()
        product_id = _require_product_id(root.name)
        return _Located(root, False, manifest.read_bytes(), "", product_id)
    if path.is_file() and path.suffix.lower() == ".zip":
        return _locate_zip(path.resolve())
    raise ValueError(
        f"{path.name} is not a Sentinel-1 SAFE folder, manifest.safe, or .zip product"
    )


def _locate_zip(path: Path) -> _Located:
    _reject_cog(path.name)
    if not path.name.lower().startswith("s1"):
        raise ValueError(f"{path.name} is not a Sentinel-1 SAFE zip (the file name must start with s1)")
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
            manifests = [name for name in names if Path(name).name.lower() == "manifest.safe"]
            if len(manifests) != 1:
                raise ValueError(
                    f"expected exactly one manifest.safe in {path.name}, found {len(manifests)}"
                )
            level = _detect_level(names, is_zip=True)
            if level == "L2":
                raise ValueError(
                    "Level-2 Sentinel-1 products (.nc) are not SLC measurements. Open the SLC SAFE zip."
                )
            if level == "L0":
                raise ValueError(
                    "Level-0 Sentinel-1 products (.dat) are not SLC measurements. Open the SLC SAFE zip."
                )
            # Prefix is the directory of the manifest entry (S1A_….SAFE/).
            member = manifests[0]
            parent = Path(member).parent.as_posix()
            prefix = "" if parent in {"", "."} else parent + "/"
            folder_name = Path(prefix[:-1]).name if prefix else path.stem
            _reject_etad(folder_name)
            _reject_cog(folder_name)
            product_id = _require_product_id(folder_name, fallback=path.stem)
            payload = archive.read(member)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name} is not a zip file") from exc
    return _Located(path, True, payload, prefix, product_id)


def _entry_names(located: _Located) -> list[str]:
    if located.is_zip:
        with zipfile.ZipFile(located.root) as archive:
            return [info.filename for info in archive.infolist() if not info.is_dir()]
    return [path.relative_to(located.root).as_posix() for path in located.root.rglob("*") if path.is_file()]


def _detect_level(names: list[str], *, is_zip: bool) -> str:
    """L1 from annotation XML (folder) or a TIFF (zip). L2 is ``.nc``. L0 is ``.dat``."""
    lower = [name.lower() for name in names]
    if any(name.endswith(".nc") for name in lower):
        return "L2"
    if any(name.endswith(".dat") for name in lower):
        return "L0"
    if is_zip and any(name.endswith((".tif", ".tiff")) for name in lower):
        return "L1"
    if any(
        _is_direct_child(name, "annotation/") and name.endswith(".xml")
        for name in lower
    ):
        return "L1"
    if any(name.endswith((".tif", ".tiff")) for name in lower):
        return "L1"
    return ""


def _discover_measurements(
    located: _Located,
    names: list[str],
) -> tuple[list[SafeMeasurement], str | None]:
    """Pair ``annotation/*.xml`` with ``measurement/*.tiff`` by basename.

    Subfolders such as ``annotation/calibration/`` are ignored. The manifest
    data-object list is not consulted.
    """
    prefix = located.archive_prefix
    annotations: dict[str, str] = {}
    measurements: list[str] = []
    quicklook: str | None = None
    preview = f"{prefix}preview/quick-look.png"
    for name in names:
        relative = _relative_to_prefix(name, prefix)
        if relative is None:
            continue
        if _is_direct_child(relative, "annotation/") and relative.lower().endswith(".xml"):
            annotations[Path(relative).stem.lower()] = relative
        elif _is_direct_child(relative, "measurement/") and relative.lower().endswith((".tif", ".tiff")):
            measurements.append(relative)
        elif relative.lower() == "preview/quick-look.png" or name.lower() == preview.lower():
            quicklook = relative
    found: list[SafeMeasurement] = []
    for href in sorted(measurements):
        swath, polarisation = band_from_measurement_name(href) or ("", "")
        found.append(
            SafeMeasurement(
                href=href,
                annotation_href=annotations.get(Path(href).stem.lower()),
                swath=swath,
                polarisation=polarisation,
            )
        )
    return found, quicklook


def _relative_to_prefix(name: str, prefix: str) -> str | None:
    if not prefix:
        return name
    if name.startswith(prefix):
        return name[len(prefix) :]
    return None


def _is_direct_child(name: str, directory: str) -> bool:
    lowered = name.lower()
    folder = directory.lower()
    if not lowered.startswith(folder):
        return False
    rest = lowered[len(folder) :]
    return bool(rest) and "/" not in rest


def _platform_fields(root: ET.Element) -> tuple[str, str, tuple[str, ...]]:
    platform = _metadata_object(root, "platform")
    if platform is None:
        raise ValueError("manifest has no platform metadata")
    family = ""
    letter = ""
    for element in _children_deep(platform, "platform"):
        # Direct children only: the instrument block repeats familyName.
        family = _direct_text(element, "familyName") or family
        letter = _direct_text(element, "number") or letter
        if family and letter:
            break
    if "SENTINEL-1" not in family.upper():
        raise ValueError(f"manifest platform is {family or 'missing'}, not Sentinel-1")
    letter = letter.strip().upper()
    if letter not in {"A", "B", "C", "D"}:
        raise ValueError(f"manifest platform number {letter or 'missing'} is not A, B, C, or D")
    mode = ""
    swaths: list[str] = []
    for instrument_mode in _children_deep(platform, "instrumentMode"):
        mode = mode or (_child_text(instrument_mode, "mode") or "")
        for swath in _children_deep(instrument_mode, "swath"):
            text = (swath.text or "").strip().upper()
            if text and text not in swaths:
                swaths.append(text)
    if not mode:
        raise ValueError("manifest has no instrument mode")
    return letter, mode.strip().upper(), tuple(swaths)


def _product_fields(root: ET.Element) -> tuple[str, tuple[str, ...], str | None]:
    section = _metadata_object(root, "generalProductInformation")
    if section is None:
        raise ValueError("manifest has no general product information")
    product_type = ""
    datatake = None
    polarisations: list[str] = []
    for element in section.iter():
        name = _local(element.tag)
        text = (element.text or "").strip()
        if name == "productType" and text and not product_type:
            product_type = text.upper()
        elif name == "missionDataTakeID" and text and datatake is None:
            datatake = text
        elif name == "transmitterReceiverPolarisation" and text:
            pol = text.upper()
            if pol not in polarisations:
                polarisations.append(pol)
    if not product_type:
        raise ValueError("manifest has no product type")
    return product_type, tuple(polarisations), datatake


def _orbit_fields(root: ET.Element) -> tuple[int | None, int | None, str]:
    section = _metadata_object(root, "measurementOrbitReference")
    if section is None:
        return None, None, ""
    absolute = _typed_int(section, "orbitNumber")
    relative = _typed_int(section, "relativeOrbitNumber")
    orbit_pass = ""
    for element in section.iter():
        if _local(element.tag) == "pass" and (element.text or "").strip():
            orbit_pass = (element.text or "").strip().upper()
            break
    return absolute, relative, orbit_pass


def _acquisition_times(root: ET.Element) -> tuple[str, str]:
    section = _metadata_object(root, "acquisitionPeriod")
    if section is None:
        return "", ""
    start = ""
    stop = ""
    for element in section.iter():
        name = _local(element.tag)
        text = (element.text or "").strip()
        if name == "startTime" and text and not start:
            start = text
        elif name == "stopTime" and text and not stop:
            stop = text
    return start, stop


def _footprint(root: ET.Element) -> tuple[tuple[float, float], ...]:
    section = _metadata_object(root, "measurementFrameSet")
    if section is None:
        return ()
    for name in ("coordinates", "posList"):
        for element in section.iter():
            if _local(element.tag) == name and (element.text or "").strip():
                return _parse_lat_lon(element.text or "")
    return ()


def _bursts(root: ET.Element) -> list[SafeBurst]:
    bursts: list[SafeBurst] = []
    for index, element in enumerate(_children_deep(root, "burst")):
        azimuth = _child_text(element, "azimuthTime") or ""
        sensing = _child_text(element, "sensingTime")
        offset = _optional_int(_child_text(element, "byteOffset"))
        bursts.append(SafeBurst(index=index, azimuth_time=azimuth, sensing_time=sensing, byte_offset=offset))
    return bursts


def _cross_check(
    product_id: str,
    *,
    platform_letter: str,
    mode: str,
    product_type: str,
    absolute_orbit: int | None,
    datatake_id: str | None,
) -> None:
    parsed = _PRODUCT_RE.match(product_id)
    if not parsed:
        raise ValueError(f"product name does not match a Sentinel-1 product: {product_id}")
    if parsed.group(1)[-1].upper() != platform_letter.upper():
        raise ValueError(f"{product_id} platform does not match manifest Sentinel-1{platform_letter}")
    name_mode = parsed.group(2).upper()
    manifest_mode = mode.upper()
    if name_mode in {"IW", "EW", "WV", "SM"}:
        if name_mode != manifest_mode:
            raise ValueError(f"{product_id} mode {name_mode} does not match manifest {manifest_mode}")
    elif manifest_mode not in {"SM", name_mode}:
        raise ValueError(f"{product_id} mode {name_mode} does not match manifest {manifest_mode}")
    if parsed.group(3).upper() != product_type.upper():
        raise ValueError(f"{product_id} type {parsed.group(3)} does not match manifest {product_type}")
    if absolute_orbit is not None and int(parsed.group(8)) != int(absolute_orbit):
        raise ValueError(f"{product_id} orbit does not match manifest orbit {absolute_orbit}")
    if datatake_id:
        if int(parsed.group(9), 16) != int(datatake_id):
            raise ValueError(f"{product_id} datatake does not match manifest {datatake_id}")


def _parse_lat_lon(text: str) -> tuple[tuple[float, float], ...]:
    numbers = [float(item) for item in text.replace(",", " ").split()]
    if len(numbers) < 2 or len(numbers) % 2:
        return ()
    return tuple((numbers[index], numbers[index + 1]) for index in range(0, len(numbers), 2))


def _format_swaths(swaths: tuple[str, ...]) -> str:
    if not swaths:
        return "no swath"
    if all(re.fullmatch(r"IW[1-3]", item) for item in swaths):
        numbers = [int(item[2]) for item in swaths]
        ordered = sorted(numbers)
        if ordered == list(range(ordered[0], ordered[0] + len(ordered))) and ordered == numbers:
            if len(ordered) == 1:
                return f"IW{ordered[0]}"
            return f"IW{ordered[0]}–{ordered[-1]}"
    return ", ".join(swaths)


def _measurement_label(product: SafeProduct, meas: SafeMeasurement) -> str:
    suffix = "" if measurement_is_present(product, meas) else " (missing file)"
    return f"{meas.swath or '?'}/{meas.polarisation or '?'}{suffix}"


def _reject_etad(name: str) -> None:
    if "_ETA_" in _name_token(name):
        raise ValueError(
            "ETAD products (_ETA_) are not opened. Choose the Sentinel-1 SLC SAFE zip."
        )


def _reject_cog(name: str) -> None:
    if "_COG" in _name_token(name):
        raise ValueError(
            "Sentinel-1 COG products are not opened. Choose the SLC SAFE zip."
        )


def _name_token(name: str) -> str:
    stem = Path(name).name
    upper = stem.upper()
    for suffix in (".ZIP", ".SAFE"):
        if upper.endswith(suffix):
            upper = upper[: -len(suffix)]
    return upper


def _require_product_id(name: str, fallback: str | None = None) -> str:
    for candidate in (name, fallback or ""):
        stem = _product_stem(candidate)
        if stem and _PRODUCT_RE.match(stem):
            _reject_etad(stem)
            return stem.upper()
    shown = _product_stem(name) or name
    raise ValueError(f"product name does not match a Sentinel-1 product: {shown}")


def _product_stem(name: str) -> str:
    stem = Path(name).name
    upper = stem.upper()
    if upper.endswith(".ZIP"):
        stem = stem[:-4]
    if stem.upper().endswith(".SAFE"):
        stem = stem[:-5]
    return stem


def _href_exists(product: SafeProduct, href: str) -> bool:
    if product.is_zip:
        return _zip_has(product.root, _member_name(product, href))
    return (product.root / _normalize_href(href)).is_file()


def _read_href(product: SafeProduct, href: str) -> bytes:
    if product.is_zip:
        member = _member_name(product, href)
        with zipfile.ZipFile(product.root) as archive:
            return archive.read(member)
    return (product.root / _normalize_href(href)).read_bytes()


def _member_name(product: SafeProduct, href: str) -> str:
    return f"{product.archive_prefix}{_normalize_href(href)}"


def _normalize_href(href: str) -> str:
    text = href.replace("\\", "/").split("?", 1)[0]
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _zip_has(path: Path, member: str) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            archive.getinfo(member)
        return True
    except (KeyError, zipfile.BadZipFile):
        return False


def _stored_offset(path: Path, member: str) -> tuple[int, int] | None:
    """Byte offset and size of an uncompressed zip entry, or ``None``."""
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            return None
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        handle = archive.fp
        handle.seek(info.header_offset)
        header = handle.read(30)
        if header[:4] != b"PK\x03\x04":
            return None
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        offset = int(info.header_offset) + 30 + name_len + extra_len
        return offset, int(info.file_size)


def _cache_zip_member(path: Path, member: str) -> Path:
    """Extract one zip member. Used when the TIFF is not stored uncompressed."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        digest = hashlib.sha256(
            f"{path.resolve()}|{member}|{info.file_size}|{info.CRC}".encode()
        ).hexdigest()[:24]
        directory = Path(tempfile.gettempdir()) / "ai-insar-safe" / digest
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / Path(member).name
        if dest.is_file() and dest.stat().st_size == info.file_size:
            return dest
        temporary = dest.with_name(dest.name + ".part")
        with archive.open(info, "r") as source, open(temporary, "wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(dest)
        return dest


def _decode_png(data: bytes) -> np.ndarray:
    """Decode an 8-bit gray, RGB, or RGBA PNG. Interlaced PNG is refused."""
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("quick-look is not a PNG")
    pos = 8
    width = height = color_type = None
    idat: list[bytes] = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression or filter_method or interlace or bit_depth != 8 or color_type not in {0, 2, 6}:
                raise ValueError("quick-look PNG is not 8-bit gray or RGB")
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    if width is None or height is None or color_type is None or not idat:
        raise ValueError("quick-look PNG is incomplete")
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    rows: list[bytes] = []
    cursor = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        if filter_type == 1:
            for index in range(stride):
                left = row[index - channels] if index >= channels else 0
                row[index] = (row[index] + left) & 255
        elif filter_type == 2:
            for index in range(stride):
                row[index] = (row[index] + previous[index]) & 255
        elif filter_type == 3:
            for index in range(stride):
                left = row[index - channels] if index >= channels else 0
                row[index] = (row[index] + ((left + previous[index]) // 2)) & 255
        elif filter_type == 4:
            for index in range(stride):
                left = row[index - channels] if index >= channels else 0
                up = previous[index]
                upper_left = previous[index - channels] if index >= channels else 0
                row[index] = (row[index] + _paeth(left, up, upper_left)) & 255
        elif filter_type != 0:
            raise ValueError(f"quick-look PNG filter {filter_type} is not supported")
        previous = row
        rows.append(bytes(row))
    array = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, channels)
    if channels == 4:
        array = array[:, :, :3]
    if channels == 1:
        return np.array(array[:, :, 0], copy=True)
    return np.array(array, copy=True)


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    distances = (abs(estimate - left), abs(estimate - up), abs(estimate - upper_left))
    if distances[0] <= distances[1] and distances[0] <= distances[2]:
        return left
    if distances[1] <= distances[2]:
        return up
    return upper_left


def _require_tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            'Reading a measurement TIFF needs tifffile: pip install "ai-insar[slc]".'
        ) from exc
    return tifffile


def _metadata_object(root: ET.Element, object_id: str) -> ET.Element | None:
    for element in root.iter():
        if _local(element.tag) == "metadataObject" and element.attrib.get("ID") == object_id:
            return element
    return None


def _first(element: ET.Element, name: str) -> ET.Element | None:
    for child in element.iter():
        if _local(child.tag) == name:
            return child
    return None


def _children_deep(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if _local(child.tag) == name]


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local(child.tag) == name:
            return child
    return None


def _direct_text(element: ET.Element, name: str) -> str | None:
    child = _direct_child(element, name)
    if child is None:
        return None
    text = (child.text or "").strip()
    return text or None


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element.iter():
        if child is element:
            continue
        if _local(child.tag) == name and (child.text or "").strip():
            return (child.text or "").strip()
    return None


def _typed_int(element: ET.Element, name: str, type_name: str = "start") -> int | None:
    for child in element.iter():
        if _local(child.tag) != name:
            continue
        if child.attrib.get("type", "start") != type_name:
            continue
        return _optional_int((child.text or "").strip())
    return None


def _optional_int(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    return int(float(text))


def _optional_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text)


def _local(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag
