"""Build a second SAFE zip from the reduced fixture, with a new acquisition day."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAFE_NAME = "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4"
FIXTURE_ZIP = ROOT / "tests" / "fixtures" / "safe" / f"{SAFE_NAME}.zip"
_VV = "measurement/s1b-iw1-slc-vv-20210401t052624-20210401t052649-026269-032297-004.tiff"
_START = "2021-04-01T05:26:22.396989"


def dated_safe_zip(directory: Path, day: str, fill: complex) -> Path:
    """Copy the fixture zip, set the manifest start day, and fill IW1 VV.

    ``day`` is ``YYYYMMDD``. The product id keeps the fixture orbit and
    datatake so the manifest cross-check still passes. The measurement TIFF
    is a small complex64 raster. The zip is stored, not extracted for the caller.
    """
    if len(day) != 8 or not day.isdigit():
        raise ValueError("day must be YYYYMMDD")
    iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
    product_id = SAFE_NAME.replace("20210401", day)
    dest = directory / f"{product_id}.zip"
    import tifffile

    buffer = io.BytesIO()
    tifffile.imwrite(buffer, np.full((4, 4), fill, dtype=np.complex64))
    pixels = buffer.getvalue()
    with zipfile.ZipFile(FIXTURE_ZIP) as source, zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as out:
        for info in source.infolist():
            if info.is_dir():
                continue
            payload = source.read(info)
            relative = info.filename.split(".SAFE/", 1)[-1]
            if relative == "manifest.safe":
                text = payload.decode("utf-8").replace(_START, f"{iso}T05:26:22.396989")
                payload = text.encode("utf-8")
            elif relative == _VV:
                payload = pixels
            out.writestr(f"{product_id}.SAFE/{relative}", payload)
    return dest
