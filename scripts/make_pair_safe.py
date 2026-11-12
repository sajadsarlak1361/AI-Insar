"""Build two same-track SAFE zips and one mismatched-track zip.

The manifest and the IW1 VV annotation come from the reduced fixture.
The annotation image size is cut to one short burst so Split can read the
cropped measurement TIFF. Relative orbit 168 is kept for the pair and set
to 54 for the mismatch.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "safe" / "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4.zip"
OUT = ROOT / "tests" / "fixtures" / "pair"
SAFE = "S1B_IW_SLC__1SDV_20210401T052622_20210401T052650_026269_032297_EFA4"
ANN = "annotation/s1b-iw1-slc-vv-20210401t052624-20210401t052649-026269-032297-004.xml"
TIFF = "measurement/s1b-iw1-slc-vv-20210401t052624-20210401t052649-026269-032297-004.tiff"
LINES = 32
SAMPLES = 48


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def shrink_annotation(payload: bytes) -> bytes:
    root = ET.fromstring(payload)
    bursts = [element for element in root.iter() if _local(element.tag) == "burst"]
    burst_list = next(element for element in root.iter() if _local(element.tag) == "burstList")
    for burst in bursts[1:]:
        burst_list.remove(burst)
    if "count" in burst_list.attrib:
        burst_list.attrib["count"] = "1"
    kept = bursts[0]
    for child in list(kept):
        name = _local(child.tag)
        if name == "firstValidSample":
            child.text = " ".join(["0"] * LINES)
        elif name == "lastValidSample":
            child.text = " ".join([str(SAMPLES - 1)] * LINES)
    for element in root.iter():
        name = _local(element.tag)
        if name == "numberOfSamples":
            element.text = str(SAMPLES)
        elif name == "numberOfLines":
            element.text = str(LINES)
        elif name == "linesPerBurst":
            element.text = str(LINES)
        elif name == "samplesPerBurst":
            element.text = str(SAMPLES)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _tiff(fill: complex) -> bytes:
    import tifffile

    column = np.linspace(0, 1, SAMPLES, dtype=np.float32)
    grid = np.ones((LINES, SAMPLES), dtype=np.complex64) * fill
    grid *= 0.2 + 0.8 * column
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, grid)
    return buffer.getvalue()


def build(day: str, relative_orbit: int, fill: complex) -> Path:
    iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
    product_id = SAFE.replace("20210401", day)
    dest = OUT / f"{product_id}.zip"
    pixels = _tiff(fill)
    with zipfile.ZipFile(FIXTURE) as source, zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as out:
        manifest = source.read(f"{SAFE}.SAFE/manifest.safe").decode("utf-8")
        manifest = manifest.replace("2021-04-01T05:26:22.396989", f"{iso}T05:26:22.396989")
        manifest = manifest.replace(">168<", f">{relative_orbit}<")
        annotation = shrink_annotation(source.read(f"{SAFE}.SAFE/{ANN}"))
        out.writestr(f"{product_id}.SAFE/manifest.safe", manifest.encode("utf-8"))
        out.writestr(f"{product_id}.SAFE/{ANN}", annotation)
        out.writestr(f"{product_id}.SAFE/{TIFF}", pixels)
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    first = build("20210401", 168, 1 + 0j)
    second = build("20210413", 168, 0 + 1j)
    mismatch = build("20210425", 54, 1 + 1j)
    print(first)
    print(second)
    print(mismatch)


if __name__ == "__main__":
    main()
