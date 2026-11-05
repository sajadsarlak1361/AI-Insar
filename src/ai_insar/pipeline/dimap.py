"""Write a BEAM-DIMAP product this build can reopen.

The ``.dim`` XML and ENVI band files follow the public DIMAP / BEAM data
model. SNAP source is not included. Complex SLC is stored as separate
``i`` and ``q`` int16 bands when every sample fits in int16, otherwise as
float32. Geographic bounds, when present, are corner tie points on WGS84.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.017453292519943295]]'
)


def write_beam_dimap(
    directory: Path | str,
    target: str,
    array: np.ndarray,
    bounds: tuple[float, float, float, float] | None = None,
    *,
    band_name: str = "band",
    unit: str = "1",
) -> Path:
    """Write ``target.dim`` and ``target.data`` under ``directory``."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stem = (target or "product").strip() or "product"
    dim_path = folder / f"{stem}.dim"
    data_dir = folder / f"{stem}.data"
    data_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    bands = _bands(image, band_name)
    rows = int(image.shape[-2])
    cols = int(image.shape[-1])
    file_xml = []
    band_xml = []
    for index, (name, values, dtype_name) in enumerate(bands):
        raw = data_dir / f"{name}.img"
        header = data_dir / f"{name}.hdr"
        envi_type = 4 if values.dtype == np.float32 else 2
        stored = values.astype(">f4" if envi_type == 4 else ">i2")
        stored.tofile(raw)
        header.write_text(
            "\n".join(
                (
                    "ENVI",
                    f"description = {{AI Insar - Unit: {unit}}}",
                    f"samples = {cols}",
                    f"lines = {rows}",
                    "bands = 1",
                    "header offset = 0",
                    "file type = ENVI Standard",
                    f"data type = {envi_type}",
                    "interleave = bsq",
                    "byte order = 1",
                    f"band names = {{ {name} }}",
                    "data gain values = {1.0}",
                    "data offset values = {0.0}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        href = f"{stem}.data/{name}.hdr"
        file_xml.append(
            f"      <Data_File><DATA_FILE_PATH href=\"{href}\"/><BAND_INDEX>{index}</BAND_INDEX></Data_File>"
        )
        band_xml.append(
            "\n".join(
                (
                    "      <Spectral_Band_Info>",
                    f"        <BAND_INDEX>{index}</BAND_INDEX>",
                    f"        <BAND_NAME>{name}</BAND_NAME>",
                    f"        <BAND_RASTER_WIDTH>{cols}</BAND_RASTER_WIDTH>",
                    f"        <BAND_RASTER_HEIGHT>{rows}</BAND_RASTER_HEIGHT>",
                    f"        <DATA_TYPE>{dtype_name}</DATA_TYPE>",
                    f"        <PHYSICAL_UNIT>{unit}</PHYSICAL_UNIT>",
                    "        <SCALING_FACTOR>1.0</SCALING_FACTOR>",
                    "        <SCALING_OFFSET>0.0</SCALING_OFFSET>",
                    "        <LOG10_SCALED>false</LOG10_SCALED>",
                    "        <NO_DATA_VALUE_USED>false</NO_DATA_VALUE_USED>",
                    "        <NO_DATA_VALUE>0.0</NO_DATA_VALUE>",
                    "      </Spectral_Band_Info>",
                )
            )
        )
    tie_xml = ""
    tie_files = ""
    if bounds is not None:
        west, south, east, north = bounds
        tie_dir = data_dir / "tie_point_grids"
        tie_dir.mkdir(parents=True, exist_ok=True)
        latitude = np.array([[north, north], [south, south]], dtype=">f4")
        longitude = np.array([[west, east], [west, east]], dtype=">f4")
        _write_tie(tie_dir / "latitude", latitude, "latitude")
        _write_tie(tie_dir / "longitude", longitude, "longitude")
        step_x = max(cols - 1, 1)
        step_y = max(rows - 1, 1)
        tie_files = f"""
    <Tie_Point_Grid_File>
      <TIE_POINT_GRID_FILE_PATH href="{stem}.data/tie_point_grids/latitude.hdr"/>
      <TIE_POINT_GRID_INDEX>0</TIE_POINT_GRID_INDEX>
    </Tie_Point_Grid_File>
    <Tie_Point_Grid_File>
      <TIE_POINT_GRID_FILE_PATH href="{stem}.data/tie_point_grids/longitude.hdr"/>
      <TIE_POINT_GRID_INDEX>1</TIE_POINT_GRID_INDEX>
    </Tie_Point_Grid_File>"""
        tie_xml = f"""
  <Tie_Point_Grids>
    <NUM_TIE_POINT_GRIDS>2</NUM_TIE_POINT_GRIDS>
    <Tie_Point_Grid_Info>
      <TIE_POINT_GRID_INDEX>0</TIE_POINT_GRID_INDEX>
      <PHYSICAL_UNIT>deg</PHYSICAL_UNIT>
      <TIE_POINT_GRID_NAME>latitude</TIE_POINT_GRID_NAME>
      <DATA_TYPE>float32</DATA_TYPE>
      <NCOLS>2</NCOLS><NROWS>2</NROWS>
      <OFFSET_X>0.5</OFFSET_X><OFFSET_Y>0.5</OFFSET_Y>
      <STEP_X>{step_x}</STEP_X><STEP_Y>{step_y}</STEP_Y>
    </Tie_Point_Grid_Info>
    <Tie_Point_Grid_Info>
      <TIE_POINT_GRID_INDEX>1</TIE_POINT_GRID_INDEX>
      <PHYSICAL_UNIT>deg</PHYSICAL_UNIT>
      <TIE_POINT_GRID_NAME>longitude</TIE_POINT_GRID_NAME>
      <DATA_TYPE>float32</DATA_TYPE>
      <NCOLS>2</NCOLS><NROWS>2</NROWS>
      <OFFSET_X>0.5</OFFSET_X><OFFSET_Y>0.5</OFFSET_Y>
      <STEP_X>{step_x}</STEP_X><STEP_Y>{step_y}</STEP_Y>
    </Tie_Point_Grid_Info>
  </Tie_Point_Grids>
  <Geoposition>
    <BAND_INDEX>0</BAND_INDEX>
    <Geoposition_Points>
      <TIE_POINT_GRID_NAME_LAT>latitude</TIE_POINT_GRID_NAME_LAT>
      <TIE_POINT_GRID_NAME_LON>longitude</TIE_POINT_GRID_NAME_LON>
    </Geoposition_Points>
  </Geoposition>"""
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<Dimap_Document name="{stem}.dim">
  <Metadata_Id>
    <METADATA_FORMAT version="2.12.1">DIMAP</METADATA_FORMAT>
    <METADATA_PROFILE>BEAM-DATAMODEL-V1</METADATA_PROFILE>
  </Metadata_Id>
  <Dataset_Id>
    <DATASET_SERIES>BEAM-PRODUCT</DATASET_SERIES>
    <DATASET_NAME>{stem}</DATASET_NAME>
  </Dataset_Id>
  <Production>
    <PRODUCT_TYPE>AI-INSAR</PRODUCT_TYPE>
  </Production>
  <Coordinate_Reference_System>
    <Horizontal_CS>
      <HORIZONTAL_CS_TYPE>GEOGRAPHIC</HORIZONTAL_CS_TYPE>
      <Geographic_CS>
        <Horizontal_Datum>
          <HORIZONTAL_DATUM_NAME>WGS-84</HORIZONTAL_DATUM_NAME>
          <Ellipsoid>
            <ELLIPSOID_NAME>WGS-84</ELLIPSOID_NAME>
            <Ellipsoid_Parameters>
              <ELLIPSOID_MAJ_AXIS unit="M">6378137.0</ELLIPSOID_MAJ_AXIS>
              <ELLIPSOID_MIN_AXIS unit="M">6356752.3</ELLIPSOID_MIN_AXIS>
            </Ellipsoid_Parameters>
          </Ellipsoid>
        </Horizontal_Datum>
      </Geographic_CS>
    </Horizontal_CS>
    <WKT>{_WGS84_WKT}</WKT>
  </Coordinate_Reference_System>
  <Raster_Dimensions>
    <NCOLS>{cols}</NCOLS>
    <NROWS>{rows}</NROWS>
    <NBANDS>{len(bands)}</NBANDS>
  </Raster_Dimensions>
  <Data_Access>
    <DATA_FILE_FORMAT>ENVI</DATA_FILE_FORMAT>
    <DATA_FILE_FORMAT_DESC>ENVI File Format</DATA_FILE_FORMAT_DESC>
    <DATA_FILE_ORGANISATION>BAND_SEPARATE</DATA_FILE_ORGANISATION>
{chr(10).join(file_xml)}{tie_files}
  </Data_Access>
  <Image_Interpretation>
{chr(10).join(band_xml)}
  </Image_Interpretation>{tie_xml}
</Dimap_Document>
"""
    dim_path.write_text(document, encoding="utf-8")
    return dim_path


def read_beam_dimap_band(dim_path: Path | str, band_index: int = 0) -> np.ndarray:
    """Read one ENVI band named by a ``.dim`` written here."""
    path = Path(dim_path)
    text = path.read_text(encoding="utf-8")
    import re

    hrefs = re.findall(r'DATA_FILE_PATH href="([^"]+)"', text)
    if band_index >= len(hrefs):
        raise ValueError(f"{path.name} has no band {band_index}")
    header = path.parent / hrefs[band_index]
    samples = _envi_int(header, "samples")
    lines = _envi_int(header, "lines")
    data_type = _envi_int(header, "data type")
    raw = header.with_suffix(".img")
    byte_order = _envi_int(header, "byte order") if "byte order" in header.read_text(encoding="utf-8") else 0
    dtype = np.dtype((">f4" if byte_order == 1 else "<f4") if data_type == 4 else (">i2" if byte_order == 1 else "<i2"))
    return np.fromfile(raw, dtype=dtype).reshape(lines, samples)


def _write_tie(stem: Path, values: np.ndarray, name: str) -> None:
    values.astype(">f4").tofile(stem.with_suffix(".img"))
    rows, cols = values.shape
    stem.with_suffix(".hdr").write_text(
        "\n".join(
            (
                "ENVI",
                "description = {AI Insar tie point}",
                f"samples = {cols}",
                f"lines = {rows}",
                "bands = 1",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bsq",
                "byte order = 1",
                f"band names = {{ {name} }}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _bands(image: np.ndarray, band_name: str) -> list[tuple[str, np.ndarray, str]]:
    if np.iscomplexobj(image):
        real = np.real(image)
        imag = np.imag(image)
        if np.nanmax(np.abs(real)) <= 32767 and np.nanmax(np.abs(imag)) <= 32767 and _integral(real) and _integral(imag):
            return [
                (f"i_{band_name}", np.rint(real).astype(np.int16), "int16"),
                (f"q_{band_name}", np.rint(imag).astype(np.int16), "int16"),
            ]
        return [
            (f"i_{band_name}", real.astype(np.float32), "float32"),
            (f"q_{band_name}", imag.astype(np.float32), "float32"),
        ]
    values = np.asarray(image, dtype=np.float32)
    return [(band_name, values, "float32")]


def _integral(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.allclose(finite, np.rint(finite)))


def _envi_int(header: Path, name: str) -> int:
    import re

    text = header.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{header.name} has no {name}")
    return int(match.group(1))
