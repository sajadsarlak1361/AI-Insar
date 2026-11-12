"""PS-linear map products: only the selected points, in WGS84.

The stacking velocity is a dense radar-grid field. Residual phase that
varies in range and is steady in azimuth draws vertical stripes through
that field. Persistent-scatterer export must not write it. A pixel is
drawn only when the PS mask says it was selected. Every other sample is
nodata or fully transparent. Nothing is interpolated between points, and
the points are not marked with a second black symbol.

Red is negative LOS (subsidence, away from the satellite). Blue is
positive LOS (uplift). White is zero.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from ai_insar.export_maps import write_png

_NODATA = np.float32(np.nan)
# PS products quote LOS in millimetres. 50 mm/yr is 5 cm/yr and 0.05 m/yr.
METRES_TO_MILLIMETRES = 1000.0


def symmetric_limits(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """Symmetric limits about zero from the robust low and high percentiles."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    peak = max(abs(lo), abs(hi))
    if peak <= 0.0:
        peak = float(np.max(np.abs(finite)))
    if peak <= 0.0:
        peak = 1.0
    return -peak, peak


def subsidence_rgb(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """RGB uint8. ``vmin`` (negative) is red, zero is white, ``vmax`` is blue."""
    field = np.asarray(values, dtype=np.float64)
    if vmax <= vmin:
        raise ValueError("colour maximum must be greater than the minimum")
    span = float(vmax) - float(vmin)
    mix = np.where(np.isfinite(field), (field - float(vmin)) / span, 0.5)
    mix = np.clip(mix, 0.0, 1.0)
    red = np.array([196, 48, 48], dtype=np.float64)
    white = np.array([244, 244, 246], dtype=np.float64)
    blue = np.array([37, 99, 196], dtype=np.float64)
    left = mix <= 0.5
    rgb = np.empty(field.shape + (3,), dtype=np.float64)
    local = np.where(left, mix / 0.5, (mix - 0.5) / 0.5)
    rgb[left] = red + (white - red) * local[left, None]
    rgb[~left] = white + (blue - white) * local[~left, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def subsidence_rgba(values: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """RGBA. Pixels outside ``valid`` or non-finite values have alpha 0."""
    keep = np.asarray(valid, dtype=bool) & np.isfinite(np.asarray(values, dtype=np.float64))
    rgb = subsidence_rgb(np.where(keep, values, 0.0), vmin, vmax)
    alpha = np.where(keep, np.uint8(255), np.uint8(0))
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def legend_image(vmin: float, vmax: float, unit: str) -> np.ndarray:
    """RGB legend: colour bar, tick labels, and Subsidence / Uplift."""
    height, width = 220, 280
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    bar_x0, bar_x1 = 36, 64
    bar_y0, bar_y1 = 28, 188
    column = np.linspace(vmax, vmin, bar_y1 - bar_y0)
    colours = subsidence_rgb(column, vmin, vmax)
    image[bar_y0:bar_y1, bar_x0:bar_x1] = colours[:, None, :]
    _text(image, 78, 8, "Uplift")
    _text(image, 78, 196, "Subsidence")
    _text(image, 78, 36, _tick(vmax, unit))
    _text(image, 78, 100, _tick(0.0, unit))
    _text(image, 78, 164, _tick(vmin, unit))
    _text(image, 8, 8, unit)
    return image


def scatter_north_up(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Place selected samples on a north-up grid. Empty cells stay NaN.

    No neighbour is filled in. Two selected samples that land in one cell
    are averaged. The returned bounds are the centres of the corner pixels,
    ``(west, south, east, north)``, with row 0 at the north.
    """
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(data) & np.isfinite(lat) & np.isfinite(lon)
    if int(np.count_nonzero(keep)) == 0:
        raise ValueError("PS export has no selected finite pixels")
    points_lat = lat[keep]
    points_lon = lon[keep]
    points_value = data[keep]
    dlat, dlon = _step(points_lat, points_lon)
    north = float(points_lat.max())
    south = float(points_lat.min())
    west = float(points_lon.min())
    east = float(points_lon.max())
    rows = max(1, int(round((north - south) / dlat)) + 1)
    cols = max(1, int(round((east - west) / dlon)) + 1)
    grid = np.zeros((rows, cols), dtype=np.float64)
    weight = np.zeros((rows, cols), dtype=np.float64)
    row_index = np.rint((north - points_lat) / dlat).astype(np.int64)
    col_index = np.rint((points_lon - west) / dlon).astype(np.int64)
    row_index = np.clip(row_index, 0, rows - 1)
    col_index = np.clip(col_index, 0, cols - 1)
    np.add.at(weight, (row_index, col_index), 1.0)
    np.add.at(grid, (row_index, col_index), points_value)
    filled = weight > 0
    grid[filled] /= weight[filled]
    grid[~filled] = np.nan
    return grid, filled, (west, south, east, north)


def geographic_edges(bounds: tuple[float, float, float, float], height: int, width: int) -> tuple[float, float, float, float]:
    """Outer pixel edges around corner-pixel centres. North-up, pixel-is-area."""
    west, south, east, north = (float(item) for item in bounds)
    if height < 1 or width < 1:
        raise ValueError("image size must be positive")
    dlat = 0.0 if height == 1 else (north - south) / (height - 1)
    dlon = 0.0 if width == 1 else (east - west) / (width - 1)
    if height == 1:
        dlat = dlon if dlon else 1.0e-6
    if width == 1:
        dlon = dlat if dlat else 1.0e-6
    return west - dlon / 2.0, south - dlat / 2.0, east + dlon / 2.0, north + dlat / 2.0


def write_float_geotiff(
    path: Path,
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    unit: str | None = None,
) -> None:
    """EPSG:4326 float32 GeoTIFF. ``bounds`` are corner-pixel centres. Nodata is NaN.

    ``unit`` is stored as a TIFF tag so a value of ``-50`` is read as mm/yr
    rather than as metres per year.
    """
    import rasterio
    from rasterio.transform import from_bounds

    field = np.asarray(values, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("float GeoTIFF input must be 2-D")
    height, width = field.shape
    west, south, east, north = geographic_edges(bounds, height, width)
    transform = from_bounds(west, south, east, north, width, height)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stored = np.where(np.isfinite(field), field, _NODATA)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=_NODATA,
    ) as dataset:
        dataset.write(stored, 1)
        tags = {"AREA_OR_POINT": "Area"}
        if unit:
            tags["UNIT"] = str(unit)
        dataset.update_tags(**tags)


def write_rgba_geotiff(path: Path, rgba: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    """EPSG:4326 RGBA GeoTIFF. Alpha 0 is empty. ``bounds`` are corner centres.

    Band 4 is unassociated alpha (ExtraSamples = 2) and the file is
    uncompressed and striped. Google Earth Pro reads that combination
    through File → Open. A GDAL mask band is not a substitute: Earth
    leaves band 4 as an undefined extra sample and the drape is gray.
    """
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_bounds

    image = np.asarray(rgba, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("RGBA GeoTIFF input must be height × width × 4")
    height, width = image.shape[:2]
    west, south, east, north = geographic_edges(bounds, height, width)
    transform = from_bounds(west, south, east, north, width, height)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        photometric="RGB",
        tiled=False,
        interleave="pixel",
    ) as dataset:
        dataset.write(np.moveaxis(image, -1, 0))
        dataset.update_tags(AREA_OR_POINT="Area")
    # Color interpretation set during creation is not written. Updating the
    # closed file stores ExtraSamples = 2 (unassociated alpha).
    with rasterio.open(destination, "r+") as dataset:
        dataset.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )


def write_ps_ground_overlay(
    path: Path,
    rgba: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> None:
    """KMZ GroundOverlay of the coloured raster.

    ``bounds`` are the outer pixel edges, west, south, east, north, the same
    numbers as the GeoTIFF's corner coordinates.
    """
    west, south, east, north = (float(item) for item in bounds)
    image = np.asarray(rgba, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("ground overlay image must be height × width × 4")
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        "  <name>PS linear overlay</name>\n"
        "  <GroundOverlay>\n"
        "    <name>PS linear</name>\n"
        "    <Icon><href>overlay.png</href></Icon>\n"
        "    <LatLonBox>\n"
        f"      <north>{north:.16f}</north>\n"
        f"      <south>{south:.16f}</south>\n"
        f"      <east>{east:.16f}</east>\n"
        f"      <west>{west:.16f}</west>\n"
        "    </LatLonBox>\n"
        "  </GroundOverlay>\n"
        "</Document></kml>\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml)
        archive.writestr("overlay.png", _png_bytes(image))


def write_ps_kmz(
    path: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    unit: str,
    coherence: np.ndarray | None = None,
    quantity: str = "LOS velocity",
) -> None:
    """One placemark per selected pixel, plus a ScreenOverlay legend."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(data) & np.isfinite(lat) & np.isfinite(lon)
    if int(np.count_nonzero(keep)) == 0:
        raise ValueError("PS KMZ has no selected finite pixels")
    colours = subsidence_rgb(data, vmin, vmax)
    rows, cols = np.nonzero(keep)
    marks = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        red, green, blue = (int(item) for item in colours[row, col])
        kml_colour = f"ff{blue:02x}{green:02x}{red:02x}"
        coh = ""
        if coherence is not None and np.isfinite(coherence[row, col]):
            coh = f"Coherence {float(coherence[row, col]):.3f}<br/>"
        description = (
            f"{escape(quantity)} {float(data[row, col]):.4f} {escape(unit)}<br/>"
            f"{coh}"
            f"Latitude {float(lat[row, col]):.8f}<br/>"
            f"Longitude {float(lon[row, col]):.8f}"
        )
        marks.append(
            "    <Placemark>\n"
            "      <Style><IconStyle>"
            f"<color>{kml_colour}</color><scale>0.7</scale>"
            "<Icon><href>dot.png</href></Icon>"
            "</IconStyle></Style>\n"
            f"      <description><![CDATA[{description}]]></description>\n"
            "      <Point><coordinates>"
            f"{float(lon[row, col]):.8f},{float(lat[row, col]):.8f},0"
            "</coordinates></Point>\n"
            "    </Placemark>"
        )
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        "  <name>PS linear</name>\n"
        "  <ScreenOverlay>\n"
        "    <name>Legend</name>\n"
        "    <Icon><href>legend.png</href></Icon>\n"
        '    <overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>\n'
        '    <screenXY x="0.02" y="0.06" xunits="fraction" yunits="fraction"/>\n'
        '    <size x="0" y="0" xunits="fraction" yunits="fraction"/>\n'
        "  </ScreenOverlay>\n"
        + "\n".join(marks)
        + "\n</Document></kml>\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    legend = legend_image(vmin, vmax, unit)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml)
        archive.writestr("legend.png", _png_bytes(legend))
        archive.writestr("dot.png", _png_bytes(_dot()))


def export_ps_linear(
    values: np.ndarray,
    valid: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    directory: Path | str,
    *,
    unit: str = "mm/yr",
    scale: float = METRES_TO_MILLIMETRES,
    vmin: float | None = None,
    vmax: float | None = None,
    coherence: np.ndarray | None = None,
    quantity: str = "LOS velocity",
    stem: str = "ps_linear",
) -> dict[str, Path]:
    """Write the float GeoTIFF, RGBA GeoTIFF, transparent PNG, legend, and KMZ.

    ``values`` are in metres or metres per year. ``scale`` converts them to
    ``unit`` for the colour scale and the placemark text. Samples outside
    ``valid`` are dropped even when they are finite, so a dense striped
    field is not exported.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    selected = np.asarray(valid, dtype=bool) & np.isfinite(np.asarray(values, dtype=np.float64))
    display = np.where(selected, np.asarray(values, dtype=np.float64) * float(scale), np.nan)
    if vmin is None or vmax is None:
        auto_min, auto_max = symmetric_limits(display)
        vmin = auto_min if vmin is None else float(vmin)
        vmax = auto_max if vmax is None else float(vmax)
    if float(vmax) <= float(vmin):
        raise ValueError("colour maximum must be greater than the minimum")
    grid, filled, bounds = scatter_north_up(latitude, longitude, display, selected)
    rgba = subsidence_rgba(grid, filled, float(vmin), float(vmax))
    float_path = folder / f"{stem}.tif"
    rgba_path = folder / f"{stem}_rgba.tif"
    png_path = folder / f"{stem}.png"
    legend_path = folder / f"{stem}_legend.png"
    kmz_path = folder / f"{stem}.kmz"
    overlay_path = folder / f"{stem}_overlay.kmz"
    write_float_geotiff(float_path, grid, bounds, unit=unit)
    write_rgba_geotiff(rgba_path, rgba, bounds)
    edges = geographic_edges(bounds, rgba.shape[0], rgba.shape[1])
    write_ps_ground_overlay(overlay_path, rgba, edges)
    write_png(png_path, rgba)
    write_png(legend_path, legend_image(float(vmin), float(vmax), unit))
    write_ps_kmz(
        kmz_path,
        latitude,
        longitude,
        display,
        selected,
        vmin=float(vmin),
        vmax=float(vmax),
        unit=unit,
        coherence=coherence,
        quantity=quantity,
    )
    return {
        "tif": float_path,
        "rgba": rgba_path,
        "png": png_path,
        "legend": legend_path,
        "kmz": kmz_path,
        "overlay": overlay_path,
    }


def radar_pixel_latlon(grid, orbit, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of each radar-pixel centre, in degrees."""
    from ai_insar.pipeline.geometry import ecef_to_geodetic, ground_points_at_height
    from ai_insar.pipeline.tops_ops import _line_times, _slant_ranges

    rows, cols = grid.shape
    row_index, col_index = np.mgrid[0:rows, 0:cols]
    azimuth = _line_times(grid.model, grid.row + row_index.ravel())
    slant = _slant_ranges(grid.model, (grid.col + col_index.ravel()).astype(np.float64))
    height = np.full(row_index.size, float(getattr(grid.model, "terrain_height_m", 0.0) or 0.0))
    ground = ground_points_at_height(orbit, azimuth, slant, height, degree=degree)
    lat, lon, _got = ecef_to_geodetic(ground)
    return np.rad2deg(lat).reshape(rows, cols), np.rad2deg(lon).reshape(rows, cols)


def _step(latitude: np.ndarray, longitude: np.ndarray) -> tuple[float, float]:
    if latitude.size == 1:
        return 1.0e-4, 1.0e-4
    dlat = np.diff(np.sort(np.unique(np.round(latitude, 8))))
    dlon = np.diff(np.sort(np.unique(np.round(longitude, 8))))
    lat_step = float(np.median(dlat)) if dlat.size else 1.0e-4
    lon_step = float(np.median(dlon)) if dlon.size else 1.0e-4
    return max(lat_step, 1.0e-8), max(lon_step, 1.0e-8)


def _tick(value: float, unit: str) -> str:
    """Plain decimal plus the unit.

    ``.3g`` prints 1000 as ``1e+03``. The legend font has no plus sign, so
    that label is drawn as ``1e 03`` and reads as 1. A 5 cm/yr subsidence
    rate is 50 mm/yr and must stay the digits 50, not a one.
    """
    number = float(value)
    if number == 0.0:
        text = "0"
    else:
        text = f"{number:.4f}".rstrip("0").rstrip(".")
        if text in {"", "-", "-0"}:
            text = "0"
    return f"{text} {unit}"


def _text(image: np.ndarray, x: int, y: int, text: str) -> None:
    cursor = x
    for character in text:
        glyph = _FONT.get(character, _FONT[" "])
        for row, bits in enumerate(glyph):
            yy = y + row
            if yy < 0 or yy >= image.shape[0]:
                continue
            for col in range(5):
                xx = cursor + col
                if xx < 0 or xx >= image.shape[1]:
                    continue
                if bits & (1 << (4 - col)):
                    image[yy, xx] = (20, 24, 32)
        cursor += 6


def _dot() -> np.ndarray:
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:16, 0:16]
    inside = (yy - 7.5) ** 2 + (xx - 7.5) ** 2 <= 36
    image[inside] = (255, 255, 255, 255)
    return image


def _png_bytes(rgb_or_rgba: np.ndarray) -> bytes:
    from io import BytesIO

    array = np.asarray(rgb_or_rgba, dtype=np.uint8)
    buffer = BytesIO()
    write_png(buffer, array)
    return buffer.getvalue()


# 5×7 glyphs, 5 bits per row, top to bottom.
_FONT: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "-": (0, 0, 0, 0x1F, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 0x04, 0),
    "/": (0x01, 0x02, 0x04, 0x08, 0x10, 0, 0),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F),
    "3": (0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "S": (0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "b": (0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x1E),
    "c": (0, 0, 0x0E, 0x10, 0x10, 0x11, 0x0E),
    "d": (0x01, 0x01, 0x01, 0x0D, 0x13, 0x11, 0x0F),
    "e": (0, 0, 0x0E, 0x11, 0x1F, 0x10, 0x0E),
    "f": (0x06, 0x08, 0x08, 0x1E, 0x08, 0x08, 0x08),
    "i": (0x04, 0, 0x0C, 0x04, 0x04, 0x04, 0x0E),
    "l": (0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "m": (0, 0, 0x1A, 0x15, 0x15, 0x15, 0x15),
    "n": (0, 0, 0x16, 0x19, 0x11, 0x11, 0x11),
    "p": (0, 0, 0x1E, 0x11, 0x1E, 0x10, 0x10),
    "r": (0, 0, 0x16, 0x19, 0x10, 0x10, 0x10),
    "s": (0, 0, 0x0E, 0x10, 0x0E, 0x01, 0x1E),
    "u": (0, 0, 0x11, 0x11, 0x11, 0x13, 0x0D),
    "y": (0, 0, 0x11, 0x11, 0x0F, 0x01, 0x0E),
    "g": (0, 0, 0x0F, 0x11, 0x0F, 0x01, 0x0E),
    "t": (0x08, 0x08, 0x1E, 0x08, 0x08, 0x09, 0x06),
}
