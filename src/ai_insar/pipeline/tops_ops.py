"""TOPS split, deburst, flat-earth, and range-Doppler resampling.

These are the published Sentinel-1 relationships: burst blocks of
``linesPerBurst``, overlap cut at the mid azimuth time, deramp with the
annotation FM rate, and right-looking zero-Doppler geolocation. They are
not a crop and not an amplitude correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai_insar.pipeline.geometry import (
    Orbit,
    ecef_to_geodetic,
    geodetic_to_ecef,
    ground_point_at_height,
)
from ai_insar.pipeline.s1_annotation import SwathModel


@dataclass
class RadarGrid:
    """A window of one swath, in full-image line and sample coordinates."""

    row: int
    col: int
    data: np.ndarray
    model: SwathModel

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.data.shape[0]), int(self.data.shape[1])


def burst_rows(model: SwathModel, burst_index: int) -> tuple[int, int]:
    """Absolute row interval of one burst, ``[start, stop)``."""
    if burst_index < 0 or burst_index >= len(model.bursts):
        raise ValueError(f"burst {burst_index} is outside 0..{len(model.bursts) - 1}")
    start = burst_index * model.lines_per_burst
    return start, start + model.lines_per_burst


def deramp(grid: RadarGrid, *, remove: bool = True) -> np.ndarray:
    """Multiply by the conjugate of the TOPS azimuth chirp, burst by burst.

    ``remove=False`` puts the chirp back (reramp). The FM rate is the
    annotation polynomial at each sample's two-way time.
    """
    out = np.array(grid.data, dtype=np.complex64, copy=True)
    rows, cols = out.shape
    model = grid.model
    coefficients = model.fm_rate
    c0 = float(coefficients[0])
    c1 = float(coefficients[1]) if coefficients.size > 1 else 0.0
    c2 = float(coefficients[2]) if coefficients.size > 2 else 0.0
    sample = grid.col + np.arange(cols, dtype=np.float64)
    delta_tau = 2.0 * sample * model.range_pixel_spacing_m / 299_792_458.0
    rate = c0 + c1 * delta_tau + c2 * delta_tau**2
    sign = -1.0 if remove else 1.0
    for burst_index, burst in enumerate(model.bursts):
        burst_row, burst_stop = burst_rows(model, burst_index)
        local_start = max(burst_row, grid.row)
        local_stop = min(burst_stop, grid.row + rows)
        if local_start >= local_stop:
            continue
        line = np.arange(local_start, local_stop, dtype=np.float64) - burst_row
        center = (model.lines_per_burst - 1) / 2.0
        eta = (line - center) * model.azimuth_time_interval_s
        phase = np.pi * rate[None, :] * (eta[:, None] ** 2)
        slc = out[local_start - grid.row : local_stop - grid.row]
        slc *= np.exp(1j * sign * phase).astype(np.complex64)
    return out


def deburst_source_rows(model: SwathModel, burst_indexes: Sequence[int] | None = None) -> np.ndarray:
    """Absolute source line of every SNAP deburst output line.

    The output time axis runs from the first selected burst's first line to
    the last selected burst's last line. Overlap is resolved by the mid time
    between the two burst times that contain the sample, which is the rule in
    SNAP ``TOPSARDeburstOp``. Bursts that are not in ``burst_indexes`` are not
    in the product, so their overlaps are kept.
    """
    if burst_indexes is None:
        indexes = list(range(len(model.bursts)))
    else:
        indexes = sorted({int(item) for item in burst_indexes})
    if not indexes or indexes[0] < 0 or indexes[-1] >= len(model.bursts):
        raise ValueError(f"burst indexes {indexes} are outside this swath")
    interval = float(model.azimuth_time_interval_s)
    lines = int(model.lines_per_burst)
    epoch = model.bursts[indexes[0]].azimuth_time

    def seconds(when: np.datetime64) -> float:
        return float((when - epoch).astype(np.int64)) * 1e-9

    first_s = [seconds(model.bursts[index].azimuth_time) for index in indexes]
    last_s = [start + (lines - 1) * interval for start in first_s]
    height = int((last_s[-1] - first_s[0]) / interval)
    if height < 1:
        raise ValueError("deburst time span is empty")
    chosen: list[int] = []
    for line in range(height):
        target = first_s[0] + line * interval
        hits: list[tuple[int, int, float, float]] = []
        for position, index in enumerate(indexes):
            if first_s[position] <= target < last_s[position]:
                local = int((target - first_s[position]) / interval + 0.5)
                local = min(max(local, 0), lines - 1)
                hits.append((index, local, first_s[position], last_s[position]))
                if len(hits) == 2:
                    break
        if not hits:
            continue
        pick = hits[0]
        if len(hits) == 2:
            mid = (hits[0][3] + hits[1][2]) / 2.0
            pick = hits[1] if target > mid else hits[0]
        chosen.append(pick[0] * lines + pick[1])
    if not chosen:
        raise ValueError("deburst kept no lines")
    return np.asarray(chosen, dtype=np.int32)


def deburst_lines(model: SwathModel, burst_indexes: Sequence[int] | None = None) -> list[tuple[int, int, int]]:
    """Local inclusive line ranges kept from each selected burst.

    Each item is ``(burst_index, first_line, last_line)``. The ranges follow
    :func:`deburst_source_rows`, so a split product does not trim its outer
    edge against a burst that was not selected.
    """
    rows = deburst_source_rows(model, burst_indexes)
    lines = int(model.lines_per_burst)
    kept: list[tuple[int, int, int]] = []
    start = int(rows[0])
    previous = start
    for absolute in rows[1:]:
        absolute = int(absolute)
        same_burst = absolute // lines == start // lines
        if absolute == previous + 1 and same_burst:
            previous = absolute
            continue
        kept.append((start // lines, start % lines, previous % lines))
        start = previous = absolute
    kept.append((start // lines, start % lines, previous % lines))
    return kept


def deburst(grids: list[RadarGrid]) -> RadarGrid:
    """Concatenate burst windows after removing the azimuth overlap.

    Every grid must be the same column window of one swath. The returned
    grid's ``row`` is the first kept absolute line.
    """
    if not grids:
        raise ValueError("deburst needs at least one burst")
    model = grids[0].model
    col = grids[0].col
    indexes = sorted({int(grid.row // model.lines_per_burst) for grid in grids})
    source_rows = deburst_source_rows(model, indexes)
    pieces: list[np.ndarray] = []
    first_absolute = None
    for absolute in source_rows:
        for grid in grids:
            if grid.col != col or grid.model.lines_per_burst != model.lines_per_burst:
                raise ValueError("deburst windows must share the column crop and swath")
            local = int(absolute) - int(grid.row)
            if 0 <= local < grid.data.shape[0]:
                if first_absolute is None:
                    first_absolute = int(absolute)
                pieces.append(np.asarray(grid.data[local], dtype=np.complex64))
                break
    if not pieces or first_absolute is None:
        raise ValueError("deburst kept no lines")
    return RadarGrid(row=int(first_absolute), col=col, data=np.stack(pieces, axis=0), model=model)


def flat_earth_phase(
    early: RadarGrid,
    later: RadarGrid,
    *,
    early_orbit: Orbit,
    later_orbit: Orbit,
    degree: int = 3,
    height_m: float = 0.0,
    dem: object | None = None,
    max_solves: int | None = None,
) -> np.ndarray:
    """Geometric reference-surface phase, later minus earlier, in radians.

    ``phi = -4π/λ (R_later - R_early)``. ``height_m`` is one ellipsoidal
    height. ``dem`` samples a different ellipsoidal height at every pixel
    and is the topographic term. A missing DEM sample stays NaN.
    """
    if early.shape != later.shape or early.row != later.row or early.col != later.col:
        raise ValueError("flat-earth phase needs two windows on the same radar grid")
    rows, cols = early.shape
    limit = rows * cols if max_solves is None else int(max_solves)
    stride = _solve_stride(rows, cols, limit)
    row_index = _sample_index(rows, stride)
    col_index = _sample_index(cols, stride)
    coarse = np.empty((row_index.size, col_index.size), dtype=np.float64)
    wavelength = early.model.wavelength_m
    for i, row in enumerate(row_index):
        absolute = early.row + int(row)
        azimuth = early.model.line_time(absolute)
        for j, col in enumerate(col_index):
            slant = early.model.slant_range_m(early.col + int(col))
            ground = ground_point_at_height(early_orbit, azimuth, slant, height_m, degree=degree)
            if dem is not None:
                lat, lon, _got = ecef_to_geodetic(ground)
                sampled_height, valid = dem.ellipsoidal_height(float(np.rad2deg(lat)), float(np.rad2deg(lon)))
                if not valid:
                    coarse[i, j] = np.nan
                    continue
                ground = ground_point_at_height(early_orbit, azimuth, slant, sampled_height, degree=degree)
            early_range = _range_at(early_orbit, azimuth, ground, degree)
            later_guess = repeat_azimuth_time(early.model, absolute, later.model)
            later_time = later_orbit.zero_doppler_time(ground, later_guess, degree)
            later_range = _range_at(later_orbit, later_time, ground, degree)
            coarse[i, j] = -4.0 * np.pi / wavelength * (later_range - early_range)
    if stride == 1 and row_index.size == rows and col_index.size == cols:
        return coarse
    return _interpolate_grid(row_index, col_index, coarse, rows, cols)


def polynomial_phase(phase: np.ndarray, degree: int, count: int) -> np.ndarray:
    """Least-squares 2-D polynomial sampled at ``count`` scattered points."""
    degree = int(degree)
    if degree < 1:
        raise ValueError("polynomial degree must be at least 1")
    rows, cols = phase.shape
    rng = np.random.default_rng(0)
    take = min(int(count), phase.size)
    choice = rng.choice(phase.size, size=take, replace=False)
    yy, xx = np.unravel_index(choice, phase.shape)
    y = yy.astype(np.float64) / max(rows - 1, 1)
    x = xx.astype(np.float64) / max(cols - 1, 1)
    columns = []
    for power_y in range(degree + 1):
        for power_x in range(degree + 1 - power_y):
            columns.append((y**power_y) * (x**power_x))
    design = np.column_stack(columns)
    values = phase[yy, xx]
    coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
    full_y = np.arange(rows, dtype=np.float64) / max(rows - 1, 1)
    full_x = np.arange(cols, dtype=np.float64) / max(cols - 1, 1)
    grid_y, grid_x = np.meshgrid(full_y, full_x, indexing="ij")
    fitted = np.zeros_like(phase)
    cursor = 0
    for power_y in range(degree + 1):
        for power_x in range(degree + 1 - power_y):
            fitted += coeff[cursor] * (grid_y**power_y) * (grid_x**power_x)
            cursor += 1
    return fitted


def subtract_phase(interferogram: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    """Return ``interferogram * exp(-j phase)``."""
    factor = np.exp(-1j * np.asarray(phase_rad, dtype=np.float64)).astype(np.complex64)
    return np.asarray(interferogram, dtype=np.complex64) * factor


def _range_at(orbit: Orbit, when: np.datetime64, ground: np.ndarray, degree: int) -> float:
    position, _velocity = orbit.interpolate(when, degree)
    return float(np.linalg.norm(ground - position))


def repeat_azimuth_time(reference: SwathModel, absolute_line: int, secondary: SwathModel) -> np.datetime64:
    """Azimuth time of the same burst index and in-burst line on ``secondary``.

    A repeat-pass zero-Doppler search has to start on the secondary
    acquisition. The reference date is a different orbit file.
    """
    burst = int(absolute_line) // int(reference.lines_per_burst)
    offset = int(absolute_line) % int(reference.lines_per_burst)
    burst = min(max(burst, 0), len(secondary.bursts) - 1)
    offset = min(max(offset, 0), int(secondary.lines_per_burst) - 1)
    return secondary.line_time(burst * int(secondary.lines_per_burst) + offset)


def radar_coordinates(
    model: SwathModel,
    orbit: Orbit,
    ground_xyz: np.ndarray,
    *,
    guess: np.datetime64,
    degree: int = 3,
) -> tuple[float, float]:
    """Return ``(absolute_line, sample)`` of a ground point in this swath."""
    when = orbit.zero_doppler_time(ground_xyz, guess, degree)
    position, _velocity = orbit.interpolate(when, degree)
    slant = float(np.linalg.norm(ground_xyz - position))
    sample = (slant - model.slant_range_m(0)) / model.range_pixel_spacing_m
    absolute = _absolute_line(model, when)
    return absolute, float(sample)


def _absolute_line(model: SwathModel, when: np.datetime64) -> float:
    """Line index using the burst interval that contains ``when``."""
    best = 0
    for index, burst in enumerate(model.bursts):
        elapsed = int((when - burst.azimuth_time).astype(np.int64)) * 1e-9
        if 0.0 <= elapsed <= model.lines_per_burst * model.azimuth_time_interval_s:
            best = index
            break
        best = index
    elapsed = int((when - model.bursts[best].azimuth_time).astype(np.int64)) * 1e-9
    return best * model.lines_per_burst + elapsed / model.azimuth_time_interval_s


def backgeocode(
    reference: RadarGrid,
    secondary: RadarGrid,
    *,
    reference_orbit: Orbit,
    secondary_orbit: Orbit,
    degree: int = 3,
    height_m: float = 0.0,
    dem: object | None = None,
    max_solves: int | None = None,
    max_residual_px: float = 0.05,
    reramp: bool = True,
    resampling: str = "BILINEAR_INTERPOLATION",
) -> np.ndarray:
    """Resample ``secondary`` onto ``reference`` by range-Doppler geolocation.

    Both images are deramped first. Each reference sample is located on the
    ellipsoid or on ``dem`` and the secondary value is resampled there, then
    reramped unless ``reramp`` is false. ``max_solves=None`` solves every
    pixel. A smaller budget interpolates the line/sample map and is kept only
    when check pixels stay within ``max_residual_px`` of a dense solve.
    """
    backgeocode.last_stride = 1
    backgeocode.last_residual_px = 0.0
    reference_deramped = deramp(reference, remove=True)
    secondary_deramped = deramp(secondary, remove=True)
    rows, cols = reference.shape
    limit = rows * cols if max_solves is None else int(max_solves)
    stride = _solve_stride(rows, cols, limit)
    residual = 0.0
    while True:
        line_s, sample_s, row_index, col_index = _locate_grid(
            reference,
            secondary,
            reference_orbit=reference_orbit,
            secondary_orbit=secondary_orbit,
            degree=degree,
            height_m=height_m,
            dem=dem,
            stride=stride,
        )
        dense = stride == 1 and row_index.size == rows and col_index.size == cols
        if not dense:
            full_line = _interpolate_grid(row_index, col_index, line_s, rows, cols)
            full_sample = _interpolate_grid(row_index, col_index, sample_s, rows, cols)
            residual = _map_residual(
                reference,
                secondary,
                full_line,
                full_sample,
                reference_orbit=reference_orbit,
                secondary_orbit=secondary_orbit,
                degree=degree,
                height_m=height_m,
                dem=dem,
                row_index=row_index,
                col_index=col_index,
            )
            if residual <= float(max_residual_px) or stride == 1:
                line_s, sample_s = full_line, full_sample
                break
            stride = max(1, stride // 2)
            continue
        residual = 0.0
        break
    backgeocode.last_stride = int(stride)
    backgeocode.last_residual_px = float(residual)
    sampled = resample_complex(secondary_deramped, line_s, sample_s, resampling)
    if not reramp:
        return sampled
    return deramp(RadarGrid(reference.row, reference.col, sampled, reference.model), remove=False)


def _locate_grid(
    reference: RadarGrid,
    secondary: RadarGrid,
    *,
    reference_orbit: Orbit,
    secondary_orbit: Orbit,
    degree: int,
    height_m: float,
    dem: object | None,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = reference.shape
    row_index = _sample_index(rows, stride)
    col_index = _sample_index(cols, stride)
    line_s = np.empty((row_index.size, col_index.size), dtype=np.float64)
    sample_s = np.empty_like(line_s)
    for i, row in enumerate(row_index):
        for j, col in enumerate(col_index):
            line_s[i, j], sample_s[i, j] = _one_secondary_coordinate(
                reference,
                secondary,
                int(row),
                int(col),
                reference_orbit=reference_orbit,
                secondary_orbit=secondary_orbit,
                degree=degree,
                height_m=height_m,
                dem=dem,
            )
    return line_s, sample_s, row_index, col_index


def _one_secondary_coordinate(
    reference: RadarGrid,
    secondary: RadarGrid,
    row: int,
    col: int,
    *,
    reference_orbit: Orbit,
    secondary_orbit: Orbit,
    degree: int,
    height_m: float,
    dem: object | None,
) -> tuple[float, float]:
    absolute = reference.row + int(row)
    azimuth = reference.model.line_time(absolute)
    slant = reference.model.slant_range_m(reference.col + int(col))
    height = float(height_m)
    ground = ground_point_at_height(reference_orbit, azimuth, slant, height, degree=degree)
    if dem is not None:
        lat, lon, _got = ecef_to_geodetic(ground)
        sampled, valid = dem.ellipsoidal_height(float(np.rad2deg(lat)), float(np.rad2deg(lon)))
        if not valid:
            return float("nan"), float("nan")
        ground = ground_point_at_height(reference_orbit, azimuth, slant, sampled, degree=degree)
    line, sample = radar_coordinates(
        secondary.model,
        secondary_orbit,
        ground,
        guess=repeat_azimuth_time(reference.model, absolute, secondary.model),
        degree=degree,
    )
    return line - secondary.row, sample - secondary.col


def _map_residual(
    reference: RadarGrid,
    secondary: RadarGrid,
    line_s: np.ndarray,
    sample_s: np.ndarray,
    *,
    reference_orbit: Orbit,
    secondary_orbit: Orbit,
    degree: int,
    height_m: float,
    dem: object | None,
    row_index: np.ndarray,
    col_index: np.ndarray,
) -> float:
    """Largest line or sample error of the interpolated map at off-lattice pixels."""
    rows, cols = reference.shape
    rng = np.random.default_rng(1)
    errors = []
    lattice_rows = set(int(item) for item in row_index)
    lattice_cols = set(int(item) for item in col_index)
    for _ in range(16):
        row = int(rng.integers(0, rows))
        col = int(rng.integers(0, cols))
        if row in lattice_rows and col in lattice_cols:
            continue
        true_line, true_sample = _one_secondary_coordinate(
            reference,
            secondary,
            row,
            col,
            reference_orbit=reference_orbit,
            secondary_orbit=secondary_orbit,
            degree=degree,
            height_m=height_m,
            dem=dem,
        )
        if not np.isfinite(true_line) or not np.isfinite(line_s[row, col]):
            continue
        errors.append(max(abs(true_line - line_s[row, col]), abs(true_sample - sample_s[row, col])))
    return 0.0 if not errors else float(max(errors))


def resample_complex(image: np.ndarray, line: np.ndarray, sample: np.ndarray, method: str) -> np.ndarray:
    """Resample a complex image. Unsupported names raise instead of changing kernel."""
    from scipy.ndimage import map_coordinates

    name = str(method).strip().upper()
    line = np.asarray(line, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.float64)
    valid = np.isfinite(line) & np.isfinite(sample)
    line = np.where(valid, line, 0.0)
    sample = np.where(valid, sample, 0.0)
    orders = {
        "NEAREST_NEIGHBOUR": 0,
        "BILINEAR_INTERPOLATION": 1,
        "CUBIC_CONVOLUTION": 3,
        "BICUBIC_INTERPOLATION": 3,
    }
    rows, cols = image.shape
    if name in orders:
        coords = np.stack((np.asarray(line, dtype=np.float64), np.asarray(sample, dtype=np.float64)))
        real = map_coordinates(np.asarray(image.real), coords, order=orders[name], mode="constant", cval=0.0)
        imag = map_coordinates(np.asarray(image.imag), coords, order=orders[name], mode="constant", cval=0.0)
        out = (real + 1j * imag).astype(np.complex64)
        out[~valid] = 0
        return out
    radius = {"BISINC_5_POINT_INTERPOLATION": 2, "BISINC_11_POINT_INTERPOLATION": 5, "BISINC_21_POINT_INTERPOLATION": 10}
    if name not in radius:
        raise ValueError(f"{method} is not an implemented resampling kernel.")
    out = _sinc_resample(image, line, sample, radius[name])
    out[~valid] = 0
    return out


def _sinc_resample(image: np.ndarray, line: np.ndarray, sample: np.ndarray, radius: int) -> np.ndarray:
    rows, cols = image.shape
    out = np.zeros(line.shape, dtype=np.complex64)
    line = np.asarray(line, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.float64)
    for row in range(line.shape[0]):
        for col in range(line.shape[1]):
            yy = line[row, col]
            xx = sample[row, col]
            if not np.isfinite(yy) or not np.isfinite(xx):
                continue
            y0 = int(np.floor(yy))
            x0 = int(np.floor(xx))
            value = 0j
            norm = 0.0
            for dy in range(-radius, radius + 1):
                iy = y0 + dy
                if iy < 0 or iy >= rows:
                    continue
                wy = float(np.sinc(yy - iy))
                for dx in range(-radius, radius + 1):
                    ix = x0 + dx
                    if ix < 0 or ix >= cols:
                        continue
                    weight = wy * float(np.sinc(xx - ix))
                    value += weight * image[iy, ix]
                    norm += weight
            if norm != 0.0:
                out[row, col] = value / norm
    return out


def _solve_stride(rows: int, cols: int, max_solves: int) -> int:
    stride = 1
    limit = max(1, int(max_solves))
    while int(np.ceil(rows / stride)) * int(np.ceil(cols / stride)) > limit:
        stride *= 2
        if stride > max(rows, cols):
            break
    return stride


def _sample_index(count: int, stride: int) -> np.ndarray:
    index = list(range(0, count, max(int(stride), 1)))
    if not index or index[-1] != count - 1:
        index.append(count - 1)
    return np.asarray(index, dtype=np.int32)


def _interpolate_grid(
    row_index: np.ndarray,
    col_index: np.ndarray,
    values: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Bilinear resample of a solved lattice onto every pixel."""
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (row_index.astype(np.float64), col_index.astype(np.float64)),
        np.asarray(values, dtype=np.float64),
        bounds_error=False,
        fill_value=None,
    )
    full_rows, full_cols = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    points = np.stack((full_rows, full_cols), axis=-1)
    return interpolator(points)


def _bilinear(image: np.ndarray, row: float, col: float) -> complex:
    if not np.isfinite(row) or not np.isfinite(col):
        return 0j
    rows, cols = image.shape
    if row < 0 or col < 0 or row > rows - 1 or col > cols - 1:
        return 0j
    row0 = int(np.floor(row))
    col0 = int(np.floor(col))
    row1 = min(row0 + 1, rows - 1)
    col1 = min(col0 + 1, cols - 1)
    wy = row - row0
    wx = col - col0
    value = (
        (1 - wy) * (1 - wx) * image[row0, col0]
        + (1 - wy) * wx * image[row0, col1]
        + wy * (1 - wx) * image[row1, col0]
        + wy * wx * image[row1, col1]
    )
    return complex(value)


def spectral_diversity_shift(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    azimuth_time_interval_s: float,
) -> float:
    """Azimuth time shift, in seconds, from the overlap spectral diversity.

    The upper and lower halves of the azimuth spectrum of the deramped
    overlap interferogram have a phase slope. Their beat phase divided by
    ``2π`` times the frequency separation is the azimuth shift. A shift of
    zero is a perfect overlap, not a claim about scene coherence.
    """
    if reference.shape != secondary.shape or reference.ndim != 2:
        raise ValueError("ESD needs two equal 2-D overlap patches")
    interferogram = np.asarray(secondary, dtype=np.complex64) * np.conj(np.asarray(reference, dtype=np.complex64))
    spectrum = np.fft.fftshift(np.fft.fft(interferogram, axis=0), axes=0)
    half = spectrum.shape[0] // 4
    if half < 1:
        raise ValueError("overlap is too short for a spectral split")
    center = spectrum.shape[0] // 2
    lower = np.mean(spectrum[center - 2 * half : center - half], axis=0)
    upper = np.mean(spectrum[center + half : center + 2 * half], axis=0)
    beat = np.mean(upper * np.conj(lower))
    # Quarter-band centres sit at ±0.375 of the sampling rate, 0.75/Δt apart.
    separation = 0.75 / float(azimuth_time_interval_s)
    return float(np.angle(beat) / (2.0 * np.pi * separation))


def burst_overlap_rows(model: SwathModel, burst_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Absolute half-open row ranges of one burst overlap.

    The first range is the end of burst ``burst_index``. The second is the
    start of the next burst. Both cover the same azimuth-time interval.
    """
    if burst_index < 0 or burst_index + 1 >= len(model.bursts):
        raise ValueError(f"burst {burst_index} has no following burst")
    interval = float(model.azimuth_time_interval_s)
    gap_s = int((model.bursts[burst_index + 1].azimuth_time - model.bursts[burst_index].azimuth_time).astype(np.int64)) * 1e-9
    first_in_burst = int(np.ceil(gap_s / interval - 1e-9))
    count = int(model.lines_per_burst) - first_in_burst
    if count < 8:
        raise ValueError("burst overlap is shorter than 8 lines")
    early = burst_index * int(model.lines_per_burst) + first_in_burst
    late = (burst_index + 1) * int(model.lines_per_burst)
    return (early, early + count), (late, late + count)


def overlap_doppler_separation_hz(model: SwathModel, burst_index: int) -> float:
    """Spectral separation, in Hz, of the two TOPS looks in one overlap.

    It is the azimuth FM rate times the burst-to-burst time. It is not the
    pulse repetition frequency.
    """
    if burst_index < 0 or burst_index + 1 >= len(model.bursts):
        raise ValueError(f"burst {burst_index} has no following burst")
    gap_s = abs(
        int((model.bursts[burst_index + 1].azimuth_time - model.bursts[burst_index].azimuth_time).astype(np.int64))
    ) * 1e-9
    rate = abs(float(model.fm_rate[0]))
    if gap_s <= 0.0 or rate <= 0.0:
        raise ValueError("FM rate and burst interval must be positive")
    return rate * gap_s


def enhanced_spectral_diversity(
    master_early: np.ndarray,
    master_late: np.ndarray,
    slave_early: np.ndarray,
    slave_late: np.ndarray,
    *,
    separation_hz: float,
    estimator: str = "Periodogram",
    coherence_threshold: float = 0.2,
) -> tuple[float, float]:
    """Azimuth shift in seconds and the double-difference coherence.

    ``Average`` is the coherence-weighted argument. ``Periodogram`` is the
    unweighted single-tone peak. Coherence below ``coherence_threshold``
    raises, and the shift is not returned as a measurement.
    """
    looks = (master_early, master_late, slave_early, slave_late)
    shape = np.asarray(master_early).shape
    if any(np.asarray(item).shape != shape for item in looks):
        raise ValueError("ESD overlap looks must share one shape")
    if float(separation_hz) <= 0.0:
        raise ValueError("spectral separation must be positive")
    name = str(estimator).strip().lower()
    if name not in {"average", "periodogram"}:
        raise ValueError(f"{estimator} is not an ESD estimator. Use Average or Periodogram.")
    ifg_early = np.asarray(slave_early, dtype=np.complex64) * np.conj(np.asarray(master_early, dtype=np.complex64))
    ifg_late = np.asarray(slave_late, dtype=np.complex64) * np.conj(np.asarray(master_late, dtype=np.complex64))
    cross = ifg_early * np.conj(ifg_late)
    weight = np.abs(cross)
    coherence = float(np.abs(np.sum(cross)) / (np.sum(weight) + 1e-12))
    if coherence < float(coherence_threshold):
        raise RuntimeError(
            f"ESD coherence {coherence:.3f} is below {float(coherence_threshold):.2f}. "
            "The azimuth shift was not applied."
        )
    if name == "average":
        phase = np.angle(np.sum(cross * weight))
    else:
        phase = np.angle(np.sum(cross))
    return float(phase / (2.0 * np.pi * float(separation_hz))), coherence


def apply_azimuth_pixel_shift(image: np.ndarray, pixels: float, azimuth_time_interval_s: float) -> np.ndarray:
    """Shift a complex image by ``pixels`` azimuth lines with a Fourier ramp."""
    spectrum = np.fft.fft(np.asarray(image, dtype=np.complex64), axis=0)
    frequency = np.fft.fftfreq(image.shape[0], d=float(azimuth_time_interval_s))
    seconds = float(pixels) * float(azimuth_time_interval_s)
    spectrum *= np.exp(-1j * 2.0 * np.pi * frequency[:, None] * seconds)
    return np.fft.ifft(spectrum, axis=0).astype(np.complex64)


def deburst_window(data: np.ndarray, model: SwathModel, row: int, col: int) -> RadarGrid:
    """Deburst one column window of a split swath that still contains overlap."""
    image = np.asarray(data)
    if image.ndim != 2:
        raise ValueError("deburst input must be a 2-D burst window")
    grids: list[RadarGrid] = []
    rows = image.shape[0]
    for index in range(len(model.bursts)):
        start, stop = burst_rows(model, index)
        local_start = start - int(row)
        local_stop = stop - int(row)
        if local_stop <= 0 or local_start >= rows:
            continue
        src0 = max(local_start, 0)
        src1 = min(local_stop, rows)
        grids.append(RadarGrid(int(row) + src0, int(col), image[src0:src1], model))
    return deburst(grids)


def square_pixel_azimuth_looks(model: SwathModel, range_looks: int, orbit: Orbit | None = None) -> int:
    """Azimuth looks that make one multilook cell nearly square on the ground."""
    if orbit is not None and orbit is not model.orbit:
        model = _with_orbit(model, orbit)
    azimuth = model.azimuth_ground_spacing_m()
    ground_range = model.ground_range_spacing_m()
    if azimuth <= 0 or ground_range <= 0:
        raise ValueError("pixel spacing must be positive")
    looks = int(np.rint(int(range_looks) * ground_range / azimuth))
    return max(1, looks)


def _with_orbit(model: SwathModel, orbit: Orbit) -> SwathModel:
    return SwathModel(
        wavelength_m=model.wavelength_m,
        azimuth_time_interval_s=model.azimuth_time_interval_s,
        range_pixel_spacing_m=model.range_pixel_spacing_m,
        slant_range_time_s=model.slant_range_time_s,
        lines_per_burst=model.lines_per_burst,
        samples_per_burst=model.samples_per_burst,
        number_of_lines=model.number_of_lines,
        incidence_rad=model.incidence_rad,
        fm_rate=model.fm_rate,
        bursts=model.bursts,
        orbit=orbit,
    )


def ellipsoid_geocode(
    grid: RadarGrid,
    orbit: Orbit,
    *,
    height_m: float = 0.0,
    out_rows: int = 8,
    out_cols: int = 8,
    degree: int = 3,
    resampling: str = "BILINEAR_INTERPOLATION",
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Inverse range-Doppler resample at one ellipsoidal height.

    This is not terrain correction. Every output coordinate uses ``height_m``.
    """
    return _geocode(
        grid,
        orbit,
        height_of=lambda _lat, _lon: (float(height_m), True),
        out_rows=out_rows,
        out_cols=out_cols,
        degree=degree,
        resampling=resampling,
        surface="ellipsoid",
    )


def terrain_correct(
    grid: RadarGrid,
    orbit: Orbit,
    dem: object,
    *,
    out_rows: int = 8,
    out_cols: int = 8,
    degree: int = 3,
    resampling: str = "BILINEAR_INTERPOLATION",
    dem_resampling: str = "BILINEAR_INTERPOLATION",
    projection: str = "geographic",
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """Inverse range-Doppler resample using a DEM height at every output coordinate.

    Returns the image, ``(west, south, east, north)`` in degrees, and the
    ellipsoidal height used at each output pixel (NaN where the DEM is no-data).
    """
    if dem is None:
        raise ValueError("Terrain correction needs a DEM. Ellipsoid height 0 is not a substitute.")

    def height_of(lat_deg: float, lon_deg: float) -> tuple[float, bool]:
        return dem.ellipsoidal_height(lat_deg, lon_deg, method=dem_resampling)

    image, bounds, heights = _geocode(
        grid,
        orbit,
        height_of=height_of,
        out_rows=out_rows,
        out_cols=out_cols,
        degree=degree,
        resampling=resampling,
        surface="dem",
        projection=projection,
    )
    return image, bounds, heights


def _geocode(
    grid: RadarGrid,
    orbit: Orbit,
    *,
    height_of,
    out_rows: int,
    out_cols: int,
    degree: int,
    resampling: str,
    surface: str,
    projection: str = "geographic",
):
    if int(out_rows) < 2 or int(out_cols) < 2:
        raise ValueError("output grid must be at least 2 by 2")
    mode = str(projection).strip().lower()
    if mode not in {"geographic", "wgs84(dd)", "utm"}:
        raise ValueError(f"{projection} is not implemented. Use geographic or UTM.")
    rows, cols = grid.shape
    lats: list[float] = []
    lons: list[float] = []
    for row, col in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
        azimuth = grid.model.line_time(grid.row + row)
        slant = grid.model.slant_range_m(grid.col + col)
        ground = ground_point_at_height(orbit, azimuth, slant, 0.0, degree=degree)
        lat, lon, _height = ecef_to_geodetic(ground)
        lats.append(float(np.rad2deg(lat)))
        lons.append(float(np.rad2deg(lon)))
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    if mode == "utm":
        from pyproj import Transformer

        zone = int((float(np.mean(lons)) + 180.0) // 6.0) + 1
        epsg = (32600 if float(np.mean(lats)) >= 0.0 else 32700) + zone
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        to_geo = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        eastings, northings = to_utm.transform(lons, lats)
        east_axis = np.linspace(min(eastings), max(eastings), int(out_cols))
        north_axis = np.linspace(max(northings), min(northings), int(out_rows))
        lat_axis = np.empty((int(out_rows), int(out_cols)))
        lon_axis = np.empty_like(lat_axis)
        for row_index, northing in enumerate(north_axis):
            lon_axis[row_index], lat_axis[row_index] = to_geo.transform(east_axis, np.full(int(out_cols), northing))
        _geocode.last_utm_zone = zone
        _geocode.last_utm_epsg = epsg
        _geocode.last_utm_bounds = (float(min(eastings)), float(min(northings)), float(max(eastings)), float(max(northings)))
    else:
        lat_axis = np.linspace(north, south, int(out_rows))
        lon_axis = np.linspace(west, east, int(out_cols))
        lat_axis, lon_axis = np.meshgrid(lat_axis, lon_axis, indexing="ij")
        _geocode.last_utm_zone = None
        _geocode.last_utm_bounds = None
    line = np.full((int(out_rows), int(out_cols)), np.nan)
    sample = np.full_like(line, np.nan)
    heights = np.full_like(line, np.nan)
    guess = grid.model.line_time(grid.row + rows // 2)
    for row_index in range(int(out_rows)):
        for col_index in range(int(out_cols)):
            lat_deg = float(lat_axis[row_index, col_index])
            lon_deg = float(lon_axis[row_index, col_index])
            height, valid = height_of(lat_deg, lon_deg)
            if not valid:
                continue
            heights[row_index, col_index] = height
            xyz = geodetic_to_ecef(np.deg2rad(lat_deg), np.deg2rad(lon_deg), np.asarray(height, dtype=np.float64))
            try:
                absolute, pixel = radar_coordinates(grid.model, orbit, xyz, guess=guess, degree=degree)
            except ValueError:
                continue
            line[row_index, col_index] = absolute - grid.row
            sample[row_index, col_index] = pixel - grid.col
    image = resample_complex(grid.data, line, sample, resampling)
    _geocode.last_surface = surface
    if surface == "ellipsoid":
        return image, (west, south, east, north)
    return image, (west, south, east, north), heights


def geocode_window(
    grid: RadarGrid,
    orbit: Orbit,
    *,
    degree: int = 3,
    height_m: float = 0.0,
    step: int = 8,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Sample the window onto the lat/lon box of its corners.

    Returns the sampled complex image and ``(west, south, east, north)`` in
    degrees. ``step`` thins the source so a burst window stays small.
    """
    rows, cols = grid.shape
    corners = []
    for row, col in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
        azimuth = grid.model.line_time(grid.row + row)
        slant = grid.model.slant_range_m(grid.col + col)
        ground = ground_point_at_height(orbit, azimuth, slant, height_m, degree=degree)
        lat, lon, _height = ecef_to_geodetic(ground)
        corners.append((float(np.rad2deg(lat)), float(np.rad2deg(lon))))
    lats = [item[0] for item in corners]
    lons = [item[1] for item in corners]
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    out_rows = max(2, rows // step)
    out_cols = max(2, cols // step)
    sampled = np.zeros((out_rows, out_cols), dtype=np.complex64)
    lat_axis = np.linspace(north, south, out_rows)
    lon_axis = np.linspace(west, east, out_cols)
    # Inverse: nearest source pixel by projecting each source sample we already
    # know, rather than solving every output coordinate. This rasterises the
    # known radar samples into the lat/lon box.
    for row in range(0, rows, step):
        azimuth = grid.model.line_time(grid.row + row)
        for col in range(0, cols, step):
            slant = grid.model.slant_range_m(grid.col + col)
            ground = ground_point_at_height(orbit, azimuth, slant, height_m, degree=degree)
            lat, lon, _height = ecef_to_geodetic(ground)
            lat_deg = float(np.rad2deg(lat))
            lon_deg = float(np.rad2deg(lon))
            out_row = int(np.clip(round((north - lat_deg) / max(north - south, 1e-9) * (out_rows - 1)), 0, out_rows - 1))
            out_col = int(np.clip(round((lon_deg - west) / max(east - west, 1e-9) * (out_cols - 1)), 0, out_cols - 1))
            sampled[out_row, out_col] = grid.data[row, col]
    return sampled, (west, south, east, north)


def write_geotiff(path, image: np.ndarray, bounds: tuple[float, float, float, float] | None = None) -> None:
    """Write a float32 GeoTIFF. Bounds are west, south, east, north in degrees.

    When bounds are present the file is EPSG:4326 via rasterio. Complex data
    is two bands, real then imaginary.
    """
    from pathlib import Path

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("GeoTIFF input must be 2-D or a stack of 2-D bands")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if np.iscomplexobj(array):
        stored = np.stack((array.real, array.imag), axis=0).astype(np.float32)
        height, width = array.shape[-2], array.shape[-1]
    else:
        stored = np.asarray(array, dtype=np.float32)
        if stored.ndim == 2:
            stored = stored[None, ...]
        height, width = stored.shape[-2], stored.shape[-1]
    if stored.ndim == 2:
        stored = stored[None, ...]
    if bounds is None:
        import tifffile

        tifffile.imwrite(destination, stored if stored.shape[0] > 1 else stored[0])
        return
    import rasterio
    from rasterio.transform import from_bounds

    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=int(stored.shape[0]),
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(stored)


def write_kmz(path, image: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    """Write a KMZ ground overlay. ``bounds`` is west, south, east, north."""
    import io
    import zipfile
    from pathlib import Path

    west, south, east, north = bounds
    amplitude = np.abs(np.asarray(image))
    finite = amplitude[np.isfinite(amplitude)]
    if finite.size == 0:
        raise ValueError("KMZ image has no finite samples")
    scale = np.percentile(finite, 98)
    gray = np.clip(amplitude / max(scale, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError:
        # Uncompressed PNG via the standard library would be long; tifffile is
        # not a KML icon. Encode a minimal PNG with a tiny writer below.
        png = _png_gray(gray)
    else:
        buffer = io.BytesIO()
        Image.fromarray(gray, mode="L").save(buffer, format="PNG")
        png = buffer.getvalue()
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <GroundOverlay>
    <name>AI Insar</name>
    <Icon><href>overlay.png</href></Icon>
    <LatLonBox>
      <north>{north:.8f}</north>
      <south>{south:.8f}</south>
      <east>{east:.8f}</east>
      <west>{west:.8f}</west>
    </LatLonBox>
  </GroundOverlay>
</kml>
"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", kml)
        archive.writestr("overlay.png", png)


def _png_gray(gray: np.ndarray) -> bytes:
    import struct
    import zlib

    height, width = gray.shape

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + gray[row].tobytes() for row in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw)),
            chunk(b"IEND", b""),
        )
    )
