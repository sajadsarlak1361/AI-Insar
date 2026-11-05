"""TOPS split, deburst, flat-earth, and range-Doppler resampling.

These are the published Sentinel-1 relationships: burst blocks of
``linesPerBurst``, overlap cut at the mid azimuth time, deramp with the
annotation FM rate, and right-looking zero-Doppler geolocation. They are
not a crop and not an amplitude correlation.
"""

from __future__ import annotations

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


def deburst_lines(model: SwathModel) -> list[tuple[int, int, int]]:
    """Absolute rows kept from each burst after the mid-time overlap cut.

    Each item is ``(burst_index, first_line, last_line)`` with both lines
    included. Lines marked invalid by ``firstValidSample < 0`` are excluded.
    """
    kept: list[tuple[int, int, int]] = []
    mids: list[np.datetime64] = []
    for index in range(len(model.bursts) - 1):
        end = model.line_time((index + 1) * model.lines_per_burst - 1)
        start_next = model.bursts[index + 1].azimuth_time
        # Midpoint in nanoseconds, avoiding datetime float division.
        delta = (start_next - end).astype(np.int64)
        mids.append(end + np.timedelta64(int(delta // 2), "ns"))
    for index, burst in enumerate(model.bursts):
        first = burst.first_valid_line
        last = burst.last_valid_line
        if index > 0:
            while first <= last and model.line_time(index * model.lines_per_burst + first) < mids[index - 1]:
                first += 1
        if index < len(model.bursts) - 1:
            while last >= first and model.line_time(index * model.lines_per_burst + last) > mids[index]:
                last -= 1
        if first <= last:
            kept.append((index, first, last))
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
    pieces: list[np.ndarray] = []
    first_absolute = None
    plan = {index: (start, stop) for index, start, stop in deburst_lines(model)}
    for grid in grids:
        if grid.col != col or grid.model is not model and grid.model.lines_per_burst != model.lines_per_burst:
            raise ValueError("deburst windows must share the column crop and swath")
        burst_index = grid.row // model.lines_per_burst
        if burst_index not in plan:
            continue
        start, stop = plan[burst_index]
        local_start = start - (grid.row - burst_index * model.lines_per_burst)
        local_stop = stop - (grid.row - burst_index * model.lines_per_burst) + 1
        local_start = max(0, local_start)
        local_stop = min(grid.data.shape[0], local_stop)
        if local_start >= local_stop:
            continue
        if first_absolute is None:
            first_absolute = burst_index * model.lines_per_burst + local_start
        pieces.append(np.asarray(grid.data[local_start:local_stop], dtype=np.complex64))
    if not pieces or first_absolute is None:
        raise ValueError("deburst kept no lines")
    return RadarGrid(row=int(first_absolute), col=col, data=np.concatenate(pieces, axis=0), model=model)


def flat_earth_phase(
    early: RadarGrid,
    later: RadarGrid,
    *,
    early_orbit: Orbit,
    later_orbit: Orbit,
    degree: int = 3,
    height_m: float = 0.0,
    max_solves: int = 4096,
) -> np.ndarray:
    """Geometric reference-surface phase, later minus earlier, in radians.

    ``phi = -4π/λ (R_later - R_early)`` on the ellipsoid (or a constant
    height). The same orbit for both dates returns zeros. Windows larger
    than ``max_solves`` are solved on a stride and interpolated; the phase
    surface is smooth, and the samples are still range-Doppler ranges.
    """
    if early.shape != later.shape or early.row != later.row or early.col != later.col:
        raise ValueError("flat-earth phase needs two windows on the same radar grid")
    rows, cols = early.shape
    stride = _solve_stride(rows, cols, max_solves)
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
    max_solves: int = 4096,
    reramp: bool = True,
) -> np.ndarray:
    """Resample ``secondary`` onto ``reference`` by range-Doppler geolocation.

    Both images are deramped first. Reference samples are located on the
    constant-height ellipsoid and the secondary value is bilinearly
    interpolated there, then reramped in the reference geometry unless
    ``reramp`` is false. A window larger than ``max_solves`` is located on a
    stride and the line/sample map is interpolated. This is not a shared crop
    and not an amplitude cross-correlation.
    """
    reference_deramped = deramp(reference, remove=True)
    secondary_deramped = deramp(secondary, remove=True)
    rows, cols = reference.shape
    stride = _solve_stride(rows, cols, max_solves)
    row_index = _sample_index(rows, stride)
    col_index = _sample_index(cols, stride)
    line_s = np.empty((row_index.size, col_index.size), dtype=np.float64)
    sample_s = np.empty_like(line_s)
    for i, row in enumerate(row_index):
        absolute = reference.row + int(row)
        azimuth = reference.model.line_time(absolute)
        for j, col in enumerate(col_index):
            slant = reference.model.slant_range_m(reference.col + int(col))
            ground = ground_point_at_height(reference_orbit, azimuth, slant, height_m, degree=degree)
            line, sample = radar_coordinates(
                secondary.model,
                secondary_orbit,
                ground,
                guess=repeat_azimuth_time(reference.model, absolute, secondary.model),
                degree=degree,
            )
            line_s[i, j] = line - secondary.row
            sample_s[i, j] = sample - secondary.col
    if not (stride == 1 and row_index.size == rows and col_index.size == cols):
        line_s = _interpolate_grid(row_index, col_index, line_s, rows, cols)
        sample_s = _interpolate_grid(row_index, col_index, sample_s, rows, cols)
    sampled = np.zeros((rows, cols), dtype=np.complex64)
    for row in range(rows):
        for col in range(cols):
            sampled[row, col] = _bilinear(secondary_deramped, float(line_s[row, col]), float(sample_s[row, col]))
    if not reramp:
        return sampled
    return deramp(RadarGrid(reference.row, reference.col, sampled, reference.model), remove=False)


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
) -> float:
    """Azimuth shift, in seconds, from the burst-overlap double difference.

    ``early`` and ``late`` are the two burst looks of the same overlap, already
    the same shape. A shift of zero means the double-difference phase is zero.
    It is not a coherence estimate.
    """
    looks = (master_early, master_late, slave_early, slave_late)
    shape = np.asarray(master_early).shape
    if any(np.asarray(item).shape != shape for item in looks):
        raise ValueError("ESD overlap looks must share one shape")
    if float(separation_hz) <= 0.0:
        raise ValueError("spectral separation must be positive")
    ifg_early = np.asarray(slave_early, dtype=np.complex64) * np.conj(np.asarray(master_early, dtype=np.complex64))
    ifg_late = np.asarray(slave_late, dtype=np.complex64) * np.conj(np.asarray(master_late, dtype=np.complex64))
    cross = ifg_early * np.conj(ifg_late)
    phase = np.angle(np.sum(cross))
    return float(phase / (2.0 * np.pi * float(separation_hz)))


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


def terrain_correct(
    grid: RadarGrid,
    orbit: Orbit,
    *,
    height_m: float = 0.0,
    out_rows: int = 8,
    out_cols: int = 8,
    degree: int = 3,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Inverse range-Doppler resample onto a WGS84 latitude/longitude grid.

    Each output coordinate is converted to ECEF at ``height_m`` and located
    in radar line and sample. Pixels outside the source window stay zero.
    This is not a forward scatter of the source samples.
    """
    if int(out_rows) < 2 or int(out_cols) < 2:
        raise ValueError("terrain-correction grid must be at least 2 by 2")
    rows, cols = grid.shape
    lats: list[float] = []
    lons: list[float] = []
    for row, col in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
        azimuth = grid.model.line_time(grid.row + row)
        slant = grid.model.slant_range_m(grid.col + col)
        ground = ground_point_at_height(orbit, azimuth, slant, height_m, degree=degree)
        lat, lon, _height = ecef_to_geodetic(ground)
        lats.append(float(np.rad2deg(lat)))
        lons.append(float(np.rad2deg(lon)))
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    if north - south <= 0.0 or east - west <= 0.0:
        raise ValueError("terrain-correction footprint has no area")
    lat_axis = np.linspace(north, south, int(out_rows))
    lon_axis = np.linspace(west, east, int(out_cols))
    sampled = np.zeros((int(out_rows), int(out_cols)), dtype=np.complex64)
    guess = grid.model.line_time(grid.row + rows // 2)
    for row_index, lat_deg in enumerate(lat_axis):
        for col_index, lon_deg in enumerate(lon_axis):
            xyz = geodetic_to_ecef(
                np.deg2rad(lat_deg),
                np.deg2rad(lon_deg),
                np.asarray(height_m, dtype=np.float64),
            )
            try:
                line, sample = radar_coordinates(grid.model, orbit, xyz, guess=guess, degree=degree)
            except ValueError:
                continue
            sampled[row_index, col_index] = _bilinear(grid.data, line - grid.row, sample - grid.col)
    return sampled, (west, south, east, north)


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
    """Write a float32 TIFF and, when bounds exist, a world file.

    ``bounds`` is west, south, east, north in degrees. Complex data is stored
    as two bands, real then imaginary. BEAM-DIMAP is not written.
    """
    import tifffile
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
        stored = array.astype(np.float32)
        height, width = stored.shape[-2], stored.shape[-1]
    tifffile.imwrite(destination, stored)
    if bounds is None:
        return
    west, south, east, north = bounds
    pixel_x = (east - west) / max(width, 1)
    pixel_y = (north - south) / max(height, 1)
    world = destination.with_suffix(".tfw")
    world.write_text(
        f"{pixel_x:.12g}\n0\n0\n{-pixel_y:.12g}\n{west + pixel_x / 2:.12g}\n{north - pixel_y / 2:.12g}\n",
        encoding="utf-8",
    )


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
