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
    ground_points_at_height,
)
from ai_insar.pipeline.s1_annotation import SwathModel, snap_utc_seconds


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


def deramp(grid: RadarGrid, *, remove: bool = True, orbit: Orbit | None = None) -> np.ndarray:
    """Remove or restore the TOPS azimuth phase, burst by burst.

    ``remove=False`` reramps. With Doppler polynomials on the swath, the phase
    is the steering-rate FM chirp about the range-dependent reference time
    plus Doppler-centroid demodulation. Otherwise the legacy centre-burst
    chirp is used.
    """
    phase = tops_phase(grid, orbit=orbit)
    sign = 1.0 if remove else -1.0
    return (np.asarray(grid.data, dtype=np.complex64) * np.exp(1j * sign * phase)).astype(np.complex64)


def tops_phase(grid: RadarGrid, orbit: Orbit | None = None) -> np.ndarray:
    """Deramp plus demodulation phase in radians, one value per pixel.

    Azimuth time is measured from the first line of the burst. The FM rate is
    ``ka * ks / (ka - ks)``, with ``ks`` the Doppler rate of the antenna
    steering. The reference time is the mid-burst time shifted by
    ``f_dc / ka``. Demodulation uses ``-2π f_dc η`` with that same azimuth
    time. Polynomials are the burst-indexed annotation records.
    """
    model = grid.model
    rows, cols = grid.shape
    if not model.fm_rates or not model.dc_estimates:
        return _legacy_chirp(grid)
    speed = _speed(model, orbit or model.orbit)
    if speed <= 0.0 or model.wavelength_m <= 0.0:
        return _legacy_chirp(grid)
    steering = 2.0 * speed * float(model.azimuth_steering_rate_rad_s) / float(model.wavelength_m)
    samples = grid.col + np.arange(cols, dtype=np.float64)
    tau = model.slant_range_time_s + 2.0 * samples * model.range_pixel_spacing_m / 299_792_458.0
    phase = np.zeros((rows, cols), dtype=np.float64)
    interval = float(model.azimuth_time_interval_s)
    eta_mid = model.lines_per_burst * interval / 2.0
    anchor = int(model.first_valid_pixel)
    for burst_index, _burst in enumerate(model.bursts):
        burst_row, burst_stop = burst_rows(model, burst_index)
        local_start = max(burst_row, grid.row)
        local_stop = min(burst_stop, grid.row + rows)
        if local_start >= local_stop:
            continue
        # A split raster addresses the first written burst as record 0.
        # The absolute SAFE burst number is not that index.
        local_burst = burst_index - int(grid.row) // int(model.lines_per_burst)
        record = local_burst if 0 <= local_burst < len(model.fm_rates) else burst_index
        fm = model.fm_rates[min(max(record, 0), len(model.fm_rates) - 1)]
        dc = model.dc_estimates[min(max(record, 0), len(model.dc_estimates) - 1)]
        ka = _polynomial(fm.coefficients, tau - fm.t0_s)
        fdc = _polynomial(dc.coefficients, tau - dc.t0_s)
        ka_anchor = float(_polynomial(fm.coefficients, _two_way(model, anchor) - fm.t0_s))
        fdc_anchor = float(_polynomial(dc.coefficients, _two_way(model, anchor) - dc.t0_s))
        if abs(ka_anchor) < 1e-6:
            continue
        rate = ka * steering / (ka - steering) if abs(steering) > 1e-6 else ka
        reference = eta_mid + fdc_anchor / ka_anchor - np.divide(fdc, ka, out=np.zeros_like(fdc), where=np.abs(ka) > 1e-6)
        line = np.arange(local_start, local_stop, dtype=np.float64) - burst_row
        eta = line * interval
        delta = eta[:, None] - reference[None, :]
        block = -np.pi * rate[None, :] * delta**2 - 2.0 * np.pi * fdc[None, :] * eta[:, None]
        phase[local_start - grid.row : local_stop - grid.row] = block
    return phase


def _two_way(model: SwathModel, sample: float) -> float:
    return float(model.slant_range_time_s + 2.0 * float(sample) * model.range_pixel_spacing_m / 299_792_458.0)


def _polynomial(coefficients: np.ndarray, delta: np.ndarray | float) -> np.ndarray:
    value = np.zeros_like(delta, dtype=np.float64)
    power = np.ones_like(delta, dtype=np.float64)
    for coefficient in np.asarray(coefficients, dtype=np.float64):
        value = value + float(coefficient) * power
        power = power * delta
    return value


def _speed(model: SwathModel, orbit: Orbit) -> float:
    when = model.bursts[len(model.bursts) // 2].azimuth_time
    _position, velocity = orbit.interpolate(when, 3)
    return float(np.linalg.norm(velocity))


def _legacy_chirp(grid: RadarGrid) -> np.ndarray:
    """Centre-burst FM chirp used when the annotation has no polynomial list."""
    model = grid.model
    rows, cols = grid.shape
    coefficients = model.fm_rate
    c0 = float(coefficients[0])
    c1 = float(coefficients[1]) if coefficients.size > 1 else 0.0
    c2 = float(coefficients[2]) if coefficients.size > 2 else 0.0
    sample = grid.col + np.arange(cols, dtype=np.float64)
    delta_tau = 2.0 * sample * model.range_pixel_spacing_m / 299_792_458.0
    rate = c0 + c1 * delta_tau + c2 * delta_tau**2
    phase = np.zeros((rows, cols), dtype=np.float64)
    for burst_index, _burst in enumerate(model.bursts):
        burst_row, burst_stop = burst_rows(model, burst_index)
        local_start = max(burst_row, grid.row)
        local_stop = min(burst_stop, grid.row + rows)
        if local_start >= local_stop:
            continue
        line = np.arange(local_start, local_stop, dtype=np.float64) - burst_row
        center = (model.lines_per_burst - 1) / 2.0
        eta = (line - center) * model.azimuth_time_interval_s
        # The legacy multiplier was exp(-j * pi * rate * eta^2) on removal,
        # which is exp(j * phi) for phi = -pi * rate * eta^2.
        phase[local_start - grid.row : local_stop - grid.row] = -np.pi * rate[None, :] * (eta[:, None] ** 2)
    return phase


def deburst_output_rows(model: SwathModel, burst_indexes: Sequence[int] | None = None) -> np.ndarray:
    """Absolute source line of every deburst output line, or ``-1`` for no-data.

    The time axis is SNAP's: burst times are ``getMJD()*86400``, the height
    is the truncated ``(lastLineTime-firstLineTime)/dt``, and an overlap
    sample belongs to the later burst only when its time is strictly after
    the mid-time. Lines outside the truncated valid-azimuth window are
    no-data. ``burst_indexes`` are 0-based positions in the annotation.
    Bursts omitted from a split are not used to trim the outer edge.
    """
    if burst_indexes is None:
        indexes = list(range(len(model.bursts)))
    else:
        indexes = sorted({int(item) for item in burst_indexes})
    if not indexes or indexes[0] < 0 or indexes[-1] >= len(model.bursts):
        raise ValueError(f"burst indexes {indexes} are outside this swath")
    interval = float(model.azimuth_time_interval_s)
    lines = int(model.lines_per_burst)
    first_s = [snap_utc_seconds(model.bursts[index].azimuth_time) for index in indexes]
    last_s = [start + (lines - 1) * interval for start in first_s]
    height = int((last_s[-1] - first_s[0]) / interval)
    if height < 1:
        raise ValueError("deburst time span is empty")
    first_valid = first_s[0] + int(model.bursts[indexes[0]].first_valid_line) * interval
    last_valid = first_s[-1] + int(model.bursts[indexes[-1]].last_valid_line) * interval
    y_min = int((first_valid - first_s[0]) / interval)
    y_max = int((last_valid - first_s[0]) / interval)
    chosen = np.full(height, -1, dtype=np.int32)
    for line in range(height):
        if line < y_min or line > y_max:
            continue
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
        chosen[line] = pick[0] * lines + pick[1]
    if int(np.count_nonzero(chosen >= 0)) == 0:
        raise ValueError("deburst kept no lines")
    return chosen


def deburst_source_rows(model: SwathModel, burst_indexes: Sequence[int] | None = None) -> np.ndarray:
    """Source lines that are copied. Leading and trailing no-data lines are omitted."""
    rows = deburst_output_rows(model, burst_indexes)
    return rows[rows >= 0]


def deburst_lines(model: SwathModel, burst_indexes: Sequence[int] | None = None) -> list[tuple[int, int, int]]:
    """Local inclusive line ranges that are copied from each selected burst.

    Each item is ``(burst_index, first_line, last_line)``. No-data lines at
    the valid-azimuth truncation are not listed here; they are zeros in
    :func:`deburst_output_rows`.
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
    width = int(grids[0].data.shape[1])
    indexes = sorted({int(grid.row // model.lines_per_burst) for grid in grids})
    source_rows = deburst_output_rows(model, indexes)
    copied = source_rows[source_rows >= 0]
    full = bool(copied.size) and all(
        any(int(grid.row) <= int(absolute) < int(grid.row) + int(grid.data.shape[0]) for grid in grids)
        for absolute in copied
    )
    pieces: list[np.ndarray] = []
    first_absolute = None
    for absolute in source_rows:
        if int(absolute) < 0:
            if full:
                if first_absolute is None:
                    first_absolute = int(copied[0])
                pieces.append(np.zeros(width, dtype=np.complex64))
            continue
        placed = False
        for grid in grids:
            if grid.col != col or grid.model.lines_per_burst != model.lines_per_burst:
                raise ValueError("deburst windows must share the column crop and swath")
            local = int(absolute) - int(grid.row)
            if 0 <= local < grid.data.shape[0]:
                if first_absolute is None:
                    first_absolute = int(absolute)
                pieces.append(np.asarray(grid.data[local], dtype=np.complex64))
                placed = True
                break
        if full and not placed:
            raise ValueError("deburst source line is missing from the burst windows")
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
    prefer_burst: int | None = None,
) -> tuple[float, float]:
    """Return ``(absolute_line, sample)`` of a ground point in this swath.

    ``prefer_burst`` selects that burst's raster copy. An overlap azimuth
    time sits in two bursts; the earlier copy is a different TOPS phase.
    Back-geocoding passes the reference burst so the secondary line stays
    on the same burst index.
    """
    when = orbit.zero_doppler_time(ground_xyz, guess, degree)
    position, _velocity = orbit.interpolate(when, degree)
    slant = float(np.linalg.norm(ground_xyz - position))
    sample = (slant - model.slant_range_m(0)) / model.range_pixel_spacing_m
    absolute = _absolute_line(model, when, prefer_burst=prefer_burst)
    return absolute, float(sample)


def _absolute_line(model: SwathModel, when: np.datetime64, prefer_burst: int | None = None) -> float:
    """Line index of ``when``.

    Without ``prefer_burst``, the first burst interval that contains the
    time is used. With it, the line is measured from that burst's first
    azimuth time even when the instant also falls in the previous burst.
    """
    if prefer_burst is not None:
        index = int(prefer_burst)
        if index < 0 or index >= len(model.bursts):
            raise ValueError(f"burst {index} is outside 0..{len(model.bursts) - 1}")
        elapsed = int((when - model.bursts[index].azimuth_time).astype(np.int64)) * 1e-9
        return index * model.lines_per_burst + elapsed / model.azimuth_time_interval_s
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
    tolerance_mm: float | None = None,
    reramp: bool = True,
    resampling: str = "BILINEAR_INTERPOLATION",
    dem_resampling: str = "BILINEAR_INTERPOLATION",
) -> np.ndarray:
    """Resample ``secondary`` onto ``reference`` by range-Doppler geolocation.

    The secondary is deramped with its own burst phase, each reference sample
    is located on the ellipsoid or on ``dem``, and the baseband value is
    resampled there. Reramp multiplies by the secondary phase bilinearly
    sampled at that same source coordinate. The reference image is not
    phase-multiplied: its Doppler stays on the reference grid, and the
    secondary is restored in the secondary's own range-Doppler time, which
    is what keeps the interpolation off the TOPS chirp without substituting
    the reference chirp. ``reramp=False`` leaves the baseband samples.
    ``max_solves=None`` solves every pixel. A smaller budget interpolates
    the line/sample map and is kept only when check pixels stay within
    ``max_residual_px`` of a dense solve.
    """
    backgeocode.last_stride = 1
    backgeocode.last_residual_px = 0.0
    backgeocode.dem_resampling = str(dem_resampling)
    secondary_phase = tops_phase(secondary, orbit=secondary_orbit)
    secondary_deramped = (
        np.asarray(secondary.data, dtype=np.complex64) * np.exp(1j * secondary_phase)
    ).astype(np.complex64)
    rows, cols = reference.shape
    if dem is not None:
        posted_line, posted_sample, reproduction = _coordinates_from_dem_posts(
            reference,
            secondary,
            reference_orbit=reference_orbit,
            secondary_orbit=secondary_orbit,
            degree=degree,
            dem=dem,
        )
        if np.isfinite(posted_line).any():
            line_s, sample_s = posted_line, posted_sample
            backgeocode.last_stride = 1
            backgeocode.last_residual_px = float(reproduction)
            backgeocode.last_geometry = "dem-triangulation"
            overlap = common_overlap_mask(reference, secondary, line_s, sample_s)
            backgeocode.last_overlap_mask = overlap
            extent = overlap_extent(overlap)
            backgeocode.last_overlap_extent = extent
            return _resample_overlap(
                secondary_deramped, secondary_phase, line_s, sample_s, overlap, extent, resampling, reramp
            )
    backgeocode.last_geometry = "radar-lattice"
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
            line_err, sample_err = _map_residual(
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
                lattice_line=line_s,
                lattice_sample=sample_s,
            )
            residual = max(line_err, sample_err)
            if tolerance_mm is None:
                accept = residual <= float(max_residual_px)
            else:
                azimuth_mm = line_err * reference.model.azimuth_ground_spacing_m() * 1000.0
                range_mm = sample_err * reference.model.range_pixel_spacing_m * 1000.0
                accept = azimuth_mm <= float(tolerance_mm) and range_mm <= float(tolerance_mm)
            if accept or stride == 1:
                line_s, sample_s = full_line, full_sample
                break
            stride = max(1, stride // 2)
            continue
        residual = 0.0
        break
    backgeocode.last_stride = int(stride)
    backgeocode.last_residual_px = float(residual)
    overlap = common_overlap_mask(reference, secondary, line_s, sample_s)
    backgeocode.last_overlap_mask = overlap
    extent = overlap_extent(overlap)
    backgeocode.last_overlap_extent = extent
    return _resample_overlap(
        secondary_deramped, secondary_phase, line_s, sample_s, overlap, extent, resampling, reramp
    )


def _resample_overlap(
    secondary_deramped: np.ndarray,
    secondary_phase: np.ndarray,
    line_s: np.ndarray,
    sample_s: np.ndarray,
    overlap: np.ndarray,
    extent: tuple[int, int, int, int] | None,
    resampling: str,
    reramp: bool,
) -> np.ndarray:
    """Resample only the common-overlap box. Everywhere else stays zero."""
    sampled = np.zeros(line_s.shape, dtype=np.complex64)
    if extent is None:
        return sampled
    r0, r1, c0, c1 = extent
    box = overlap[r0:r1, c0:c1]
    sampled_box = resample_complex(
        secondary_deramped,
        line_s[r0:r1, c0:c1],
        sample_s[r0:r1, c0:c1],
        resampling,
        mask=box,
    )
    if reramp:
        phase_at_source = resample_real(
            secondary_phase,
            line_s[r0:r1, c0:c1],
            sample_s[r0:r1, c0:c1],
            resampling,
            mask=box,
        )
        sampled_box = (sampled_box * np.exp(-1j * phase_at_source)).astype(np.complex64)
        sampled_box[~box] = 0
    sampled[r0:r1, c0:c1] = sampled_box
    return sampled


def _coordinates_from_dem_posts(
    reference: RadarGrid,
    secondary: RadarGrid,
    *,
    reference_orbit: Orbit,
    secondary_orbit: Orbit,
    degree: int,
    dem: object,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Secondary line and sample by triangulating DEM posts.

    Each post is turned into an ECEF point at its own ellipsoidal height and
    located in both images by zero Doppler and slant range. Those scattered
    radar coordinates are linearly interpolated onto the reference grid, with
    range and azimuth scaled by their pixel spacing so the triangles follow
    the ground. Pixels outside the posts stay NaN and are left out of the
    common overlap.
    """
    rows, cols = reference.shape
    empty_line = np.full((rows, cols), np.nan, dtype=np.float64)
    empty_sample = np.full((rows, cols), np.nan, dtype=np.float64)
    lat_deg, lon_deg, height = _dem_posts_covering(reference, reference_orbit, degree, dem)
    if height.size < 3:
        return empty_line, empty_sample, 0.0
    xyz = geodetic_to_ecef(np.deg2rad(lat_deg), np.deg2rad(lon_deg), height)
    lines = int(reference.model.lines_per_burst)
    first_burst = int(reference.row) // lines
    last_burst = (int(reference.row) + rows - 1) // lines
    ref_line, ref_sample, sec_line, sec_sample = _paired_radar_index(
        reference.model,
        reference_orbit,
        secondary.model,
        secondary_orbit,
        xyz,
        degree,
        first_burst=first_burst,
        last_burst=last_burst,
    )
    finite = (
        np.isfinite(ref_line)
        & np.isfinite(ref_sample)
        & np.isfinite(sec_line)
        & np.isfinite(sec_sample)
        & (np.abs(ref_line) < 1.0e7)
        & (np.abs(ref_sample) < 1.0e7)
    )
    if int(np.count_nonzero(finite)) < 3:
        return empty_line, empty_sample, 0.0
    ref_line = ref_line[finite]
    ref_sample = ref_sample[finite]
    sec_line = sec_line[finite]
    sec_sample = sec_sample[finite]
    key = np.round(np.column_stack((ref_line, ref_sample)), 5)
    _unique, index = np.unique(key, axis=0, return_index=True)
    ref_line = ref_line[index]
    ref_sample = ref_sample[index]
    sec_line = sec_line[index]
    sec_sample = sec_sample[index]
    if ref_line.size < 3:
        return empty_line, empty_sample, 0.0
    azimuth_scale = float(reference.model.azimuth_ground_spacing_m(degree))
    range_scale = float(reference.model.range_pixel_spacing_m)
    points = np.column_stack((ref_line * azimuth_scale, ref_sample * range_scale))
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError

    try:
        line_of = LinearNDInterpolator(points, sec_line - secondary.row, fill_value=np.nan)
        sample_of = LinearNDInterpolator(points, sec_sample - secondary.col, fill_value=np.nan)
    except QhullError:
        return empty_line, empty_sample, 0.0
    line_s = np.full((rows, cols), np.nan, dtype=np.float64)
    sample_s = np.full((rows, cols), np.nan, dtype=np.float64)
    for row0 in range(0, rows, 128):
        row1 = min(rows, row0 + 128)
        yy, xx = np.mgrid[row0:row1, 0:cols]
        query = np.column_stack(
            (
                (reference.row + yy.ravel()) * azimuth_scale,
                (reference.col + xx.ravel()) * range_scale,
            )
        )
        line_s[row0:row1] = np.asarray(line_of(query), dtype=np.float64).reshape(row1 - row0, cols)
        sample_s[row0:row1] = np.asarray(sample_of(query), dtype=np.float64).reshape(row1 - row0, cols)
    reproduced_line = np.asarray(line_of(points), dtype=np.float64)
    reproduced_sample = np.asarray(sample_of(points), dtype=np.float64)
    truth_line = sec_line - secondary.row
    truth_sample = sec_sample - secondary.col
    ok = np.isfinite(reproduced_line) & np.isfinite(reproduced_sample)
    if not np.any(ok):
        return line_s, sample_s, 0.0
    reproduction = float(
        max(
            np.max(np.abs(reproduced_line[ok] - truth_line[ok])),
            np.max(np.abs(reproduced_sample[ok] - truth_sample[ok])),
        )
    )
    return line_s, sample_s, reproduction


def _dem_posts_covering(reference: RadarGrid, orbit: Orbit, degree: int, dem: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitude, longitude, and ellipsoidal height of the DEM posts around the window."""
    rows, cols = reference.shape
    corner_row = np.array([0, 0, rows - 1, rows - 1, rows // 2], dtype=np.int64)
    corner_col = np.array([0, cols - 1, 0, cols - 1, cols // 2], dtype=np.int64)
    azimuth = _line_times(reference.model, reference.row + corner_row)
    slant = _slant_ranges(reference.model, reference.col + corner_col.astype(np.float64))
    ground = ground_points_at_height(orbit, azimuth, slant, np.zeros(corner_row.size), degree=degree)
    lat, lon, _height = ecef_to_geodetic(ground)
    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    dem_lat = np.asarray(dem.lat_deg, dtype=np.float64)
    dem_lon = np.asarray(dem.lon_deg, dtype=np.float64)
    lat_step = float(np.max(np.abs(np.diff(dem_lat)))) if dem_lat.size > 1 else 0.01
    lon_step = float(np.max(np.abs(np.diff(dem_lon)))) if dem_lon.size > 1 else 0.01
    lat_min = float(np.min(lat_deg)) - 20.0 * lat_step
    lat_max = float(np.max(lat_deg)) + 20.0 * lat_step
    lon_min = float(np.min(lon_deg)) - 20.0 * lon_step
    lon_max = float(np.max(lon_deg)) + 20.0 * lon_step
    lat_index = np.flatnonzero((dem_lat >= lat_min) & (dem_lat <= lat_max))
    lon_index = np.flatnonzero((dem_lon >= lon_min) & (dem_lon <= lon_max))
    if lat_index.size == 0 or lon_index.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    post_lat = dem_lat[lat_index]
    post_lon = dem_lon[lon_index]
    mesh_lat, mesh_lon = np.meshgrid(post_lat, post_lon, indexing="ij")
    method = str(getattr(backgeocode, "dem_resampling", "BILINEAR_INTERPOLATION"))
    height, valid = dem.ellipsoidal_heights(mesh_lat, mesh_lon, method=method)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(height)
    if not np.any(keep):
        return np.empty(0), np.empty(0), np.empty(0)
    return mesh_lat[keep], mesh_lon[keep], height[keep]


def _paired_radar_index(
    reference: SwathModel,
    reference_orbit: Orbit,
    secondary: SwathModel,
    secondary_orbit: Orbit,
    xyz: np.ndarray,
    degree: int,
    first_burst: int = 0,
    last_burst: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Absolute line and sample in both images.

    A post that falls in a burst overlap is kept once for each burst, because
    each burst has its own azimuth time. The two images use the same burst index.
    """
    empty = np.empty(0, dtype=np.float64)
    count = int(xyz.shape[0])
    if count == 0 or not reference.bursts:
        return empty, empty, empty, empty
    ref_lines: list[np.ndarray] = []
    ref_samples: list[np.ndarray] = []
    sec_lines: list[np.ndarray] = []
    sec_samples: list[np.ndarray] = []
    bursts = min(len(reference.bursts), len(secondary.bursts))
    start_burst = max(0, int(first_burst))
    stop_burst = bursts - 1 if last_burst is None else min(bursts - 1, int(last_burst))
    for burst in range(start_burst, stop_burst + 1):
        ref_guess = _burst_guess(reference, burst, count)
        sec_guess = _burst_guess(secondary, burst, count)
        try:
            ref_when = reference_orbit.zero_doppler_times(xyz, ref_guess, degree)
            sec_when = secondary_orbit.zero_doppler_times(xyz, sec_guess, degree)
        except ValueError:
            continue
        ref_position, _ref_velocity = reference_orbit.interpolate(ref_when, degree)
        sec_position, _sec_velocity = secondary_orbit.interpolate(sec_when, degree)
        ref_slant = np.linalg.norm(xyz - ref_position, axis=1)
        sec_slant = np.linalg.norm(xyz - sec_position, axis=1)
        burst_index = np.full(count, burst, dtype=np.int64)
        ref_line = _lines_from_times(reference, ref_when, burst_index)
        sec_line = _lines_from_times(secondary, sec_when, burst_index)
        ref_sample = (ref_slant - reference.slant_range_m(0)) / reference.range_pixel_spacing_m
        sec_sample = (sec_slant - secondary.slant_range_m(0)) / secondary.range_pixel_spacing_m
        start_line = burst * reference.lines_per_burst
        stop_line = start_line + reference.lines_per_burst
        sec_start = burst * secondary.lines_per_burst
        sec_stop = sec_start + secondary.lines_per_burst
        inside = (
            (ref_line >= start_line - 2)
            & (ref_line <= stop_line + 2)
            & (sec_line >= sec_start - 2)
            & (sec_line <= sec_stop + 2)
        )
        if not np.any(inside):
            continue
        ref_lines.append(ref_line[inside])
        ref_samples.append(ref_sample[inside])
        sec_lines.append(sec_line[inside])
        sec_samples.append(sec_sample[inside])
    if not ref_lines:
        return empty, empty, empty, empty
    return (
        np.concatenate(ref_lines),
        np.concatenate(ref_samples),
        np.concatenate(sec_lines),
        np.concatenate(sec_samples),
    )


def _burst_guess(model: SwathModel, burst: int, count: int) -> np.ndarray:
    start = model.bursts[burst].azimuth_time
    mid_ticks = int(round((model.lines_per_burst // 2) * model.azimuth_time_interval_s * 1e9))
    return np.full(count, start + np.timedelta64(mid_ticks, "ns"), dtype="datetime64[ns]")


def common_overlap_mask(
    reference: RadarGrid,
    secondary: RadarGrid,
    source_line: np.ndarray,
    source_sample: np.ndarray,
) -> np.ndarray:
    """Pixels whose reference sample and secondary source both lie in a burst's valid window.

    ``source_line`` and ``source_sample`` are coordinates in the secondary
    raster. Outside this mask nothing is resampled.
    """
    rows, cols = reference.shape
    row = reference.row + np.arange(rows, dtype=np.int64)[:, None]
    col = reference.col + np.arange(cols, dtype=np.int64)[None, :]
    reference_ok = _inside_valid_footprint(reference.model, np.broadcast_to(row, (rows, cols)), np.broadcast_to(col, (rows, cols)))
    line = np.asarray(source_line, dtype=np.float64)
    sample = np.asarray(source_sample, dtype=np.float64)
    finite = np.isfinite(line) & np.isfinite(sample)
    inside_raster = (
        (line >= 0.0)
        & (line <= secondary.data.shape[0] - 1)
        & (sample >= 0.0)
        & (sample <= secondary.data.shape[1] - 1)
    )
    absolute_line = np.zeros(line.shape, dtype=np.int64)
    absolute_sample = np.zeros(sample.shape, dtype=np.int64)
    absolute_line[finite] = secondary.row + np.rint(line[finite]).astype(np.int64)
    absolute_sample[finite] = secondary.col + np.rint(sample[finite]).astype(np.int64)
    secondary_ok = finite & inside_raster & _inside_valid_footprint(secondary.model, absolute_line, absolute_sample)
    return reference_ok & secondary_ok


def overlap_extent(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Half-open bounding box ``(row0, row1, col0, col1)`` of a boolean mask."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def overlap_metadata(mask: np.ndarray) -> dict[str, int]:
    """Counts and the bounding box stored with the coregistered product."""
    extent = overlap_extent(mask)
    if extent is None:
        return {"count": 0, "rows": 0, "cols": 0, "row_start": 0, "row_end": 0, "col_start": 0, "col_end": 0}
    row0, row1, col0, col1 = extent
    return {
        "count": int(np.count_nonzero(mask)),
        "rows": row1 - row0,
        "cols": col1 - col0,
        "row_start": row0,
        "row_end": row1,
        "col_start": col0,
        "col_end": col1,
    }


def _inside_valid_footprint(model: SwathModel, absolute_line: np.ndarray, absolute_sample: np.ndarray) -> np.ndarray:
    """True where the absolute line and sample sit in that burst's valid window."""
    lines = int(model.lines_per_burst)
    burst = np.asarray(absolute_line, dtype=np.int64) // lines
    local = np.asarray(absolute_line, dtype=np.int64) - burst * lines
    count = len(model.bursts)
    known = (burst >= 0) & (burst < count)
    index = np.clip(burst, 0, max(count - 1, 0))
    first_line = np.array([item.first_valid_line for item in model.bursts], dtype=np.int64)
    last_line = np.array([item.last_valid_line for item in model.bursts], dtype=np.int64)
    first_sample = np.array([item.first_valid_sample for item in model.bursts], dtype=np.int64)
    last_sample = np.array([item.last_valid_sample for item in model.bursts], dtype=np.int64)
    sample = np.asarray(absolute_sample, dtype=np.int64)
    return (
        known
        & (local >= first_line[index])
        & (local <= last_line[index])
        & (sample >= first_sample[index])
        & (sample <= last_sample[index])
    )


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
    # One batched zero-Doppler solve for the whole lattice. A Python call per
    # node was about 10 ms, which is hours at an 8-pixel spacing.
    mesh_row, mesh_col = np.meshgrid(row_index, col_index, indexing="ij")
    line_flat, sample_flat = _secondary_coordinates(
        reference,
        secondary,
        mesh_row.ravel(),
        mesh_col.ravel(),
        reference_orbit=reference_orbit,
        secondary_orbit=secondary_orbit,
        degree=degree,
        height_m=height_m,
        dem=dem,
    )
    return (
        line_flat.reshape(mesh_row.shape),
        sample_flat.reshape(mesh_row.shape),
        row_index,
        col_index,
    )


def _secondary_coordinates(
    reference: RadarGrid,
    secondary: RadarGrid,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    reference_orbit: Orbit,
    secondary_orbit: Orbit,
    degree: int,
    height_m: float,
    dem: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Secondary line and sample for many reference pixels.

    This is :func:`_one_secondary_coordinate` with the orbit polynomial, the
    ellipsoid intersection, and the zero-Doppler Newton step evaluated for
    every pixel at once.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    count = rows.shape[0]
    absolute = int(reference.row) + rows
    azimuth = _line_times(reference.model, absolute)
    slant = _slant_ranges(reference.model, int(reference.col) + cols.astype(np.float64))
    height = np.full(count, float(height_m), dtype=np.float64)
    ground = ground_points_at_height(reference_orbit, azimuth, slant, height, degree=degree)
    valid = np.ones(count, dtype=bool)
    if dem is not None:
        ground, valid = _ground_on_dem(ground, azimuth, slant, reference_orbit, dem, degree, valid)
    line = np.full(count, np.nan, dtype=np.float64)
    sample = np.full(count, np.nan, dtype=np.float64)
    if not np.any(valid):
        return line, sample
    chosen = np.flatnonzero(valid)
    burst = absolute[chosen] // int(reference.model.lines_per_burst)
    guess = _repeat_times(reference.model, absolute[chosen], secondary.model)
    when = secondary_orbit.zero_doppler_times(ground[chosen], guess, degree)
    position, _velocity = secondary_orbit.interpolate(when, degree)
    slant_m = np.linalg.norm(ground[chosen] - position, axis=1)
    sample[chosen] = (slant_m - secondary.model.slant_range_m(0)) / secondary.model.range_pixel_spacing_m
    line[chosen] = _lines_from_times(secondary.model, when, burst)
    return line - secondary.row, sample - secondary.col


_DEM_HEIGHT_TOLERANCE_M = 1e-3
_DEM_HEIGHT_MAX_UPDATES = 30


def _ground_on_dem(ground, azimuth, slant, orbit, dem, degree, valid):
    """Put each ground point on the DEM and stop when the height settles.

    A point is finished when the DEM sample moved by less than 1 mm and the
    point's own ellipsoidal height already matches that sample. The same rule
    is used for one pixel and for a batch, so the two solvers stay together.
    """
    method = str(getattr(backgeocode, "dem_resampling", "BILINEAR_INTERPOLATION"))
    count = int(ground.shape[0])
    previous = np.full(count, np.nan, dtype=np.float64)
    azimuth = np.asarray(azimuth)
    slant = np.asarray(slant, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).reshape(count)
    for _ in range(_DEM_HEIGHT_MAX_UPDATES):
        lat, lon, got = ecef_to_geodetic(ground)
        sampled, inside = dem.ellipsoidal_heights(
            np.rad2deg(np.asarray(lat, dtype=np.float64)).reshape(count),
            np.rad2deg(np.asarray(lon, dtype=np.float64)).reshape(count),
            method=method,
        )
        sampled = np.asarray(sampled, dtype=np.float64).reshape(count)
        inside = np.asarray(inside, dtype=bool).reshape(count)
        got = np.asarray(got, dtype=np.float64).reshape(count)
        valid = valid & inside & np.isfinite(sampled)
        settled = (
            valid
            & np.isfinite(previous)
            & (np.abs(sampled - previous) < _DEM_HEIGHT_TOLERANCE_M)
            & (np.abs(got - sampled) < _DEM_HEIGHT_TOLERANCE_M)
        )
        moving = valid & ~settled
        if not np.any(moving):
            break
        updated = ground_points_at_height(
            orbit,
            azimuth,
            slant,
            np.where(moving, sampled, 0.0),
            degree=degree,
        )
        ground = np.where(moving[:, None], updated, ground)
        previous = np.where(moving, sampled, previous)
    return ground, valid


def _line_times(model: SwathModel, absolute: np.ndarray) -> np.ndarray:
    """Azimuth time of many absolute lines, using the scalar nanosecond rounding."""
    lines = np.asarray(absolute, dtype=np.int64)
    burst_index = lines // int(model.lines_per_burst)
    if np.any(burst_index < 0) or np.any(burst_index >= len(model.bursts)):
        raise ValueError("line is outside the swath")
    inside = lines % int(model.lines_per_burst)
    starts = np.array(
        [np.datetime64(item.azimuth_time, "ns").astype(np.int64) for item in model.bursts],
        dtype=np.int64,
    )
    ticks = np.rint(inside.astype(np.float64) * float(model.azimuth_time_interval_s) * 1e9).astype(np.int64)
    return (starts[burst_index] + ticks).astype("datetime64[ns]")


def _slant_ranges(model: SwathModel, sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float64)
    delay = float(model.slant_range_time_s) + 2.0 * sample * float(model.range_pixel_spacing_m) / 299_792_458.0
    return 299_792_458.0 * delay / 2.0


def _repeat_times(reference: SwathModel, absolute: np.ndarray, secondary: SwathModel) -> np.ndarray:
    burst = np.asarray(absolute, dtype=np.int64) // int(reference.lines_per_burst)
    offset = np.asarray(absolute, dtype=np.int64) % int(reference.lines_per_burst)
    burst = np.clip(burst, 0, len(secondary.bursts) - 1)
    offset = np.clip(offset, 0, int(secondary.lines_per_burst) - 1)
    return _line_times(secondary, burst * int(secondary.lines_per_burst) + offset)


def _lines_from_times(model: SwathModel, when: np.ndarray, burst_index: np.ndarray) -> np.ndarray:
    burst_index = np.asarray(burst_index, dtype=np.int64)
    if np.any(burst_index < 0) or np.any(burst_index >= len(model.bursts)):
        raise ValueError(f"burst index is outside 0..{len(model.bursts) - 1}")
    starts = np.array(
        [np.datetime64(item.azimuth_time, "ns").astype(np.int64) for item in model.bursts],
        dtype=np.int64,
    )
    elapsed_ns = np.asarray(when, dtype="datetime64[ns]").astype(np.int64) - starts[burst_index]
    elapsed = elapsed_ns.astype(np.float64) * 1e-9
    return burst_index * int(model.lines_per_burst) + elapsed / float(model.azimuth_time_interval_s)


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
        # Same stopping rule as the batched solver: keep reading the DEM until
        # the sample and the point's ellipsoidal height agree to 1 mm.
        settled, alive = _ground_on_dem(
            np.asarray(ground, dtype=np.float64).reshape(1, 3),
            np.asarray([azimuth]),
            np.asarray([slant], dtype=np.float64),
            reference_orbit,
            dem,
            degree,
            np.ones(1, dtype=bool),
        )
        if not bool(alive[0]):
            return float("nan"), float("nan")
        ground = settled[0]
    burst = int(absolute) // int(reference.model.lines_per_burst)
    line, sample = radar_coordinates(
        secondary.model,
        secondary_orbit,
        ground,
        guess=repeat_azimuth_time(reference.model, absolute, secondary.model),
        degree=degree,
        prefer_burst=burst,
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
    lattice_line: np.ndarray | None = None,
    lattice_sample: np.ndarray | None = None,
) -> tuple[float, float]:
    """Largest line error and sample error, in pixels.

    Knots are compared with the solved lattice, including the border knots.
    A few edge and interior pixels are compared with a fresh solve. The two
    axes are kept separate so a millimetre in azimuth is not the same pixel
    count as a millimetre in range.
    """
    line_err = 0.0
    sample_err = 0.0
    if lattice_line is not None and lattice_sample is not None:
        predicted_line = line_s[np.ix_(row_index, col_index)]
        predicted_sample = sample_s[np.ix_(row_index, col_index)]
        finite = np.isfinite(lattice_line) & np.isfinite(predicted_line)
        if np.any(finite):
            line_err = float(np.max(np.abs(predicted_line[finite] - lattice_line[finite])))
            sample_err = float(np.max(np.abs(predicted_sample[finite] - lattice_sample[finite])))
    rows, cols = reference.shape
    checks = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1), (rows // 2, 0), (rows // 2, cols - 1)]
    rng = np.random.default_rng(1)
    for _ in range(16):
        checks.append((int(rng.integers(0, rows)), int(rng.integers(0, cols))))
    for row, col in checks:
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
        line_err = max(line_err, abs(true_line - line_s[row, col]))
        sample_err = max(sample_err, abs(true_sample - sample_s[row, col]))
    return line_err, sample_err


def resample_real(
    image: np.ndarray,
    line: np.ndarray,
    sample: np.ndarray,
    method: str,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Resample one real array with the same kernel as :func:`resample_complex`.

    Polynomial kernels stay in float64. The deramp phase is thousands of
    radians, and a complex64 round trip would quantise it before the reramp.
    """
    from scipy.ndimage import map_coordinates

    name = str(method).strip().upper()
    orders = {
        "NEAREST_NEIGHBOUR": 0,
        "BILINEAR_INTERPOLATION": 1,
        "CUBIC_CONVOLUTION": 3,
        "BICUBIC_INTERPOLATION": 3,
    }
    values = np.asarray(image, dtype=np.float64)
    line = np.asarray(line, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.float64)
    valid = np.isfinite(line) & np.isfinite(sample)
    keep = valid if mask is None else valid & np.asarray(mask, dtype=bool)
    if name in orders:
        out = np.zeros(line.shape, dtype=np.float64)
        if np.any(keep):
            coords = np.stack((line[keep], sample[keep]))
            out[keep] = map_coordinates(values, coords, order=orders[name], mode="constant", cval=0.0)
        return out
    if name.startswith("BISINC_"):
        # The chirp phase is thousands of radians. Keep it in float64; a
        # complex64 round trip quantises it before the reramp.
        return _hann_sinc_resample(values, line, sample, name, real=True, mask=keep)
    sampled = resample_complex(values + 0j, line, sample, method)
    return np.asarray(sampled.real, dtype=np.float64)


def resample_complex(
    image: np.ndarray,
    line: np.ndarray,
    sample: np.ndarray,
    method: str,
    mask: np.ndarray | None = None,
) -> np.ndarray:
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
    keep = valid if mask is None else valid & np.asarray(mask, dtype=bool)
    if name in orders:
        out = np.zeros(line.shape, dtype=np.complex64)
        if np.any(keep):
            coords = np.stack((line[keep], sample[keep]))
            real = map_coordinates(np.asarray(image.real), coords, order=orders[name], mode="constant", cval=0.0)
            imag = map_coordinates(np.asarray(image.imag), coords, order=orders[name], mode="constant", cval=0.0)
            out[keep] = (real + 1j * imag).astype(np.complex64)
        return out
    if name not in _BISINC_HALF:
        raise ValueError(f"{method} is not an implemented resampling kernel.")
    out = _hann_sinc_resample(image, line, sample, name, real=False, mask=keep)
    out[~keep] = 0
    return out


# Half-width of the raised-cosine window. Five taps span ±2 samples.
_BISINC_HALF = {
    "BISINC_5_POINT_INTERPOLATION": 2,
    "BISINC_11_POINT_INTERPOLATION": 5,
    "BISINC_21_POINT_INTERPOLATION": 10,
}
_HANN_BINS = 4096
_HANN_LUT: dict[int, np.ndarray] = {}


def _hann_sinc_lut(half: int) -> np.ndarray:
    """Weights for fractional offsets in ``[0, 1)``.

    ``w(u) = sinc(u) · 0.5 · (1 + cos(π u / half))`` for ``|u| <= half``,
    else 0. Taps are ``floor(x) + offset`` with ``offset`` from ``-half``
    to ``+half``. A tap past the window is already zero, including the
    extra fractional tap an unwindowed sinc would keep.
    """
    cached = _HANN_LUT.get(half)
    if cached is not None:
        return cached
    frac = np.arange(_HANN_BINS, dtype=np.float64) / _HANN_BINS
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    delta = frac[:, None] - offsets[None, :]
    window = 0.5 * (1.0 + np.cos(np.pi * np.clip(delta / half, -1.0, 1.0)))
    weights = np.sinc(delta) * window
    weights[np.abs(delta) > half] = 0.0
    cached = np.ascontiguousarray(weights, dtype=np.float64)
    _HANN_LUT[half] = cached
    return cached


def _hann_sinc_resample(
    image: np.ndarray,
    line: np.ndarray,
    sample: np.ndarray,
    method: str,
    *,
    real: bool,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Separable Hann-windowed sinc. Each axis is normalised on its own."""
    half = _BISINC_HALF[str(method).strip().upper()]
    lut = _hann_sinc_lut(half)
    values = np.asarray(image)
    line = np.array(line, dtype=np.float64, copy=True)
    sample = np.array(sample, dtype=np.float64, copy=True)
    if mask is not None:
        dropped = ~np.asarray(mask, dtype=bool)
        line[dropped] = np.nan
        sample[dropped] = np.nan
    if real:
        real_part = np.ascontiguousarray(values, dtype=np.float64)
        imag_part = np.zeros((1, 1), dtype=np.float64)
    else:
        real_part = np.ascontiguousarray(np.real(values), dtype=np.float64)
        imag_part = np.ascontiguousarray(np.imag(values), dtype=np.float64)
    out_real, out_imag = _hann_apply(real_part, imag_part, line, sample, lut, half, 0 if real else 1)
    if real:
        return out_real
    return (out_real + 1j * out_imag).astype(np.complex64)


def _hann_apply(real_part, imag_part, line, sample, lut, half, complex_flag):
    try:
        return _hann_apply_numba(real_part, imag_part, line, sample, lut, int(half), int(complex_flag))
    except Exception:
        return _hann_apply_numpy(real_part, imag_part, line, sample, lut, int(half), int(complex_flag))


def _hann_apply_numpy(real_part, imag_part, line, sample, lut, half, complex_flag):
    bins = lut.shape[0]
    taps = lut.shape[1]
    height, width = real_part.shape
    rows, cols = line.shape
    out_real = np.zeros((rows, cols), dtype=np.float64)
    out_imag = np.zeros((rows, cols), dtype=np.float64)
    offsets = np.arange(-half, half + 1)
    for row in range(rows):
        for col in range(cols):
            y = line[row, col]
            x = sample[row, col]
            if not np.isfinite(y) or not np.isfinite(x):
                continue
            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            wy = lut[min(int((y - y0) * bins), bins - 1)].copy()
            wx = lut[min(int((x - x0) * bins), bins - 1)].copy()
            for index, offset in enumerate(offsets):
                if y0 + offset < 0 or y0 + offset >= height:
                    wy[index] = 0.0
                if x0 + offset < 0 or x0 + offset >= width:
                    wx[index] = 0.0
            sy = float(np.sum(wy))
            sx = float(np.sum(wx))
            if sy == 0.0 or sx == 0.0:
                continue
            wy /= sy
            wx /= sx
            acc_r = 0.0
            acc_i = 0.0
            for iy, oy in enumerate(offsets):
                if wy[iy] == 0.0:
                    continue
                yy = y0 + int(oy)
                for ix, ox in enumerate(offsets):
                    if wx[ix] == 0.0:
                        continue
                    xx = x0 + int(ox)
                    weight = wy[iy] * wx[ix]
                    acc_r += weight * real_part[yy, xx]
                    if complex_flag:
                        acc_i += weight * imag_part[yy, xx]
            out_real[row, col] = acc_r
            out_imag[row, col] = acc_i
    return out_real, out_imag


def _hann_apply_numba(real_part, imag_part, line, sample, lut, half, complex_flag):
    import numba

    if not hasattr(_hann_apply_numba, "compiled"):

        @numba.njit(parallel=True, cache=True)
        def _kernel(real_part, imag_part, line, sample, lut, half, complex_flag):
            bins = lut.shape[0]
            height = real_part.shape[0]
            width = real_part.shape[1]
            rows = line.shape[0]
            cols = line.shape[1]
            out_real = np.zeros((rows, cols), dtype=np.float64)
            out_imag = np.zeros((rows, cols), dtype=np.float64)
            for row in numba.prange(rows):
                for col in range(cols):
                    y = line[row, col]
                    x = sample[row, col]
                    if not np.isfinite(y) or not np.isfinite(x):
                        continue
                    y0 = int(np.floor(y))
                    x0 = int(np.floor(x))
                    iy = int((y - y0) * bins)
                    ix = int((x - x0) * bins)
                    if iy < 0:
                        iy = 0
                    if ix < 0:
                        ix = 0
                    if iy >= bins:
                        iy = bins - 1
                    if ix >= bins:
                        ix = bins - 1
                    sy = 0.0
                    sx = 0.0
                    for tap in range(lut.shape[1]):
                        offset = tap - half
                        yy = y0 + offset
                        xx = x0 + offset
                        if 0 <= yy < height:
                            sy += lut[iy, tap]
                        if 0 <= xx < width:
                            sx += lut[ix, tap]
                    if sy == 0.0 or sx == 0.0:
                        continue
                    acc_r = 0.0
                    acc_i = 0.0
                    for ty in range(lut.shape[1]):
                        yy = y0 + ty - half
                        if yy < 0 or yy >= height:
                            continue
                        wy = lut[iy, ty] / sy
                        if wy == 0.0:
                            continue
                        for tx in range(lut.shape[1]):
                            xx = x0 + tx - half
                            if xx < 0 or xx >= width:
                                continue
                            weight = wy * (lut[ix, tx] / sx)
                            acc_r += weight * real_part[yy, xx]
                            if complex_flag != 0:
                                acc_i += weight * imag_part[yy, xx]
                    out_real[row, col] = acc_r
                    out_imag[row, col] = acc_i
            return out_real, out_imag

        _hann_apply_numba.compiled = _kernel
    return _hann_apply_numba.compiled(real_part, imag_part, line, sample, lut, half, complex_flag)


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
    """Cubic resample of a solved lattice onto every pixel.

    A smooth range-Doppler field is poorly served by bilinear steps of tens
    of pixels: those steps were several millimetres. Cubic needs four knots
    on each axis; a smaller lattice stays linear.
    """
    from scipy.interpolate import RegularGridInterpolator

    method = "cubic" if row_index.size >= 4 and col_index.size >= 4 else "linear"
    interpolator = RegularGridInterpolator(
        (row_index.astype(np.float64), col_index.astype(np.float64)),
        np.asarray(values, dtype=np.float64),
        method=method,
        bounds_error=False,
        fill_value=None,
        **(
            {"solver_args": {"rtol": 1e-12, "atol": 1e-11}}
            if method == "cubic"
            else {}
        ),
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


def valid_overlap_rows(model: SwathModel, burst_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Overlap rows after the invalid-line margins SNAP drops.

    ``burst_overlap_rows`` is the time overlap. The early burst loses the lines
    after its last valid line, and the next burst loses the lines before its
    first valid line. Both returned ranges have the same length.
    """
    early, late = burst_overlap_rows(model, burst_index)
    drop_start = int(model.bursts[burst_index + 1].first_valid_line)
    drop_end = int(model.lines_per_burst) - 1 - int(model.bursts[burst_index].last_valid_line)
    trimmed = (
        (early[0] + drop_start, early[1] - drop_end),
        (late[0] + drop_start, late[1] - drop_end),
    )
    if trimmed[0][1] - trimmed[0][0] < 8 or trimmed[0][1] - trimmed[0][0] != trimmed[1][1] - trimmed[1][0]:
        raise ValueError("trimmed burst overlap is not a usable rectangle")
    return trimmed


def spectral_separation_hz(model: SwathModel, burst_index: int, overlap_lines: int) -> np.ndarray:
    """Per-sample spectral separation ``kt(range) · (linesPerBurst − overlap) · Δt``.

    ``kt = ka · ks / (ka − ks)``. ``ka`` is the FM polynomial at ``burst_index``
    in the list the product still carries. On a split swath SNAP indexes that
    list by the burst number inside the raster, which is the first polynomial
    when the list was not cut down to the written bursts. One value per range
    sample, in Hz. The azimuth interval is inside the cycle time, so the
    values are hertz, not hertz times a line.
    """
    if burst_index < 0 or burst_index >= len(model.fm_rates):
        raise ValueError(f"FM polynomial {burst_index} is not in this swath")
    if int(overlap_lines) < 0 or int(overlap_lines) >= int(model.lines_per_burst):
        raise ValueError("overlap line count must be inside one burst")
    interval = float(model.azimuth_time_interval_s)
    samples = np.arange(int(model.samples_per_burst), dtype=np.float64)
    tau = model.slant_range_time_s + 2.0 * samples * float(model.range_pixel_spacing_m) / 299_792_458.0
    fm = model.fm_rates[int(burst_index)]
    ka = _polynomial(fm.coefficients, tau - fm.t0_s)
    speed = _speed(model, model.orbit)
    steering = 2.0 * speed * float(model.azimuth_steering_rate_rad_s) / float(model.wavelength_m)
    kt = ka * steering / (ka - steering) if abs(steering) > 1e-6 else ka
    cycle = (int(model.lines_per_burst) - int(overlap_lines)) * interval
    return np.asarray(kt * cycle, dtype=np.float64)


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


def early_look_coherence(master: np.ndarray, slave: np.ndarray) -> np.ndarray:
    """5×5 coherence of one overlap look, the window SNAP's ESD weight uses.

    The sum is the early-burst interferogram ``master · conj(slave)``. A pixel
    whose box meets the edge uses the smaller box. A zero-power box stays 0.
    """
    master_c = np.asarray(master)
    slave_c = np.asarray(slave)
    if master_c.shape != slave_c.shape or master_c.ndim != 2:
        raise ValueError("coherence looks must be the same 2-D shape")
    height, width = master_c.shape
    real = master_c.real * slave_c.real + master_c.imag * slave_c.imag
    imag = master_c.imag * slave_c.real - master_c.real * slave_c.imag
    master_power = master_c.real.astype(np.float64) ** 2 + master_c.imag.astype(np.float64) ** 2
    slave_power = slave_c.real.astype(np.float64) ** 2 + slave_c.imag.astype(np.float64) ** 2

    def _cumsum(values: np.ndarray) -> np.ndarray:
        padded = np.zeros((height + 1, width + 1), dtype=np.float64)
        padded[1:, 1:] = np.asarray(values, dtype=np.float64)
        return padded.cumsum(0).cumsum(1)

    real_sum, imag_sum, master_sum, slave_sum = (
        _cumsum(real),
        _cumsum(imag),
        _cumsum(master_power),
        _cumsum(slave_power),
    )
    coherence = np.zeros((height, width), dtype=np.float64)
    half = 2
    cols = np.arange(width)
    col0 = np.maximum(cols - half, 0)
    col1 = np.minimum(cols + half, width - 1) + 1
    # Same clipped 5×5 box as the per-row sum, evaluated a few rows at a time.
    for row_start in range(0, height, 64):
        row_stop = min(height, row_start + 64)
        rows = np.arange(row_start, row_stop)[:, None]
        row0 = np.maximum(rows - half, 0)
        row1 = np.minimum(rows + half, height - 1) + 1
        real_box = real_sum[row1, col1] - real_sum[row0, col1] - real_sum[row1, col0] + real_sum[row0, col0]
        imag_box = imag_sum[row1, col1] - imag_sum[row0, col1] - imag_sum[row1, col0] + imag_sum[row0, col0]
        master_box = master_sum[row1, col1] - master_sum[row0, col1] - master_sum[row1, col0] + master_sum[row0, col0]
        slave_box = slave_sum[row1, col1] - slave_sum[row0, col1] - slave_sum[row1, col0] + slave_sum[row0, col0]
        good = (master_box != 0.0) & (slave_box != 0.0)
        coherence[row_start:row_stop, :] = np.where(
            good,
            np.sqrt((real_box**2 + imag_box**2) / np.maximum(master_box * slave_box, 1e-30)),
            0.0,
        )
    return coherence


def esd_periodogram_shift_px(
    phase_rad: np.ndarray,
    weight: np.ndarray,
    separation_hz: np.ndarray,
    azimuth_time_interval_s: float,
    *,
    search_peak_hz: float | None = None,
) -> float:
    """SNAP periodogram: maximise the real part of the weighted double difference.

    The model phase is ``2π · separation · shift_px · Δt``. Nine candidates are
    refined until the step and the objective change are both below 1e-4, which
    is the operator's stopping rule. The return is in pixels.
    """
    phase = np.asarray(phase_rad, dtype=np.float64).ravel()
    weights = np.asarray(weight, dtype=np.float64).ravel()
    separation = np.asarray(separation_hz, dtype=np.float64).ravel()
    if phase.shape != weights.shape or phase.shape != separation.shape:
        raise ValueError("ESD phase, weight, and spectral separation must share one length")
    interval = float(azimuth_time_interval_s)
    if interval <= 0.0:
        raise ValueError("azimuth interval must be positive")
    peak = float(np.max(np.abs(separation))) if search_peak_hz is None else float(search_peak_hz)
    if peak <= 0.0 or float(np.sum(weights)) <= 0.0:
        return 0.0
    lower = -0.5 / (interval * peak)
    step = (0.5 / (interval * peak)) * 2.0 / 8.0
    previous = float(np.finfo(float).tiny)
    best_index = 0
    two_pi_dt = 2.0 * np.pi * interval
    for _ in range(40):
        candidates = lower + step * np.arange(9, dtype=np.float64)
        model = phase - (two_pi_dt * separation) * candidates[:, None]
        objective = np.sum(np.cos(model) * weights, axis=1)
        finite = np.where(np.isfinite(objective), objective, -np.inf)
        best_index = int(np.argmax(finite))
        best = float(finite[best_index])
        if not np.isfinite(best):
            best = float(np.finfo(float).tiny)
            best_index = 0
        if abs(previous - best) < 1e-4 and step < 1e-4:
            return float(lower + step * best_index)
        previous = best
        center = lower + step * best_index
        span = step * 2.0
        lower = center - step
        step = span / 8.0
    return float(lower + step * best_index)


def overlap_block_shifts(
    master_early: np.ndarray,
    master_late: np.ndarray,
    slave_early: np.ndarray,
    slave_late: np.ndarray,
    *,
    separation_hz: float | np.ndarray,
    azimuth_time_interval_s: float,
    n_blocks: int = 10,
    coherence_threshold: float = 0.3,
) -> list[dict[str, float | int | bool]]:
    """SNAP's per-block ESD periodogram, in pixels, for one overlap.

    The phase is ``arg((m_early · conj(s_early)) · conj(m_late · conj(s_late)))``.
    The weight is ``sqrt(coherence)`` of the early look's 5×5 interferogram when
    that coherence is above ``coherence_threshold``, and 0 otherwise. This is
    the Inv Quadratic weight. It is not the amplitude of the double difference.
    A block whose weight is all zero is not a measurement. The stored shift is
    the periodogram in pixels. ``shift_s`` is that shift times the azimuth
    interval. Nothing is added back onto the slave.

    ``separation_hz`` is either one value or one value per full-swath sample.
    SNAP looks the per-sample array up at ``block · block_width + column``,
    counting columns from sample 0 of the swath rather than from the first
    valid sample of the overlap.
    """
    looks = (master_early, master_late, slave_early, slave_late)
    shape = np.asarray(master_early).shape
    if any(np.asarray(item).shape != shape for item in looks):
        raise ValueError("ESD overlap looks must share one shape")
    if int(n_blocks) < 1:
        raise ValueError("ESD needs at least one range block")
    interval = float(azimuth_time_interval_s)
    if interval <= 0.0:
        raise ValueError("azimuth interval must be positive")
    width = int(shape[1])
    block_width = width // int(n_blocks)
    if block_width < 1:
        raise ValueError(f"overlap width {width} cannot be split into {n_blocks} blocks")
    separation = np.asarray(separation_hz, dtype=np.float64).reshape(-1)
    if separation.size != 1 and separation.size < int(n_blocks) * block_width:
        raise ValueError("spectral separation must be a scalar or one value per swath sample")
    # The search window is half an ambiguity of the largest separation in the
    # swath, which is what the operator computes once per overlap.
    search_peak = float(np.max(np.abs(separation)))
    reports: list[dict[str, float | int | bool]] = []
    master_e = np.asarray(master_early)
    master_l = np.asarray(master_late)
    slave_e = np.asarray(slave_early)
    slave_l = np.asarray(slave_late)
    for index in range(int(n_blocks)):
        col0 = index * block_width
        col1 = (index + 1) * block_width
        early = master_e[:, col0:col1] * np.conj(slave_e[:, col0:col1])
        late = master_l[:, col0:col1] * np.conj(slave_l[:, col0:col1])
        cross = early * np.conj(late)
        coherence = float(np.abs(np.sum(cross)) / (np.sum(np.abs(cross)) + 1e-12))
        look = early_look_coherence(master_e[:, col0:col1], slave_e[:, col0:col1])
        weight = np.where(look > float(coherence_threshold), np.sqrt(look), 0.0)
        if separation.size == 1:
            block_separation = np.full(weight.size, float(separation[0]), dtype=np.float64)
        else:
            # SNAP chopSpectralSeparation: column block*width, not the valid-sample origin.
            sample = index * block_width + np.arange(block_width)
            block_separation = np.broadcast_to(separation[sample], weight.shape).ravel().copy()
        shift_px = esd_periodogram_shift_px(
            np.angle(cross),
            weight,
            block_separation,
            interval,
            search_peak_hz=search_peak,
        )
        mean_weight = float(np.mean(weight))
        reports.append(
            {
                "block": index,
                "col0": col0,
                "col1": col1,
                "coherence": coherence,
                "weight": mean_weight,
                "shift_px": shift_px,
                "shift_s": shift_px * interval,
                "accepted": mean_weight > 0.0,
            }
        )
    return reports


def esd_applied_shift_px(reports: list[dict[str, float | int | bool]]) -> float:
    """Negated weight-mean of the block periodograms, in pixels.

    SNAP writes this value after the block list as the mean azimuth shift and
    applies it to the secondary. The weight is the mean per-pixel block weight.
    """
    if not reports:
        return 0.0
    shifts = np.array([float(item["shift_px"]) for item in reports], dtype=np.float64)
    weights = np.array([float(item["weight"]) for item in reports], dtype=np.float64)
    total = float(np.sum(weights))
    if total == 0.0:
        return 0.0
    return float(-np.sum(shifts * weights) / total)


def apply_azimuth_pixel_shift(image: np.ndarray, pixels: float, azimuth_time_interval_s: float) -> np.ndarray:
    """Shift a complex image by ``pixels`` azimuth lines with a Fourier ramp."""
    spectrum = np.fft.fft(np.asarray(image, dtype=np.complex64), axis=0)
    frequency = np.fft.fftfreq(image.shape[0], d=float(azimuth_time_interval_s))
    seconds = float(pixels) * float(azimuth_time_interval_s)
    spectrum *= np.exp(-1j * 2.0 * np.pi * frequency[:, None] * seconds)
    return np.fft.ifft(spectrum, axis=0).astype(np.complex64)


def shift_secondary_in_overlap(
    grid: RadarGrid,
    azimuth_px: float,
    range_px: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Range-shift, deramp, azimuth-shift the baseband and the phase, then reramp.

    The range ramp is applied to the still-chirped lines. Deramp uses that
    burst's phase, and the azimuth ramp is applied to the baseband and to the
    phase before the phase is put back. Each burst is transformed on its own
    so the ramp does not cross a burst boundary. Only the common-overlap box
    is touched. Pixels outside the mask, and pixels that were zero before the
    shift, stay zero.
    """
    data = np.asarray(grid.data)
    valid = (np.real(data) != 0) | (np.imag(data) != 0)
    if mask is not None and mask.shape == data.shape:
        valid = valid & np.asarray(mask, dtype=bool)
    out = np.zeros(data.shape, dtype=np.complex64)
    extent = overlap_extent(valid)
    if extent is None:
        return out
    r0, r1, c0, c1 = extent
    if azimuth_px == 0.0 and range_px == 0.0:
        out[r0:r1, c0:c1] = np.where(valid[r0:r1, c0:c1], data[r0:r1, c0:c1], 0)
        return out
    phase_full = tops_phase(grid)
    lines = int(grid.model.lines_per_burst)
    interval = float(grid.model.azimuth_time_interval_s)
    burst0 = (grid.row + r0) // lines
    burst1 = (grid.row + r1 - 1) // lines
    for burst in range(int(burst0), int(burst1) + 1):
        row_a = max(r0, burst * lines - grid.row)
        row_b = min(r1, (burst + 1) * lines - grid.row)
        if row_b <= row_a:
            continue
        tile = np.asarray(data[row_a:row_b, c0:c1], dtype=np.complex64)
        tile = _range_shift(tile, float(range_px))
        phase = phase_full[row_a:row_b, c0:c1]
        baseband = tile * np.exp(1j * phase)
        baseband = _azimuth_shift(baseband, float(azimuth_px), interval)
        shifted_phase = _azimuth_shift(phase.astype(np.complex64), float(azimuth_px), interval).real
        corrected = (baseband * np.exp(-1j * shifted_phase)).astype(np.complex64)
        tile_valid = valid[row_a:row_b, c0:c1]
        corrected[~tile_valid] = 0
        out[row_a:row_b, c0:c1] = corrected
    return out


def _range_shift(image: np.ndarray, pixels: float) -> np.ndarray:
    if not pixels:
        return np.asarray(image, dtype=np.complex64)
    spectrum = np.fft.fft(np.asarray(image, dtype=np.complex64), axis=1)
    frequency = np.fft.fftfreq(image.shape[1])
    spectrum *= np.exp(-1j * 2.0 * np.pi * frequency[None, :] * float(pixels))
    return np.fft.ifft(spectrum, axis=1).astype(np.complex64)


def _azimuth_shift(image: np.ndarray, pixels: float, azimuth_time_interval_s: float) -> np.ndarray:
    if not pixels:
        return np.asarray(image, dtype=np.complex64)
    return apply_azimuth_pixel_shift(image, float(pixels), float(azimuth_time_interval_s))


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
        fm_rates=model.fm_rates,
        dc_estimates=model.dc_estimates,
        azimuth_steering_rate_rad_s=model.azimuth_steering_rate_rad_s,
        first_valid_pixel=model.first_valid_pixel,
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
    if surface == "dem":
        lats, lons = _footprint_latlon(grid, orbit, height_of, degree)
    else:
        lats, lons = _footprint_latlon(grid, orbit, lambda _lat, _lon: (0.0, True), degree, height_m=0.0)
    if not lats:
        raise ValueError(
            "Range-Doppler terrain correction found no DEM height on the source footprint. "
            "The output grid was not filled with zeros."
        )
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
                local_line, local_sample = _raster_coordinates(grid, orbit, xyz, guess=guess, degree=degree)
            except ValueError:
                continue
            line[row_index, col_index] = local_line
            sample[row_index, col_index] = local_sample
    image = resample_complex(grid.data, line, sample, resampling)
    _geocode.last_surface = surface
    _geocode.last_warning = ""
    if surface == "dem":
        image = np.asarray(image, dtype=np.complex64).copy()
        inside = (
            np.isfinite(line)
            & np.isfinite(sample)
            & (line >= -0.5)
            & (line <= rows - 0.5)
            & (sample >= -0.5)
            & (sample <= cols - 0.5)
        )
        image[~inside] = np.nan + 1j * np.nan
        valid_count = int(np.count_nonzero(np.isfinite(np.real(image))))
        _geocode.last_valid = valid_count
        if valid_count == 0:
            finite_sample = sample[np.isfinite(sample)]
            span = (
                f"{float(finite_sample.min()):.1f} to {float(finite_sample.max()):.1f}"
                if finite_sample.size
                else "undefined"
            )
            message = (
                "Range-Doppler terrain correction produced no valid output pixels. "
                f"Inverse source columns were {span}, outside 0..{cols - 1}. "
                "Invalid samples are NaN, not 0."
            )
            _geocode.last_warning = message
            raise ValueError(message)
    if surface == "ellipsoid":
        return image, (west, south, east, north)
    return image, (west, south, east, north), heights


def _footprint_latlon(grid, orbit, height_of, degree, height_m=None):
    """Latitude and longitude of the source border, at the DEM height.

    A zero-elevation box sits hundreds of metres from the same pixels on a
    real DEM. Each border sample is geocoded, the DEM is read there, and the
    point is geocoded again until the height stops moving.
    """
    rows, cols = grid.shape
    samples = (
        (0, 0),
        (0, cols - 1),
        (rows - 1, 0),
        (rows - 1, cols - 1),
        (0, cols // 2),
        (rows - 1, cols // 2),
        (rows // 2, 0),
        (rows // 2, cols - 1),
        (rows // 2, cols // 2),
    )
    lats: list[float] = []
    lons: list[float] = []
    for row, col in samples:
        azimuth = grid.model.line_time(grid.row + row)
        slant = grid.model.slant_range_m(grid.col + col)
        height = 0.0 if height_m is None else float(height_m)
        lat_deg = lon_deg = None
        for _ in range(12):
            ground = ground_point_at_height(orbit, azimuth, slant, height, degree=degree)
            lat, lon, got = ecef_to_geodetic(ground)
            lat_deg = float(np.rad2deg(lat))
            lon_deg = float(np.rad2deg(lon))
            if height_m is not None:
                break
            sampled, valid = height_of(lat_deg, lon_deg)
            if not valid:
                lat_deg = None
                break
            if abs(float(sampled) - height) < 1e-3 and abs(float(got) - float(sampled)) < 1e-3:
                height = float(sampled)
                ground = ground_point_at_height(orbit, azimuth, slant, height, degree=degree)
                lat, lon, _got = ecef_to_geodetic(ground)
                lat_deg = float(np.rad2deg(lat))
                lon_deg = float(np.rad2deg(lon))
                break
            height = float(sampled)
        if lat_deg is not None:
            lats.append(lat_deg)
            lons.append(lon_deg)
    return lats, lons


def _raster_coordinates(grid, orbit, xyz, *, guess, degree):
    """Line and sample in this raster, measured from its first line and sample.

    ``grid.row`` and ``grid.col`` are the subset or deburst origin. The slant
    range is relative to that first sample, and the azimuth time is relative
    to that first line, not to sample 0 of the uncut swath.
    """
    when = orbit.zero_doppler_time(xyz, guess, degree)
    position, _velocity = orbit.interpolate(when, degree)
    slant = float(np.linalg.norm(xyz - position))
    sample = (slant - grid.model.slant_range_m(grid.col)) / grid.model.range_pixel_spacing_m
    first = grid.model.line_time(int(grid.row))
    elapsed = int((when - first).astype(np.int64)) * 1e-9
    line = elapsed / float(grid.model.azimuth_time_interval_s)
    return float(line), float(sample)


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
