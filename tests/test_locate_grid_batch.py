"""The batched geometry lattice matches the one-pixel solver."""

from __future__ import annotations

import numpy as np

from ai_insar.pipeline.dem import ElevationModel, GeoidGrid
from ai_insar.pipeline.geometry import Orbit
from ai_insar.pipeline.s1_annotation import BurstTiming, SwathModel
from ai_insar.pipeline.tops_ops import RadarGrid, _locate_grid, _one_secondary_coordinate


def _orbit() -> Orbit:
    epoch = np.datetime64("2021-06-30T05:20:00")
    count = 40
    times = epoch + np.arange(count) * np.timedelta64(10, "s")
    rate = 0.00106
    angle = rate * np.arange(count) * 10.0
    radius = 7_078_137.0
    position = np.stack((radius * np.cos(angle), radius * np.sin(angle), np.full(count, 1.0e5)), axis=1)
    velocity = np.stack(
        (-radius * rate * np.sin(angle), radius * rate * np.cos(angle), np.zeros(count)),
        axis=1,
    )
    return Orbit(times, position, velocity)


def _models(orbit: Orbit) -> tuple[SwathModel, SwathModel]:
    start = np.datetime64("2021-06-30T05:22:30")
    interval = 0.002055556
    lines = 40
    bursts = tuple(
        BurstTiming(start + np.timedelta64(int(round(index * lines * interval * 1e9)), "ns"), 0, lines - 1, 0, 200)
        for index in range(2)
    )
    later = start + np.timedelta64(12, "D")
    secondary_bursts = tuple(
        BurstTiming(later + np.timedelta64(int(round(index * lines * interval * 1e9)), "ns"), 0, lines - 1, 0, 200)
        for index in range(2)
    )
    common = dict(
        wavelength_m=0.05546576,
        azimuth_time_interval_s=interval,
        range_pixel_spacing_m=2.329562,
        slant_range_time_s=0.006,
        lines_per_burst=lines,
        samples_per_burst=200,
        number_of_lines=lines * 2,
        incidence_rad=0.6,
        fm_rate=np.array([-2000.0, 0.0, 0.0]),
        orbit=orbit,
    )
    return (
        SwathModel(bursts=bursts, **common),
        SwathModel(bursts=secondary_bursts, **common),
    )


def _dem() -> ElevationModel:
    geoid = GeoidGrid(
        lat_deg=np.array([10.0, -10.0]),
        lon_deg=np.array([0.0, 20.0]),
        undulation_m=np.array([[1.5, 1.6], [1.4, 1.7]]),
        source="test",
        model="EGM2008",
    )
    return ElevationModel(
        lat_deg=np.linspace(5.0, -8.0, 8),
        lon_deg=np.linspace(5.0, 15.0, 9),
        orthometric_m=np.full((8, 9), 250.0),
        nodata=-32768.0,
        orthometric_datum="EGM2008",
        geoid=geoid,
        provenance="test",
    )


def test_batched_lattice_matches_the_scalar_solver_with_and_without_dem():
    orbit = _orbit()
    reference_model, secondary_model = _models(orbit)
    # The secondary date needs its own state vectors, shifted by the repeat.
    later = Orbit(
        orbit.time + np.timedelta64(12, "D"),
        orbit.position_m + np.array([0.0, 0.0, 30.0]),
        orbit.velocity_m_s,
    )
    reference = RadarGrid(0, 0, np.zeros((24, 16)), reference_model)
    secondary = RadarGrid(2, 3, np.zeros((24, 16)), secondary_model)
    for dem in (None, _dem()):
        lines, samples, row_index, col_index = _locate_grid(
            reference,
            secondary,
            reference_orbit=orbit,
            secondary_orbit=later,
            degree=3,
            height_m=0.0,
            dem=dem,
            stride=4,
        )
        assert np.isfinite(lines).all()
        worst = 0.0
        for i, row in enumerate(row_index):
            for j, col in enumerate(col_index):
                scalar_line, scalar_sample = _one_secondary_coordinate(
                    reference,
                    secondary,
                    int(row),
                    int(col),
                    reference_orbit=orbit,
                    secondary_orbit=later,
                    degree=3,
                    height_m=0.0,
                    dem=dem,
                )
                worst = max(worst, abs(scalar_line - lines[i, j]), abs(scalar_sample - samples[i, j]))
        assert worst < 1e-9


def test_common_overlap_is_the_valid_sample_intersection_and_resample_skips_the_rest():
    from ai_insar.pipeline.tops_ops import common_overlap_mask, resample_complex

    orbit = _orbit()
    reference_model, _secondary_model = _models(orbit)
    reference_model.bursts[0].last_valid_sample = 4
    rows, cols = 6, 8
    line = np.broadcast_to(np.arange(rows, dtype=np.float64)[:, None], (rows, cols)).copy()
    sample = np.broadcast_to(np.arange(cols, dtype=np.float64)[None, :], (rows, cols)).copy()
    reference = RadarGrid(0, 0, np.zeros((rows, cols)), reference_model)
    secondary = RadarGrid(0, 0, np.ones((rows, cols)), reference_model)
    mask = common_overlap_mask(reference, secondary, line, sample)
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()
    image = np.ones((rows, cols), dtype=np.complex64)
    resampled = resample_complex(image, line, sample, "BILINEAR_INTERPOLATION", mask=mask)
    assert resampled[0, 0] != 0
    assert resampled[0, 6] == 0
