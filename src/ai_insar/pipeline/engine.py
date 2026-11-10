"""Stage-by-stage workspace used by the GUI and the headless runner.

Each ``compute_*`` function performs one numbered stage and clears the
products of the stages that depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ai_insar.conventions import DEFAULT_MASK_FRACTION, phase_from_los
from ai_insar.pipeline.coherence import coherence_stack
from ai_insar.pipeline.detect import detect_displacement
from ai_insar.pipeline.filtering import filter_stack, goldstein_filter
from ai_insar.pipeline.interferogram import date_pairs, form_interferograms
from ai_insar.pipeline.synthetic import SLCStack
from ai_insar.safe_stack import SafeInput
from ai_insar.pipeline.timeseries import TIMESERIES_LABELS, TIMESERIES_METHODS, invert_timeseries
from ai_insar.pipeline.unwrap import (
    UNWRAP_LABELS,
    reliability,
    remove_ramp,
    residue_count,
    unwrap_quality_guided,
    unwrap_separable,
)

LogFn = Callable[[str], None]


class PipelineError(RuntimeError):
    """A stage was started before the data it needs exists."""


@dataclass
class PipelineConfig:
    """Tunable parameters for the seven processing stages."""

    max_date_span: int = 3
    goldstein_alpha: float = 0.5
    goldstein_window: int = 32
    goldstein_overlap: float = 0.5
    goldstein_spectrum_smooth: int = 3
    coherence_window: int = 5
    coherence_azimuth_window: int | None = None
    unwrap_min_coherence: float = 0.2
    ramp_border_px: int = 4
    detection_sigma: float = 3.5
    detection_peak_fraction: float = DEFAULT_MASK_FRACTION
    detection_min_coherence: float = 0.3
    detection_min_pixels: int = 12
    timeseries_method: str = "sbas"
    ps_da_max: float = 0.4
    ps_min_coherence: float = 0.5
    nsbas_weight: float = 0.25
    unwrap_method: str = "separable"
    unwrap_sigma: float = 2.0

    def __post_init__(self) -> None:
        if self.max_date_span < 1:
            raise ValueError("max_date_span must be >= 1")
        if not 0.0 <= self.goldstein_alpha <= 1.0:
            raise ValueError("goldstein_alpha must be between 0 and 1")
        if self.goldstein_window < 8:
            raise ValueError("goldstein_window must be >= 8")
        if not 0.0 <= self.goldstein_overlap < 1.0:
            raise ValueError("goldstein_overlap must be in [0, 1)")
        if self.goldstein_spectrum_smooth < 1:
            raise ValueError("goldstein_spectrum_smooth must be >= 1")
        if self.coherence_window < 1:
            raise ValueError("coherence_window must be >= 1")
        if self.coherence_azimuth_window is not None and self.coherence_azimuth_window < 1:
            raise ValueError("coherence_azimuth_window must be >= 1")
        if not 0.0 <= self.unwrap_min_coherence <= 1.0:
            raise ValueError("unwrap_min_coherence must be between 0 and 1")
        if self.detection_sigma <= 0:
            raise ValueError("detection_sigma must be positive")
        if not 0.0 < self.detection_peak_fraction < 1.0:
            raise ValueError("detection_peak_fraction must be between 0 and 1")
        if self.detection_min_pixels < 1:
            raise ValueError("detection_min_pixels must be >= 1")
        if self.timeseries_method not in TIMESERIES_METHODS:
            known = ", ".join(TIMESERIES_METHODS)
            raise ValueError(f"timeseries_method must be one of {known}")
        if self.ps_da_max <= 0:
            raise ValueError("ps_da_max must be positive")
        if self.nsbas_weight <= 0:
            raise ValueError("nsbas_weight must be positive")
        if self.unwrap_method not in ("quality", "separable"):
            raise ValueError("unwrap_method must be 'quality' or 'separable'")
        if self.unwrap_sigma <= 0:
            raise ValueError("unwrap_sigma must be positive")


@dataclass
class Workspace:
    """SLC stack plus the products of each stage. Missing products are ``None``."""

    stack: SLCStack | None = None
    config: PipelineConfig = field(default_factory=PipelineConfig)
    pairs: list[tuple[int, int]] | None = None
    interferograms: np.ndarray | None = None
    filtered: np.ndarray | None = None
    coherence: np.ndarray | None = None
    mean_coherence: np.ndarray | None = None
    unwrapped: np.ndarray | None = None
    los_m: np.ndarray | None = None
    velocity_m_yr: np.ndarray | None = None
    mask: np.ndarray | None = None
    dem_m: np.ndarray | None = None
    topo_baseline_perp_m: float = 0.0
    topo_slant_range_m: float = 800_000.0
    topo_incidence_deg: float = 35.0
    safe_inputs: list[SafeInput] = field(default_factory=list)
    reference_index: int | None = None
    swath_model: object | None = None
    precise_orbit: object | None = None
    secondary_orbit: object | None = None
    reference_grid: object | None = None
    secondary_grid: object | None = None
    geocoded: np.ndarray | None = None
    geocode_bounds: tuple | None = None
    elevation: object | None = None
    esd_azimuth_shift_s: float | None = None
    crop_without_coregistration: bool = False
    common_overlap_mask: np.ndarray | None = None
    common_overlap_extent: tuple[int, int, int, int] | None = None
    common_overlap_meta: dict | None = None

    def clear_products(self) -> None:
        self.pairs = None
        self.interferograms = None
        self.filtered = None
        self.coherence = None
        self.mean_coherence = None
        self.unwrapped = None
        self.los_m = None
        self.velocity_m_yr = None
        self.mask = None


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)


def _require_stack(workspace: Workspace) -> SLCStack:
    if workspace.stack is None:
        raise PipelineError("Load or generate an SLC stack first.")
    return workspace.stack


def restrict_array(array: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
    """Zero every sample outside the common overlap. Other shapes are unchanged."""
    if array is None or mask is None or array.shape[-2:] != mask.shape:
        return array
    return np.where(mask, array, 0)


def apply_overlap_to_workspace(workspace: Workspace) -> None:
    """Keep rasters that share the overlap grid inside that mask."""
    mask = workspace.common_overlap_mask
    if mask is None:
        return
    stack = workspace.stack
    if stack is not None and stack.data.shape[-2:] == mask.shape:
        stack.data = np.where(mask, stack.data, 0).astype(stack.data.dtype, copy=False)
    for name in ("interferograms", "filtered", "coherence", "unwrapped", "los_m", "mean_coherence"):
        setattr(workspace, name, restrict_array(getattr(workspace, name), mask))
    for name in ("reference_grid", "secondary_grid"):
        grid = getattr(workspace, name, None)
        if grid is not None and getattr(grid, "data", None) is not None and grid.data.shape == mask.shape:
            grid.data = np.where(mask, grid.data, 0).astype(grid.data.dtype, copy=False)


def _overlap_crop(array: np.ndarray, workspace: Workspace) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    meta = workspace.common_overlap_meta
    mask = workspace.common_overlap_mask
    if not meta or mask is None or array.shape[-2:] != mask.shape or int(meta.get("count", 0)) == 0:
        return array, None
    row0, row1 = int(meta["row_start"]), int(meta["row_end"])
    col0, col1 = int(meta["col_start"]), int(meta["col_end"])
    tile_mask = mask[row0:row1, col0:col1]
    if array.ndim == 2:
        tile = np.where(tile_mask, array[row0:row1, col0:col1], 0)
    else:
        tile = np.where(tile_mask, array[..., row0:row1, col0:col1], 0)
    return tile, (row0, row1, col0, col1)


def _overlap_paste(shape: tuple[int, ...], tile: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=tile.dtype)
    row0, row1, col0, col1 = box
    if tile.ndim == 2:
        out[row0:row1, col0:col1] = tile
    else:
        out[..., row0:row1, col0:col1] = tile
    return out


def compute_interferograms(workspace: Workspace, log: LogFn | None = None) -> None:
    """Stage 2. Form ``later * conj(earlier)`` on the small-baseline network."""
    stack = _require_stack(workspace)
    if stack.n_dates < 2:
        raise PipelineError("Need at least two SLC dates to form an interferogram.")
    reference = getattr(workspace, "reference_index", None)
    if reference is None:
        workspace.pairs = date_pairs(stack.n_dates, workspace.config.max_date_span)
        workspace.interferograms = form_interferograms(stack.data, workspace.pairs)
        relation = "Each is later × conjugate(earlier)."
    else:
        reference = int(reference)
        if reference < 0 or reference >= stack.n_dates:
            raise PipelineError(f"reference index {reference} is outside the stack.")
        pairs: list[tuple[int, int]] = []
        planes = []
        for secondary in range(stack.n_dates):
            if secondary == reference:
                continue
            earlier, later = sorted((secondary, reference))
            pairs.append((earlier, later))
            planes.append(stack.data[reference] * np.conj(stack.data[secondary]))
        workspace.pairs = pairs
        workspace.interferograms = np.stack(planes).astype(np.complex64)
        chronological = all(
            np.allclose(plane, stack.data[later] * np.conj(stack.data[earlier]))
            for plane, (earlier, later) in zip(workspace.interferograms, pairs)
        )
        if chronological:
            relation = "Each is later × conjugate(earlier)."
        else:
            relation = (
                f"Each is reference date index {reference} × conjugate(secondary). "
                "The selected reference is not the later date, so this is not later × conjugate(earlier)."
            )
    workspace.interferograms = restrict_array(workspace.interferograms, workspace.common_overlap_mask)
    backend = getattr(form_interferograms, "last_backend", "numpy")
    backend_label = {"insardev": "InSARdev", "numba": "Numba", "numpy": "NumPy"}.get(backend, backend)
    workspace.filtered = None
    workspace.coherence = None
    workspace.mean_coherence = None
    workspace.unwrapped = None
    workspace.los_m = None
    workspace.velocity_m_yr = None
    workspace.mask = None
    times = stack.times_year()
    spans = [abs(times[later] - times[earlier]) for earlier, later in workspace.pairs]
    _log(
        log,
        f"Interferograms: {len(workspace.pairs)} pairs "
        f"(max date span {workspace.config.max_date_span}, "
        f"temporal baselines {min(spans) * 365.25:.0f}–{max(spans) * 365.25:.0f} days). "
        f"{relation} "
        f"Backend: {backend_label}.",
    )
    _log_truth_phase(stack, workspace.pairs, workspace.interferograms, log)


def compute_filter(workspace: Workspace, log: LogFn | None = None) -> None:
    """Stage 3. Goldstein-filter every interferogram. Phase for unwrapping comes from this."""
    if workspace.interferograms is None:
        raise PipelineError("Form interferograms before filtering.")
    cfg = workspace.config
    tile, box = _overlap_crop(workspace.interferograms, workspace)
    workspace.filtered = filter_stack(
        tile,
        alpha=cfg.goldstein_alpha,
        window=cfg.goldstein_window,
        overlap=cfg.goldstein_overlap,
        spectrum_smooth=cfg.goldstein_spectrum_smooth,
    )
    if box is not None:
        workspace.filtered = _overlap_paste(workspace.interferograms.shape, workspace.filtered, box)
    workspace.unwrapped = None
    workspace.los_m = None
    workspace.velocity_m_yr = None
    workspace.mask = None
    residues = [
        residue_count(np.angle(workspace.filtered[index]))
        for index in range(workspace.filtered.shape[0])
    ]
    filter_backend = getattr(goldstein_filter, "last_backend", "numpy")
    _log(
        log,
        f"Goldstein filter: alpha={cfg.goldstein_alpha:.2f}, window={cfg.goldstein_window}. "
        f"Wrapped residues after filtering: {sum(residues)} "
        f"(per pair min {min(residues)}, max {max(residues)}). "
        f"Backend: batched FFT ({filter_backend}).",
    )


def compute_coherence(workspace: Workspace, log: LogFn | None = None) -> None:
    """Stage 4. Coherence from the original SLC pairs, not from the filtered phase."""
    stack = _require_stack(workspace)
    if workspace.pairs is None:
        raise PipelineError("Form interferograms before estimating coherence.")
    cfg = workspace.config
    azimuth = cfg.coherence_azimuth_window
    slc, box = _overlap_crop(stack.data, workspace)
    workspace.coherence = coherence_stack(
        slc,
        workspace.pairs,
        cfg.coherence_window,
        azimuth_window=azimuth,
    )
    if box is not None:
        workspace.coherence = _overlap_paste(
            (workspace.coherence.shape[0], stack.data.shape[-2], stack.data.shape[-1]),
            workspace.coherence,
            box,
        )
    workspace.mean_coherence = np.mean(workspace.coherence, axis=0).astype(np.float32)
    workspace.mean_coherence = restrict_array(workspace.mean_coherence, workspace.common_overlap_mask)
    workspace.unwrapped = None
    workspace.los_m = None
    workspace.velocity_m_yr = None
    workspace.mask = None
    finite = workspace.coherence[np.isfinite(workspace.coherence)]
    azimuth_size = cfg.coherence_window if azimuth is None else int(azimuth)
    _log(
        log,
        f"Coherence: {cfg.coherence_window}×{azimuth_size} "
        f"window on each SLC pair. Median {float(np.median(finite)):.3f}, "
        f"5th percentile {float(np.percentile(finite, 5)):.3f}.",
    )


def compute_unwrap(workspace: Workspace, log: LogFn | None = None) -> None:
    """Stage 5. Unwrap the filtered phase, then remove a border plane.

    The default method is the fast separable unwrap. Quality-guided remains
    available by setting ``unwrap_method`` to ``quality``.
    """
    if workspace.filtered is None or workspace.coherence is None:
        raise PipelineError("Filter the interferograms and estimate coherence before unwrapping.")
    cfg = workspace.config
    n_pairs, height, width = workspace.filtered.shape
    unwrapped = np.empty((n_pairs, height, width), dtype=np.float64)
    for index in range(n_pairs):
        phase = np.angle(workspace.filtered[index]).astype(np.float64)
        coh = workspace.coherence[index].astype(np.float64)
        if cfg.unwrap_method == "separable":
            field = unwrap_separable(
                phase,
                coh,
                sigma=cfg.unwrap_sigma,
                min_coherence=cfg.unwrap_min_coherence,
            )
        else:
            quality = reliability(phase, coh)
            # Low-coherence pixels are barriers: the integrator does not walk through them.
            quality = np.where(coh >= cfg.unwrap_min_coherence, quality, -np.inf)
            field = unwrap_quality_guided(phase, quality, min_quality=0.0)
        field = remove_ramp(field, coh, border_px=cfg.ramp_border_px)
        unwrapped[index] = field
    workspace.unwrapped = unwrapped
    workspace.los_m = None
    workspace.velocity_m_yr = None
    workspace.mask = None
    valid = np.isfinite(unwrapped)
    _log(
        log,
        f"Unwrapped filtered phase with {UNWRAP_LABELS.get(cfg.unwrap_method, cfg.unwrap_method)} "
        f"and removed a border plane ({cfg.ramp_border_px} px). "
        f"Finite pixels: {100.0 * valid.mean():.1f}%.",
    )


def compute_sbas(workspace: Workspace, log: LogFn | None = None) -> None:
    """Stage 6 using SBAS only. Prefer :func:`compute_timeseries` in the GUI."""
    workspace.config.timeseries_method = "sbas"
    compute_timeseries(workspace, log)


def compute_timeseries(workspace: Workspace, log: LogFn | None = None) -> None:
    """Stage 6. LOS series and velocity with the configured method.

    The default method is SBAS. Stacking, NSBAS, persistent-scatterer linear
    velocity, sequential phase linking, and a one-year seasonal fit write the
    same ``los_m`` and ``velocity_m_yr`` arrays. Seasonal's velocity is the
    linear coefficient; its displacement series also includes the annual term.
    """
    stack = _require_stack(workspace)
    if workspace.unwrapped is None or workspace.pairs is None or workspace.coherence is None:
        raise PipelineError("Unwrap the interferograms before the time-series stage.")
    method = workspace.config.timeseries_method
    los_m, velocity = invert_timeseries(
        method,
        unwrapped_phase=workspace.unwrapped,
        pairs=workspace.pairs,
        times_year=stack.times_year(),
        wavelength_m=stack.wavelength_m,
        coherence=workspace.coherence,
        slc=stack.data,
        da_max=workspace.config.ps_da_max,
        ps_min_coherence=workspace.config.ps_min_coherence,
        nsbas_weight=workspace.config.nsbas_weight,
    )
    workspace.los_m = los_m
    workspace.velocity_m_yr = velocity
    workspace.mask = None
    finite = velocity[np.isfinite(velocity)]
    label = TIMESERIES_LABELS.get(method, method)
    if finite.size == 0:
        _log(log, f"{label}: no finite velocities.")
    else:
        _log(
            log,
            f"{label}: displacement at each date (first date = 0) and a linear LOS velocity. "
            f"Velocity min {float(finite.min()):.4f}, max {float(finite.max()):.4f} m/yr. "
            "Negative velocity is motion away from the radar.",
        )
    if stack.truth_velocity_m_yr is not None and finite.size:
        peak = _peak_pixel(stack.truth_velocity_m_yr)
        est = float(velocity[peak])
        truth = float(stack.truth_velocity_m_yr[peak])
        if np.isfinite(est):
            _log(
                log,
                f"At the synthetic peak pixel (row {peak[0]}, col {peak[1]}): "
                f"estimated {est:.4f} m/yr, truth {truth:.4f} m/yr.",
            )


def compute_detection(workspace: Workspace, log: LogFn | None = None) -> None:
    """Stage 7. Mark coherent pixels whose velocity is large relative to the scene."""
    if workspace.velocity_m_yr is None or workspace.mean_coherence is None:
        raise PipelineError("Run a time-series method before displacement detection.")
    cfg = workspace.config
    workspace.mask = detect_displacement(
        workspace.velocity_m_yr,
        workspace.mean_coherence,
        sigma_k=cfg.detection_sigma,
        peak_fraction=cfg.detection_peak_fraction,
        min_coherence=cfg.detection_min_coherence,
        min_pixels=cfg.detection_min_pixels,
    )
    count = int(workspace.mask.sum())
    _log(
        log,
        f"Displacement mask: {count} pixels "
        f"(|z| ≥ {cfg.detection_sigma:g} and |velocity| ≥ "
        f"{100 * cfg.detection_peak_fraction:.0f}% of the coherent peak, "
        f"coherence ≥ {cfg.detection_min_coherence:g}).",
    )


def run_all(workspace: Workspace, log: LogFn | None = None) -> None:
    """Run stages 2–7. The workspace must already hold an SLC stack."""
    stack = _require_stack(workspace)
    _log(
        log,
        f"SLC stack {stack.data.shape} {stack.data.dtype}, "
        f"wavelength {stack.wavelength_m:.6f} m, sensor {stack.sensor}.",
    )
    if stack.description:
        _log(log, stack.description)
    if stack.truth_velocity_m_yr is not None:
        peak = _peak_pixel(stack.truth_velocity_m_yr)
        value = float(stack.truth_velocity_m_yr[peak])
        _log(
            log,
            f"Synthetic truth peak LOS velocity {value:.4f} m/yr at "
            f"row {peak[0]}, col {peak[1]} (negative = away from the radar).",
        )
    compute_interferograms(workspace, log)
    compute_filter(workspace, log)
    compute_coherence(workspace, log)
    compute_unwrap(workspace, log)
    compute_timeseries(workspace, log)
    compute_detection(workspace, log)


def _peak_pixel(velocity: np.ndarray) -> tuple[int, int]:
    absolute = np.abs(np.asarray(velocity, dtype=np.float64))
    index = int(np.argmax(absolute))
    row, col = np.unravel_index(index, absolute.shape)
    return int(row), int(col)


def _log_truth_phase(
    stack: SLCStack,
    pairs: list[tuple[int, int]],
    interferograms: np.ndarray,
    log: LogFn | None,
) -> None:
    if stack.truth_los_m is None or log is None:
        return
    earlier, later = pairs[0]
    true_phase = phase_from_los(stack.truth_los_m, stack.wavelength_m)
    row, col = _peak_pixel(stack.truth_velocity_m_yr) if stack.truth_velocity_m_yr is not None else (
        stack.height // 2,
        stack.width // 2,
    )
    predicted = float(true_phase[later, row, col] - true_phase[earlier, row, col])
    wrapped = float(np.angle(interferograms[0, row, col]))
    _log(
        log,
        f"First pair dates {earlier}→{later} at the peak pixel: "
        f"deformation phase {predicted:.2f} rad "
        f"({predicted / (2 * np.pi):.2f} fringes, positive = subsidence), "
        f"wrapped SLC interferogram {wrapped:.2f} rad.",
    )
