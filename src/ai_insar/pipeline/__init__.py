"""InSAR processing stages used by the GUI, the CLI, and the tests."""

from ai_insar.pipeline.coherence import coherence_map, coherence_stack
from ai_insar.pipeline.detect import detect_displacement
from ai_insar.pipeline.engine import (
    PipelineConfig,
    Workspace,
    compute_coherence,
    compute_detection,
    compute_filter,
    compute_interferograms,
    compute_sbas,
    compute_unwrap,
    run_all,
)
from ai_insar.pipeline.interferogram import date_pairs, form_interferograms
from ai_insar.pipeline.sbas import design_matrix, invert_sbas
from ai_insar.pipeline.synthetic import SLCStack, generate_demo_stack, generate_subsidence_stack
from ai_insar.pipeline.unwrap import remove_ramp, unwrap_quality_guided

__all__ = [
    "PipelineConfig",
    "SLCStack",
    "Workspace",
    "coherence_map",
    "coherence_stack",
    "compute_coherence",
    "compute_detection",
    "compute_filter",
    "compute_interferograms",
    "compute_sbas",
    "compute_unwrap",
    "date_pairs",
    "design_matrix",
    "detect_displacement",
    "form_interferograms",
    "generate_demo_stack",
    "generate_subsidence_stack",
    "invert_sbas",
    "remove_ramp",
    "run_all",
    "unwrap_quality_guided",
]
