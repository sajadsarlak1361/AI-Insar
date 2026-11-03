"""Headless self-check against the built-in subsidence bowl."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai_insar.pipeline.engine import Workspace, run_all
from ai_insar.pipeline.synthetic import SLCStack, generate_demo_stack


@dataclass
class CheckResult:
    """Numbers printed by ``ai-insar check`` and the pass/fail decision."""

    passed: bool
    messages: list[str] = field(default_factory=list)
    correlation: float = 0.0
    relative_error: float = 1.0
    core_recall: float = 0.0
    outside_false_positive_rate: float = 1.0
    estimated_peak_m_yr: float = 0.0
    truth_peak_m_yr: float = 0.0
    mask_pixels: int = 0

    def render(self) -> str:
        lines = list(self.messages)
        lines.append(
            "Peak LOS velocity "
            f"{self.estimated_peak_m_yr:.4f} m/yr "
            f"(truth {self.truth_peak_m_yr:.4f}, relative error {self.relative_error:.2%})."
        )
        lines.append(
            f"Velocity correlation {self.correlation:.3f}. "
            f"Core recall {self.core_recall:.3f}. "
            f"Outside false-positive rate {self.outside_false_positive_rate:.3f}. "
            f"Mask pixels {self.mask_pixels}."
        )
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def recovery_metrics(stack: SLCStack, workspace: Workspace) -> CheckResult:
    """Compare a finished workspace to the synthetic ground truth on ``stack``."""
    if stack.truth_velocity_m_yr is None or workspace.velocity_m_yr is None or workspace.mask is None:
        raise ValueError("recovery metrics need a synthetic stack and a finished SBAS run")
    truth = np.asarray(stack.truth_velocity_m_yr, dtype=np.float64)
    estimate = np.asarray(workspace.velocity_m_yr, dtype=np.float64)
    peak = int(np.argmax(np.abs(truth)))
    row, col = np.unravel_index(peak, truth.shape)
    row = int(row)
    col = int(col)
    r0, r1 = max(0, row - 1), min(truth.shape[0], row + 2)
    c0, c1 = max(0, col - 1), min(truth.shape[1], col + 2)
    truth_peak = float(truth[r0:r1, c0:c1].mean())
    est_peak = float(estimate[r0:r1, c0:c1].mean())
    relative = abs(est_peak - truth_peak) / max(abs(truth_peak), 1e-6)
    correlation = float(np.corrcoef(truth.ravel(), estimate.ravel())[0, 1])

    magnitude = np.abs(truth)
    maximum = float(magnitude.max())
    core = magnitude >= 0.5 * maximum
    outside = magnitude < 0.15 * maximum
    mask = np.asarray(workspace.mask, dtype=bool)
    core_recall = float(np.logical_and(mask, core).sum() / max(int(core.sum()), 1))
    outside_rate = float(np.logical_and(mask, outside).sum() / max(int(outside.sum()), 1))
    sign_ok = est_peak * truth_peak > 0
    passed = bool(
        sign_ok
        and relative <= 0.40
        and correlation >= 0.80
        and core_recall >= 0.70
        and outside_rate <= 0.15
    )
    return CheckResult(
        passed=passed,
        correlation=correlation,
        relative_error=relative,
        core_recall=core_recall,
        outside_false_positive_rate=outside_rate,
        estimated_peak_m_yr=est_peak,
        truth_peak_m_yr=truth_peak,
        mask_pixels=int(mask.sum()),
    )


def run_self_check(stack: SLCStack | None = None) -> CheckResult:
    """Generate the demo stack (unless one is provided), process it, and score it."""
    scene = stack if stack is not None else generate_demo_stack()
    logs: list[str] = []
    workspace = Workspace(stack=scene)
    run_all(workspace, log=logs.append)
    result = recovery_metrics(scene, workspace)
    result.messages = logs
    return result
