"""Validation package: falsification, walk-forward, significance, sensitivity."""

from retest.validation.random_baseline import (
    baseline_passes,
    random_signals,
    run_random_baseline,
)
from retest.validation.sensitivity import SweepResult, sweep_parameter
from retest.validation.significance import (
    deflated_sharpe,
    sharpe_confidence_interval,
    sharpe_ratio,
    significance_report,
)
from retest.validation.walk_forward import (
    WindowResult,
    walk_forward_evaluate,
    walk_forward_windows,
)

__all__ = [
    "SweepResult",
    "WindowResult",
    "baseline_passes",
    "deflated_sharpe",
    "random_signals",
    "run_random_baseline",
    "sharpe_confidence_interval",
    "sharpe_ratio",
    "significance_report",
    "sweep_parameter",
    "walk_forward_evaluate",
    "walk_forward_windows",
]
