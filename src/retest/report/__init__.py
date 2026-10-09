"""Report package: equity plots and metrics tables."""

from retest.report.equity_curve import plot_equity_curve
from retest.report.metrics_table import max_drawdown, metrics_table

__all__ = ["max_drawdown", "metrics_table", "plot_equity_curve"]
