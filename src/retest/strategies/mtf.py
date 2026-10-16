"""Multi-timeframe retest: same break-then-return rules, finer level timeframes.

The daily-level pipeline starves trade counts. This variant keeps the exact
retest engine (levels -> break -> first return -> limit) but detects levels on
H2/H4/H1 bars resampled from H1 so there are many more breakable levels while
execution stays on H1. All no-lookahead gating is shared with the core engine.
"""

from __future__ import annotations

import pandas as pd

from retest.signals.retest import RetestParams, _generate

_DAY = pd.Timedelta(days=1)


def resample_ohlc(h1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate H1 bars into a higher timeframe (right edge = H1 close stamps)."""
    out = h1.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def generate_mtf_retest(
    hourly: pd.DataFrame,
    levels_frame: pd.DataFrame,
    params: RetestParams | None = None,
    level_session: pd.Timedelta = _DAY,
) -> pd.DataFrame:
    """Retest signals with levels from ``levels_frame`` and H1 break/return/entry."""
    return _generate(hourly, levels_frame, params or RetestParams.from_config(), level_session)
