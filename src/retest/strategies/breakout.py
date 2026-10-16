"""Breakout-continuation signals: trade the break itself, no return wait.

Structurally different from the retest: the first H1 close beyond a level
(+ tolerance, fresh cross) fires a market entry at the next bar open. Stop sits
beyond the break-bar extreme, target at the nearest known opposing level with an
ATR fallback. Reuses the shared level/ATR/no-lookahead machinery; no daily
confirmations, no return scan - many more signals by construction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from retest.levels.detection import atr_series, detect_levels, filter_levels
from retest.signals.retest import (
    SIGNAL_COLUMNS,
    RetestParams,
    _available_from,
    _empty_signals,
    _in_session,
    _known_levels,
    _price_series,
)

_DAY = pd.Timedelta(days=1)


def generate_breakout_signals(
    hourly: pd.DataFrame,
    levels_frame: pd.DataFrame,
    params: RetestParams | None = None,
    level_session: pd.Timedelta = _DAY,
) -> pd.DataFrame:
    """First fresh close beyond each level -> market entry next bar."""
    params = params or RetestParams.from_config()
    levels = detect_levels(
        levels_frame,
        lookback=params.swing_lookback,
        cluster_ticks=params.cluster_ticks,
        tick_size=params.tick_size,
        atr_period=params.atr_period,
        min_swing_magnitude_atr=params.min_swing_magnitude_atr,
    )
    if params.min_swings_per_level > 1 or params.min_level_spacing_atr > 0:
        levels = filter_levels(
            levels,
            levels_frame,
            min_swings=params.min_swings_per_level,
            min_spacing_atr=params.min_level_spacing_atr,
            atr_period=params.atr_period,
        )
    if not levels or hourly.empty:
        return _empty_signals()
    available = {
        id(lv): _available_from(levels_frame, lv.formed_at, params.swing_lookback, level_session)
        for lv in levels
    }
    resistances = [lv for lv in levels if lv.kind == "resistance"]
    supports = [lv for lv in levels if lv.kind == "support"]
    atr_h = atr_series(hourly, params.atr_period).shift(1)
    highs = hourly["high"].to_numpy()
    lows = hourly["low"].to_numpy()
    closes = hourly["close"].to_numpy()

    rows: list[dict[str, Any]] = []
    for level in levels:
        start = available[id(level)]
        if start is None or hourly.index[-1] < start:
            continue
        price_at = _price_series(
            level, levels_frame, params.swing_lookback, hourly.index, level_session
        )
        i0 = int(hourly.index.searchsorted(start))
        for p in range(i0, len(hourly)):
            a = float(atr_h.iloc[p])
            if not a > 0 or p == 0:
                continue
            tol = params.touch_tolerance_atr * a
            price = float(price_at[p])
            if level.kind == "resistance":
                fresh = closes[p] > price + tol and closes[p - 1] <= price + tol
            else:
                fresh = closes[p] < price - tol and closes[p - 1] >= price - tol
            if not fresh:
                continue  # level not freshly broken yet
            direction = "long" if level.kind == "resistance" else "short"
            ts = hourly.index[p]
            if not _in_session(ts, params):
                break  # first break is the event; dead regardless of session
            if params.max_level_age_days > 0:
                latest = max(level.swing_times) if level.swing_times else level.formed_at
                if (pd.Timestamp(ts) - pd.Timestamp(latest)).days > params.max_level_age_days:
                    break
            ref = float(closes[p])
            buf = params.stop_buffer_atr * a
            if direction == "long":
                stop = float(lows[p]) - buf
                known = _known_levels(
                    resistances, available, levels_frame, params, ts, level_session
                )
                above = [r for r in known if r > ref]
                target = min(above) if above else ref + params.fta_fallback_atr * a
                risk, reward = ref - stop, target - ref
            else:
                stop = float(highs[p]) + buf
                known = _known_levels(supports, available, levels_frame, params, ts, level_session)
                below = [s for s in known if s < ref]
                target = max(below) if below else ref - params.fta_fallback_atr * a
                risk, reward = stop - ref, ref - target
            if not risk > 0 or reward < params.min_reward_risk * risk:
                break
            if params.min_stop_atr > 0 and risk < params.min_stop_atr * a:
                break
            rows.append(
                {
                    "timestamp": ts,
                    "direction": direction,
                    "entry_price": ref,
                    "stop_price": float(stop),
                    "target_price": float(target),
                    "level_price": price,
                    "sweep_extreme": float("nan"),
                    "touches": 0.0,
                    "confirmations": [],
                    "expected_r": float(reward / risk),
                    "entry_mode": "market",
                    "expiry_bars": 0.0,
                }
            )
            break  # level is dead after its first break
    if not rows:
        return _empty_signals()
    out = pd.DataFrame(rows)
    return out.sort_values("timestamp").reset_index(drop=True)[SIGNAL_COLUMNS]
