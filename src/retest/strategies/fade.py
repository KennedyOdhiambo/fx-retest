"""Level-fade signals: trade touches of daily levels against the approach.

The mirror image of breakout-continuation: support touches are bought, resistance
touches are sold, entry is a limit order sitting at the level, stop beyond the
touch episode's extreme, target the nearest known opposing level (ATR fallback).
A level dies once it actually breaks (close beyond tolerance), so fades only
trade levels that are still intact. Same no-lookahead availability rules.
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
    touch_episodes,
)

_DAY = pd.Timedelta(days=1)


def generate_fade_signals(
    hourly: pd.DataFrame,
    levels_frame: pd.DataFrame,
    params: RetestParams | None = None,
    level_session: pd.Timedelta = _DAY,
) -> pd.DataFrame:
    """Limit-entry fades at intact level touches (first ``max_returns_per_level`` episodes)."""
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

    max_eps = max(params.max_returns_per_level, 1)
    rows: list[dict[str, Any]] = []
    for level in levels:
        start = available[id(level)]
        if start is None or hourly.index[-1] < start:
            continue
        price_at = _price_series(
            level, levels_frame, params.swing_lookback, hourly.index, level_session
        )
        i0 = int(hourly.index.searchsorted(start))
        episodes, _bpos = touch_episodes(
            price_at, level.kind, hourly, i0, params.touch_tolerance_atr, params.atr_period
        )
        if not episodes:
            continue
        direction = "short" if level.kind == "resistance" else "long"
        for ep in episodes[:max_eps]:
            p0 = ep[0]
            ts = hourly.index[p0]
            a = float(atr_h.iloc[p0])
            if not a > 0:
                continue
            if not _in_session(ts, params):
                continue
            if params.max_level_age_days > 0:
                latest = max(level.swing_times) if level.swing_times else level.formed_at
                if (pd.Timestamp(ts) - pd.Timestamp(latest)).days > params.max_level_age_days:
                    continue
            limit = float(price_at[p0])
            buf = params.stop_buffer_atr * a
            if direction == "long":
                stop = min(float(lows[p]) for p in ep) - buf
                known = _known_levels(
                    resistances, available, levels_frame, params, ts, level_session
                )
                above = [r for r in known if r > limit]
                target = min(above) if above else limit + params.fta_fallback_atr * a
                risk, reward = limit - stop, target - limit
            else:
                stop = max(float(highs[p]) for p in ep) + buf
                known = _known_levels(supports, available, levels_frame, params, ts, level_session)
                below = [s for s in known if s < limit]
                target = max(below) if below else limit - params.fta_fallback_atr * a
                risk, reward = stop - limit, limit - target
            if not risk > 0 or reward < params.min_reward_risk * risk:
                continue
            if params.min_stop_atr > 0 and risk < params.min_stop_atr * a:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "direction": direction,
                    "entry_price": limit,
                    "stop_price": float(stop),
                    "target_price": float(target),
                    "level_price": limit,
                    "sweep_extreme": float("nan"),
                    "touches": float(len(ep)),
                    "confirmations": [],
                    "expected_r": float(reward / risk),
                    "entry_mode": "limit",
                    "expiry_bars": float(params.limit_expiry_bars),
                }
            )
    if not rows:
        return _empty_signals()
    out = pd.DataFrame(rows)
    return out.sort_values("timestamp").reset_index(drop=True)[SIGNAL_COLUMNS]
