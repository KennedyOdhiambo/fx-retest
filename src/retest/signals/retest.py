"""Retest signals: breakout-then-return off higher-timeframe levels (strategy.md).

Daily levels/targets/confirmations, H1 break/return/entry/stop. Up to
``max_returns_per_level`` signals per level (default 1: first return only),
touch-graded confirmations, limit entry so the existing limit-fill engine
executes unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from retest.config import get_config
from retest.confirmation.sfp import detect_sfp
from retest.levels.detection import atr_series, detect_levels, filter_levels

_DAY = pd.Timedelta(days=1)  # daily bars are open-stamped 24h sessions


@dataclass
class RetestParams:
    swing_lookback: int = 5
    cluster_ticks: int = 10
    tick_size: float = 0.0001
    min_swing_magnitude_atr: float = 0.3
    touch_tolerance_atr: float = 0.1
    atr_period: int = 14
    max_break_return_bars: int = 24
    touch_gates: dict[int, int] = field(default_factory=lambda: {3: 0, 2: 1, 1: 2})
    pin_wick_body_ratio: float = 2.0
    bias_sfp_wick_atr: float = 0.1
    bias_sfp_close_atr: float = 0.05
    fib_lookback: int = 20
    fib_ratios: tuple[float, ...] = (0.382, 0.5, 0.618)
    fib_proximity_atr: float = 0.2
    special_fib_rule: bool = True
    atr_stretch_period: int = 20
    atr_stretch_multiple: float = 1.0
    stop_buffer_atr: float = 0.1
    fta_fallback_atr: float = 2.0
    min_reward_risk: float = 1.0
    min_stop_atr: float = 0.25
    limit_expiry_bars: int = 24
    max_returns_per_level: int = 1  # how many return episodes may trade per level
    max_level_age_days: int = 0  # 0 = off; drop levels older than this at signal time
    session_enabled: bool = False
    entry_start_hour_utc: int = 7
    entry_end_hour_utc: int = 20
    drop_stretch_confirmation: bool = False  # True: pattern/fib only
    min_swings_per_level: int = 1  # 1 = off; 2 = equal highs/lows only
    min_level_spacing_atr: float = 0.0  # 0 = off; drop weaker of close levels
    ban_one_touch: bool = False  # True: trade 2-3 touch setups only
    long_extra_confirmations: int = 0  # longs must clear required + this
    atr_regime_lookback: int = 0  # 0 = off; else bars for H1 ATR median
    atr_regime_mult: float = 1.0  # require cur ATR >= mult x median

    @classmethod
    def from_config(cls) -> RetestParams:
        cfg = get_config()
        lvl = cfg.section("levels")
        ret = cfg.section("retest")
        ses = cfg.section("session")
        return cls(
            swing_lookback=int(lvl.get("swing_lookback", 5)),
            cluster_ticks=int(lvl.get("cluster_ticks", 10)),
            tick_size=float(lvl.get("tick_size", 0.0001)),
            min_swing_magnitude_atr=float(lvl.get("min_swing_magnitude_atr", 0.3)),
            touch_tolerance_atr=float(lvl.get("touch_tolerance_atr", 0.1)),
            atr_period=int(cfg.get("atr.period", 14)),
            max_break_return_bars=int(ret.get("max_break_return_bars", 24)),
            touch_gates={int(k): int(v) for k, v in ret.get("touch_gates", {}).items()}
            or {3: 0, 2: 1, 1: 2},
            pin_wick_body_ratio=float(ret.get("pin_wick_body_ratio", 2.0)),
            bias_sfp_wick_atr=float(ret.get("bias_sfp_wick_atr", 0.1)),
            bias_sfp_close_atr=float(ret.get("bias_sfp_close_atr", 0.05)),
            fib_lookback=int(ret.get("fib_lookback", 20)),
            fib_ratios=tuple(float(r) for r in ret.get("fib_ratios", [0.382, 0.5, 0.618])),
            fib_proximity_atr=float(ret.get("fib_proximity_atr", 0.2)),
            special_fib_rule=bool(ret.get("special_fib_rule", True)),
            atr_stretch_period=int(ret.get("atr_stretch_period", 20)),
            atr_stretch_multiple=float(ret.get("atr_stretch_multiple", 1.0)),
            stop_buffer_atr=float(ret.get("stop_buffer_atr", 0.1)),
            fta_fallback_atr=float(ret.get("fta_fallback_atr", 2.0)),
            min_reward_risk=float(ret.get("min_reward_risk", 1.0)),
            min_stop_atr=float(ret.get("min_stop_atr", 0.25)),
            limit_expiry_bars=int(ret.get("limit_expiry_bars", 24)),
            max_returns_per_level=int(ret.get("max_returns_per_level", 1)),
            max_level_age_days=int(
                ret.get("max_level_age_days", lvl.get("max_level_age_days", 0))
            ),
            session_enabled=bool(ses.get("enabled", False)),
            entry_start_hour_utc=int(ses.get("entry_start_hour_utc", 7)),
            entry_end_hour_utc=int(ses.get("entry_end_hour_utc", 20)),
            min_swings_per_level=int(lvl.get("min_swings_per_level", 1)),
            min_level_spacing_atr=float(lvl.get("min_level_spacing_atr", 0.0)),
            ban_one_touch=bool(ret.get("ban_one_touch", False)),
            long_extra_confirmations=int(ret.get("long_extra_confirmations", 0)),
            atr_regime_lookback=int(ret.get("atr_regime_lookback", 0)),
            atr_regime_mult=float(ret.get("atr_regime_mult", 1.0)),
        )


SIGNAL_COLUMNS = [
    "timestamp",
    "direction",
    "entry_price",
    "stop_price",
    "target_price",
    "level_price",
    "sweep_extreme",
    "touches",
    "confirmations",
    "expected_r",
    "entry_mode",
    "expiry_bars",
]


def _in_session(ts: Any, params: RetestParams) -> bool:
    """Return-bar must sit in the liquid entry window (UTC, weekdays only)."""
    if not params.session_enabled:
        return True
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    if int(t.weekday()) >= 5:
        return False
    if int(t.weekday()) == 4 and int(t.hour) >= params.entry_end_hour_utc:
        return False
    h = int(t.hour)
    start, end = params.entry_start_hour_utc, params.entry_end_hour_utc
    if start <= end:
        return start <= h < end
    return h >= start or h < end


def pin_bar(df: pd.DataFrame, pos: int, direction: str, ratio: float = 2.0) -> bool:
    """Pin bar at row ``pos``: dominant wick >= ``ratio`` x body, against it."""
    if pos < 0 or pos >= len(df):
        return False
    o, h, lo, c = (float(df.iloc[pos][k]) for k in ("open", "high", "low", "close"))
    body = abs(o - c)
    if not body > 0:
        return False
    if direction == "long":
        return (min(o, c) - lo) >= ratio * body and (h - max(o, c)) <= (min(o, c) - lo)
    return (h - max(o, c)) >= ratio * body and (min(o, c) - lo) <= (h - max(o, c))


def engulfing(df: pd.DataFrame, pos: int, direction: str) -> bool:
    """Two-bar engulfing at row ``pos`` in ``direction``."""
    if pos < 1 or pos >= len(df):
        return False
    po, pc = float(df.iloc[pos - 1]["open"]), float(df.iloc[pos - 1]["close"])
    o, c = float(df.iloc[pos]["open"]), float(df.iloc[pos]["close"])
    if direction == "long":
        return pc < po and c > o and o <= pc and c >= po
    return pc > po and c < o and o >= pc and c <= po


def fib_grid(daily: pd.DataFrame, end_pos: int, lookback: int, ratios: tuple) -> list[float]:
    """Retracement levels of the deep high->low over ``lookback`` bars ending at ``end_pos``."""
    window = daily.iloc[max(0, end_pos - lookback + 1) : end_pos + 1]
    if window.empty:
        return []
    hi, lo = float(window["high"].max()), float(window["low"].min())
    if not hi > lo:
        return []
    return [hi - r * (hi - lo) for r in ratios]


def daily_confirmations(
    daily: pd.DataFrame,
    level_price: float,
    direction: str,
    ret_ts: pd.Timestamp,
    params: RetestParams,
    session: pd.Timedelta = _DAY,
) -> list[str]:
    """Higher-timeframe confirmations known before the return bar (completed bars only)."""
    pos = _completed_daily_pos(daily, ret_ts, session)
    if pos < 1:
        return []
    out: list[str] = []
    if (
        pin_bar(daily, pos, direction, params.pin_wick_body_ratio)
        or engulfing(daily, pos, direction)
        or bool(
            detect_sfp(
                daily.iloc[: pos + 1],
                level_price,
                direction,
                params.bias_sfp_wick_atr,
                params.bias_sfp_close_atr,
                params.atr_period,
            ).iloc[-1]
        )
    ):
        out.append("d1_pattern")
    atr_d = atr_series(daily, params.atr_period)
    a = float(atr_d.iloc[pos])
    if a > 0:
        grid = fib_grid(daily, pos, params.fib_lookback, params.fib_ratios)
        if grid and min(abs(level_price - f) for f in grid) <= params.fib_proximity_atr * a:
            out.append("fib")
    sma = daily["close"].rolling(params.atr_stretch_period).mean()
    atr_s = atr_series(daily, params.atr_stretch_period)
    if (
        not params.drop_stretch_confirmation
        and pos >= params.atr_stretch_period
        and float(atr_s.iloc[pos]) > 0
    ):
        if abs(level_price - float(sma.iloc[pos])) >= params.atr_stretch_multiple * float(
            atr_s.iloc[pos]
        ):
            out.append("atr_stretch")
    return out


def fib_override(
    daily: pd.DataFrame,
    level_price: float,
    ret_ts: pd.Timestamp,
    params: RetestParams,
    session: pd.Timedelta = _DAY,
) -> float | None:
    """Nearest fib within proximity, else None."""
    pos = _completed_daily_pos(daily, ret_ts, session)
    if pos < 1:
        return None
    a = float(atr_series(daily, params.atr_period).iloc[pos])
    if not a > 0:
        return None
    grid = fib_grid(daily, pos, params.fib_lookback, params.fib_ratios)
    near = [f for f in grid if abs(level_price - f) <= params.fib_proximity_atr * a]
    return min(near, key=lambda f: abs(level_price - f)) if near else None


def _return_episodes(
    price_at: np.ndarray,
    hourly: pd.DataFrame,
    bpos: int,
    deadline: int,
    direction: str,
    params: RetestParams,
) -> list[tuple[int, pd.Timestamp]]:
    """First bar of each return to the broken level: the bar must overlap the
    tolerance band around the level AND close on the broken side (a bar that
    closes back through has re-broken, not returned)."""
    atr = atr_series(hourly, params.atr_period).shift(1).to_numpy()
    highs = hourly["high"].to_numpy()
    lows = hourly["low"].to_numpy()
    closes = hourly["close"].to_numpy()
    out: list[tuple[int, pd.Timestamp]] = []
    prev = -2
    for p in range(bpos + 1, deadline + 1):
        tol = params.touch_tolerance_atr * atr[p]
        if not tol > 0:
            prev = -2
            continue
        price = float(price_at[p])
        overlap = lows[p] <= price + tol and highs[p] >= price - tol
        if direction == "long":
            held = closes[p] >= price - tol
        else:
            held = closes[p] <= price + tol
        if overlap and held and p > prev + 1:
            out.append((p, hourly.index[p]))
        prev = p if (overlap and held) else -2
    return out


def touch_episodes(
    price_at: np.ndarray,
    kind: str,
    hourly: pd.DataFrame,
    i0: int,
    tolerance_atr: float,
    atr_period: int,
) -> tuple[list[list[int]], int | None]:
    """Approach episodes from ``i0`` and the first break-bar position (or None).

    Returns (episodes, break_pos): episodes are lists of consecutive bar
    positions touching the level; the break bar ends the timeline. The forming
    swing counts as touch #1, so touches = 1 + episodes before the break.
    """
    atr = atr_series(hourly, atr_period).shift(1).to_numpy()
    highs = hourly["high"].to_numpy()
    lows = hourly["low"].to_numpy()
    closes = hourly["close"].to_numpy()
    episodes: list[list[int]] = []
    current: list[int] = []
    for p in range(i0, len(hourly)):
        tol = tolerance_atr * atr[p]
        if not tol > 0:
            if current:
                episodes.append(current)
                current = []
            continue
        price = float(price_at[p])
        if kind == "resistance":
            touched = highs[p] >= price - tol
            broke = closes[p] > price + tol
        else:
            touched = lows[p] <= price + tol
            broke = closes[p] < price - tol
        if broke:
            if current:
                episodes.append(current)
            return episodes, p
        if touched:
            if current and p > current[-1] + 1:
                episodes.append(current)
                current = []
            current.append(p)
        elif current:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes, None


def generate_retest_signals(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    params: RetestParams | None = None,
) -> pd.DataFrame:
    """Breakout-then-return signals, one per level at most (Daily levels, H1 exec)."""
    return _generate(hourly, daily, params or RetestParams.from_config(), _DAY)


def _generate(
    exec_frame: pd.DataFrame,
    levels_frame: pd.DataFrame,
    params: RetestParams,
    level_session: pd.Timedelta,
) -> pd.DataFrame:
    """Core engine: levels/targets/confirmations from ``levels_frame``,
    break/return/entry/stop on ``exec_frame``."""
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
    if not levels or exec_frame.empty:
        return _empty_signals()
    available = {
        id(lv): _available_from(levels_frame, lv.formed_at, params.swing_lookback, level_session)
        for lv in levels
    }
    resistances = [lv for lv in levels if lv.kind == "resistance"]
    supports = [lv for lv in levels if lv.kind == "support"]
    atr_h = atr_series(exec_frame, params.atr_period).shift(1)
    highs = exec_frame["high"].to_numpy()
    lows = exec_frame["low"].to_numpy()

    rows: list[dict[str, Any]] = []
    for level in levels:
        start = available[id(level)]
        if start is None or exec_frame.index[-1] < start:
            continue
        price_at = _price_series(
            level, levels_frame, params.swing_lookback, exec_frame.index, level_session
        )
        i0 = int(exec_frame.index.searchsorted(start))
        episodes, bpos = touch_episodes(
            price_at, level.kind, exec_frame, i0, params.touch_tolerance_atr, params.atr_period
        )
        if bpos is None:
            continue  # never broke: not a retest level
        # break direction decides the trade (resistance broken up -> long)
        if level.kind == "resistance":
            direction = "long"
        else:
            direction = "short"
        touches = 1 + sum(1 for ep in episodes if ep[-1] < bpos)
        if params.ban_one_touch and touches < 2:
            continue
        required = params.touch_gates.get(min(touches, 3), 0)
        if direction == "long":
            required += params.long_extra_confirmations
        # first N return episodes trade (N=1 keeps the first-return-only rule)
        deadline = min(bpos + params.max_break_return_bars, len(exec_frame) - 1)
        returns = _return_episodes(price_at, exec_frame, bpos, deadline, direction, params)
        if not returns:
            continue
        for p, ret_ts in returns[: max(params.max_returns_per_level, 1)]:
            if not _in_session(ret_ts, params):
                continue
            if params.max_level_age_days > 0:
                latest_swing = (
                    max(level.swing_times) if level.swing_times else level.formed_at
                )
                if (
                    pd.Timestamp(ret_ts) - pd.Timestamp(latest_swing)
                ).days > params.max_level_age_days:
                    continue
            limit = _price_asof(level, levels_frame, params.swing_lookback, ret_ts, level_session)
            factors = daily_confirmations(
                levels_frame, limit, direction, ret_ts, params, level_session
            )
            if len(factors) < required:
                continue
            a = float(atr_h.iloc[p])
            if not a > 0:
                continue
            if params.atr_regime_lookback > 0:
                regime_start = max(0, p - params.atr_regime_lookback)
                med = (
                    float(atr_h.iloc[regime_start:p].median()) if p > regime_start else 0.0
                )
                if not med > 0 or a < params.atr_regime_mult * med:
                    continue  # dead board: chop filter
            if params.special_fib_rule:
                override = fib_override(levels_frame, limit, ret_ts, params, level_session)
                if override is not None:
                    limit = float(override)
            buf = params.stop_buffer_atr * a
            if direction == "long":
                stop = float(lows[p] - buf)
                level_now = _known_levels(
                    resistances, available, levels_frame, params, ret_ts, level_session
                )
                above = [r for r in level_now if r > limit]
                target = min(above) if above else limit + params.fta_fallback_atr * a
                risk, reward = limit - stop, target - limit
                extreme = float(lows[p])
            else:
                stop = float(highs[p] + buf)
                level_now = _known_levels(
                    supports, available, levels_frame, params, ret_ts, level_session
                )
                below = [s for s in level_now if s < limit]
                target = max(below) if below else limit - params.fta_fallback_atr * a
                risk, reward = stop - limit, limit - target
                extreme = float(highs[p])
            if not risk > 0 or reward < params.min_reward_risk * risk:
                continue
            if params.min_stop_atr > 0 and risk < params.min_stop_atr * a:
                continue  # stop inside noise: leverage fantasy, skip
            rows.append(
                {
                    "timestamp": ret_ts,
                    "direction": direction,
                    "entry_price": float(limit),
                    "stop_price": float(stop),
                    "target_price": float(target),
                    "level_price": float(limit),
                    "sweep_extreme": extreme,
                    "touches": touches,
                    "confirmations": factors,
                    "expected_r": float(reward / risk),
                    "entry_mode": "limit",
                    "expiry_bars": float(params.limit_expiry_bars),
                }
            )
    if not rows:
        return _empty_signals()
    out = pd.DataFrame(rows)
    return out.sort_values("timestamp").reset_index(drop=True)[SIGNAL_COLUMNS]


def _known_levels(
    candidates: list[Any],
    available: dict[int, pd.Timestamp | None],
    daily: pd.DataFrame,
    params: RetestParams,
    ret_ts: pd.Timestamp,
    session: pd.Timedelta = _DAY,
) -> list[float]:
    """Prices of candidate levels a trader could already draw at ``ret_ts``."""
    out = []
    for lv in candidates:
        avail = available[id(lv)]
        if avail is None or avail > ret_ts:
            continue
        out.append(_price_asof(lv, daily, params.swing_lookback, ret_ts, session))
    return out


def _available_from(
    daily: pd.DataFrame,
    formed_at: pd.Timestamp,
    lookback: int,
    session: pd.Timedelta = _DAY,
) -> pd.Timestamp | None:
    """First moment the swing at ``formed_at`` is knowable.

    Bars are open-stamped: the last right-side confirmation bar
    (``formed_at + lookback``) closes one session after its timestamp.
    Using the bar's own timestamp would front-run its close.
    """
    if formed_at not in daily.index:
        return None
    pos = daily.index.get_loc(formed_at) + lookback
    if pos >= len(daily):
        return None
    return daily.index[pos] + session


def _completed_daily_pos(
    daily: pd.DataFrame, ret_ts: pd.Timestamp, session: pd.Timedelta = _DAY
) -> int:
    """Index of the last level-frame bar whose session has closed at ``ret_ts``."""
    close_times = daily.index + session
    return int(close_times.searchsorted(ret_ts, side="right")) - 1


def _price_asof(
    level: Any,
    daily: pd.DataFrame,
    lookback: int,
    ts: pd.Timestamp,
    session: pd.Timedelta = _DAY,
) -> float:
    """Cluster mean using only swings confirmed by ``ts`` (no future swings)."""
    total, n = 0.0, 0
    for swing_ts, price in zip(level.swing_times, level.swing_prices, strict=False):
        conf = _available_from(daily, swing_ts, lookback, session)
        if conf is not None and conf <= ts:
            total += price
            n += 1
    return total / n if n else float(level.price)


def _price_series(
    level: Any,
    daily: pd.DataFrame,
    lookback: int,
    hourly_index: pd.DatetimeIndex,
    session: pd.Timedelta = _DAY,
) -> np.ndarray:
    """Per-bar level price: mean of swings confirmed by each bar's timestamp."""
    pairs: list[tuple[pd.Timestamp, float]] = []
    for swing_ts, price in zip(level.swing_times, level.swing_prices, strict=False):
        conf = _available_from(daily, swing_ts, lookback, session)
        if conf is not None:
            pairs.append((conf, float(price)))
    if not pairs:
        return np.full(len(hourly_index), float(level.price), dtype=float)
    pairs.sort(key=lambda item: item[0])
    conf_idx = pd.DatetimeIndex([c for c, _ in pairs])
    cum = np.cumsum(np.array([p for _, p in pairs], dtype=float))
    n = conf_idx.searchsorted(hourly_index, side="right")
    out = np.empty(len(hourly_index), dtype=float)
    out[n == 0] = float(level.price)
    nonzero = n > 0
    out[nonzero] = cum[n[nonzero] - 1] / n[nonzero]
    return out


def _empty_signals() -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "direction": pd.Series(dtype=object),
            "entry_price": pd.Series(dtype=float),
            "stop_price": pd.Series(dtype=float),
            "target_price": pd.Series(dtype=float),
            "level_price": pd.Series(dtype=float),
            "sweep_extreme": pd.Series(dtype=float),
            "touches": pd.Series(dtype=float),
            "confirmations": pd.Series(dtype=object),
            "expected_r": pd.Series(dtype=float),
            "entry_mode": pd.Series(dtype=object),
            "expiry_bars": pd.Series(dtype=float),
        }
    )
    return out[SIGNAL_COLUMNS]
