"""Backtest engine: market/limit entries, stop/target/warning exits, costs, sizing.

A signal at bar ``t`` resolves on the execution frame (H1 or M5): market
orders fill at bar ``t+1``'s open; limit orders fill at the limit price if a
bar trades through it within ``expiry_bars`` (unfilled limits expire with no
trade). Exits per bar, checked in order: session flat (EOD), stop (worst case
first), target, warning-sign close back through the sweep extreme, timeout.
One position at a time; the warning exit applies only when the signal carries
``sweep_extreme``. When ``session_enabled``, fills only happen inside the
entry window and open positions are forced flat at ``force_flat_hour_utc``,
so nothing is carried overnight and nothing trades the illiquid window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from retest.config import get_config


@dataclass
class BacktestParams:
    spread_pips: float = 1.0
    slippage_pips: float = 0.5
    exit_slippage_pips: float = 1.0
    pip_size: float = 0.0001
    risk_per_trade_pct: float = 1.0
    min_equity: float = 10000.0
    max_hold_bars: int = 120
    session_enabled: bool = False
    entry_start_hour_utc: int = 0
    entry_end_hour_utc: int = 20
    force_flat_hour_utc: int = 21

    @classmethod
    def from_config(cls, pair: str | None = None) -> BacktestParams:
        bt = get_config().section("backtest")
        ses = get_config().section("session")
        sizes = bt.get("pip_sizes", {})
        pip_size = (
            float(sizes.get(pair, bt.get("pip_size", 0.0001)))
            if pair
            else float(bt.get("pip_size", 0.0001))
        )
        return cls(
            spread_pips=float(bt.get("spread_pips", 1.0)),
            slippage_pips=float(bt.get("slippage_pips", 0.5)),
            exit_slippage_pips=float(bt.get("exit_slippage_pips", 1.0)),
            pip_size=pip_size,
            risk_per_trade_pct=float(bt.get("risk_per_trade_pct", 1.0)),
            min_equity=float(bt.get("min_equity", 10000)),
            max_hold_bars=int(bt.get("max_hold_bars", 120)),
            session_enabled=bool(ses.get("enabled", False)),
            entry_start_hour_utc=int(ses.get("entry_start_hour_utc", 0)),
            entry_end_hour_utc=int(ses.get("entry_end_hour_utc", 20)),
            force_flat_hour_utc=int(ses.get("force_flat_hour_utc", 21)),
        )


@dataclass
class BacktestResult:
    trades_df: pd.DataFrame
    equity_curve: pd.Series


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    params: BacktestParams | None = None,
    max_hold_bars: int | None = None,
) -> BacktestResult:
    params = params or BacktestParams.from_config()
    max_hold = params.max_hold_bars if max_hold_bars is None else max_hold_bars
    empty_trades = pd.DataFrame(
        columns=[
            "signal_ts",
            "entry_ts",
            "exit_ts",
            "direction",
            "entry_price",
            "exit_price",
            "stop_price",
            "target_price",
            "units",
            "pnl",
            "exit_reason",
        ]
    )
    if signals.empty:
        return BacktestResult(empty_trades, pd.Series(params.min_equity, index=df.index))
    by_bar = {ts: grp for ts, grp in signals.groupby("timestamp")}
    cost = (params.spread_pips + params.slippage_pips) * params.pip_size
    exit_cost = params.exit_slippage_pips * params.pip_size

    equity = float(params.min_equity)
    equity_at = pd.Series(float("nan"), index=df.index)
    trades: list[dict[str, Any]] = []
    open_pos: dict[str, Any] | None = None

    for i in range(len(df)):
        ts = df.index[i]
        bar = df.iloc[i]
        if open_pos is not None and i >= int(open_pos["entry_idx"]):
            if params.session_enabled and _should_flat(ts, open_pos, params):
                exit_price = _eod_price(str(open_pos["direction"]), bar, exit_cost)
                signed = 1.0 if open_pos["direction"] == "long" else -1.0
                pnl = signed * (exit_price - open_pos["entry_price"]) * open_pos["units"]
                equity += pnl
                trades.append(
                    {
                        **open_pos,
                        "exit_ts": ts,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "exit_reason": "eod",
                    }
                )
                equity_at[ts] = equity
                open_pos = None
            else:
                outcome = _check_exit(open_pos, bar, i, max_hold, exit_cost)
                if outcome is not None:
                    exit_price, reason = outcome
                    signed = 1.0 if open_pos["direction"] == "long" else -1.0
                    pnl = signed * (exit_price - open_pos["entry_price"]) * open_pos["units"]
                    equity += pnl
                    trades.append(
                        {
                            **open_pos,
                            "exit_ts": ts,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "exit_reason": reason,
                        }
                    )
                    equity_at[ts] = equity
                    open_pos = None
        if open_pos is None and ts in by_bar:
            if params.session_enabled and _no_entry_bar(ts, params):
                continue  # exit-only bar: no new entries
            for _, s in by_bar[ts].iterrows():
                if i + 1 >= len(df):
                    break
                mode = s.get("entry_mode", "market")
                expiry = int(s.get("expiry_bars", 0) or 0)
                if mode == "limit":
                    fill, fill_idx = _limit_fill(
                        str(s["direction"]),
                        float(s["entry_price"]),
                        df,
                        i,
                        expiry,
                        cost,
                        params if params.session_enabled else None,
                    )
                    if fill is None:
                        continue  # expired unfilled: no trade
                else:
                    fill_idx = i + 1
                    if params.session_enabled and not _in_entry_window(df.index[fill_idx], params):
                        continue
                    fill = _fill_price(str(s["direction"]), df.iloc[i + 1], cost)
                risk_dist = abs(float(s["stop_price"]) - fill)
                if not risk_dist > 0:
                    continue
                open_pos = {
                    "signal_ts": ts,
                    "entry_ts": df.index[fill_idx],
                    "entry_idx": fill_idx,
                    "direction": str(s["direction"]),
                    "entry_price": fill,
                    "stop_price": float(s["stop_price"]),
                    "target_price": float(s["target_price"]),
                    "sweep_extreme": (
                        float(s["sweep_extreme"])
                        if "sweep_extreme" in s and pd.notna(s["sweep_extreme"])
                        else None
                    ),
                }
                open_pos["units"] = equity * params.risk_per_trade_pct / 100.0 / risk_dist
                break  # one position at a time
    if open_pos is not None:  # force-close at the final close
        signed = 1.0 if open_pos["direction"] == "long" else -1.0
        exit_price = float(df["close"].iloc[-1])
        pnl = signed * (exit_price - open_pos["entry_price"]) * open_pos["units"]
        equity += pnl
        trades.append(
            {
                **open_pos,
                "exit_ts": df.index[-1],
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": "timeout",
            }
        )
        equity_at[df.index[-1]] = equity
    equity_at.iloc[0] = params.min_equity
    curve = equity_at.ffill()
    curve.name = "equity"
    trades_df = (
        pd.DataFrame(trades).drop(columns=["entry_idx", "sweep_extreme"])
        if trades
        else empty_trades
    )
    return BacktestResult(trades_df, curve)


def _fill_price(direction: str, bar: pd.Series, cost: float) -> float:
    if direction == "long":
        return float(bar["open"]) + cost
    return float(bar["open"]) - cost


def _limit_fill(
    direction: str,
    limit: float,
    df: pd.DataFrame,
    i: int,
    expiry: int,
    cost: float,
    params: BacktestParams | None = None,
) -> tuple[float | None, int]:
    """First liquid bar after ``i`` (up to ``expiry`` bars out) trading through ``limit``.

    With session params, illiquid bars are skipped (no fill in the 21-00 UTC
    window) and the search stops at the next flat bar, so resting orders never
    fill overnight — new slate each morning.
    """
    last = min(i + expiry, len(df) - 1)
    if params is not None:
        for k in range(i + 1, last + 1):
            if _is_dead_bar(df.index[k], params):
                last = k - 1
                break
    for j in range(i + 1, last + 1):
        if params is not None and not _in_entry_window(df.index[j], params):
            continue
        bar = df.iloc[j]
        touched = bar["low"] <= limit if direction == "long" else bar["high"] >= limit
        if touched:
            fill = limit + cost if direction == "long" else limit - cost
            return fill, j
    return None, -1


def _as_utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _hour_utc(ts: Any) -> int:
    return int(_as_utc(ts).hour)


def _weekday_utc(ts: Any) -> int:
    return int(_as_utc(ts).weekday())  # Monday=0 … Sunday=6


def _in_entry_window(ts: Any, params: BacktestParams) -> bool:
    if _weekday_utc(ts) >= 5:
        return False  # no weekend entries (crypto prints weekend bars)
    t = _as_utc(ts)
    if int(t.weekday()) == 4 and int(t.hour) >= params.entry_end_hour_utc:
        return False  # Friday late: flat into the weekend, no new risk
    start, end = params.entry_start_hour_utc, params.entry_end_hour_utc
    h = int(t.hour)
    if start <= end:
        return start <= h < end
    return h >= start or h < end  # overnight window, e.g. 22-06


def _is_flat_bar(ts: Any, params: BacktestParams) -> bool:
    return _hour_utc(ts) == params.force_flat_hour_utc


def _is_dead_bar(ts: Any, params: BacktestParams) -> bool:
    """Resting orders die here: flat bar, weekend bar, or Friday close."""
    if _is_flat_bar(ts, params):
        return True
    return not _in_entry_window(ts, params) and (
        _weekday_utc(ts) >= 5 or _hour_utc(ts) >= params.entry_end_hour_utc
    )


def _no_entry_bar(ts: Any, params: BacktestParams) -> bool:
    return _is_flat_bar(ts, params) or not _in_entry_window(ts, params)


def _should_flat(ts: Any, pos: dict[str, Any], params: BacktestParams) -> bool:
    """Flat at 21:00, Friday 20:00 into the weekend, or any later-date bar.

    The date backstop covers instruments with no 21:00 bar (metals halt
    21:00-22:00 UTC): the next tradable bar closes the position instead of
    carrying it for days.
    """
    t = _as_utc(ts)
    if _is_flat_bar(ts, params):
        return True
    if int(t.weekday()) == 4 and int(t.hour) >= params.entry_end_hour_utc:
        return True  # Friday close: flat before the weekend
    if int(t.weekday()) >= 5:
        return True  # weekend bar (crypto): stay flat
    entry_day = _as_utc(pos["entry_ts"]).date()
    return t.date() > entry_day


def _eod_price(direction: str, bar: pd.Series, exit_cost: float) -> float:
    if direction == "long":
        return float(bar["open"]) - exit_cost
    return float(bar["open"]) + exit_cost


def _check_exit(
    pos: dict[str, Any], bar: pd.Series, i: int, max_hold_bars: int, exit_cost: float = 0.0
) -> tuple[float, str] | None:
    if pos["direction"] == "long":
        if bar["low"] <= pos["stop_price"]:
            return pos["stop_price"] - exit_cost, "stop"
        if bar["high"] >= pos["target_price"]:
            return pos["target_price"] - exit_cost, "target"
        if pos["sweep_extreme"] is not None and bar["close"] < pos["sweep_extreme"]:
            return float(bar["close"]) - exit_cost, "warning"
    else:
        if bar["high"] >= pos["stop_price"]:
            return pos["stop_price"] + exit_cost, "stop"
        if bar["low"] <= pos["target_price"]:
            return pos["target_price"] + exit_cost, "target"
        if pos["sweep_extreme"] is not None and bar["close"] > pos["sweep_extreme"]:
            return float(bar["close"]) + exit_cost, "warning"
    if i - pos["entry_idx"] >= max_hold_bars:
        return float(bar["close"]), "timeout"
    return None
