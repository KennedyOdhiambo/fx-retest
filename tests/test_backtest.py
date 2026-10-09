"""Backtest tests: market/limit fills, stop-first exits, warning exit, costs."""

import pandas as pd
import pytest

from retest.backtest.engine import BacktestParams, run_backtest
from retest.data.schema import validate_ohlc

FREE = BacktestParams(
    spread_pips=0.0,
    slippage_pips=0.0,
    exit_slippage_pips=0.0,
    risk_per_trade_pct=1.0,
    min_equity=10000.0,
)


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 0.0
    df.index.name = "timestamp"
    return validate_ohlc(df)


def _sig(
    ts: pd.Timestamp,
    direction: str,
    stop: float,
    target: float,
    sweep_extreme: float | None = None,
) -> pd.DataFrame:
    row: dict = {
        "timestamp": ts,
        "direction": direction,
        "entry_price": 0.0,  # reference only: engine fills at next open
        "stop_price": stop,
        "target_price": target,
    }
    if sweep_extreme is not None:
        row["sweep_extreme"] = sweep_extreme
    return pd.DataFrame([row])


def test_market_fill_at_next_open_with_exact_pnl() -> None:
    df = _frame(
        [
            (1.1000, 1.1010, 1.0990, 1.1000),  # signal bar
            (1.1000, 1.1120, 1.0990, 1.1110),  # entry 1.10, target 1.11 hit
            (1.1110, 1.1120, 1.1100, 1.1110),
        ]
    )
    res = run_backtest(df, _sig(df.index[0], "long", 1.0950, 1.1100), FREE)
    assert len(res.trades_df) == 1
    t = res.trades_df.iloc[0]
    assert t["entry_price"] == 1.1000
    assert t["exit_reason"] == "target"
    # risk 0.005 -> 20000 units; reward 0.01 -> pnl 200
    assert t["units"] == pytest.approx(20000.0)
    assert t["pnl"] == pytest.approx(200.0)


def test_stop_wins_same_bar_ambiguity() -> None:
    df = _frame(
        [
            (1.1000, 1.1010, 1.0990, 1.1000),
            (1.1000, 1.1150, 1.0900, 1.1120),  # touches both: stop first
            (1.1120, 1.1130, 1.1110, 1.1120),
        ]
    )
    res = run_backtest(df, _sig(df.index[0], "long", 1.0950, 1.1100), FREE)
    t = res.trades_df.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["pnl"] == pytest.approx(-100.0)  # -0.005 x 20000 units


def test_warning_exit_on_close_through_extreme() -> None:
    df = _frame(
        [
            (1.1000, 1.1010, 1.0980, 1.1005),  # signal bar
            (1.1005, 1.1020, 1.0995, 1.1010),  # entry, nothing hit
            (1.1010, 1.1015, 1.0996, 1.0985),  # close < sweep extreme 1.0990? no...
            (1.0985, 1.0990, 1.0970, 1.0975),  # close 1.0975 < 1.0980 extreme
        ]
    )
    sig = _sig(df.index[0], "long", 1.0940, 1.1100, sweep_extreme=1.0980)
    res = run_backtest(df, sig, FREE)
    t = res.trades_df.iloc[0]
    assert t["exit_reason"] == "warning"
    assert t["exit_ts"] == df.index[3]


def test_no_warning_without_sweep_extreme() -> None:
    df = _frame(
        [
            (1.1000, 1.1010, 1.0980, 1.1005),
            (1.1005, 1.1020, 1.0995, 1.1010),
            (1.1010, 1.1015, 1.0996, 1.0975),  # would warn, but no extreme carried
            (1.0975, 1.0980, 1.0970, 1.0978),
        ]
    )
    sig = _sig(df.index[0], "long", 1.0940, 1.1100)
    res = run_backtest(df, sig, FREE, max_hold_bars=120)
    assert res.trades_df.iloc[0]["exit_reason"] != "warning"


def test_costs_shift_fill_against_trader() -> None:
    pricey = BacktestParams(
        spread_pips=10.0,
        slippage_pips=0.0,
        pip_size=0.0001,
        risk_per_trade_pct=1.0,
        min_equity=10000.0,
    )
    df = _frame(
        [
            (1.1000, 1.1010, 1.0990, 1.1000),
            (1.1000, 1.1120, 1.0990, 1.1110),
            (1.1110, 1.1120, 1.1100, 1.1110),
        ]
    )
    res = run_backtest(df, _sig(df.index[0], "long", 1.0950, 1.1100), pricey)
    t = res.trades_df.iloc[0]
    assert t["entry_price"] == 1.1010  # open + 10-pip cost
    assert t["pnl"] < 200.0


def test_one_position_at_a_time() -> None:
    df = _frame(
        [
            (1.1000, 1.1010, 1.0990, 1.1000),
            (1.1000, 1.1010, 1.0990, 1.1000),
            (1.1000, 1.1120, 1.0990, 1.1110),
            (1.1110, 1.1120, 1.1100, 1.1110),
        ]
    )
    sigs = pd.concat(
        [_sig(df.index[0], "long", 1.0950, 1.1100), _sig(df.index[0], "short", 1.1050, 1.0900)]
    )
    res = run_backtest(df, sigs, FREE)
    assert len(res.trades_df) == 1
    assert res.trades_df.iloc[0]["direction"] == "long"


def _limit_sig(ts: pd.Timestamp, stop: float, target: float, expiry: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "direction": "long",
                "entry_price": 1.1000,
                "stop_price": stop,
                "target_price": target,
                "entry_mode": "limit",
                "expiry_bars": float(expiry),
            }
        ]
    )


def test_limit_fills_on_touch_and_expires_otherwise() -> None:
    df = _frame(
        [
            (1.1020, 1.1030, 1.1010, 1.1020),  # signal bar
            (1.1015, 1.1020, 1.1010, 1.1015),  # no touch of 1.10
            (1.1010, 1.1012, 1.0995, 1.1000),  # touches 1.10 -> fill
            (1.1000, 1.1120, 1.0995, 1.1110),  # target 1.11 hit
            (1.1110, 1.1120, 1.1100, 1.1110),
        ]
    )
    res = run_backtest(df, _limit_sig(df.index[0], 1.0950, 1.1100), FREE)
    assert len(res.trades_df) == 1
    t = res.trades_df.iloc[0]
    assert t["entry_price"] == 1.1000
    assert t["entry_ts"] == df.index[2]
    assert t["exit_reason"] == "target"

    df2 = _frame(
        [
            (1.1020, 1.1030, 1.1010, 1.1020),
            (1.1015, 1.1020, 1.1010, 1.1015),
            (1.1012, 1.1020, 1.1010, 1.1015),
            (1.1015, 1.1020, 1.1010, 1.1015),
        ]
    )
    res2 = run_backtest(df2, _limit_sig(df2.index[0], 1.0950, 1.1100, expiry=2), FREE)
    assert res2.trades_df.empty  # never touched within expiry: no trade


def test_exit_slippage_applies_adversely() -> None:
    pricey_exit = BacktestParams(
        spread_pips=0.0,
        slippage_pips=0.0,
        exit_slippage_pips=10.0,
        pip_size=0.0001,
        risk_per_trade_pct=1.0,
        min_equity=10000.0,
    )
    df = _frame(
        [
            (1.1000, 1.1010, 1.0990, 1.1000),
            (1.1000, 1.1120, 1.0990, 1.1110),
            (1.1110, 1.1120, 1.1100, 1.1110),
        ]
    )
    res = run_backtest(df, _sig(df.index[0], "long", 1.0950, 1.1100), pricey_exit)
    t = res.trades_df.iloc[0]
    assert t["exit_price"] == pytest.approx(1.1100 - 0.0010)  # target minus 10-pip slip
    assert t["pnl"] == pytest.approx((1.1090 - 1.1000) * t["units"])


def _hourly_frame(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 0.0},
        index=idx,
    )
    df.index.name = "timestamp"
    return validate_ohlc(df)


def test_session_forces_flat_at_eod() -> None:
    params = BacktestParams(
        spread_pips=0.0,
        slippage_pips=0.0,
        exit_slippage_pips=0.0,
        risk_per_trade_pct=1.0,
        min_equity=10000.0,
        max_hold_bars=120,
        session_enabled=True,
        entry_start_hour_utc=0,
        entry_end_hour_utc=20,
        force_flat_hour_utc=21,
    )
    df = _hourly_frame("2024-01-01 10:00+00:00", 15)  # 10:00 -> 00:00 next day
    sig = _sig(df.index[0], "long", 1.05, 1.20)  # never touched: must survive to EOD
    res = run_backtest(df, sig, params)
    assert len(res.trades_df) == 1
    t = res.trades_df.iloc[0]
    assert t["exit_reason"] == "eod"
    assert pd.Timestamp(t["exit_ts"]).hour == 21


def test_session_blocks_entries_in_illiquid_window() -> None:
    params = BacktestParams(
        spread_pips=0.0,
        slippage_pips=0.0,
        exit_slippage_pips=0.0,
        risk_per_trade_pct=1.0,
        min_equity=10000.0,
        session_enabled=True,
        entry_start_hour_utc=0,
        entry_end_hour_utc=20,
        force_flat_hour_utc=21,
    )
    df = _hourly_frame("2024-01-01 22:00+00:00", 4)  # 22,23,00,01: all illiquid
    sig = _sig(df.index[0], "long", 1.09, 1.20)
    res = run_backtest(df, sig, params)
    assert res.trades_df.empty  # market fill at 23:00 blocked


def test_session_no_exit_before_entry_bar() -> None:
    """Limit fill several bars out: pre-fill bars must not stop the trade."""
    params = BacktestParams(
        spread_pips=0.0,
        slippage_pips=0.0,
        exit_slippage_pips=0.0,
        risk_per_trade_pct=1.0,
        min_equity=10000.0,
        session_enabled=True,
        entry_start_hour_utc=0,
        entry_end_hour_utc=20,
        force_flat_hour_utc=21,
    )
    idx = pd.date_range("2024-01-03 10:00+00:00", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10],
            "high": [1.10, 1.25, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10],
            "low": [1.10, 1.10, 1.10, 1.10, 1.09, 1.10, 1.10, 1.10],
            "close": [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10],
            "volume": 0.0,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    df = validate_ohlc(df)
    sig = pd.DataFrame(
        [
            {
                "timestamp": idx[0],
                "direction": "long",
                "entry_price": 1.09,  # fills at 14:00 bar, not before
                "stop_price": 1.05,
                "target_price": 1.20,
                "entry_mode": "limit",
                "expiry_bars": 10.0,
            }
        ]
    )
    res = run_backtest(df, sig, params)
    assert len(res.trades_df) == 1
    t = res.trades_df.iloc[0]
    assert t["entry_ts"] == idx[4]  # first touch of 1.09
    assert pd.Timestamp(t["exit_ts"]) >= pd.Timestamp(t["entry_ts"])


def test_session_friday_flat_before_weekend() -> None:
    params = BacktestParams(
        spread_pips=0.0,
        slippage_pips=0.0,
        exit_slippage_pips=0.0,
        risk_per_trade_pct=1.0,
        min_equity=10000.0,
        max_hold_bars=120,
        session_enabled=True,
        entry_start_hour_utc=0,
        entry_end_hour_utc=20,
        force_flat_hour_utc=21,
    )
    df = _hourly_frame("2024-01-05 10:00+00:00", 15)  # Friday 10:00 -> Saturday
    sig = _sig(df.index[0], "long", 1.05, 1.20)
    res = run_backtest(df, sig, params)
    assert len(res.trades_df) == 1
    t = res.trades_df.iloc[0]
    assert t["exit_reason"] == "eod"
    assert pd.Timestamp(t["exit_ts"]).weekday() == 4  # Friday, never weekend


def test_session_winner_runs_loser_cut_overnight() -> None:
    def _params() -> BacktestParams:
        return BacktestParams(
            spread_pips=0.0,
            slippage_pips=0.0,
            exit_slippage_pips=0.0,
            risk_per_trade_pct=1.0,
            min_equity=10000.0,
            max_hold_bars=120,
            session_enabled=True,
            entry_start_hour_utc=0,
            entry_end_hour_utc=20,
            force_flat_hour_utc=21,
            hold_winners_overnight=True,
        )

    # Wednesday 10:00 -> Thursday 02:00; 21:00 flat bar decides.
    idx = pd.date_range("2024-01-03 10:00+00:00", periods=17, freq="h", tz="UTC")
    base = {"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.10, "volume": 0.0}
    # winner: 21:00 open well above entry
    wdf = pd.DataFrame(base, index=idx)
    wdf.loc[idx[0], ["open", "high", "low", "close"]] = [1.10, 1.11, 1.09, 1.10]
    wdf.loc[idx[1], ["open"]] = 1.10
    wdf.loc[idx[11], ["open"]] = 1.15  # 21:00 bar open: winner runs
    wdf.index.name = "timestamp"
    wdf = validate_ohlc(wdf)
    res = run_backtest(wdf, _sig(idx[0], "long", 1.05, 1.50), _params())
    assert pd.Timestamp(res.trades_df.iloc[0]["exit_ts"]) > idx[11]

    # loser: 21:00 open below entry -> cut
    ldf = pd.DataFrame(base, index=idx)
    ldf.loc[idx[11], ["open"]] = 1.08
    ldf.index.name = "timestamp"
    ldf = validate_ohlc(ldf)
    res = run_backtest(ldf, _sig(idx[0], "long", 1.05, 1.50), _params())
    t = res.trades_df.iloc[0]
    assert t["exit_reason"] == "eod"
    assert t["exit_ts"] == idx[11]
