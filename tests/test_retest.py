"""Retest tests: break-then-return grading, confirmations, expiry."""

from dataclasses import replace

import pandas as pd

from retest.data.schema import validate_ohlc
from retest.signals.retest import (
    RetestParams,
    daily_confirmations,
    engulfing,
    fib_grid,
    generate_retest_signals,
    pin_bar,
)

LB = 2


def _daily(peak_high: float = 1.15, pin: bool = False) -> pd.DataFrame:
    idx = pd.date_range("2023-12-01", periods=15, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.10, "high": 1.12, "low": 1.10, "close": 1.10, "volume": 0.0}, index=idx
    )
    df.index.name = "timestamp"
    df.loc[idx[7], ["open", "high", "low", "close"]] = [1.12, peak_high, 1.12, 1.12]
    df.loc[idx[6], ["high"]] = 1.121
    df.loc[idx[8], ["high"]] = 1.121
    if pin:
        df.loc[idx[14], ["open", "high", "low", "close"]] = [1.112, 1.116, 1.09, 1.115]
    return validate_ohlc(df)


def _hourly(n: int = 100, special: dict[int, tuple] | None = None) -> pd.DataFrame:
    """Flat 1.14 H1 bars (ATR ~= 0.004, tolerance 0.0004 vs level 1.15)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.14, "high": 1.1420, "low": 1.1380, "close": 1.14, "volume": 0.0},
        index=idx,
    )
    df.index.name = "timestamp"
    for i, (o, h, lo, c) in (special or {}).items():
        df.iloc[i] = [o, h, lo, c, 0.0]
    return validate_ohlc(df)


def _params(**kw) -> RetestParams:
    return replace(
        RetestParams(),
        swing_lookback=LB,
        tick_size=0.001,
        min_swing_magnitude_atr=0.0,
        atr_period=3,  # short frame: keep ATR warmed up
        atr_stretch_period=3,
        **kw,
    )


def test_pin_and_engulfing() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.10, 1.10, 1.112],
            "high": [1.11, 1.11, 1.116],
            "low": [1.09, 1.09, 1.09],
            "close": [1.095, 1.095, 1.115],
            "volume": 0.0,
        },
        index=idx,
    )
    assert pin_bar(df, 2, "long")
    assert not pin_bar(df, 2, "short")
    assert not pin_bar(df, 0, "long")
    eng = pd.DataFrame(
        {
            "open": [1.10, 1.09],
            "high": [1.11, 1.12],
            "low": [1.09, 1.085],
            "close": [1.09, 1.11],
            "volume": 0.0,
        },
        index=idx[:2],
    )
    assert engulfing(eng, 1, "long")
    assert not engulfing(eng, 1, "short")


def test_fib_grid_marks_deep_legs() -> None:
    grid = fib_grid(_daily(), 14, 20, (0.5,))
    assert grid == [1.15 - 0.5 * (1.15 - 1.10)]


def test_three_touch_break_return_trades_clean() -> None:
    hourly = _hourly(
        special={
            10: (1.14, 1.1498, 1.1380, 1.1480),  # touch episode 1
            11: (1.1480, 1.1498, 1.1380, 1.1480),
            30: (1.14, 1.1498, 1.1380, 1.1480),  # touch episode 2
            50: (1.14, 1.1510, 1.1380, 1.1505),  # break up
            55: (1.1490, 1.1510, 1.1490, 1.1500),  # return within window
        }
    )
    sig = generate_retest_signals(hourly, _daily(), _params())
    assert len(sig) == 1
    row = sig.iloc[0]
    assert row["direction"] == "long"
    assert row["entry_price"] == 1.15
    assert row["entry_mode"] == "limit"
    assert row["touches"] == 3
    assert row["timestamp"] == hourly.index[55]


def test_two_touch_needs_confirmation() -> None:
    hourly = _hourly(
        special={
            10: (1.14, 1.1498, 1.1380, 1.1480),
            50: (1.14, 1.1510, 1.1380, 1.1505),
            55: (1.1490, 1.1510, 1.1490, 1.1500),
        }
    )
    # stretch gate disabled to isolate the daily-pattern confirmation
    assert generate_retest_signals(
        hourly, _daily(), _params(atr_stretch_multiple=1e9)
    ).empty  # no pin on daily
    sig = generate_retest_signals(hourly, _daily(pin=True), _params(atr_stretch_multiple=1e9))
    assert len(sig) == 1
    assert sig.iloc[0]["touches"] == 2
    assert "d1_pattern" in sig.iloc[0]["confirmations"]


def test_late_return_expires() -> None:
    hourly = _hourly(
        n=120,
        special={
            10: (1.14, 1.1498, 1.1380, 1.1480),
            30: (1.14, 1.1498, 1.1380, 1.1480),
            50: (1.14, 1.1510, 1.1380, 1.1505),
            90: (1.1490, 1.1510, 1.1490, 1.1500),  # 40 bars after break > 24
        },
    )
    assert generate_retest_signals(hourly, _daily(), _params()).empty


def test_no_break_no_signal() -> None:
    assert generate_retest_signals(_hourly(), _daily(), _params()).empty


def _flat_daily(n: int = 40, start: str = "2023-12-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.10, "high": 1.12, "low": 1.10, "close": 1.10, "volume": 0.0}, index=idx
    )
    df.index.name = "timestamp"
    return df


def test_level_not_usable_before_confirmation_bar_closes() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    daily = _flat_daily(30, "2024-01-01")
    daily.loc[idx[5], ["open", "high", "low", "close"]] = [1.12, 1.15, 1.12, 1.12]
    daily.loc[idx[4], ["high"]] = 1.121
    daily.loc[idx[6], ["high"]] = 1.121
    daily = validate_ohlc(daily)
    # swing on idx[5] is confirmed by idx[7], whose session closes 2024-01-09 00:00

    def hourly(touches: list[str], break_ts: str) -> pd.DataFrame:
        hidx = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")
        h = pd.DataFrame(
            {"open": 1.14, "high": 1.1420, "low": 1.1380, "close": 1.14, "volume": 0.0},
            index=hidx,
        )
        h.index.name = "timestamp"
        for ts in touches:
            h.loc[pd.Timestamp(ts, tz="UTC"), ["open", "high", "low", "close"]] = [
                1.14,
                1.1498,
                1.1380,
                1.1480,
            ]
        h.loc[pd.Timestamp(break_ts, tz="UTC"), ["open", "high", "low", "close"]] = [
            1.1490,
            1.1510,
            1.1480,
            1.1510,
        ]
        b = hidx.get_loc(pd.Timestamp(break_ts, tz="UTC"))
        h.iloc[b + 2] = [1.1495, 1.1510, 1.1460, 1.1500, 0.0]  # return
        return validate_ohlc(h)

    # break inside the last confirmation bar: the swing is not knowable yet
    assert generate_retest_signals(
        hourly(["2024-01-08 02:00", "2024-01-08 06:00"], "2024-01-08 12:00"), daily, _params()
    ).empty
    # same shape one day later: level available, signal fires
    sig = generate_retest_signals(
        hourly(["2024-01-09 02:00", "2024-01-09 06:00"], "2024-01-09 12:00"), daily, _params()
    )
    assert len(sig) == 1


def test_confirmations_ignore_forming_daily_bar() -> None:
    daily = _daily(pin=True)  # pin on Dec 15; session closes Dec 16 00:00
    params = _params(atr_stretch_multiple=1e9, fib_proximity_atr=0.0)
    during = pd.Timestamp("2023-12-15 12:00", tz="UTC")
    after = pd.Timestamp("2023-12-16 12:00", tz="UTC")
    assert "d1_pattern" not in daily_confirmations(daily, 1.15, "long", during, params)
    assert "d1_pattern" in daily_confirmations(daily, 1.15, "long", after, params)


def test_first_return_only_no_retry() -> None:
    idx = pd.date_range("2023-12-01", periods=40, freq="D", tz="UTC")
    daily = _flat_daily()
    daily.loc[idx[5], ["open", "high", "low", "close"]] = [1.12, 1.15, 1.12, 1.12]
    daily.loc[idx[4], ["high"]] = 1.121
    daily.loc[idx[6], ["high"]] = 1.121
    daily.loc[idx[32], ["open", "high", "low", "close"]] = [1.112, 1.116, 1.09, 1.115]  # pin
    daily = validate_ohlc(daily)

    hidx = pd.date_range("2023-12-20", periods=400, freq="h", tz="UTC")
    h = pd.DataFrame(
        {"open": 1.14, "high": 1.1420, "low": 1.1380, "close": 1.14, "volume": 0.0}, index=hidx
    )
    h.index.name = "timestamp"
    h.loc[pd.Timestamp("2023-12-25 02:00", tz="UTC"), ["open", "high", "low", "close"]] = [
        1.14,
        1.1498,
        1.1380,
        1.1480,
    ]  # 1 touch episode -> 1 confirmation required
    h.loc[pd.Timestamp("2024-01-02 10:00", tz="UTC"), ["open", "high", "low", "close"]] = [
        1.1490,
        1.1510,
        1.1480,
        1.1510,
    ]  # break
    h.loc[pd.Timestamp("2024-01-02 12:00", tz="UTC"), ["open", "high", "low", "close"]] = [
        1.1495,
        1.1510,
        1.1460,
        1.1500,
    ]  # return #1 (pin bar still forming -> no confirmation)
    h.loc[pd.Timestamp("2024-01-02 13:00", tz="UTC"), ["open", "high", "low", "close"]] = [
        1.1515,
        1.1520,
        1.1510,
        1.1515,
    ]  # episode break
    h.loc[pd.Timestamp("2024-01-03 10:00", tz="UTC"), ["open", "high", "low", "close"]] = [
        1.1495,
        1.1510,
        1.1460,
        1.1500,
    ]  # return #2 (pin now completed -- but stale: first return only)
    hourly = validate_ohlc(h)
    assert generate_retest_signals(hourly, daily, _params(atr_stretch_multiple=1e9)).empty


def test_level_price_ignores_swings_not_yet_confirmed() -> None:
    idx = pd.date_range("2023-12-01", periods=40, freq="D", tz="UTC")
    daily = _flat_daily()
    daily.loc[idx[5], ["open", "high", "low", "close"]] = [1.12, 1.1500, 1.12, 1.12]
    daily.loc[idx[4], ["high"]] = 1.121
    daily.loc[idx[6], ["high"]] = 1.121
    daily.loc[idx[20], ["open", "high", "low", "close"]] = [1.12, 1.1520, 1.12, 1.12]
    daily.loc[idx[19], ["high"]] = 1.121
    daily.loc[idx[21], ["high"]] = 1.121
    daily = validate_ohlc(daily)
    # second swing joins the cluster only on 2023-12-24; the trade is on 12-15

    hidx = pd.date_range("2023-12-09", periods=300, freq="h", tz="UTC")
    h = pd.DataFrame(
        {"open": 1.14, "high": 1.1420, "low": 1.1380, "close": 1.14, "volume": 0.0}, index=hidx
    )
    h.index.name = "timestamp"
    for ts in ("2023-12-10 02:00", "2023-12-10 06:00"):
        h.loc[pd.Timestamp(ts, tz="UTC"), ["open", "high", "low", "close"]] = [
            1.14,
            1.1498,
            1.1380,
            1.1480,
        ]
    h.loc[pd.Timestamp("2023-12-15 12:00", tz="UTC"), ["open", "high", "low", "close"]] = [
        1.1490,
        1.1510,
        1.1480,
        1.1510,
    ]
    h.loc[pd.Timestamp("2023-12-15 14:00", tz="UTC"), ["open", "high", "low", "close"]] = [
        1.1495,
        1.1510,
        1.1460,
        1.1500,
    ]
    hourly = validate_ohlc(h)
    sig = generate_retest_signals(hourly, daily, _params())
    assert len(sig) == 1
    assert sig.iloc[0]["entry_price"] == 1.1500
    assert sig.iloc[0]["level_price"] == 1.1500
