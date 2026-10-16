"""Strategy variant tests: MTF resample/retest wiring, breakout, fade."""

from dataclasses import replace

import pandas as pd

from retest.data.schema import validate_ohlc
from retest.signals.retest import RetestParams, generate_retest_signals
from retest.strategies.breakout import generate_breakout_signals
from retest.strategies.fade import generate_fade_signals
from retest.strategies.mtf import generate_mtf_retest, resample_ohlc

LB = 2
DAY = pd.Timedelta(days=1)


def _daily(peak: float = 1.15, trough: float = 1.10) -> pd.DataFrame:
    idx = pd.date_range("2023-12-01", periods=15, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.14, "high": 1.142, "low": 1.138, "close": 1.14, "volume": 0.0}, index=idx
    )
    df.index.name = "timestamp"
    df.loc[idx[7], "high"] = peak  # swing high on its own bar
    df.loc[idx[6], "high"] = 1.141
    df.loc[idx[8], "high"] = 1.141
    df.loc[idx[10], "low"] = trough  # swing low on its own bar
    df.loc[idx[9], "low"] = 1.139
    df.loc[idx[11], "low"] = 1.139
    return validate_ohlc(df)


def _hourly(n: int = 24 * 5, special: dict[int, tuple] | None = None) -> pd.DataFrame:
    """Flat 1.14 H1 bars (ATR ~= 0.003); overrides via ``special``."""
    idx = pd.date_range("2024-01-03", periods=n, freq="h", tz="UTC")
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
        atr_period=3,
        touch_tolerance_atr=0.3,
        min_stop_atr=0.0,
        min_reward_risk=0.0,
        fta_fallback_atr=2.0,
        stop_buffer_atr=0.25,
        **kw,
    )


def test_resample_ohlc_aggregates() -> None:
    idx = pd.date_range("2024-01-03", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "high": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
            "low": [0.95, 1.05, 1.15, 1.25, 1.35, 1.45],
            "close": [1.02, 1.12, 1.22, 1.32, 1.42, 1.52],
            "volume": 0.0,
        },
        index=idx,
    )
    out = resample_ohlc(df, "2h")
    assert len(out) == 3
    assert out.iloc[0]["open"] == 1.0 and out.iloc[0]["close"] == 1.12
    assert out.iloc[0]["high"] == 1.15 and out.iloc[0]["low"] == 0.95
    assert out.iloc[2]["close"] == 1.52


def test_mtf_retest_wiring_matches_core() -> None:
    h, d = _hourly(), _daily()
    p = _params()
    a = generate_mtf_retest(h, d, p, DAY)
    b = generate_retest_signals(h, d, p)
    pd.testing.assert_frame_equal(a, b)


def test_breakout_emits_on_fresh_close_beyond() -> None:
    d = _daily()
    h = _hourly(special={50: (1.148, 1.154, 1.147, 1.152)})
    sig = generate_breakout_signals(h, d, _params(), DAY)
    assert len(sig) == 1
    row = sig.iloc[0]
    assert row["direction"] == "long"
    assert row["entry_mode"] == "market"
    assert row["entry_price"] == 1.152
    assert row["stop_price"] < 1.147  # beyond break-bar low
    assert row["target_price"] > row["entry_price"]


def test_breakout_no_signal_when_never_fresh() -> None:
    # level already broken before H1 data starts: price sits above, no fresh cross
    d = _daily()
    idx = pd.date_range("2024-01-03", periods=60, freq="h", tz="UTC")
    h = pd.DataFrame(
        {"open": 1.16, "high": 1.162, "low": 1.158, "close": 1.16, "volume": 0.0}, index=idx
    )
    h.index.name = "timestamp"
    sig = generate_breakout_signals(validate_ohlc(h), d, _params(), DAY)
    assert sig.empty


def test_breakout_session_gate() -> None:
    d = _daily()
    h = _hourly(special={50: (1.148, 1.154, 1.147, 1.152)})  # 2024-01-05 02:00 UTC
    p = _params(session_enabled=True, entry_start_hour_utc=3, entry_end_hour_utc=20)
    assert generate_breakout_signals(h, d, p, DAY).empty
    p2 = _params(session_enabled=True, entry_start_hour_utc=2, entry_end_hour_utc=20)
    assert len(generate_breakout_signals(h, d, p2, DAY)) == 1


def test_fade_emits_limit_at_support_touch() -> None:
    d = _daily()
    h = _hourly(special={40: (1.105, 1.106, 1.098, 1.104)})  # touch 1.10, close above
    sig = generate_fade_signals(h, d, _params(), DAY)
    longs = sig[sig["direction"] == "long"]
    assert len(longs) == 1
    row = longs.iloc[0]
    assert row["entry_mode"] == "limit"
    assert abs(row["entry_price"] - 1.10) < 1e-9
    assert row["stop_price"] < 1.098


def test_fade_skips_broken_level() -> None:
    d = _daily()
    h = _hourly(special={40: (1.105, 1.106, 1.098, 1.096)})  # closes beyond: break, not touch
    sig = generate_fade_signals(h, d, _params(), DAY)
    assert sig.empty
