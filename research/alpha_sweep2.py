"""Alpha sweep v2: structural variants (MTF retest, breakout continuation).

Usage:
  uv run python alpha_sweep2.py smoke    # trade counts + full-sample for every variant
  uv run python alpha_sweep2.py sweep    # batched knob sweep over PLAYBOOK configs

Results append to alpha_sweep2_results.csv.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from retest.backtest.engine import BacktestParams, run_backtest
from retest.config import get_config
from retest.data.loader import load_ohlc
from retest.data.schema import empty_ohlc
from retest.portfolio import (
    apply_group_filter,
    apply_usd_cap,
    combine_trades,
    portfolio_equity,
)
from retest.report.metrics_table import metrics_table
from retest.signals.retest import RetestParams, _generate
from retest.strategies.breakout import generate_breakout_signals
from retest.strategies.mtf import resample_ohlc
from retest.validation.significance import significance_report

from alpha_sweep import _summarize  # reuse summary plumbing

START_EQUITY = 10000.0
RESULTS_CSV = str(Path(__file__).resolve().parent / "results" / "alpha_sweep2_results.csv")

_D = pd.Timedelta(days=1)
_H1 = pd.Timedelta(hours=1)
_H2 = pd.Timedelta(hours=2)
_H4 = pd.Timedelta(hours=4)

FRAMES: dict = {}


def _no_fetch(*_a, **_k):
    return empty_ohlc()


def universe() -> list[str]:
    override = os.environ.get("SWEEP_UNIVERSE")
    if override:
        return [p for p in override.split(",") if p]
    return list(get_config().get("data.tradingview.universe"))


def _init_worker(pairs: list[str]) -> None:
    global FRAMES
    for p in pairs:
        h1 = load_ohlc(p, granularity="H1", fetcher=_no_fetch)
        FRAMES[p] = {
            "h1": h1,
            "d": load_ohlc(p, fetcher=_no_fetch),
            "h2": resample_ohlc(h1, "2h"),
            "h4": resample_ohlc(h1, "4h"),
        }


def _signals(strat: str, fr: dict, rp: RetestParams) -> pd.DataFrame:
    if strat == "retest_d":
        return _generate(fr["h1"], fr["d"], rp, _D)
    if strat == "retest_h4":
        return _generate(fr["h1"], fr["h4"], rp, _H4)
    if strat == "retest_h2":
        return _generate(fr["h1"], fr["h2"], rp, _H2)
    if strat == "retest_h1":
        return _generate(fr["h1"], fr["h1"], rp, _H1)
    if strat == "breakout_d":
        return generate_breakout_signals(fr["h1"], fr["d"], rp, _D)
    if strat == "breakout_h4":
        return generate_breakout_signals(fr["h1"], fr["h4"], rp, _H4)
    if strat == "breakout_h2":
        return generate_breakout_signals(fr["h1"], fr["h2"], rp, _H2)
    if strat == "breakout_h1":
        return generate_breakout_signals(fr["h1"], fr["h1"], rp, _H1)
    if strat == "fade_d":
        from retest.strategies.fade import generate_fade_signals

        return generate_fade_signals(fr["h1"], fr["d"], rp, _D)
    raise ValueError(strat)


def run_configs(pool, configs: list[tuple[str, str, dict, dict]]) -> list[dict]:
    """configs: list of (label, strat, r_over, bt_over)."""
    pairs = universe()
    tasks = []
    for label, strat, r, b in configs:
        tasks += [(label, strat, r, b, p) for p in pairs]
    collected = {label: {"per": {}, "sigs": 0} for label, _, _, _ in configs}
    t0 = time.time()
    for lbl, strat, r, b, pair, trades, ns in pool.map(_task2, tasks):
        collected[lbl]["per"][pair] = trades
        collected[lbl]["sigs"] += ns
    rows = []
    for label, strat, r, b in configs:
        row, _trades = _summarize(label, collected[label]["per"], r, b, collected[label]["sigs"])
        rows.append(row)
    return rows


def _task2(args):
    label, strat, r_over, bt_over, pair = args
    fr = FRAMES[pair]
    rp = replace(RetestParams.from_config(), **r_over)
    bp = replace(BacktestParams.from_config(pair), **bt_over)
    sig = _signals(strat, fr, rp)
    res = run_backtest(fr["h1"], sig, bp)
    t = res.trades_df.copy()
    t["symbol"] = pair
    return label, strat, r_over, bt_over, pair, t, len(sig)


def smoke_configs() -> list[tuple[str, str, dict, dict]]:
    strats = [
        "retest_d",
        "retest_h4",
        "retest_h2",
        "retest_h1",
        "breakout_d",
        "breakout_h4",
        "breakout_h2",
        "breakout_h1",
    ]
    return [(f"{s} | smoke", s, {}, {}) for s in strats]


def sweep_configs() -> list[tuple[str, str, dict, dict]]:
    s = "breakout_d"
    cfgs: list[tuple[str, str, dict, dict]] = [
        ("c1 cluster10+fta30", {"cluster_ticks": 10, "fta_fallback_atr": 3.0}),
        ("c2 +rr15", {"cluster_ticks": 10, "fta_fallback_atr": 3.0, "min_reward_risk": 1.5}),
        ("c3 +tol015", {"cluster_ticks": 10, "fta_fallback_atr": 3.0, "touch_tolerance_atr": 0.15}),
        ("c4 +start7", {"cluster_ticks": 10, "fta_fallback_atr": 3.0, "entry_start_hour_utc": 7}),
        ("c5 fta30+rr15", {"fta_fallback_atr": 3.0, "min_reward_risk": 1.5}),
        ("c6 rr15+tol015", {"min_reward_risk": 1.5, "touch_tolerance_atr": 0.15}),
        (
            "c7 all5",
            {
                "cluster_ticks": 10,
                "fta_fallback_atr": 3.0,
                "min_reward_risk": 1.5,
                "touch_tolerance_atr": 0.15,
                "entry_start_hour_utc": 7,
            },
        ),
        (
            "c8 c1+rr15+start7",
            {
                "cluster_ticks": 10,
                "fta_fallback_atr": 3.0,
                "min_reward_risk": 1.5,
                "entry_start_hour_utc": 7,
            },
        ),
        (
            "c9 buf05+c1",
            {"cluster_ticks": 10, "fta_fallback_atr": 3.0, "stop_buffer_atr": 0.5},
        ),
        (
            "c10 buf05+c2",
            {
                "cluster_ticks": 10,
                "fta_fallback_atr": 3.0,
                "min_reward_risk": 1.5,
                "stop_buffer_atr": 0.5,
            },
        ),
        (
            "c11 c1+minswing075",
            {"cluster_ticks": 10, "fta_fallback_atr": 3.0, "min_swing_magnitude_atr": 0.75},
        ),
        (
            "c12 c2+lb5",
            {
                "cluster_ticks": 10,
                "fta_fallback_atr": 3.0,
                "min_reward_risk": 1.5,
                "swing_lookback": 5,
            },
        ),
    ]
    return [(f"{s} | {c[0]}", s, c[1], c[2] if len(c) > 2 else {}) for c in cfgs]


def h1idx_configs() -> list[tuple[str, str, dict, dict]]:
    """One-knob + combo sweep for breakout_h1 on the index universe."""
    s = "breakout_h1"
    cfgs: list[tuple[str, dict]] = [
        ("base", {}),
        ("cluster10", {"cluster_ticks": 10}),
        ("fta20", {"fta_fallback_atr": 2.0}),
        ("fta30", {"fta_fallback_atr": 3.0}),
        ("rr15", {"min_reward_risk": 1.5}),
        ("minswing075", {"min_swing_magnitude_atr": 0.75}),
        ("tol015", {"touch_tolerance_atr": 0.15}),
        ("buf05", {"stop_buffer_atr": 0.5}),
        ("c1 cl10+fta30", {"cluster_ticks": 10, "fta_fallback_atr": 3.0}),
        (
            "c2 c1+rr15",
            {"cluster_ticks": 10, "fta_fallback_atr": 3.0, "min_reward_risk": 1.5},
        ),
        (
            "c3 c1+minswing075",
            {"cluster_ticks": 10, "fta_fallback_atr": 3.0, "min_swing_magnitude_atr": 0.75},
        ),
    ]
    return [(f"{s} | {lbl}", s, r, {}) for lbl, r in cfgs]


def fade_configs() -> list[tuple[str, str, dict, dict]]:
    s = "fade_d"
    cfgs: list[tuple[str, str, dict, dict]] = [
        ("base", s, {}, {}),
        ("eps2", s, {"max_returns_per_level": 2}, {}),
        ("eps3", s, {"max_returns_per_level": 3}, {}),
        ("start7", s, {"entry_start_hour_utc": 7}, {}),
        ("fta15", s, {"fta_fallback_atr": 1.5}, {}),
        ("fta30", s, {"fta_fallback_atr": 3.0}, {}),
        ("rr075", s, {"min_reward_risk": 0.75}, {}),
        ("rr15", s, {"min_reward_risk": 1.5}, {}),
        ("minswing05", s, {"min_swing_magnitude_atr": 0.5}, {}),
        ("minswing075", s, {"min_swing_magnitude_atr": 0.75}, {}),
        ("cluster3", s, {"cluster_ticks": 3}, {}),
        ("cluster10", s, {"cluster_ticks": 10}, {}),
        ("lb5", s, {"swing_lookback": 5}, {}),
        ("lb6", s, {"swing_lookback": 6}, {}),
        ("minstop025", s, {"min_stop_atr": 0.25}, {}),
        ("minstop05", s, {"min_stop_atr": 0.5}, {}),
        ("tol015", s, {"touch_tolerance_atr": 0.15}, {}),
        ("age0", s, {"max_level_age_days": 0}, {}),
        ("buf01", s, {"stop_buffer_atr": 0.1}, {}),
        ("buf05", s, {"stop_buffer_atr": 0.5}, {}),
        ("hold60", s, {}, {"max_hold_bars": 60}),
        ("hold240", s, {}, {"max_hold_bars": 240}),
    ]
    return [(f"{s} | {lbl}", st, r, b) for lbl, st, r, b in cfgs]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    workers = min(10, (os.cpu_count() or 4))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(universe(),)
    ) as pool:
        if mode == "smoke":
            rows = run_configs(pool, smoke_configs())
        elif mode == "sweep":
            rows = run_configs(pool, sweep_configs())
        elif mode == "fade":
            rows = run_configs(pool, fade_configs())
        elif mode == "h1idx":
            rows = run_configs(pool, h1idx_configs())
        else:
            print(f"unknown mode: {mode}")
            sys.exit(1)
    df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("r_over", "bt_over")} for r in rows]
    )
    print("\n=== summary ===")
    print(df.sort_values("pnl", ascending=False).to_string(index=False))
    _append_2(rows)


def _append_2(rows: list[dict]) -> None:
    flat = [
        {
            **{k: v for k, v in r.items() if k not in ("r_over", "bt_over")},
            "r_over": str(r["r_over"]),
            "bt_over": str(r["bt_over"]),
            "universe": os.environ.get("SWEEP_UNIVERSE", "fx"),
        }
        for r in rows
    ]
    df = pd.DataFrame(flat)
    try:
        old = pd.read_csv(RESULTS_CSV)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(RESULTS_CSV, index=False)


if __name__ == "__main__":
    main()
