"""Daily-history long-run test: 2002-2026 daily bars, breakout + retest families.

Every verdict so far was judged on ~1.8y of H1 (~4 OOS quarters). This harness
judges the same structural families on the full daily history (FX/metals/oil
to 2002, indices to 2003): one signal pass per symbol over the whole frame -
safe, the level-availability guard is purely backward-looking - one backtest
pass, then trades are sliced by year/quarter for the regime view.

Usage:
  uv run python sweep_daily.py quick   # 2 symbols, smoke
  uv run python sweep_daily.py         # full universe

No tuning in this run: config defaults (plus session_enabled=False, since the
H1 session rules do not apply to 24h-stamped daily bars).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

from retest.backtest.engine import BacktestParams, run_backtest
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
from retest.validation.random_baseline import baseline_passes, run_random_baseline
from retest.validation.significance import significance_report

START_EQUITY = 10000.0
START_DATE = "2002-05-01"
_DAY = pd.Timedelta(days=1)

INDICES = ["SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR", "UK100_GBP", "JP225_USD"]
FX_BOOK = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
    "XAU_USD",
    "XAG_USD",
    "WTI_USD",
    "BTC_USD",
]
DAILY_UNIVERSE = FX_BOOK + INDICES

FRAMES: dict = {}


def _no_fetch(*_a, **_k):
    return empty_ohlc()


def universe() -> list[str]:
    override = os.environ.get("SWEEP_UNIVERSE")
    if override:
        return [p for p in override.split(",") if p]
    return DAILY_UNIVERSE


def _init_worker(pairs: list[str]) -> None:
    global FRAMES
    for p in pairs:
        d = load_ohlc(p, fetcher=_no_fetch)
        FRAMES[p] = d[d.index >= pd.Timestamp(START_DATE, tz=d.index.tz)]


def _signals(strat: str, d: pd.DataFrame, rp: RetestParams) -> pd.DataFrame:
    if strat == "breakout_daily":
        return generate_breakout_signals(d, d, rp, _DAY)
    if strat == "retest_daily":
        return _generate(d, d, rp, _DAY)
    raise ValueError(strat)


def _task(args):
    strat, r_over, bt_over, pair = args
    d = FRAMES[pair]
    rp = replace(RetestParams.from_config(), session_enabled=False, **r_over)
    bp = replace(BacktestParams.from_config(pair), session_enabled=False, **bt_over)
    sig = _signals(strat, d, rp)
    res = run_backtest(d, sig, bp)
    t = res.trades_df.copy()
    t["symbol"] = pair
    return pair, t, len(sig)


def _rb_task(args):
    strat, r_over, bt_over, n_sig, pair = args
    d = FRAMES[pair]
    bp = replace(BacktestParams.from_config(pair), session_enabled=False, **bt_over)
    _ = strat, r_over  # gate mirrors this config's signal count on the same frame
    pnls = run_random_baseline(d, max(n_sig, 1), bp, 20, 0, max_hold_bars=bp.max_hold_bars)
    passed, stats = baseline_passes(pnls)
    return pair, n_sig, stats, passed


def _filtered(trades: pd.DataFrame) -> pd.DataFrame:
    t = apply_group_filter(trades) if not trades.empty else trades
    return apply_usd_cap(t) if not t.empty else t


def _subset_metrics(trades: pd.DataFrame, label: str) -> None:
    t = _filtered(trades)
    if t.empty:
        print(f"{label:<28} no trades", flush=True)
        return
    eq = portfolio_equity(t, START_EQUITY)
    m = metrics_table(t, eq, START_EQUITY).set_index("metric")["value"].to_dict()
    ret = eq.resample("D").last().ffill().pct_change().dropna()
    rep = significance_report(ret)
    print(
        f"{label:<28} n={int(m.get('n_trades', 0)):>4} pnl={m.get('total_pnl', 0.0):>9.0f} "
        f"pf={m.get('profit_factor', float('nan')):>5.2f} wr={m.get('win_rate', float('nan')):.2f} "
        f"mdd={m.get('max_drawdown', float('nan')):.3f} sharpe={rep['sharpe']:>6.2f} "
        f"[{rep['ci_low']:>6.2f}]",
        flush=True,
    )


def _yearly(trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    t = trades.assign(year=trades["entry_ts"].dt.year, win=trades["pnl"] > 0)
    g = t.groupby("year").agg(n=("pnl", "size"), pnl=("pnl", "sum"), wr=("win", "mean"))
    g["pnl"] = g["pnl"].round(0)
    g["wr"] = g["wr"].round(2)
    print(g.to_string(), flush=True)
    pos = int((g["pnl"] > 0).sum())
    print(f"positive years: {pos}/{len(g)}", flush=True)


def _quarterly(trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    t = trades.copy()
    t["q"] = t["entry_ts"].dt.tz_localize(None).dt.to_period("Q")
    g = t.groupby("q")["pnl"].sum()
    pos = int((g > 0).sum())
    print(
        f"positive quarters: {pos}/{len(g)} ({100.0 * pos / len(g):.0f}%), "
        f"worst={g.min():.0f}, best={g.max():.0f}, median={g.median():.0f}",
        flush=True,
    )
    print("worst 5 quarters:", ", ".join(f"{k}:{v:.0f}" for k, v in g.nsmallest(5).items()), flush=True)


def _configs() -> list[tuple[str, str, dict, dict]]:
    return [
        ("breakout_daily | base", "breakout_daily", {}, {}),
        (
            "breakout_daily | c1 fx-knobs",
            "breakout_daily",
            {"cluster_ticks": 10, "fta_fallback_atr": 3.0},
            {},
        ),
        ("breakout_daily | hold30", "breakout_daily", {}, {"max_hold_bars": 30}),
        ("retest_daily | base", "retest_daily", {}, {}),
    ]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    pairs = ["EUR_USD", "SPX500_USD"] if mode == "quick" else universe()
    configs = _configs()
    if mode == "quick":
        configs = configs[:2]
    elif mode == "costs":
        # robustness: heavy costs (~3x spread, 2x slippage everywhere)
        configs = [
            (
                "breakout_daily | c1 heavycosts",
                "breakout_daily",
                {"cluster_ticks": 10, "fta_fallback_atr": 3.0},
                {"spread_pips": 3.0, "slippage_pips": 1.0, "exit_slippage_pips": 2.0},
            )
        ]
    print(f"universe: {len(pairs)} symbols | {pairs[0]}.. | window {START_DATE} -> data end", flush=True)
    workers = min(10, (os.cpu_count() or 4))
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(pairs,)
    ) as pool:
        for label, strat, r_over, bt_over in configs:
            tasks = [(strat, r_over, bt_over, p) for p in pairs]
            per: dict[str, pd.DataFrame] = {}
            n_sigs: dict[str, int] = {}
            for pair, t, ns in pool.map(_task, tasks):
                per[pair] = t
                n_sigs[pair] = ns
            print(f"\n=== {label} ({time.time() - t0:.0f}s) ===", flush=True)
            combined = combine_trades(per)
            _subset_metrics(combined, "pooled:all")
            _subset_metrics(
                combined[combined["symbol"].isin(INDICES)] if not combined.empty else combined,
                "pooled:indices",
            )
            _subset_metrics(
                combined[~combined["symbol"].isin(INDICES)] if not combined.empty else combined,
                "pooled:fx+metals",
            )
            sym = []
            for p in pairs:
                t = per[p]
                sym.append(
                    {
                        "symbol": p,
                        "n": len(t),
                        "pnl": round(float(t["pnl"].sum()), 0) if len(t) else 0.0,
                        "wr": round(float((t["pnl"] > 0).mean()), 2) if len(t) else float("nan"),
                    }
                )
            print(pd.DataFrame(sym).to_string(index=False), flush=True)
            filt = _filtered(combined)
            _yearly(filt)
            _quarterly(filt)
            rb = (
                list(pool.map(_rb_task, [(strat, r_over, bt_over, n_sigs[p], p) for p in pairs]))
                if mode != "quick"
                else []
            )
            if rb:
                n_pass = sum(1 for *_, ok in rb if ok)
                for pair, ns, stats, ok in rb:
                    print(
                        f"{pair:<11} n_sig={ns:>4} rand_mean={stats['mean']:>9.0f} "
                        f"p={stats['p']:.3f} -> {'PASS' if ok else 'FAIL'}",
                        flush=True,
                    )
                print(f"random gate: {n_pass}/{len(pairs)} pairs pass (need all)", flush=True)


if __name__ == "__main__":
    main()
