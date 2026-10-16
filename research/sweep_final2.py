"""Final-stage validation for strategy variants (breakout_d / fade_d / retest_*).

Usage:
  uv run python sweep_final2.py <strat> '<r_over json>' ['<bt_over json>']

Runs: full sample + per-symbol, OOS last 25%, walk-forward windows (9mo/3mo),
random baseline per pair.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from alpha_sweep2 import FRAMES, _init_worker, _signals, universe
from retest.backtest.engine import BacktestParams, run_backtest
from retest.portfolio import (
    apply_group_filter,
    apply_usd_cap,
    combine_trades,
    portfolio_equity,
)
from retest.report.metrics_table import metrics_table
from retest.signals.retest import RetestParams
from retest.validation.random_baseline import baseline_passes, run_random_baseline
from retest.validation.significance import significance_report
from retest.validation.walk_forward import walk_forward_windows

START_EQUITY = 10000.0
OOS_FRACTION = 0.25
IS_YEARS = 0.75
OOS_YEARS = 0.25

STRAT = "breakout_d"
R_OVER: dict = {}
BT_OVER: dict = {}


def _params(pair: str):
    rp = replace(RetestParams.from_config(), **R_OVER)
    bp = replace(BacktestParams.from_config(pair), **BT_OVER)
    return rp, bp


def _task(args):
    strat, mode, r_over, bt_over, pair = args
    fr = FRAMES[pair]
    rp = replace(RetestParams.from_config(), **r_over)
    bp = replace(BacktestParams.from_config(pair), **bt_over)
    hourly = fr["h1"]
    if mode == "full":
        sig = _signals(strat, fr, rp)
        res = run_backtest(hourly, sig, bp)
        t = res.trades_df.copy()
        t["symbol"] = pair
        return mode, pair, t, len(sig)
    if mode == "oos":
        sig = _signals(strat, fr, rp)
        split = hourly.index[int(len(hourly) * (1.0 - OOS_FRACTION))]
        sig = sig[sig["timestamp"] >= split]
        res = run_backtest(hourly, sig, bp)
        t = res.trades_df.copy()
        t["symbol"] = pair
        return mode, pair, t, len(sig)
    if mode == "wf":
        ist = pd.Timedelta(days=365.25 * IS_YEARS)
        oost = pd.Timedelta(days=365.25 * OOS_YEARS)
        daily = fr["d"]
        out = []
        for is_start, is_end, oos_start, oos_end in walk_forward_windows(
            hourly.index[0], hourly.index[-1], ist, oost
        ):
            is_h = hourly[(hourly.index >= is_start) & (hourly.index < is_end)]
            oos_h = hourly[(hourly.index >= oos_start) & (hourly.index < oos_end)]
            if oos_h.empty:
                continue
            h = pd.concat([is_h, oos_h])
            fr2 = dict(fr)
            fr2["h1"] = h
            fr2["d"] = daily[(daily.index >= is_start) & (daily.index < oos_end)]
            if "h2" in fr:
                fr2["h2"] = fr["h2"][(fr["h2"].index >= is_start) & (fr["h2"].index < oos_end)]
            if "h4" in fr:
                fr2["h4"] = fr["h4"][(fr["h4"].index >= is_start) & (fr["h4"].index < oos_end)]
            sig = _signals(strat, fr2, rp)
            sig = sig[sig["timestamp"] >= oos_h.index[0]]
            res = run_backtest(h, sig, bp)
            t = res.trades_df.copy()
            t["symbol"] = pair
            t["window"] = str(oos_start.date())
            out.append(t)
        return mode, pair, (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), 0
    if mode == "rb":
        sig = _signals(strat, fr, rp)
        pnls = run_random_baseline(hourly, max(len(sig), 1), bp, 20, 0)
        passed, stats = baseline_passes(pnls)
        return mode, pair, stats, (len(sig), passed)
    raise ValueError(mode)


def _pool_metrics(per: dict[str, pd.DataFrame], label: str):
    trades = combine_trades(per)
    trades = apply_group_filter(trades)
    trades = apply_usd_cap(trades)
    eq = portfolio_equity(trades, START_EQUITY)
    m = metrics_table(trades, eq, START_EQUITY).set_index("metric")["value"].to_dict()
    ret = eq.resample("D").last().ffill().pct_change().dropna()
    rep = significance_report(ret)
    print(
        f"{label:<30} n={int(m.get('n_trades', 0)):>4} pnl={m.get('total_pnl', 0.0):>9.1f} "
        f"pf={m.get('profit_factor', float('nan')):>6.2f} wr={m.get('win_rate', float('nan')):.2f} "
        f"mdd={m.get('max_drawdown', float('nan')):.3f} sharpe={rep['sharpe']:>6.2f} "
        f"[{rep['ci_low']:>6.2f}] psr={rep['deflated_psr']:.2f}",
        flush=True,
    )
    return trades


def main() -> None:
    global STRAT, R_OVER, BT_OVER
    if len(sys.argv) > 1:
        STRAT = sys.argv[1]
    if len(sys.argv) > 2:
        R_OVER = json.loads(sys.argv[2])
    if len(sys.argv) > 3:
        BT_OVER = json.loads(sys.argv[3])
    print(f"validating strat={STRAT} r_over={R_OVER} bt_over={BT_OVER}", flush=True)
    workers = min(10, (os.cpu_count() or 4))
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(universe(),)
    ) as pool:
        pairs = universe()

        full = {p: t for _, p, t, _ in pool.map(_task, [(STRAT, "full", R_OVER, BT_OVER, p) for p in pairs])}
        print(f"--- full ({time.time() - t0:.0f}s) ---")
        trades = _pool_metrics(full, f"full:{STRAT}")
        sym = []
        for p in pairs:
            t = trades[trades["symbol"] == p]
            sym.append({
                "symbol": p,
                "n": len(t),
                "pnl": float(t["pnl"].sum()) if len(t) else 0.0,
                "wr": float((t["pnl"] > 0).mean()) if len(t) else float("nan"),
            })
        print(pd.DataFrame(sym).to_string(index=False))

        oos = {p: t for _, p, t, _ in pool.map(_task, [(STRAT, "oos", R_OVER, BT_OVER, p) for p in pairs])}
        print(f"\n--- OOS last {OOS_FRACTION:.0%} ({time.time() - t0:.0f}s) ---")
        _pool_metrics(oos, f"oos:{STRAT}")

        wf = list(pool.map(_task, [(STRAT, "wf", R_OVER, BT_OVER, p) for p in pairs]))
        all_wf = pd.concat([t for _, _, t, _ in wf if not t.empty], ignore_index=True)
        print(f"\n--- walk-forward ({time.time() - t0:.0f}s) ---")
        if all_wf.empty:
            print("no wf trades")
        else:
            for win, grp in all_wf.groupby("window"):
                print(
                    f"{win}: n={len(grp):>4} pnl={grp['pnl'].sum():>9.1f} "
                    f"wr={(grp['pnl'] > 0).mean():.2f}"
                )
            _pool_metrics({p: t for _, p, t, _ in wf}, f"wf-pooled:{STRAT}")

        print(f"\n--- random baseline ({time.time() - t0:.0f}s) ---")
        n_pass = 0
        for _, p, stats, extra in pool.map(_task, [(STRAT, "rb", R_OVER, BT_OVER, p) for p in pairs]):
            n_sig, passed = extra
            n_pass += bool(passed)
            print(
                f"{p:<8} n_sig={n_sig:>4} rand_mean={stats['mean']:>9.2f} p={stats['p']:.3f} "
                f"-> {'PASS' if passed else 'FAIL'}"
            )
        print(f"random gate: {n_pass}/{len(pairs)} pairs pass (need all)")


if __name__ == "__main__":
    main()
