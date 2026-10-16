"""Final-stage validation for the chosen sweep config.

Runs, for the CHOSEN config (overrides on top of current config.yaml):
  1. full-sample portfolio metrics + per-symbol table
  2. OOS split (last 25% of each pair's H1 history), pooled + per-symbol
  3. walk-forward windows (9mo IS / 3mo OOS rolling), pooled per window
  4. random-entry baseline per pair (falsification gate)

Usage: uv run python sweep_final.py
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

from alpha_sweep import FRAMES, _init_worker, universe
from retest.backtest.engine import BacktestParams, run_backtest
from retest.data.loader import load_ohlc  # noqa: F401  (workers re-init FRAMES)
from retest.portfolio import (
    apply_group_filter,
    apply_usd_cap,
    combine_trades,
    portfolio_equity,
)
from retest.report.metrics_table import metrics_table
from retest.signals.retest import RetestParams, generate_retest_signals
from retest.validation.random_baseline import (
    baseline_passes,
    run_random_baseline,
)
from retest.validation.significance import significance_report
from retest.validation.walk_forward import walk_forward_windows

START_EQUITY = 10000.0

CHOSEN_LABEL = "CHOSEN"
CHOSEN_R: dict = {}
CHOSEN_BT: dict = {}
OOS_FRACTION = 0.25
IS_YEARS = 0.75
OOS_YEARS = 0.25


def _load_choice() -> None:
    """Usage: sweep_final.py LABEL '{"entry_start_hour_utc": 7, ...}' ['{bt json}']"""
    global CHOSEN_LABEL, CHOSEN_R, CHOSEN_BT
    if len(sys.argv) > 1:
        CHOSEN_LABEL = sys.argv[1]
    if len(sys.argv) > 2:
        CHOSEN_R = json.loads(sys.argv[2])
        if "touch_gates" in CHOSEN_R:
            CHOSEN_R["touch_gates"] = {int(k): int(v) for k, v in CHOSEN_R["touch_gates"].items()}
    if len(sys.argv) > 3:
        CHOSEN_BT = json.loads(sys.argv[3])


def _params(pair: str, r_over: dict, bt_over: dict):
    rp = replace(RetestParams.from_config(), **r_over)
    bp = replace(BacktestParams.from_config(pair), **bt_over)
    return rp, bp


def _task(args):
    mode, r_over, bt_over, pair = args
    hourly, daily = FRAMES[pair]
    rp, bp = _params(pair, r_over, bt_over)
    if mode == "full":
        sig = generate_retest_signals(hourly, daily, rp)
        res = run_backtest(hourly, sig, bp)
        trades = res.trades_df.copy()
        trades["symbol"] = pair
        return pair, trades
    if mode == "oos":
        sig = generate_retest_signals(hourly, daily, rp)
        split = hourly.index[int(len(hourly) * (1.0 - OOS_FRACTION))]
        sig = sig[sig["timestamp"] >= split]
        res = run_backtest(hourly, sig, bp)
        trades = res.trades_df.copy()
        trades["symbol"] = pair
        return pair, trades
    if mode == "wf":
        ist = pd.Timedelta(days=365.25 * IS_YEARS)
        oost = pd.Timedelta(days=365.25 * OOS_YEARS)
        out = []
        for is_start, is_end, oos_start, oos_end in walk_forward_windows(
            hourly.index[0], hourly.index[-1], ist, oost
        ):
            is_h = hourly[(hourly.index >= is_start) & (hourly.index < is_end)]
            oos_h = hourly[(hourly.index >= oos_start) & (hourly.index < oos_end)]
            is_d = daily[(daily.index >= is_start) & (daily.index < is_end)]
            oos_d = daily[(daily.index >= oos_start) & (daily.index < oos_end)]
            if oos_h.empty:
                continue
            h = pd.concat([is_h, oos_h])
            d = pd.concat([is_d, oos_d]) if not is_d.empty else oos_d
            sig = generate_retest_signals(h, d, rp)
            sig = sig[sig["timestamp"] >= oos_h.index[0]]
            res = run_backtest(h, sig, bp)
            trades = res.trades_df.copy()
            trades["symbol"] = pair
            trades["window"] = str(oos_start.date())
            out.append(trades)
        return pair, pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if mode == "rb":
        sig = generate_retest_signals(hourly, daily, rp)
        pnls = run_random_baseline(hourly, max(len(sig), 1), bp, 20, 0)
        passed, stats = baseline_passes(pnls)
        return pair, len(sig), stats, passed
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
        f"{label:<28} n={int(m.get('n_trades', 0)):>4} pnl={m.get('total_pnl', 0.0):>9.1f} "
        f"pf={m.get('profit_factor', float('nan')):>6.2f} wr={m.get('win_rate', float('nan')):.2f} "
        f"mdd={m.get('max_drawdown', float('nan')):.3f} sharpe={rep['sharpe']:>6.2f} "
        f"[{rep['ci_low']:>6.2f}] psr={rep['deflated_psr']:.2f}",
        flush=True,
    )
    return trades, eq


def main() -> None:
    _load_choice()
    workers = min(10, (os.cpu_count() or 4))
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(universe(),)
    ) as pool:
        pairs = universe()

        # 1) full sample
        full = {
            p: t
            for p, t in pool.map(
                _task, [("full", CHOSEN_R, CHOSEN_BT, p) for p in pairs]
            )
        }
        print(f"--- full sample ({time.time() - t0:.0f}s) ---")
        trades, _eq = _pool_metrics(full, "full:" + CHOSEN_LABEL)
        sym_rows = []
        for p in pairs:
            t = trades[trades["symbol"] == p]
            if t.empty:
                sym_rows.append({"symbol": p, "n": 0, "pnl": 0.0, "wr": float("nan")})
                continue
            sym_rows.append(
                {
                    "symbol": p,
                    "n": len(t),
                    "pnl": float(t["pnl"].sum()),
                    "wr": float((t["pnl"] > 0).mean()),
                }
            )
        print(pd.DataFrame(sym_rows).to_string(index=False))

        # 2) OOS split
        oos = {
            p: t
            for p, t in pool.map(
                _task, [("oos", CHOSEN_R, CHOSEN_BT, p) for p in pairs]
            )
        }
        print(f"\n--- OOS last {OOS_FRACTION:.0%} ({time.time() - t0:.0f}s) ---")
        _pool_metrics(oos, "oos:" + CHOSEN_LABEL)
        oos_rows = []
        for p in pairs:
            t = oos[p]
            oos_rows.append(
                {"symbol": p, "n": len(t), "pnl": float(t["pnl"].sum()) if len(t) else 0.0}
            )
        print(pd.DataFrame(oos_rows).to_string(index=False))

        # 3) walk-forward windows
        wf_parts = list(
            pool.map(_task, [("wf", CHOSEN_R, CHOSEN_BT, p) for p in pairs])
        )
        all_wf = pd.concat([t for _, t in wf_parts if not t.empty], ignore_index=True)
        print(f"\n--- walk-forward windows ({time.time() - t0:.0f}s) ---")
        if all_wf.empty:
            print("no wf trades")
        else:
            for win, grp in all_wf.groupby("window"):
                print(
                    f"{win}: n={len(grp):>3} pnl={grp['pnl'].sum():>8.1f} "
                    f"wr={(grp['pnl'] > 0).mean():.2f}"
                )
            _pool_metrics({p: t for p, t in wf_parts}, "wf-pooled:" + CHOSEN_LABEL)

        # 4) random baseline
        print(f"\n--- random baseline per pair ({time.time() - t0:.0f}s) ---")
        rb_rows = list(
            pool.map(_task, [("rb", CHOSEN_R, CHOSEN_BT, p) for p in pairs])
        )
        for p, n_sig, stats, passed in rb_rows:
            print(
                f"{p:<8} n_sig={n_sig:>3} rand_mean={stats['mean']:>8.2f} p={stats['p']:.3f} "
                f"-> {'PASS' if passed else 'FAIL'}"
            )


if __name__ == "__main__":
    main()
