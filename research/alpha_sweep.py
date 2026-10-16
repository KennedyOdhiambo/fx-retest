"""Parallel alpha sweep: search retest config space over the full universe.

Usage:
  uv run python alpha_sweep.py single   # one-knob-at-a-time batch from current config
  uv run python alpha_sweep.py combo    # hand-picked combos
  uv run python alpha_sweep.py final    # full sample + OOS + random gate for CHOSEN

Writes alpha_sweep_results.csv. Selection metric: portfolio pnl / sharpe.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from retest.signals.retest import RetestParams, generate_retest_signals
from retest.validation.significance import significance_report

START_EQUITY = 10000.0
RESULTS_CSV = str(Path(__file__).resolve().parent / "results" / "alpha_sweep_results.csv")

FRAMES: dict = {}


def _no_fetch(*_a, **_k):
    return empty_ohlc()


def universe() -> list[str]:
    return list(get_config().get("data.tradingview.universe"))


def _init_worker(pairs: list[str]) -> None:
    global FRAMES
    for p in pairs:
        FRAMES[p] = (
            load_ohlc(p, granularity="H1", fetcher=_no_fetch),
            load_ohlc(p, fetcher=_no_fetch),
        )


def _task(args):
    label, r_over, bt_over, pair = args
    hourly, daily = FRAMES[pair]
    rp = replace(RetestParams.from_config(), **r_over)
    bp = replace(BacktestParams.from_config(pair), **bt_over)
    sig = generate_retest_signals(hourly, daily, rp)
    res = run_backtest(hourly, sig, bp)
    t = res.trades_df.copy()
    t["symbol"] = pair
    return label, pair, t, len(sig)


def _summarize(label, per, r_over, bt_over, n_sigs, quiet=False):
    trades = combine_trades(per)
    trades = apply_group_filter(trades)
    trades = apply_usd_cap(trades)
    eq = portfolio_equity(trades, START_EQUITY)
    m = metrics_table(trades, eq, START_EQUITY).set_index("metric")["value"].to_dict()
    ret = eq.resample("D").last().ffill().pct_change().dropna()
    rep = significance_report(ret)
    row = {
        "label": label,
        "n_sigs": n_sigs,
        "n": int(m.get("n_trades", 0)),
        "pnl": float(m.get("total_pnl", 0.0)),
        "pf": float(m.get("profit_factor", np.nan)),
        "wr": float(m.get("win_rate", np.nan)),
        "mdd": float(m.get("max_drawdown", np.nan)),
        "sharpe": float(rep["sharpe"]),
        "sharpe_lo": float(rep["ci_low"]),
        "psr": float(rep["deflated_psr"]),
        "r_over": r_over,
        "bt_over": bt_over,
    }
    if not quiet:
        print(
            f"{label:<40} n={row['n']:>4} pnl={row['pnl']:>9.1f} sharpe={row['sharpe']:>6.2f} "
            f"[{row['sharpe_lo']:>6.2f}] pf={row['pf']:>6.2f} wr={row['wr']:.2f} mdd={row['mdd']:.3f}",
            flush=True,
        )
    return row, trades


def evaluate_many(pool, configs: list[tuple[str, dict, dict]]) -> list[tuple[dict, pd.DataFrame]]:
    """configs: list of (label, r_over, bt_over). Batches all pairs x configs for balance."""
    pairs = universe()
    tasks = [(label, r, b, p) for (label, r, b) in configs for p in pairs]
    collected: dict[str, dict] = {}
    n_done: dict[str, int] = {}
    for label, _r, _b in configs:
        collected[label] = {"per": {}, "sigs": 0}
        n_done[label] = 0
    t0 = time.time()
    for fut in as_completed([pool.submit(_task, t) for t in tasks]):
        label, pair, trades, ns = fut.result()
        collected[label]["per"][pair] = trades
        collected[label]["sigs"] += ns
        n_done[label] += 1
        if n_done[label] == len(pairs):
            print(f"[{time.time() - t0:5.0f}s] {label} done", flush=True)
    out = []
    for label, r, b in configs:
        c = collected[label]
        row, trades = _summarize(label, c["per"], r, b, c["sigs"])
        out.append((row, trades))
    return out


def _append(rows: list[dict]) -> None:
    flat = [
        {
            **{k: v for k, v in r.items() if k not in ("r_over", "bt_over")},
            "r_over": str(r["r_over"]),
            "bt_over": str(r["bt_over"]),
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


def single_configs() -> list[tuple[str, dict, dict]]:
    cfgs = [("baseline(current config)", {}, {})]
    for name, values in [
        ("swing_lookback", [3, 5, 6, 8]),
        ("cluster_ticks", [3, 10, 20]),
        ("min_swing_magnitude_atr", [0.15, 0.5, 0.75]),
        ("touch_tolerance_atr", [0.15, 0.45]),
        ("max_break_return_bars", [12, 36, 48]),
        ("min_reward_risk", [0.75, 1.5, 2.0]),
        ("min_stop_atr", [0.5, 1.0]),
        ("stop_buffer_atr", [0.4, 0.6]),
        ("fta_fallback_atr", [1.5, 3.0]),
        ("limit_expiry_bars", [12, 48]),
        ("max_level_age_days", [0, 120]),
        ("ban_one_touch", [True]),
        ("min_swings_per_level", [2]),
        ("long_extra_confirmations", [1]),
        ("special_fib_rule", [False]),
        ("entry_start_hour_utc", [0, 7, 13]),
    ]:
        for v in values:
            cfgs.append((f"{name}={v}", {name: v}, {}))
    cfgs.append(("gates:strategy.md", {"touch_gates": {3: 0, 2: 1, 1: 2}}, {}))
    cfgs.append(("gates:strict", {"touch_gates": {3: 1, 2: 1, 1: 2}}, {}))
    for name, values in [
        ("max_hold_bars", [60, 240]),
        ("breakeven_at_r", [1.0]),
        ("hold_winners_overnight", [False]),
    ]:
        for v in values:
            cfgs.append((f"bt:{name}={v}", {}, {name: v}))
    return cfgs


_STRICT = {3: 1, 2: 1, 1: 2}
_MD = {3: 0, 2: 1, 1: 2}
_C8 = {
    "entry_start_hour_utc": 7,
    "touch_gates": _STRICT,
    "max_break_return_bars": 36,
    "special_fib_rule": False,
}
COMBO_CONFIGS: list[tuple[str, dict, dict]] = [
    ("D1 C8+rr15", {**_C8, "min_reward_risk": 1.5}, {}),
    ("D2 C8+minswing075", {**_C8, "min_swing_magnitude_atr": 0.75}, {}),
    ("D3 C8+minstop05", {**_C8, "min_stop_atr": 0.5}, {}),
    ("D4 C8+minstop10", {**_C8, "min_stop_atr": 1.0}, {}),
    ("D5 C8+ban1", {**_C8, "ban_one_touch": True}, {}),
    ("D6 C8+break48", {**_C8, "max_break_return_bars": 48}, {}),
    ("D7 C8+lookback8", {**_C8, "swing_lookback": 8}, {}),
    ("D8 C8+age120", {**_C8, "max_level_age_days": 120}, {}),
    ("D9 C8+end16", {**_C8, "entry_end_hour_utc": 16}, {}),
    ("D10 C8+gates2", {**_C8, "touch_gates": {3: 2, 2: 2, 1: 2}}, {}),
    ("D11 C8+longextra1", {**_C8, "long_extra_confirmations": 1}, {}),
    ("D12 C8+hold240", _C8, {"max_hold_bars": 240}),
    ("D13 C8+tol015", {**_C8, "touch_tolerance_atr": 0.15}, {}),
    ("D14 C8+buf01", {**_C8, "stop_buffer_atr": 0.1}, {}),
    ("D15 C8+rr2", {**_C8, "min_reward_risk": 2.0}, {}),
]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    workers = min(10, (os.cpu_count() or 4))
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(universe(),)
    ) as pool:
        if mode == "single":
            results = evaluate_many(pool, single_configs())
        elif mode == "combo":
            results = evaluate_many(pool, COMBO_CONFIGS)
        else:
            print(f"unknown mode: {mode}")
            sys.exit(1)
    rows = [r for r, _ in results]
    _append(rows)
    print(f"\n{len(rows)} configs in {time.time() - t0:.0f}s -> {RESULTS_CSV}")
    print("\n=== sorted by pnl ===")
    for r in sorted(rows, key=lambda x: -x["pnl"])[:15]:
        print(
            f"{r['label']:<40} n={r['n']:>4} pnl={r['pnl']:>9.1f} sharpe={r['sharpe']:>6.2f} "
            f"[{r['sharpe_lo']:>6.2f}] pf={r['pf']:>6.2f} wr={r['wr']:.2f}"
        )
    print("\n=== sorted by sharpe (n>=20) ===")
    for r in sorted([x for x in rows if x["n"] >= 20], key=lambda x: -x["sharpe"])[:15]:
        print(
            f"{r['label']:<40} n={r['n']:>4} pnl={r['pnl']:>9.1f} sharpe={r['sharpe']:>6.2f} "
            f"[{r['sharpe_lo']:>6.2f}] pf={r['pf']:>6.2f} wr={r['wr']:.2f}"
        )


if __name__ == "__main__":
    main()
