"""Bund (FGBL) 5-minute test: intraday entries/breakouts off higher-TF levels.

Data: EUREX FGBL1! continuous via the TV feed. M5 history is fetch-limited
(~1.4 months anonymously, ~29 trading days) - a short window, judge accordingly.
H1 since 2025, D since 2007. Single symbol: pooled == symbol.

Exec frame is always M5 (break/return detection, ATR, stops, targets, costs).
Level frames vary (D/H4/H1). An optional higher-TF trend gate keeps only
signals aligned with the level-frame trend (close vs SMA, completed bars only).

Modes:
  uv run python research/sweep_bund.py smoke
  uv run python research/sweep_bund.py gate  '<cfg json>'
  uv run python research/sweep_bund.py final '<cfg json>' [costs]

cfg json: {"strat": "breakout"|"retest", "level": "d"|"h4"|"h1",
           "gates": [["d", 50], ["h4", 100]], "r_over": {}, "bt_over": {}}
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
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
from retest.strategies.mtf import resample_ohlc
from retest.validation.random_baseline import baseline_passes, run_random_baseline
from retest.validation.significance import significance_report
from retest.validation.walk_forward import walk_forward_windows

START_EQUITY = 10000.0
PAIR = "FGBL"
_DAY = pd.Timedelta(days=1)
_H4 = pd.Timedelta(hours=4)
_H1 = pd.Timedelta(hours=1)
_M15 = pd.Timedelta(minutes=15)
_M5 = pd.Timedelta(minutes=5)

_DELTAS = {"d": _DAY, "h4": _H4, "h1": _H1, "m15": _M15, "m5": _M5}


def _no_fetch(*_a, **_k):
    return empty_ohlc()


def load_frames() -> dict[str, pd.DataFrame]:
    h1 = load_ohlc(PAIR, granularity="H1", fetcher=_no_fetch)
    m5 = load_ohlc(PAIR, granularity="M5", fetcher=_no_fetch)
    return {
        "m5": m5,
        "m15": resample_ohlc(m5, "15min"),
        "h1": h1,
        "h4": resample_ohlc(h1, "4h"),
        "d": load_ohlc(PAIR, granularity="D", fetcher=_no_fetch),
    }


def _params(r_over: dict, bt_over: dict) -> tuple[RetestParams, BacktestParams]:
    rp = replace(RetestParams.from_config(), **r_over)
    bp = replace(BacktestParams.from_config(PAIR), **bt_over)
    return rp, bp


def gen_signals(fr: dict, strat: str, level: str, rp: RetestParams) -> pd.DataFrame:
    if strat == "breakout":
        return generate_breakout_signals(fr["m5"], fr[level], rp, _DELTAS[level])
    if strat == "retest":
        return _generate(fr["m5"], fr[level], rp, _DELTAS[level])
    raise ValueError(strat)


def apply_trend_gates(sig: pd.DataFrame, fr: dict, gates: list) -> pd.DataFrame:
    """Keep only signals aligned with each gate's trend: (frame, sma_len)."""
    out = sig
    for frame_key, sma_len in gates:
        frame = fr[str(frame_key)]
        if out.empty or frame.empty:
            return out.iloc[0:0]
        c = frame["close"].astype(float)
        sma = c.rolling(int(sma_len), min_periods=int(sma_len)).mean()
        close_t = (frame.index + _DELTAS[frame_key]).tz_convert(None).to_numpy()
        sig_t = out["timestamp"].dt.tz_convert(None).to_numpy()
        pos = np.searchsorted(close_t, sig_t, side="left") - 1
        up = (c > sma).to_numpy()
        ok = sma.notna().to_numpy()
        keep = []
        for p, d in zip(pos, out["direction"].to_numpy()):
            if p < 0 or not ok[p]:
                keep.append(False)
            else:
                keep.append(bool(up[p]) if d == "long" else not bool(up[p]))
        out = out[np.asarray(keep)]
    return out


def _filtered(trades: pd.DataFrame) -> pd.DataFrame:
    t = apply_group_filter(trades) if not trades.empty else trades
    return apply_usd_cap(t) if not t.empty else t


def _report(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"{label:<36} no trades", flush=True)
        return
    t = _filtered(trades)
    eq = portfolio_equity(t, START_EQUITY)
    m = metrics_table(t, eq, START_EQUITY).set_index("metric")["value"].to_dict()
    ret = eq.resample("D").last().ffill().pct_change().dropna()
    rep = significance_report(ret)
    print(
        f"{label:<36} n={int(m.get('n_trades', 0)):>4} pnl={m.get('total_pnl', 0.0):>9.0f} "
        f"pf={m.get('profit_factor', float('nan')):>5.2f} wr={m.get('win_rate', float('nan')):.2f} "
        f"mdd={m.get('max_drawdown', float('nan')):.3f} sharpe={rep['sharpe']:>6.2f} "
        f"[{rep['ci_low']:>6.2f}]",
        flush=True,
    )


def run_one(fr: dict, strat: str, level: str, r_over: dict, bt_over: dict, gates: list):
    rp, bp = _params(r_over, bt_over)
    sig = gen_signals(fr, strat, level, rp)
    n_raw = len(sig)
    if gates:
        sig = apply_trend_gates(sig, fr, gates)
    res = run_backtest(fr["m5"], sig, bp)
    t = res.trades_df.copy()
    t["symbol"] = PAIR
    return t, n_raw, len(sig), bp


def _weekly_table(trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    t = trades.copy()
    wk = t["entry_ts"].dt.tz_localize(None).dt.to_period("W")
    g = t.assign(wk=wk, win=t["pnl"] > 0).groupby("wk").agg(
        n=("pnl", "size"), pnl=("pnl", "sum"), wr=("win", "mean")
    )
    g["pnl"] = g["pnl"].round(0)
    g["wr"] = g["wr"].round(2)
    print(g.to_string(), flush=True)


def mode_smoke() -> None:
    fr = load_frames()
    print(f"frames: m5={len(fr['m5'])} h1={len(fr['h1'])} h4={len(fr['h4'])} d={len(fr['d'])}", flush=True)
    print(f"m5 window: {fr['m5'].index[0]} -> {fr['m5'].index[-1]}\n", flush=True)
    for strat in ("breakout", "retest"):
        for level in ("d", "h4", "h1", "m15", "m5"):
            t, n_raw, n_kept, _ = run_one(fr, strat, level, {}, {}, [])
            print(f"{strat} | {level} levels | signals={n_raw}", flush=True)
            _report(t, f"  {strat} | {level}")


def mode_gate(cfg: dict) -> None:
    fr = load_frames()
    t, n_raw, n_kept, _ = run_one(fr, cfg["strat"], cfg["level"], cfg.get("r_over", {}), cfg.get("bt_over", {}), [])
    print(f"{cfg['strat']} | {cfg['level']} | ungated | signals={n_raw}", flush=True)
    _report(t, "  ungated")
    for gate in cfg.get("gates", []):
        t, n_raw, n_kept, _ = run_one(
            fr, cfg["strat"], cfg["level"], cfg.get("r_over", {}), cfg.get("bt_over", {}), [gate]
        )
        print(f"{cfg['strat']} | {cfg['level']} | gate {gate} | signals {n_raw}->{n_kept}", flush=True)
        _report(t, f"  gate {gate[0].upper()}-SMA{gate[1]}")


def mode_final(cfg: dict, costs: bool) -> None:
    fr = load_frames()
    raw_over = dict(cfg.get("bt_over", {}))
    if costs:
        raw_over.update({"spread_pips": 3.0, "slippage_pips": 1.5, "exit_slippage_pips": 2.0})
    bt_over = raw_over
    gates = cfg.get("gates", [])
    r_over = cfg.get("r_over", {})
    strat, level = cfg["strat"], cfg["level"]
    tag = f"{strat}|{level}" + (f"|gate{gates}" if gates else "")

    t, n_raw, n_kept, _bp = run_one(fr, strat, level, r_over, bt_over, gates)
    print(f"\n=== final: {tag} | signals {n_raw}->{n_kept} | costs={'heavy' if costs else 'base'} ===", flush=True)
    _report(t, "full")
    print("\nper week:", flush=True)
    _weekly_table(_filtered(t))
    if not t.empty:
        print("\nexit reasons:", t["exit_reason"].value_counts().to_dict(), flush=True)

    # OOS last 25%
    rp, _ = _params(r_over, bt_over)
    m5 = fr["m5"]
    split = m5.index[int(len(m5) * 0.75)]
    sig = gen_signals(fr, strat, level, rp)
    if gates:
        sig = apply_trend_gates(sig, fr, gates)
    soos = sig[sig["timestamp"] >= split]
    res = run_backtest(m5, soos, _params(r_over, bt_over)[1])
    t_oos = res.trades_df.copy()
    t_oos["symbol"] = PAIR
    print(f"\nOOS last 25% (from {split.date()}):", flush=True)
    _report(t_oos, "oos")

    # walk-forward: 14 calendar days IS, 7 days OOS rolling
    ist, oost = pd.Timedelta(days=14), pd.Timedelta(days=7)
    fr2_base = fr
    rows = []
    for is_start, is_end, oos_start, oos_end in walk_forward_windows(m5.index[0], m5.index[-1], ist, oost):
        oos_m5 = fr["m5"][(fr["m5"].index >= oos_start) & (fr["m5"].index < oos_end)]
        if oos_m5.empty:
            continue
        fr2 = {
            "m5": fr["m5"][(fr["m5"].index >= is_start) & (fr["m5"].index < oos_end)],
            "h1": fr["h1"][(fr["h1"].index >= is_start) & (fr["h1"].index < oos_end)],
            "h4": fr["h4"][(fr["h4"].index >= is_start) & (fr["h4"].index < oos_end)],
            "d": fr["d"][(fr["d"].index >= is_start) & (fr["d"].index < oos_end)],
        }
        rp2, _ = _params(r_over, bt_over)
        sig2 = gen_signals(fr2, strat, level, rp2)
        if gates:
            sig2 = apply_trend_gates(sig2, fr2, gates)
        sig2 = sig2[sig2["timestamp"] >= oos_m5.index[0]]
        if sig2.empty:
            continue
        res2 = run_backtest(fr2["m5"], sig2, _params(r_over, bt_over)[1])
        t2 = res2.trades_df.copy()
        t2["symbol"] = PAIR
        t2["window"] = str(oos_start.date())
        rows.append(t2)
    print("\nwalk-forward (14d IS / 7d OOS):", flush=True)
    if rows:
        wf = pd.concat(rows, ignore_index=True)
        for win, grp in wf.groupby("window"):
            print(f"{win}: n={len(grp):>3} pnl={grp['pnl'].sum():>8.0f} wr={(grp['pnl'] > 0).mean():.2f}", flush=True)
        _report(wf, "wf pooled")
    else:
        print("no wf trades", flush=True)

    # random gate on the full frame
    print("\nrandom baseline:", flush=True)
    bp_gate = _params(r_over, bt_over)[1]
    pnls = run_random_baseline(m5, max(n_kept, 1), bp_gate, 20, 0, max_hold_bars=bp_gate.max_hold_bars)
    passed, stats = baseline_passes(pnls)
    print(
        f"n_sig={n_kept} rand_mean={stats['mean']:>9.0f} p={stats['p']:.3f} -> "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True,
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        mode_smoke()
    elif mode == "gate":
        mode_gate(json.loads(sys.argv[2]))
    elif mode == "final":
        mode_final(json.loads(sys.argv[2]), costs=len(sys.argv) > 3 and sys.argv[3] == "costs")
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
