"""Vectorized daily portfolio backtester for cross-instrument studies.

Complementary to the event engine (which is single-position, order-based and
was built for the retest/breakout families). This tool is for portfolio-level
studies over 50-70 instruments and decades of dailies: signals on closes,
position applied to the next day's close-to-close return, equal 1/N weights,
per-instrument risk scaling k = clip(vol_target / ann_vol, 0.2, 3.0), costs
charged per unit of position change. Honesty rails: signals only use data up
to and including bar t; the crossing return is day t+1; random gate by
block-bootstrapping returns against the same positions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data/cache")
TRADING_DAYS = 252
START_EQUITY = 100_000.0


def load_daily(name: str, start: str = "2005-01-01") -> pd.DataFrame:
    df = pd.read_parquet(CACHE / f"{name}_D.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    df = df[df["close"] > 0]  # drop artifact prints (e.g. CFD pinned near/below zero)
    return df.sort_index()


# ---------------------------------------------------------------- signals

def donchian_state(df: pd.DataFrame, entry: int = 50, exit_: int = 20) -> np.ndarray:
    """Classic channel breakout state machine: +1 / -1 / 0, exits to flat."""
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(c)
    state = np.zeros(n)
    st = 0
    warm = max(entry, exit_) + 1
    for t in range(warm, n):
        if st == 0:
            if c[t] > hi[t - entry : t].max():
                st = 1
            elif c[t] < lo[t - entry : t].min():
                st = -1
        elif st == 1:
            if c[t] < lo[t - exit_ : t].min():
                st = 0
        else:
            if c[t] > hi[t - exit_ : t].max():
                st = 0
        state[t] = st
    return state


def tsmom_monthly(close: pd.Series, lookback: int = 252) -> np.ndarray:
    """Sign of 12-month return, evaluated at the first bar of each month."""
    sig = np.sign(close - close.shift(lookback))
    sig = sig.where(close.shift(lookback).notna(), 0.0)
    periods = close.index.tz_localize(None).to_period("M")
    pos = sig.groupby(periods).transform("first")
    return pos.to_numpy()


def rsi_wilder(close: pd.Series, n: int = 2) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0.0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0.0, np.nan))


def meanrev_long(close: pd.Series, rsi_entry: float = 5.0, exit_ma: int = 5,
                 max_hold: int = 10, sma_filter: int | None = None) -> np.ndarray:
    """Long-only dip-buy: enter after close with RSI2 < entry, exit on close > SMA5."""
    c = close.to_numpy()
    rsi = rsi_wilder(close).to_numpy()
    ma = close.rolling(exit_ma).mean().to_numpy()
    sma200 = close.rolling(200).mean().to_numpy() if sma_filter else None
    n = len(c)
    pos = np.zeros(n)
    st, hold = 0, 0
    for t in range(1, n):
        if st == 0:
            if rsi[t] == rsi[t] and rsi[t] < rsi_entry:  # not NaN
                if sma200 is None or (sma200[t] == sma200[t] and c[t] > sma200[t]):
                    st = 1
                    hold = 0
        else:
            hold += 1
            if c[t] > ma[t] or hold >= max_hold:
                st = 0
        pos[t] = st
    return pos


# ---------------------------------------------------------------- simulation

def simulate(frames: dict[str, pd.DataFrame], positions: dict[str, np.ndarray],
             cost_bps: float = 3.0, vol_target: float = 0.10, vol_win: int = 20,
             kmin: float = 0.2, kmax: float = 3.0,
             funding: dict[str, pd.Series] | None = None) -> dict:
    """Equal 1/N weights x per-instrument vol scaling; costs on position changes.

    funding: optional per-instrument daily funding rate (sum of the day's
    events). Perp positions pay it when long / receive it when short:
    contribution -= pos.shift(1) * funding / N.
    """
    union = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in frames.values()])))
    port = pd.Series(0.0, index=union)
    costs = pd.Series(0.0, index=union)
    episodes: list[tuple[str, int, float]] = []
    costs_by_name: dict[str, float] = {}
    funding_by_name: dict[str, float] = {}
    n_inst = len(frames)
    for name, df in frames.items():
        c = df["close"]
        ret = c.pct_change().fillna(0.0)
        vol = c.pct_change().rolling(vol_win).std() * np.sqrt(TRADING_DAYS)
        k = (vol_target / vol).clip(kmin, kmax).fillna(0.0)
        pos = pd.Series(positions[name], index=df.index, dtype=float)
        eff = (k * pos).shift(1).fillna(0.0) / n_inst
        contrib = eff * ret
        if funding is not None and name in funding:
            f = funding[name].reindex(df.index).fillna(0.0)
            fund = -(pos.shift(1).fillna(0.0) * f) / n_inst
            contrib = contrib + fund
            funding_by_name[name] = float(fund.sum())
        pnl = contrib.reindex(union).fillna(0.0)
        port += pnl
        dpos = pos.diff().abs().fillna(pos.abs())
        cost = ((dpos * k).shift(1).fillna(0.0) * (cost_bps / 1e4) / n_inst).reindex(union).fillna(0.0)
        costs += cost
        costs_by_name[name] = float(cost.sum())
        # episodes for trade-level stats (a sign flip closes one trade and opens the next).
        # day t's pnl uses pos[t-1] (eff is shifted), so a trade entered at t=r earns
        # from t=r+1 through t=exit(inclusive).
        p = pos.to_numpy()
        e = contrib.to_numpy()
        run_start = None
        prev = 0.0
        for t in range(len(p)):
            if p[t] != prev:
                if prev != 0 and run_start is not None:
                    episodes.append((name, int(np.sign(prev)), float(np.nansum(e[run_start + 1 : t + 1]))))
                    run_start = None
                if p[t] != 0:
                    run_start = t
                prev = p[t]
        if run_start is not None and prev != 0:
            episodes.append((name, int(np.sign(prev)), float(np.nansum(e[run_start + 1 :]))))
    net = port - costs
    equity = (1.0 + net).cumprod() * START_EQUITY
    return {"net": net, "gross": port, "costs": costs, "equity": equity, "episodes": episodes,
            "gross_pnl": float(port.sum()), "cost_total": float(costs.sum()), "n_inst": n_inst,
            "costs_by_name": costs_by_name, "funding_by_name": funding_by_name}


def random_gate(frames: dict[str, pd.DataFrame], positions: dict[str, np.ndarray],
                cost_bps: float = 3.0, n_draws: int = 200, block: int = 20,
                vol_target: float = 0.10, vol_win: int = 20, kmin: float = 0.2,
                kmax: float = 3.0, seed: int = 7,
                funding: dict[str, pd.Series] | None = None) -> dict:
    """Block-shuffle each instrument's returns against the same positions.

    The funding term depends only on positions, so it is a constant added to
    both the actual and the null distribution."""
    rng = np.random.default_rng(seed)
    n_inst = len(frames)
    null = np.zeros(n_draws)
    actual = 0.0
    cost_total = 0.0
    funding_total = 0.0
    for name, df in frames.items():
        c = df["close"]
        ret = c.pct_change().fillna(0.0).to_numpy()
        vol = c.pct_change().rolling(vol_win).std() * np.sqrt(TRADING_DAYS)
        k = (vol_target / vol).clip(kmin, kmax).fillna(0.0)
        pos = pd.Series(positions[name], index=df.index, dtype=float)
        eff = ((k * pos).shift(1).fillna(0.0) / n_inst).to_numpy()
        dpos = pos.diff().abs().fillna(pos.abs())
        cost_total += float(((dpos * k).shift(1).fillna(0.0) * (cost_bps / 1e4) / n_inst).sum())
        if funding is not None and name in funding:
            f = funding[name].reindex(df.index).fillna(0.0)
            funding_total += float((-(pos.shift(1).fillna(0.0) * f) / n_inst).sum())
        n = len(ret)
        nb = n // block
        usable = nb * block
        r_use = ret[:usable]
        eff_use = eff[:usable]
        actual += float(eff_use @ r_use)
        idx = np.arange(usable).reshape(nb, block)
        order = np.argsort(rng.random((n_draws, nb)), axis=1)
        r_sh = r_use[idx[order]].reshape(n_draws, usable)
        null += r_sh @ eff_use
    actual_net = actual + funding_total - cost_total
    null_net = null + funding_total - cost_total
    p = (float(np.sum(null_net >= actual_net)) + 1.0) / (n_draws + 1.0)
    return {"actual_net": actual_net, "null_median": float(np.median(null_net)),
            "p_value": p, "cost_total": cost_total, "funding_total": funding_total}


# ---------------------------------------------------------------- reporting

def summary(res: dict, label: str, periods_per_year: int = TRADING_DAYS) -> dict:
    net = res["net"]
    eq = res["equity"]
    n = len(net)
    final_mult = float(eq.iloc[-1] / START_EQUITY)
    ann_ret = final_mult ** (periods_per_year / max(n, 1)) - 1.0 if final_mult > 0 else -1.0
    ann_vol = float(net.std() * np.sqrt(periods_per_year))
    sharpe = float(net.mean() / net.std() * np.sqrt(periods_per_year)) if net.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())
    eps = pd.Series([p for _, _, p in res["episodes"]])
    wins = eps[eps > 0].sum()
    losses = eps[eps < 0].sum()
    pf = wins / abs(losses) if losses < 0 else float("inf")
    wr = float((eps > 0).mean()) if len(eps) else float("nan")
    yearly = (1.0 + net).resample("YE").prod() - 1.0
    pos_years = int((yearly > 0).sum())
    q = (1.0 + net).resample("QE").prod() - 1.0
    pos_q = int((q > 0).sum())
    print(
        f"{label:<26} eq={eq.iloc[-1] / START_EQUITY:>7.2f}x cagr={ann_ret:>6.1%} vol={ann_vol:>5.1%} "
        f"sharpe={sharpe:>5.2f} mdd={dd:>6.1%} trades={len(eps):>5} pf={pf:>5.2f} wr={wr:.2f} "
        f"yrs+={pos_years:>2}/{len(yearly)} q+={pos_q:>2}/{len(q)}",
        flush=True,
    )
    return {"label": label, "cagr": ann_ret, "sharpe": sharpe, "mdd": dd, "pf": pf, "wr": wr,
            "trades": len(eps), "final_mult": float(eq.iloc[-1] / START_EQUITY),
            "pos_years": pos_years, "n_years": len(yearly), "pos_q": pos_q, "n_q": len(q)}


def yearly_table(res: dict) -> None:
    net = res["net"]
    yearly = (1.0 + net).resample("YE").prod() - 1.0
    yearly.index = yearly.index.year
    print("  " + "  ".join(f"{y}:{v:+.0%}" for y, v in yearly.items()), flush=True)
