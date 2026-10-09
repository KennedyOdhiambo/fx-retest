"""Multi-asset portfolio: one universe, pooled trades, shared equity.

Each instrument is signaled and backtested standalone at fixed fractional risk
(1% of ``starting_equity`` per trade, so every trade risks the same dollars and
R multiples stay comparable). Correlated groups (see ``portfolio.groups`` in
config: AUD+NZD, XAU+XAG, …) never double the same directional bet — ``skip``
keeps the higher expected-R trade, ``halve`` keeps both at half size.
Portfolio equity = starting equity + cumulative realized PnL by exit time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from retest.backtest.engine import BacktestParams, run_backtest
from retest.report.metrics_table import metrics_table
from retest.signals.retest import RetestParams, generate_retest_signals
from retest.validation.significance import significance_report


@dataclass
class PortfolioResult:
    trades_df: pd.DataFrame  # all trades, with symbol + r_multiple
    equity_curve: pd.Series
    symbol_metrics: pd.DataFrame
    portfolio_metrics: pd.DataFrame
    significance: dict


def run_portfolio(
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    signal_params: RetestParams | None = None,
    backtest_params: BacktestParams | None = None,
    starting_equity: float = 10000.0,
    exec_frames: dict[str, pd.DataFrame] | None = None,
) -> PortfolioResult:
    signal_params = signal_params or RetestParams.from_config()
    exec_frames = exec_frames or {}
    per_symbol: dict[str, pd.DataFrame] = {}
    for symbol, (hourly, daily) in frames.items():
        sig = generate_retest_signals(hourly, daily, signal_params)
        bt = backtest_params if backtest_params is not None else BacktestParams.from_config(symbol)
        res = run_backtest(exec_frames.get(symbol, hourly), sig, bt)
        trades = res.trades_df.copy()
        trades["symbol"] = symbol
        per_symbol[symbol] = trades
    trades_df = combine_trades(per_symbol)
    trades_df = apply_group_filter(trades_df)
    trades_df = apply_usd_cap(trades_df)
    equity_curve = portfolio_equity(trades_df, starting_equity)
    sym_rows = []
    for symbol in per_symbol:
        trades = trades_df[trades_df["symbol"] == symbol]
        curve = portfolio_equity(trades, starting_equity)
        m = metrics_table(trades, curve, starting_equity).set_index("metric")["value"]
        sym_rows.append({"symbol": symbol, **m.to_dict()})
    symbol_metrics = pd.DataFrame(sym_rows)
    portfolio_metrics = metrics_table(trades_df, equity_curve, starting_equity)
    if trades_df.empty:
        significance: dict = {
            "n": 0.0,
            "sharpe": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "deflated_psr": 0.0,
            "enough_data": False,
        }
    else:
        daily_returns = equity_curve.resample("D").last().ffill().pct_change().dropna()
        significance = significance_report(daily_returns)
    return PortfolioResult(trades_df, equity_curve, symbol_metrics, portfolio_metrics, significance)


def group_of(symbol: str, groups: list[list[str]] | None = None) -> int | None:
    """Group index for ``symbol`` (None = ungrouped, always trades)."""
    if groups is None:
        from retest.config import get_config as _gc

        groups = _gc().get("portfolio.groups", [])
    for i, grp in enumerate(groups or []):
        if symbol in grp:
            return i
    return None


def apply_group_filter(
    trades_df: pd.DataFrame,
    groups: list[list[str]] | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    """One directional bet per correlated group at a time (ex-ante only).

    Same group + same direction + overlapping [entry, exit] windows compete:
    ``skip`` keeps the higher ``expected_r`` (earlier entry breaks ties, NaN
    counts as worst — e.g. random-baseline signals carry none); ``halve``
    keeps both at half size. Opposite directions are hedged — always kept.
    """
    if trades_df.empty:
        return trades_df
    from retest.config import get_config as _gc

    if groups is None:
        groups = _gc().get("portfolio.groups", [])
    if mode is None:
        mode = str(_gc().get("portfolio.mode", "skip"))
    if not groups:
        return trades_df
    t = trades_df.copy()
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True)
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True)
    t["_grp"] = t["symbol"].map(lambda s: group_of(s, groups))
    t["_er"] = pd.to_numeric(t.get("expected_r", float("nan")), errors="coerce")
    t = t.sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)
    if mode == "halve":
        involved: set[int] = set()
        rows = t.to_dict("records")
        for i in range(len(rows)):
            if rows[i]["_grp"] is None:
                continue
            for j in range(i + 1, len(rows)):
                if rows[j]["entry_ts"] > rows[i]["exit_ts"]:
                    break
                if (
                    rows[j]["_grp"] == rows[i]["_grp"]
                    and rows[j]["direction"] == rows[i]["direction"]
                    and rows[j]["entry_ts"] <= rows[i]["exit_ts"]
                    and rows[i]["entry_ts"] <= rows[j]["exit_ts"]
                ):
                    involved.add(i)
                    involved.add(j)
        for i in involved:
            t.loc[t.index[i], "pnl"] = float(t.loc[t.index[i], "pnl"]) * 0.5
            if "units" in t.columns:
                t.loc[t.index[i], "units"] = float(t.loc[t.index[i], "units"]) * 0.5
        return t.drop(columns=["_grp", "_er"]).reset_index(drop=True)
    kept: list[dict] = []
    for _, row in t.iterrows():
        r = row.to_dict()
        if r["_grp"] is None:
            kept.append(r)
            continue
        rival_idx = None
        for k, kr in enumerate(kept):
            if (
                kr["_grp"] == r["_grp"]
                and kr["direction"] == r["direction"]
                and r["entry_ts"] <= kr["exit_ts"]
                and kr["entry_ts"] <= r["exit_ts"]
            ):
                rival_idx = k
                break
        if rival_idx is None:
            kept.append(r)
            continue
        old = kept[rival_idx]
        old_er = old["_er"] if pd.notna(old["_er"]) else float("-inf")
        new_er = r["_er"] if pd.notna(r["_er"]) else float("-inf")
        if new_er > old_er:
            kept[rival_idx] = r
    out = pd.DataFrame(kept)
    if out.empty:
        return trades_df.iloc[0:0]
    return out.drop(columns=["_grp", "_er"]).sort_values("exit_ts").reset_index(drop=True)


def usd_direction(symbol: str, direction: str) -> str | None:
    """USD-leg side of a trade: ``USD_Y`` long = long USD; ``X_USD`` long =
    short USD; crosses carry no USD leg (None)."""
    if symbol.startswith("USD_"):
        return direction
    if symbol.endswith("_USD"):
        return "short" if direction == "long" else "long"
    return None


def apply_usd_cap(trades_df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Cap concurrent same-side USD-leg bets (ex-ante, expected-R priority).

    A dollar shock hits all 12 USD pairs at once — beyond ``cap`` same-side
    concurrent trades, keep the highest ``expected_r``. Crosses (no USD leg)
    and opposite-side trades always pass.
    """
    if trades_df.empty:
        return trades_df
    from retest.config import get_config as _gc

    if cap is None:
        cap = int(_gc().get("portfolio.max_usd_same_direction", 0))
    if not cap or cap <= 0:
        return trades_df
    t = trades_df.copy()
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True)
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True)
    t["_usd"] = [usd_direction(s, d) for s, d in zip(t["symbol"], t["direction"])]
    t["_er"] = pd.to_numeric(t.get("expected_r", float("nan")), errors="coerce")
    t = t.sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)
    kept: list[dict] = []
    for _, row in t.iterrows():
        r = row.to_dict()
        if r["_usd"] is None:
            kept.append(r)
            continue
        overlapping = [
            k
            for k in kept
            if k["_usd"] == r["_usd"] and k["entry_ts"] <= r["entry_ts"] <= k["exit_ts"]
        ]
        if len(overlapping) < cap:
            kept.append(r)
            continue
        cand = overlapping + [r]
        cand.sort(
            key=lambda x: (x["_er"] if pd.notna(x["_er"]) else float("-inf"), x["entry_ts"])
        )
        drop = cand[0]
        if drop is r:
            continue  # weakest — sits out
        kept = [k for k in kept if not (k is drop)]
        kept.append(r)
    out = pd.DataFrame(kept)
    if out.empty:
        return trades_df.iloc[0:0]
    return out.drop(columns=["_usd", "_er"]).sort_values("exit_ts").reset_index(drop=True)


def combine_trades(per_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pool per-symbol trades: R multiple per trade, sorted by exit time."""
    parts = []
    for symbol, trades in per_symbol.items():
        if trades.empty:
            continue
        t = trades.copy()
        t["symbol"] = symbol
        risk = (t["entry_price"] - t["stop_price"]).abs()
        signed = t.apply(
            lambda r: (
                (r["exit_price"] - r["entry_price"]) * (1.0 if r["direction"] == "long" else -1.0)
            ),
            axis=1,
        )
        t["r_multiple"] = (signed / risk.replace(0.0, float("nan"))).fillna(0.0)
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("exit_ts").reset_index(drop=True)


def portfolio_equity(trades_df: pd.DataFrame, starting_equity: float) -> pd.Series:
    """Step equity curve from realized PnL ordered by exit time."""
    if trades_df.empty:
        return pd.Series([starting_equity])
    ordered = trades_df.sort_values("exit_ts")
    curve = pd.Series(
        starting_equity + ordered["pnl"].to_numpy().cumsum(),
        index=pd.DatetimeIndex(ordered["exit_ts"]),
    )
    curve.name = "equity"
    return curve
