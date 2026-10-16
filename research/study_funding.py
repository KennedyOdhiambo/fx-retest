"""Study C: crypto funding carry (cash-and-carry yield) on Binance USDT perps.

Tradable construction: long spot + short equal-notional perp. PnL = funding
received by the short leg (positive when funding > 0); price risk is hedged,
basis/venue risk remains. We measure the funding stream itself net of fees.

  policy A: hold the hedge continuously for the whole history
  policy B: hold only while trailing 7d average funding > 0.003% per event
            (fee charged every time the position toggles)

Fees: $10 (0.1% of a $10k book) per round trip/toggle. Funding = sum of the
per-8h events per day.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data/cache")
NOTIONAL = 10_000.0
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
THRESHOLD = 0.00003  # 0.003% per funding event


def load_funding(sym: str) -> pd.Series:
    df = pd.read_parquet(CACHE / f"{sym}_funding.parquet")
    f = df["funding"]
    if f.index.tz is None:
        f.index = f.index.tz_localize("UTC")
    return f.sort_index()


def main():
    rows = []
    daily: dict[str, pd.Series] = {}
    daily_b: dict[str, pd.Series] = {}
    print(f"{'sym':<9} {'start':<11} {'days':>5} {'annA%':>7} {'shA':>5} {'pos%':>5} "
          f"{'annB%':>7} {'togB':>5} {'worst30d$':>9} {'maxDD$':>8}", flush=True)
    for sym in SYMS:
        f = load_funding(sym)
        d = f.resample("D").sum().dropna()
        daily[sym] = d
        yrs = len(d) / 365.0
        sh_a = float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else float("nan")
        pos_days = float((d > 0).mean()) * 100
        pnl_a = d * NOTIONAL
        cum_a = pnl_a.cumsum()
        maxdd_a = float((cum_a - cum_a.cummax()).min())
        trail = f.rolling("7D").mean().resample("D").last().reindex(d.index)
        hold = (trail > THRESHOLD).fillna(False)
        pnl_b = d.where(hold, 0.0) * NOTIONAL
        toggles = int((hold.astype(int).diff().abs().fillna(0) > 0).sum())
        net_b = pnl_b.sum() - toggles * 10.0
        ann_b = net_b / NOTIONAL / yrs * 100
        daily_b[sym] = (d.where(hold, 0.0)
                        - (hold.astype(int).diff().abs().fillna(0) > 0) * 10.0 / NOTIONAL)
        worst30 = float(pnl_a.rolling(30).sum().min())
        # net policy A (one entry + one exit)
        net_a = pnl_a.sum() - 20.0
        ann_a_net = net_a / NOTIONAL / yrs * 100
        rows.append((sym, ann_a_net, sh_a, pos_days, ann_b, toggles, worst30, maxdd_a,
                     str(d.index[0].date()), len(d)))
        print(f"{sym:<9} {str(d.index[0].date()):<11} {len(d):>5} {ann_a_net:>7.2f} {sh_a:>5.2f} "
              f"{pos_days:>5.0f} {ann_b:>7.2f} {toggles:>5} {worst30:>9.1f} {maxdd_a:>8.1f}",
              flush=True)

    for tag, dd in [("A always", daily), ("B threshold", daily_b)]:
        panel = pd.DataFrame(dd).dropna()
        port = panel.mean(axis=1) * 365 * 100  # annualized % of book
        pnl = panel.mean(axis=1) * NOTIONAL
        cum = pnl.cumsum()
        sh = float(pnl.mean() / pnl.std() * np.sqrt(365)) if pnl.std() > 0 else float("nan")
        print(f"\nportfolio {tag}: {panel.index[0].date()} -> {panel.index[-1].date()} "
              f"({len(panel)} days) annualized={port.mean():.2f}% sharpe={sh:.2f} "
              f"total${pnl.sum():.0f} maxDD${float((cum - cum.cummax()).min()):.0f} "
              f"worst30d${float(pnl.rolling(30).sum().min()):.1f}", flush=True)
    panel = pd.DataFrame(daily)
    if len(panel) >= 700:
        yearly = (panel.mean(axis=1) * 365 * 100).resample("YE").mean()
        yearly.index = yearly.index.year
        print("portfolio A yearly mean ann%: " + "  ".join(f"{y}:{v:+.1f}" for y, v in yearly.items()), flush=True)

    # sensitivity: threshold x toggle-cost, pooled book, from common start
    panel_all = pd.DataFrame({s: load_funding(s).resample("D").sum().dropna() for s in SYMS}).dropna()
    raw = pd.DataFrame({s: load_funding(s) for s in SYMS})
    trail = raw.rolling("7D").mean().resample("D").last().reindex(panel_all.index)
    print(f"\nsensitivity (pooled, {panel_all.index[0].date()} -> {panel_all.index[-1].date()}):", flush=True)
    print(f"{'thresh/event':>14} {'cost$/tog':>9} {'ann%':>7} {'maxDD$':>8} {'worst30d$':>10} {'toggles':>8}", flush=True)
    for th in [0.0, 0.00003, 0.0001]:
        hold = (trail > th).fillna(False) if th > 0 else pd.DataFrame(True, index=panel_all.index, columns=SYMS)
        pnl = (panel_all.where(hold, 0.0) * NOTIONAL)
        tog = (hold.astype(int).diff().abs().fillna(0) > 0)
        for cost in [10.0, 20.0]:
            pnl_net = pnl.mean(axis=1) - (tog * cost).sum(axis=1) / len(SYMS)
            yrs = len(panel_all) / 365.0
            ann = float(pnl_net.sum() / NOTIONAL / yrs) * 100
            cum = pnl_net.cumsum()
            print(f"{th * 100:>13.3f}% {cost:>9.0f} {ann:>7.2f} "
                  f"{float((cum - cum.cummax()).min()):>8.1f} "
                  f"{float(pnl_net.rolling(30).sum().min()):>10.1f} {int(tog.sum().sum()):>8}",
                  flush=True)

    # split halves robustness (policy A / policy B 0.003% at $10 per toggle)
    hold_b = (trail > 0.00003).fillna(False)
    tog_b = (hold_b.astype(int).diff().abs().fillna(0) > 0) * 10.0
    mid = panel_all.index[len(panel_all) // 2]
    print("\nsplit halves (ann% / sharpe):", flush=True)
    for half, sel in [("1st", panel_all.index < mid), ("2nd", panel_all.index >= mid)]:
        pa = panel_all.loc[sel].mean(axis=1)
        pb = (panel_all.where(hold_b, 0.0).loc[sel].mean(axis=1)
              - tog_b.loc[sel].sum(axis=1) / len(SYMS) / NOTIONAL)
        for tag, s in [("A always", pa), ("B thresh", pb)]:
            yrs = max(len(s), 1) / 365.0
            ann = float(s.sum() / yrs) * 100
            sh = float(s.mean() / s.std() * np.sqrt(365)) if s.std() > 0 else float("nan")
            print(f"  {half} half {tag:<9} {s.index[0].date()} -> {s.index[-1].date()} "
                  f"ann={ann:>6.2f}% sharpe={sh:>5.2f}", flush=True)


if __name__ == "__main__":
    main()
