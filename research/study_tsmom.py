"""Study A: trend/momentum breadth — canonical Donchian 50/20 state machine and
12-month time-series momentum across every asset class in the cache.

Portfolio-level, vol-scaled equal weights (vecbt.simulate); costs per position
change; block-bootstrap random gate against the same positions. Rules are
fixed/canonical a priori — no per-instrument knobs, no cherry-picking.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vecbt

# documented data-artifact mask: the WTI CFD pinned at 0.009 on the day the
# front contract settled at -$37 (roll artifact); dropped so the instrument is
# flat across the event instead of booking a fake -99.95%.
MASKS = {"WTI_USD": ("2020-04-19", "2020-04-19")}


def load(name: str) -> pd.DataFrame:
    df = vecbt.load_daily(name)
    if name in MASKS:
        a, b = (pd.Timestamp(x, tz="UTC") for x in MASKS[name])
        df = df[(df.index < a) | (df.index >= b + pd.Timedelta(days=1))]
    return df

CLASSES = {
    "equity_index": ["SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR", "UK100_GBP", "JP225_USD"],
    "commodity": ["XAU_USD", "XAG_USD", "WTI_USD", "BRENT_USD", "COPPER_USD", "PLATINUM_USD",
                  "PALLADIUM_USD", "NATGAS_USD", "CORN_USD", "SOYBEAN_USD", "WHEAT_USD",
                  "COFFEE_USD", "SUGAR_USD", "COCOA_USD", "COTTON_USD"],
    "fx": ["AUD_USD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_USD", "GBP_JPY", "GBP_USD",
           "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY", "USD_NOK"],
    "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
               "AVAXUSDT", "LINKUSDT", "LTCUSDT"],
    "bond": ["FGBL"],
    "us_stock": ["AAPL", "AMD", "AMZN", "BA", "BAC", "CAT", "DIS", "GE", "GOOGL", "HD", "INTC",
                 "JNJ", "JPM", "KO", "MA", "META", "MSFT", "NFLX", "NVDA", "PFE", "PG", "QCOM",
                 "T", "TSLA", "UNH", "V", "XOM"],
}


def build(rule: str):
    frames, positions = {}, {}
    for _cls, names in CLASSES.items():
        for name in names:
            try:
                df = load(name)
            except FileNotFoundError:
                print(f"  missing {name}")
                continue
            if len(df) < 750:
                print(f"  thin {name}: {len(df)} bars")
                continue
            frames[name] = df
            positions[name] = (vecbt.donchian_state(df) if rule == "donchian"
                               else vecbt.tsmom_monthly(df["close"]))
    return frames, positions


def leg_split(res):
    longs = sum(p for _, s, p in res["episodes"] if s > 0)
    shorts = sum(p for _, s, p in res["episodes"] if s < 0)
    n_long = sum(1 for _, s, _ in res["episodes"] if s > 0)
    n_short = len(res["episodes"]) - n_long
    return longs, shorts, n_long, n_short


def main():
    for rule in ("donchian_50_20", "tsmom_252"):
        base = "donchian" if rule.startswith("donchian") else "tsmom"
        frames, positions = build(base)
        print(f"\n=== {rule} | {len(frames)} instruments | window 2005-01-01 -> 2026 ===", flush=True)
        for cls in CLASSES:
            subset = [n for n in CLASSES[cls] if n in frames]
            if not subset:
                continue
            f = {n: frames[n] for n in subset}
            p = {n: positions[n] for n in subset}
            res = vecbt.simulate(f, p, cost_bps=3.0)
            vecbt.summary(res, f"{rule} {cls}")
            longs, shorts, nl, ns = leg_split(res)
            print(f"    legs: long {nl} tr {longs:+.3f} | short {ns} tr {shorts:+.3f} | "
                  f"cost {res['cost_total']:.3f}", flush=True)
        res_all = vecbt.simulate(frames, positions, cost_bps=3.0)
        vecbt.summary(res_all, f"{rule} ALL 3bps")
        vecbt.summary(vecbt.simulate(frames, positions, cost_bps=6.0), f"{rule} ALL 6bps")
        vecbt.yearly_table(res_all)
        gate = vecbt.random_gate(frames, positions, cost_bps=3.0)
        print(f"    gate: p={gate['p_value']:.3f} actual={gate['actual_net']:+.3f} "
              f"null_med={gate['null_median']:+.3f}", flush=True)

    # retail classic: long-only 200-day SMA trend filter on indices + stocks
    names = CLASSES["equity_index"] + CLASSES["us_stock"]
    frames = {}
    for n in names:
        try:
            df = load(n)
        except FileNotFoundError:
            continue
        if len(df) >= 750:
            frames[n] = df
    pos = {n: (df["close"] > df["close"].rolling(200).mean()).astype(float).to_numpy()
           for n, df in frames.items()}
    print(f"\n=== sma200_longonly | {len(frames)} instruments | equity indices + US stocks ===", flush=True)
    res = vecbt.simulate(frames, pos, cost_bps=3.0, kmin=1.0, kmax=1.0)
    vecbt.summary(res, "sma200 eq+stk")
    exposure = np.mean([float((p > 0).mean()) for p in pos.values()])
    print(f"    exposure={exposure:.1%}", flush=True)
    gate = vecbt.random_gate(frames, pos, cost_bps=3.0, kmin=1.0, kmax=1.0)
    print(f"    gate: p={gate['p_value']:.3f} actual={gate['actual_net']:+.3f} "
          f"null_med={gate['null_median']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
