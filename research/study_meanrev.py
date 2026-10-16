"""Study B: daily mean reversion — canonical Connors RSI(2) dip-buying, long
only, on the 6 equity indices and the 27 US stocks; exit on close > SMA5 or
after 10 bars; optional 200-day trend filter. Benchmarked against equal-weight
buy-and-hold of the same instruments. Plus the XAU/XAG ratio z-score spread
probe (the most classic pair in the book).

Rules are canonical and fixed a priori; both RSI 5 and 10 thresholds reported.
The relevant falsification for a long-only timing overlay is vs buy-and-hold
and vs block-shuffled returns with identical positions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vecbt

INDICES = ["SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR", "UK100_GBP", "JP225_USD"]
STOCKS = ["AAPL", "AMD", "AMZN", "BA", "BAC", "CAT", "DIS", "GE", "GOOGL", "HD", "INTC",
          "JNJ", "JPM", "KO", "MA", "META", "MSFT", "NFLX", "NVDA", "PFE", "PG", "QCOM",
          "T", "TSLA", "UNH", "V", "XOM"]


def load_set(names):
    frames = {}
    for n in names:
        try:
            df = vecbt.load_daily(n)
        except FileNotFoundError:
            continue
        if len(df) >= 750:
            frames[n] = df
    return frames


def positions_for(frames, rsi_entry, sma_filter=None, max_hold=10, exit_ma=5):
    return {n: vecbt.meanrev_long(df["close"], rsi_entry=rsi_entry, exit_ma=exit_ma,
                                  max_hold=max_hold, sma_filter=sma_filter)
            for n, df in frames.items()}


def exposure(positions):
    tot = sum(len(p) for p in positions.values())
    act = sum(float((p > 0).sum()) for p in positions.values())
    return act / tot


def run_block(frames, label_prefix):
    n_frames = len(frames)
    print(f"\n--- {label_prefix}: {n_frames} instruments ---", flush=True)
    for tag, rsi_entry, sma in [("rsi5", 5.0, None), ("rsi10", 10.0, None),
                                ("rsi5_200", 5.0, 200), ("rsi10_200", 10.0, 200)]:
        pos = positions_for(frames, rsi_entry, sma)
        res = vecbt.simulate(frames, pos, cost_bps=3.0, kmin=1.0, kmax=1.0)
        vecbt.summary(res, f"mr_{tag} {label_prefix}")
        print(f"    exposure={exposure(pos):.1%} cost={res['cost_total']:.3f}", flush=True)
    bh = {n: np.ones(len(df)) for n, df in frames.items()}
    vecbt.summary(vecbt.simulate(frames, bh, cost_bps=3.0, kmin=1.0, kmax=1.0),
                  f"buyhold {label_prefix}")


def main():
    idx = load_set(INDICES)
    stk = load_set(STOCKS)
    both = {**idx, **stk}
    run_block(idx, "indices")
    run_block(stk, "stocks")
    run_block(both, "all")

    # falsification on the headline variant (rsi5_200, all instruments)
    pos = positions_for(both, 5.0, 200)
    gate = vecbt.random_gate(both, pos, cost_bps=3.0, kmin=1.0, kmax=1.0)
    print(f"\nMR gate rsi5_200 all: p={gate['p_value']:.3f} actual={gate['actual_net']:+.3f} "
          f"null_med={gate['null_median']:+.3f}", flush=True)

    # ---- XAU/XAG ratio z-score spread probe -------------------------------
    xau = vecbt.load_daily("XAU_USD")["close"]
    xag = vecbt.load_daily("XAG_USD")["close"]
    pair = pd.concat({"xau": xau, "xag": xag}, axis=1).dropna()
    ratio = (pair["xau"] / pair["xag"]).rename("close")
    z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()
    zz = z.to_numpy()
    pos = np.zeros(len(ratio))
    st = 0
    for t in range(len(zz)):
        if np.isnan(zz[t]):
            pos[t] = st
            continue
        if st == 0:
            if zz[t] < -1.0:
                st = 1
            elif zz[t] > 1.0:
                st = -1
        elif st == 1 and zz[t] > 1.0:
            st = -1
        elif st == -1 and zz[t] < -1.0:
            st = 1
        elif abs(zz[t]) < 0.2:
            st = 0
        pos[t] = st
    frame = {"XAU_XAG": pd.DataFrame({"close": ratio})}
    res = vecbt.simulate(frame, {"XAU_XAG": pos}, cost_bps=3.0, kmin=1.0, kmax=1.0)
    vecbt.summary(res, "ratio_z60 XAU/XAG")
    gate = vecbt.random_gate(frame, {"XAU_XAG": pos}, cost_bps=3.0, kmin=1.0, kmax=1.0)
    print(f"    ratio gate: p={gate['p_value']:.3f} actual={gate['actual_net']:+.3f} "
          f"null_med={gate['null_median']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
