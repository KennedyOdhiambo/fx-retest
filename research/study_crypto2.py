"""Study D2: detail pass on the two gate-passers from the crypto grid.

- obv_30 (OBV trend) and mom30_highvol (volume-gated momentum): yearly tables,
  per-symbol sharpes, long/short leg split.
- OBV parameter neighborhood: lookback 10-60, an OBV-vs-SMA variant, and a
  volume-weighted-return variant (vwr) — does the effect survive the
  neighborhood or is it knife-edge?
- OBV on each symbol's own full history (BTC etc. back to 2019).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vecbt
from study_crypto import COST, FRAMES, FUNDING, KMAX, KMIN, SYMS, VOL_T, sig_mom_volgate, sig_obv


def sim(positions, cost=COST, frames=None):
    f = frames or FRAMES
    p = {k: v for k, v in positions.items() if k in f}
    return vecbt.simulate(f, p, cost_bps=cost, vol_target=VOL_T, kmin=KMIN, kmax=KMAX,
                          funding=FUNDING)


def yearly(net: pd.Series, label: str):
    yr = (1.0 + net).resample("YE").prod() - 1.0
    yr.index = yr.index.year
    print(f"  {label:<18} " + "  ".join(f"{y}:{v:+.0%}" for y, v in yr.items()), flush=True)


def sharpe_of(net: pd.Series) -> float:
    sd = net.std()
    return float(net.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def details(name, positions):
    res = sim(positions)
    vecbt.summary(res, name, periods_per_year=365)
    yearly(res["net"], name)
    longs = sum(p for _, s, p in res["episodes"] if s > 0)
    shorts = sum(p for _, s, p in res["episodes"] if s < 0)
    print(f"  legs: long {longs:+.2f} short {shorts:+.2f} | funding "
          f"{sum(res['funding_by_name'].values()):+.2f}", flush=True)
    per = {}
    eps_by = {}
    for nm, _, pnl in res["episodes"]:
        eps_by[nm] = eps_by.get(nm, 0.0) + pnl
    for s in SYMS:
        # single-symbol sim on its own full history
        rr = sim({s: positions[s]}, frames={s: FRAMES[s]})
        per[s] = sharpe_of(rr["net"])
    line = "  ".join(f"{s[:3]}:{v:+.2f}" for s, v in per.items())
    print(f"  per-sym sharpe (own history): {line}", flush=True)
    pos_count = sum(1 for nm in SYMS if eps_by.get(nm, 0.0) - res["costs_by_name"].get(nm, 0.0) > 0)
    print(f"  net-positive symbols (pooled window): {pos_count}/10", flush=True)


def obv_variant(df, k):
    return sig_obv(df, k)


def obv_ma_variant(df, k=30):
    d = np.sign(df["close"].diff()).fillna(0.0) * df["volume"]
    obv = d.cumsum()
    return np.sign(obv - obv.rolling(k).mean()).fillna(0.0).to_numpy()


def vwr_variant(df, k=20):
    r = df["close"].pct_change()
    num = (r * df["volume"]).rolling(k).sum()
    den = df["volume"].rolling(k).sum()
    return np.sign(num / den).fillna(0.0).to_numpy()


def main():
    print("=== detail: OBV trend ===", flush=True)
    details("obv_30", {s: sig_obv(FRAMES[s], 30) for s in SYMS})

    print("\n=== detail: mom30_highvol ===", flush=True)
    details("mom30_highvol", {s: sig_mom_volgate(FRAMES[s], 30, 0.5, 20, "high") for s in SYMS})

    print("\n=== OBV parameter neighborhood (pooled, 5bps) ===", flush=True)
    for k in [10, 15, 20, 30, 45, 60]:
        res = sim({s: obv_variant(FRAMES[s], k) for s in SYMS})
        print(f"  obv_{k:<3} " + _oneline(res), flush=True)
    res = sim({s: obv_ma_variant(FRAMES[s], 30) for s in SYMS})
    print("  obv_vs_sma30 " + _oneline(res), flush=True)
    for k in [10, 20, 30]:
        res = sim({s: vwr_variant(FRAMES[s], k) for s in SYMS})
        print(f"  vwr_{k:<3} " + _oneline(res), flush=True)

    print("\n=== finalists kill tests (365-annualized) ===", flush=True)
    finals = {
        "obv_30": {s: sig_obv(FRAMES[s], 30) for s in SYMS},
        "vwr_30": {s: vwr_variant(FRAMES[s], 30) for s in SYMS},
        "mom30_highvol": {s: sig_mom_volgate(FRAMES[s], 30, 0.5, 20, "high") for s in SYMS},
    }
    for name, pos in finals.items():
        print(f"\n{name}:", flush=True)
        for cb in [10.0, 20.0]:
            res = sim(pos, cost=cb)
            print(f"  cost{cb:>2.0f} " + _oneline(res), flush=True)
        gate = vecbt.random_gate(FRAMES, pos, cost_bps=COST, vol_target=VOL_T, kmin=KMIN,
                                 kmax=KMAX, funding=FUNDING)
        print(f"  gate p={gate['p_value']:.3f} actual={gate['actual_net']:+.2f} "
              f"null={gate['null_median']:+.2f}", flush=True)
        res = sim(pos)
        net = res["net"]
        n3 = len(net) // 3
        thirds = [sharpe_of(net.iloc[i * n3:(i + 1) * n3]) for i in range(3)]
        print("  thirds: " + "  ".join(f"{t:+.2f}" for t in thirds), flush=True)
        loo = []
        for drop in SYMS:
            f = {k: v for k, v in FRAMES.items() if k != drop}
            pp = {k: v for k, v in pos.items() if k != drop}
            rr = vecbt.simulate(f, pp, cost_bps=COST, vol_target=VOL_T, kmin=KMIN, kmax=KMAX,
                                funding=FUNDING)
            loo.append(sharpe_of(rr["net"]))
        print(f"  leave-one-out: min {min(loo):+.2f} max {max(loo):+.2f}", flush=True)
        lo = {s: np.clip(p, 0, None) for s, p in pos.items()}
        print("  long-only: " + _oneline(sim(lo)), flush=True)


def _oneline(res) -> str:
    net = res["net"]
    sd = net.std()
    sh = float(net.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0
    eps = pd.Series([p for _, _, p in res["episodes"]])
    pf = eps[eps > 0].sum() / abs(eps[eps < 0].sum())
    mid = len(net) // 2
    h1 = sharpe_of(net.iloc[:mid])
    h2 = sharpe_of(net.iloc[mid:])
    return f"sharpe {sh:+.2f} pf {pf:.2f} trades {len(eps):>5} halves {h1:+.2f}/{h2:+.2f}"


if __name__ == "__main__":
    main()
