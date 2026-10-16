"""Study D3: horizon scan + recent robustness for the crypto volume-momentum
finalists (vwr = volume-weighted return momentum, obv = OBV trend).

Checks whether the 30-day horizon is a plateau or a cherry-pick spike, and how
the last 12 months alone look. Pooled sim, 5 bps, funding charged.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vecbt
from study_crypto import COST, FRAMES, FUNDING, KMAX, KMIN, SYMS, VOL_T, sig_obv
from study_crypto2 import vwr_variant


def sharpe_of(net: pd.Series) -> float:
    sd = net.std()
    return float(net.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def sim(positions):
    return vecbt.simulate(FRAMES, positions, cost_bps=COST, vol_target=VOL_T, kmin=KMIN,
                          kmax=KMAX, funding=FUNDING)


def row(label: str, res) -> None:
    net = res["net"]
    eps = pd.Series([p for _, _, p in res["episodes"]])
    pf = eps[eps > 0].sum() / abs(eps[eps < 0].sum())
    last = sharpe_of(net.iloc[-365:])
    mid = len(net) // 2
    h1, h2 = sharpe_of(net.iloc[:mid]), sharpe_of(net.iloc[mid:])
    print(f"{label:<14} sharpe {sharpe_of(net):+.2f}  pf {pf:.2f}  trades {len(eps):>5}  "
          f"halves {h1:+.2f}/{h2:+.2f}  last12m {last:+.2f}", flush=True)


def main():
    print("=== vwr horizon scan (long/short) ===", flush=True)
    for k in [20, 25, 30, 35, 40, 45]:
        row(f"vwr_{k}", sim({s: vwr_variant(FRAMES[s], k) for s in SYMS}))
    print("\n=== vwr horizon scan (long-only) ===", flush=True)
    for k in [20, 25, 30, 35, 40, 45]:
        pos = {s: np.clip(vwr_variant(FRAMES[s], k), 0, None) for s in SYMS}
        row(f"vwr_{k}_lo", sim(pos))
    print("\n=== obv horizon scan (long-only) ===", flush=True)
    for k in [20, 25, 30, 35, 40, 45]:
        pos = {s: np.clip(sig_obv(FRAMES[s], k), 0, None) for s in SYMS}
        row(f"obv_{k}_lo", sim(pos))


if __name__ == "__main__":
    main()
