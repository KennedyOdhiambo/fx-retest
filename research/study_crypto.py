"""Study D: crypto-only lab on Binance USDT perps (10 symbols, 2019-2026).

Data: {SYM}_PF_D.parquet (fapi klines: OHLC + volume + quote volume + trades +
taker buy base/quote) and {SYM}_funding.parquet. Signals on daily closes,
applied next bar; equal 1/N, per-instrument vol scaling to 25% standalone
(kmin 0.05, kmax 3.0); costs 5 bps one-way per position change (Binance taker
= 5 bps); funding charged on the position (longs pay, shorts receive).

Pre-declared selection rule for kill-tests: pooled Sharpe >= 0.60, >= 7/10
symbols positive, both sample halves positive. Everything else is exploration
and gets reported either way.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vecbt

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
VOL_T = 0.25
KMIN, KMAX = 0.05, 3.0
COST = 5.0


def load(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/cache/{sym}_PF_D.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def load_funding(sym: str) -> pd.Series:
    f = pd.read_parquet(f"data/cache/{sym}_funding.parquet")["funding"]
    if f.index.tz is None:
        f.index = f.index.tz_localize("UTC")
    return f.resample("D").sum()


FRAMES = {s: load(s) for s in SYMS}
FUNDING = {s: load_funding(s) for s in SYMS}


# ------------------------------------------------------------------ signals

def sig_donchian(df, e=55, x=20):
    return vecbt.donchian_state(df, entry=e, exit_=x)


def _channel_entries(df, e, mult, cond):
    """Donchian with a volume condition on entries only."""
    hi, lo, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    vma = df["volume"].rolling(20).mean().to_numpy()
    v = df["volume"].to_numpy()
    state = np.zeros(n)
    st = 0
    for t in range(e + 1, n):
        if st == 0:
            if c[t] > hi[t - e:t].max() and cond(v[t], vma[t], mult):
                st = 1
            elif c[t] < lo[t - e:t].min() and cond(v[t], vma[t], mult):
                st = -1
        elif st == 1:
            if c[t] < lo[t - 20:t].min():
                st = 0
        else:
            if c[t] > hi[t - 20:t].max():
                st = 0
        state[t] = st
    return state


def sig_dc_volconfirm(df, e=55, mult=1.5):
    return _channel_entries(df, e, mult, lambda v, vm, m: v > m * vm)


def sig_dc_drybreak(df, e=20, mult=0.7):
    return _channel_entries(df, e, mult, lambda v, vm, m: v < m * vm)


def sig_sma_cross(df, f=20, s=100):
    c = df["close"]
    return np.sign(c.rolling(f).mean() - c.rolling(s).mean()).fillna(0.0).to_numpy()


def sig_mom(df, k=30):
    c = df["close"]
    return np.sign(c - c.shift(k)).fillna(0.0).to_numpy()


def sig_mom_volgate(df, k=30, vz=0.5, win=20, side="high"):
    c = df["close"]
    vr = (df["volume"] - df["volume"].rolling(win).mean()) / df["volume"].rolling(win).std()
    s = np.sign(c - c.shift(k))
    keep = (vr > vz) if side == "high" else (vr < -vz)
    return s.where(keep, 0.0).fillna(0.0).to_numpy()


def sig_bb_break(df, n=20, k=2.0):
    c = df["close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    cn, man, upn, dnn = c.to_numpy(), ma.to_numpy(), (ma + k * sd).to_numpy(), (ma - k * sd).to_numpy()
    state = np.zeros(len(cn))
    st = 0
    for t in range(n, len(cn)):
        if st == 0:
            if cn[t] > upn[t]:
                st = 1
            elif cn[t] < dnn[t]:
                st = -1
        elif (st == 1 and cn[t] < man[t]) or (st == -1 and cn[t] > man[t]):
            st = 0
        state[t] = st
    return state


def sig_bb_revert(df, n=20, k=2.0):
    c = df["close"]
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    cn, man, dnn = c.to_numpy(), ma.to_numpy(), (ma - k * sd).to_numpy()
    pos = np.zeros(len(cn))
    st = 0
    for t in range(n, len(cn)):
        if st == 0 and cn[t] < dnn[t]:
            st = 1
        elif st == 1 and cn[t] > man[t]:
            st = 0
        pos[t] = st
    return pos


def sig_rsi2(df, th=10.0, hold=10, exit_ma=10):
    return vecbt.meanrev_long(df["close"], rsi_entry=th, exit_ma=exit_ma, max_hold=hold)


def sig_drop_buy(df, z0=-2.0, hold=3, win=60):
    r = df["close"].pct_change()
    z = ((r - r.rolling(win).mean()) / r.rolling(win).std()).to_numpy()
    n = len(z)
    pos = np.zeros(n)
    st, left = 0, 0
    for t in range(win + 1, n):
        if st == 0 and z[t] == z[t] and z[t] < z0:
            st, left = 1, hold
        elif st == 1:
            left -= 1
            if left <= 0:
                st = 0
        pos[t] = st
    return pos


def sig_revert_1d(df):
    r = df["close"].pct_change()
    return -np.sign(r).fillna(0.0).to_numpy()


def sig_volspike_revert(df, vz=2.0, hold=3, win=20):
    r = df["close"].pct_change()
    vr = (df["volume"] - df["volume"].rolling(win).mean()) / df["volume"].rolling(win).std()
    trig = ((vr > vz) & (r < 0)).to_numpy()
    n = len(trig)
    pos = np.zeros(n)
    st, left = 0, 0
    for t in range(win + 1, n):
        if st == 0 and trig[t]:
            st, left = 1, hold
        elif st == 1:
            left -= 1
            if left <= 0:
                st = 0
        pos[t] = st
    return pos


def _taker_imbalance(df):
    return (2.0 * df["taker_buy_base"] - df["volume"]) / df["volume"]


def sig_taker_mom(df, k=5):
    imb = _taker_imbalance(df).rolling(k).sum()
    return np.sign(imb).fillna(0.0).to_numpy()


def sig_taker_fade(df, z=2.0, hold=3, win=20):
    imb = _taker_imbalance(df)
    zz = ((imb - imb.rolling(win).mean()) / imb.rolling(win).std()).to_numpy()
    n = len(zz)
    pos = np.zeros(n)
    st, left, direction = 0, 0, 0.0
    for t in range(win + 1, n):
        if st == 0:
            if zz[t] == zz[t] and abs(zz[t]) > z:
                st, left, direction = 1, hold, -float(np.sign(zz[t]))
        else:
            left -= 1
            if left <= 0:
                st = 0
        pos[t] = direction if st == 1 else 0.0
    return pos


def sig_obv(df, k=30):
    d = np.sign(df["close"].diff()).fillna(0.0) * df["volume"]
    obv = d.cumsum()
    return np.sign(obv - obv.shift(k)).fillna(0.0).to_numpy()


def sig_vwap_mom(df, k=10):
    pv = (df["close"] * df["volume"]).rolling(k).sum()
    v = df["volume"].rolling(k).sum()
    return np.sign(df["close"] - pv / v).fillna(0.0).to_numpy()


VARIANTS = {
    # trend
    "dc_55_20": lambda df: sig_donchian(df, 55, 20),
    "dc_100_50": lambda df: sig_donchian(df, 100, 50),
    "sma_20_100": lambda df: sig_sma_cross(df, 20, 100),
    "sma_50_200": lambda df: sig_sma_cross(df, 50, 200),
    "mom_7": lambda df: sig_mom(df, 7),
    "mom_30": lambda df: sig_mom(df, 30),
    "mom_90": lambda df: sig_mom(df, 90),
    "bb_break_20": lambda df: sig_bb_break(df, 20, 2.0),
    # volume family
    "dc_volconfirm": lambda df: sig_dc_volconfirm(df, 55, 1.5),
    "dc_drybreak": lambda df: sig_dc_drybreak(df, 20, 0.7),
    "mom30_highvol": lambda df: sig_mom_volgate(df, 30, 0.5, 20, "high"),
    "mom30_lowvol": lambda df: sig_mom_volgate(df, 30, 0.5, 20, "low"),
    "vwap_mom_10": lambda df: sig_vwap_mom(df, 10),
    "obv_30": lambda df: sig_obv(df, 30),
    # taker flow
    "taker_mom_5": lambda df: sig_taker_mom(df, 5),
    "taker_fade_2z": lambda df: sig_taker_fade(df, 2.0, 3, 20),
    # mean reversion
    "rsi2_10": lambda df: sig_rsi2(df, 10.0, 10, 10),
    "drop_z2_hold3": lambda df: sig_drop_buy(df, -2.0, 3, 60),
    "bb_revert_20": lambda df: sig_bb_revert(df, 20, 2.0),
    "volspike_revert": lambda df: sig_volspike_revert(df, 2.0, 3, 20),
    "revert_1d": sig_revert_1d,
}


# ------------------------------------------------------------------ audit

def audit():
    panel = pd.DataFrame({s: FRAMES[s]["close"] for s in SYMS}).dropna()
    rets = panel.pct_change().dropna()
    print(f"=== AUDIT | pooled window {panel.index[0].date()} -> {panel.index[-1].date()} "
          f"({len(rets)} days) ===", flush=True)
    vols = {s: rets[s].std() * np.sqrt(365) for s in SYMS}
    print("ann vol: " + "  ".join(f"{s[:3]}:{v:.0%}" for s, v in vols.items()), flush=True)
    cc = rets.corr().values[np.triu_indices(len(SYMS), 1)]
    print(f"mean cross-corr {cc.mean():.2f} | mean corr to BTC "
          f"{np.mean([rets[s].corr(rets['BTCUSDT']) for s in SYMS if s != 'BTCUSDT']):.2f}", flush=True)
    ac = "  ".join(f"l{k}:{np.mean([rets[s].autocorr(k) for s in SYMS]):+.3f}" for k in [1, 2, 3, 5])
    print(f"return autocorr (pooled): {ac}", flush=True)
    print(f"|ret| autocorr l1 {np.mean([rets[s].abs().autocorr(1) for s in SYMS]):+.2f} "
          f"(vol clustering)", flush=True)
    wd = rets.groupby(rets.index.dayofweek).mean().mean(axis=1) * 100
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("weekday mean %/d: " + "  ".join(f"{d}:{v:+.3f}" for d, v in
          zip(days, wd, strict=False)), flush=True)

    volp = pd.DataFrame({s: np.log1p(FRAMES[s]["volume"]) for s in SYMS}).dropna()
    r_abs = pd.DataFrame({s: rets[s].abs() for s in SYMS})
    vz = (volp - volp.mean()) / volp.std()
    r_abs_z = (r_abs - r_abs.mean()) / r_abs.std()
    common = vz.index.intersection(r_abs_z.index)
    print(f"volume: |ret| vs log-vol corr (pooled z) "
          f"{np.mean([vz.loc[common, s].corr(r_abs_z.loc[common, s]) for s in SYMS]):+.2f} | "
          f"log-vol autocorr l1 {np.mean([volp[s].autocorr(1) for s in SYMS]):+.2f}", flush=True)

    imb = pd.DataFrame({s: _taker_imbalance(FRAMES[s]) for s in SYMS}).dropna()
    rats = "  ".join(f"{s[:3]}:{imb[s].mean():+.3f}" for s in SYMS)
    print(f"taker imbalance mean: {rats}", flush=True)
    print(f"taker: mean std {imb.std().mean():.3f} | corr(imb, ret) pooled "
          f"{np.mean([imb[s].corr(rets[s].reindex(imb.index)) for s in SYMS]):+.2f} | "
          f"autocorr l1 {np.mean([imb[s].autocorr(1) for s in SYMS]):+.2f}", flush=True)

    fund = pd.DataFrame({s: FUNDING[s] for s in SYMS}).dropna()
    print(f"funding: mean {fund.mean().mean() * 365:.1%}/yr | % positive days "
          f"{(fund > 0).mean().mean():.0%} | BTC {fund['BTCUSDT'].mean() * 365:.1%}/yr", flush=True)


# ------------------------------------------------------------------ grid

def half_sharpes(net: pd.Series) -> tuple[float, float]:
    mid = len(net) // 2
    out = []
    for part in (net.iloc[:mid], net.iloc[mid:]):
        sd = part.std()
        out.append(float(part.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0)
    return out[0], out[1]


def run_grid():
    print(f"\n=== GRID | {len(VARIANTS)} variants | cost {COST:.0f}bps | vol_t {VOL_T} | "
          f"funding charged ===", flush=True)
    header = (f"{'variant':<16} {'sharpe':>6} {'pf':>5} {'trades':>6} {'mdd':>7} {'h1':>5} {'h2':>5} "
              f"{'pos/10':>6} {'fund$':>7} {'cost$':>6}")
    print(header, flush=True)
    results = {}
    for vname, fn in VARIANTS.items():
        pos = {}
        for s in SYMS:
            try:
                p = fn(FRAMES[s])
                p = np.nan_to_num(np.asarray(p, dtype=float))
                pos[s] = p[: len(FRAMES[s])] if len(p) > len(FRAMES[s]) else p
            except Exception as exc:  # noqa: BLE001
                print(f"  {vname} {s} FAIL {type(exc).__name__}: {exc}", flush=True)
                pos = None
                break
        if pos is None:
            continue
        res = vecbt.simulate(FRAMES, pos, cost_bps=COST, vol_target=VOL_T, kmin=KMIN,
                             kmax=KMAX, funding=FUNDING)
        s = vecbt.summary(res, vname, periods_per_year=365)
        h1, h2 = half_sharpes(res["net"])
        eps_by = {}
        for nm, _, pnl in res["episodes"]:
            eps_by[nm] = eps_by.get(nm, 0.0) + pnl
        pos_syms = sum(1 for nm in SYMS
                       if eps_by.get(nm, 0.0) - res["costs_by_name"].get(nm, 0.0) > 0)
        fund_tot = sum(res["funding_by_name"].values())
        print(f"{vname:<16} {s['sharpe']:>6.2f} {s['pf']:>5.2f} {s['trades']:>6} {s['mdd']:>7.1%} "
              f"{h1:>5.2f} {h2:>5.2f} {pos_syms:>5}/10 {fund_tot:>7.2f} {res['cost_total']:>6.2f}",
              flush=True)
        results[vname] = {"res": res, "pos": pos, "summary": s, "h1": h1, "h2": h2,
                          "pos_syms": pos_syms}
    return results


def kill_tests(results):
    print("\n=== KILL TESTS (rule: sharpe>=0.60, pos>=7/10, both halves>0) ===", flush=True)
    for vname, r in results.items():
        s = r["summary"]
        if not (s["sharpe"] >= 0.60 and r["pos_syms"] >= 7 and r["h1"] > 0 and r["h2"] > 0):
            continue
        print(f"\n{vname}: sharpe {s['sharpe']:.2f}, {r['pos_syms']}/10 symbols, "
              f"halves {r['h1']:.2f}/{r['h2']:.2f}", flush=True)
        for cb in [10.0, 20.0]:
            rr = vecbt.simulate(FRAMES, r["pos"], cost_bps=cb, vol_target=VOL_T, kmin=KMIN,
                                kmax=KMAX, funding=FUNDING)
            vecbt.summary(rr, f"  {vname} cost{cb:.0f}", periods_per_year=365)
        gate = vecbt.random_gate(FRAMES, r["pos"], cost_bps=COST, vol_target=VOL_T,
                                 kmin=KMIN, kmax=KMAX, funding=FUNDING)
        print(f"  gate: p={gate['p_value']:.3f} actual={gate['actual_net']:+.2f} "
              f"null_med={gate['null_median']:+.2f}", flush=True)
        loo = []
        for drop in SYMS:
            frames = {k: v for k, v in FRAMES.items() if k != drop}
            poss = {k: v for k, v in r["pos"].items() if k != drop}
            rr = vecbt.simulate(frames, poss, cost_bps=COST, vol_target=VOL_T, kmin=KMIN,
                                kmax=KMAX, funding=FUNDING)
            net = rr["net"]
            sd = net.std()
            loo.append(float(net.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0)
        print(f"  leave-one-out sharpe: min {min(loo):+.2f} max {max(loo):+.2f}", flush=True)


def benchmark():
    print("\n=== benchmarks ===", flush=True)
    pos = {s: np.ones(len(FRAMES[s])) for s in SYMS}
    res = vecbt.simulate(FRAMES, pos, cost_bps=COST, vol_target=VOL_T, kmin=KMIN, kmax=KMAX,
                         funding=FUNDING)
    vecbt.summary(res, "buyhold_all_long", periods_per_year=365)
    print(f"  funding drag on B&H: {sum(res['funding_by_name'].values()):+.2f} "
          f"(portfolio return units)", flush=True)
    pos = {s: -np.ones(len(FRAMES[s])) for s in SYMS}
    res = vecbt.simulate(FRAMES, pos, cost_bps=COST, vol_target=VOL_T, kmin=KMIN, kmax=KMAX,
                         funding=FUNDING)
    vecbt.summary(res, "buyhold_all_short", periods_per_year=365)


def main():
    audit()
    results = run_grid()
    kill_tests(results)
    benchmark()


if __name__ == "__main__":
    main()
