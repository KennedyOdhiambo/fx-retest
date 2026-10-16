"""Fetch Binance USDT-perp (fapi) klines with the full field set — volume,
quote volume, trade count, taker buy volume — for the 10 perps used in the
crypto studies. Daily + H1, cached as {SYM}_PF_D.parquet / {SYM}_PF_H1.parquet
(the pre-existing {SYM}_D/H1 files are Binance *spot*; perps are the tradable
instrument and match the funding data).

Rate limits: /fapi/v1/klines weight 10 at limit=1500; ~0.25s pacing keeps us
well under 2400 weight/min. Re-runs skip files that already exist.
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
BASE = "https://fapi.binance.com/fapi/v1/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch(symbol: str, interval: str, start_ms: int) -> list:
    rows = []
    while True:
        url = f"{BASE}?symbol={symbol}&interval={interval}&limit=1500&startTime={start_ms}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read().decode())
        if not batch:
            break
        rows += batch
        if len(batch) < 1500:
            break
        start_ms = batch[-1][0] + 1
        time.sleep(0.25)
    return rows


def to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["close_time", "ignore"]).set_index("open_time")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    return df.sort_index()


def main():
    out = Path("data/cache")
    start = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)
    for sym in SYMS:
        for tf, name in [("1d", "D"), ("1h", "H1")]:
            path = out / f"{sym}_PF_{name}.parquet"
            if path.exists():
                print(f"skip {path}")
                continue
            try:
                df = to_df(fetch(sym, tf, start))
                df.to_parquet(path)
                print(f"OK {sym} {tf}: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}",
                      flush=True)
            except Exception as e:
                print(f"FAIL {sym} {tf}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
