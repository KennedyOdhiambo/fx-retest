# Research — structural variants vs the retest book

Ad-hoc (but reproducible) research tooling used to stress-test the retest
strategy and two structural rewrites on cached H1/D data, across two
universes: the 11-pair FX/metal/BTC book and six index CFDs, plus (2026-10)
cross-asset breadth studies — trend/momentum, mean reversion, crypto funding
carry — over 71 instruments in six asset classes, and a crypto-only lab on
Binance USDT perps (volume-momentum on daily bars, H1 data staged for next
phase). Run everything **from the repo root** (data loading uses relative
cache paths).

## Scripts

| script | what it does |
|---|---|
| `alpha_sweep.py` | one-knob + combo sweeps for the daily-level **retest** book (single|combo modes) |
| `alpha_sweep2.py` | smoke + knob sweeps for the structural variants: MTF retest, **breakout-continuation**, **fade** (`smoke|sweep|fade|h1idx` modes; `SWEEP_UNIVERSE=SYM1,SYM2,...` overrides the universe) |
| `sweep_final.py` | kill-test for a retest config: full sample, OOS last 25%, walk-forward windows, random-entry gate |
| `sweep_final2.py` | same kill-test for the variants: `python research/sweep_final2.py <strat> '<r_over json>' ['<bt_over json>']` |
| `sweep_daily.py` | daily-history long-run test (2002-2026): breakout + retest on daily bars, session off, one signal pass per symbol, yearly/quarterly regime view (`quick|full|costs` modes) |
| `sweep_bund.py` | Euro-Bund (FGBL) intraday test: M5 execution, levels from D/H4/H1/M15/M5, optional HTF SMA trend gates (`smoke|gate|final` modes) |
| `vecbt.py` | vectorized daily portfolio backtester for breadth studies: signals on closes, applied next bar, equal 1/N weights, per-instrument vol scaling, costs per position change, block-bootstrap random gate |
| `study_tsmom.py` | **Study A (trend breadth)**: Donchian 50/20 + 12m TSMOM across 71 instruments / 6 classes, 2005-2026, gate + cost sensitivity + SMA200 long-only retail filter |
| `study_meanrev.py` | **Study B (mean reversion)**: Connors RSI(2) dip-buy on 6 indices + 27 US stocks vs equal-weight buy-and-hold; XAU/XAG ratio z-score spread probe |
| `study_funding.py` | **Study C (funding carry)**: Binance USDT-perp funding, hedged spot+perp carry, always-on vs threshold policies, fee/threshold sensitivity, split halves |
| `fetch_binance.py` | pulls full-history fapi klines (D + H1) with volume/quote-volume/trades/taker-buy fields → `{SYM}_PF_{D,H1}.parquet` (perp data for the crypto lab) |
| `study_crypto.py` | **Study D (crypto grid)**: deep audit + 21 technical variants (trend/volume/taker-flow/MR) on 10 Binance perps, funding charged, kill-test block for qualifiers |
| `study_crypto2.py` | **Study D2**: detail pass on the gate-passers (yearly, per-symbol, legs) + OBV/VWR parameter neighborhood + full kill tests (gate, cost stress, thirds, LOO, long-only) |
| `study_crypto3.py` | **Study D3**: horizon scan (20-45d) + last-12m robustness for the volume-momentum finalists |

Variant strategies themselves live in `src/retest/strategies/`
(`mtf.py`, `breakout.py`, `fade.py`), tested in `tests/test_strategies.py`.

## Results

### FX universe (2026-10, ~1.8y H1 history, 11 pairs)

- All in-sample darlings die out-of-sample, again. Best retest config
  (+1703, sharpe 1.07 in-sample) went to walk-forward −686.
- Fine-timeframe levels (H1/H2/H4) are noise in every family.
- **Breakout-continuation on daily levels** is the best raw material found:
  165→172 trades (6x the retest book), +1452 in-sample, random gate passes
  11/11 — but its walk-forward is one regime (Q4-2025) wearing a trench coat
  (strip that window and it loses).
- Fade at level touches: negative across almost all 22 knob settings.

### Index universe (2026-10, six index CFDs)

Six OANDA index CFDs (SPX500, NAS100, US30, DE30, UK100, JP225; daily back to
2003, H1 from 2025-01; pip size = 1 index point in `config.yaml`) ran the same
battery. The terrain flips: the retest book dies on indices (28 trades, −13%),
while fine-timeframe breakouts — dead on FX — come alive in-sample
(breakout_h1 + `touch_tolerance_atr: 0.15`: 538 trades, +5911, sharpe 1.16,
random gate 6/6). Same wall: OOS last 25% −1011, walk-forward decays
monotonically (+2576/+730/−306/−852 through 2026-Q3). FX-tuned knobs
(`fta_fallback_atr=3`, `cluster_ticks=10`) do not transfer.

Nothing in either universe earned adoption into `config.yaml`.

### Daily history (2026-10, 2002-2026 daily bars, 17 symbols)

The decisive test: the same families on the full daily history (11-pair book +
6 indices, ~99 quarters), one signal pass per symbol (availability guard is
backward-looking), session rules off, nothing tuned.

- **The H1 index breakout "edge" was a regime artifact.** On 24 years of
  daily bars, breakout on indices loses badly: −7216, PF 0.80 (base) / −4934,
  PF 0.87 (H1-selected knobs) / −7425, PF 0.80 (3x costs). The 538-trade
  +5911 H1 window did not generalize to one symbol's own history.
- **Retest family on daily: dead** — 182 trades in 24 years, PF 0.54, −6254.
  The strategy has no time horizon on which it works.
- **One 24-year survivor**: daily breakout on FX+metals (11 symbols) — PF 1.06
  base → 1.19 with the H1-selected knobs, sharpe ~0.2-0.3, 44-46% positive
  quarters, survives ~3x costs (PF 1.13). But it is concentrated in
  metals/energy/crypto momentum (XAU +3885, XAG +1005, WTI +1266, BTC +904 at
  heavy costs) while USD_JPY (−1647) and USD_CAD (−962) bleed. This is the
  classic commodity trend-momentum effect captured weakly — a seed for a
  *different* system, not a deployment candidate and not this strategy.
- Random gate 17/17 pairs pass on every daily config (engine honest).

Verdict: the retest book is shelved; the only non-curve-fit lead is daily
breakout momentum on commodities, with a measured edge too small to trade
as-is.

### Bund (FGBL) 5-minute test (2026-10, ~29 trading days of M5, 1.8y H1)

Euro-Bund via `EUREX:FGBL1!` (TV feed; anonymous M5 history is capped at
~1.4 months, so this is a smoke-level verdict by construction). M5 execution
throughout (entries/exits/ATR/stops); level frames D/H4/H1/M15/M5; optional
D/H4 SMA trend gates on signals.

- Everything loses: breakout on H1 levels −548 (33 trades, PF 0.76), M15
  levels −1502 (PF 0.51), M5 levels −2752 (PF 0.36). Retest family: 0–13
  trades, negative.
- HTF trend gates roughly halve the damage (PF 0.36 → 0.70, WR 30% → 41%)
  but never flip it; kill test on the least-bad variant (M5 levels + D-SMA20
  gate): −832, PF 0.70, 1 of 5 weeks positive; at 3x costs PF 0.22. Random
  gate passes.
- Longer-sample cross-check (same family, H1 execution, 1.8y): H1-level
  breakout −537 with WF pooled −923; D-level breakout flat noise (26 trades).
- Verdict: the Bund joins the list — no edge on any timeframe tested. The M5
  window is too short for more than a smoke verdict, but nothing here
  justifies deeper-history engineering.

Raw evidence: `results/` (CSV of every config tried, appended; the `universe`
column tags the source book — rows before 2026-10-16 are FX) and `logs/`
(one log per sweep/validation run). Note: `logs/final2_fade.log` was produced
by a spawn-worker bug that ignored the strategy argument and re-ran breakout_d;
`logs/final2_fade_real.log` is the correct fade validation.

### Cross-asset breadth studies (2026-10, `vecbt.py` portfolio sim)

After the retest book was shelved, three *different* families were tested
across every asset class with cached history — trend/momentum, mean reversion,
funding carry — on daily closes, next-bar execution, equal 1/N weights,
per-instrument vol scaling, 3 bps per position change (6 bps sensitivity), and
a block-bootstrap random gate (shuffle each instrument's returns against the
same positions; p = P[shuffled ≥ actual]). The universe was extended with 12
commodity futures fetched from the TV feed (COPPER_USD←COMEX:HG1!,
NATGAS_USD←NYMEX:NG1!, PLATINUM_USD/PALLADIUM_USD←NYMEX, BRENT_USD←NYMEX:BZ1!,
CORN/SOYBEAN/WHEAT←CBOT, COFFEE/SUGAR/COCOA/COTTON←ICEUS; deep history to the
70s/80s). Data note: the WTI CFD pinned at 0.009 for one day during the
Apr-2020 negative-oil episode (roll artifact; masked — the real series, CL1!,
settled at −$37.63); the 2026 natgas −46% day was verified genuine (NYMEX NG1!
identical).

**Study A — trend breadth (`study_tsmom.py`, `logs/tsmom.log`).** Canonical
Donchian 50/20 state machine (±1, exit to flat on the 20-day reverse channel)
and 12-month TSMOM (monthly sign), 71 instruments (6 indices, 15 commodities,
12 FX, 10 perps, FGBL, 27 US stocks), 2005-2026.

- Donchian ALL: Sharpe 0.10, PF 1.13, gate p=0.94 (actual *worse* than the
  shuffled null). TSMOM ALL: Sharpe 0.23, PF 1.47, gate p=0.81 (null median
  higher). No alignment edge at portfolio level; cost-robust (low turnover)
  but edge-less.
- Classes: crypto donchian 0.67 (8y), us_stock tsmom 0.56 (long-biased),
  equity_index tsmom 0.37 (long leg only, +0.95 long / −0.25 short);
  commodities ≈ 0; FX ±0.18; FGBL negative.
- Retail classic SMA200 long-only filter (indices+stocks): Sharpe 0.84,
  PF 2.70, MDD −20%, 65% exposure — but gate p=0.87: the timing adds nothing
  over random days with the same exposure; it is beta shaping, not edge.

**Study B — mean reversion (`study_meanrev.py`, `logs/meanrev.log`).**
Connors RSI(2) dip-buy: enter after close with RSI2<5, exit close>SMA5 or 10
bars, long-only, optional SMA200 filter; 6 indices + 27 stocks; vs
equal-weight buy-and-hold of the same instruments.

- rsi5 + SMA200 filter, all 33: Sharpe 0.61, PF 1.81, WR 71%, exposure 5.2%,
  MDD −5.2% (stocks-only Sharpe 0.78). Gate p=0.005 — the only statistically
  significant signal alignment in the whole program.
- But versus buy-and-hold on the same universe (Sharpe 0.71 all / 0.91
  stocks, survivorship-inflated): levering MR to B&H vol reproduces B&H
  returns with shallower drawdowns. A real timing effect, not an edge beyond
  being long — at best a capital-efficient participation sleeve.
- XAU/XAG ratio z-score(60) spread probe: dead (Sharpe −0.03, gate p=0.51).

**Study C — crypto funding carry (`study_funding.py`, `logs/funding.log`).**
Binance USDT-perp funding, 10 symbols, 2020-09→2026-10 (BTC/ETH from 2019).
Hedged carry (long spot + short equal-notional perp); PnL = funding received
by the short leg; fees assumed $10-20 per toggle. Policy A holds always;
policy B holds only while trailing-7d funding > 0.003% per event.

- Policy A: **9.51%/yr, Sharpe 6.1**, maxDD $520 (on a $10k/symbol book),
  positive every year except 2022 (−4%). Policy B: **10.19%/yr net,
  Sharpe 7.8**, maxDD $113, worst 30d −$29.
- Robust: 7.7-10.2%/yr net across thresholds (0-0.01%/event) and fee
  assumptions; split halves +12.6/15.2%/yr (1st) and +6.5/5.2%/yr (2nd) —
  positive in both, yield compressing.
- Per-symbol (A): BTC 11.5, ETH 13.7, LTC 14.7, XRP 14.5, LINK 13.7, ADA
  13.4, DOGE 12.3 vs SOL 0.2, BNB −0.1, AVAX 6.8 — the weak ones carry deep
  negative-funding regimes (SOL maxDD −$4.3k per $10k); the threshold policy
  avoids those.
- Verdict: the only family in the whole program with persistent positive
  expectancy. Risks are real but non-statistical: venue/counterparty, margin
  management on the short perp leg, spot+perp execution, yield compression.

**Study D — crypto-only lab (`fetch_binance.py`, `study_crypto*.py`,
`logs/crypto_*.log`).** Binance USDT perps, 10 majors, full perp history from
listing (2019-09→2026-10 for BTC; most alts 2020). Data refetched via
`fetch_binance.py` with the full fapi field set — volume, quote volume, trade
count, taker-buy volume — cached as `{SYM}_PF_D/H1.parquet` (the older
`{SYM}_D/H1` files are Binance *spot* and are no longer used by crypto sims).
All sims: signals on daily closes applied next bar, equal 1/N, vol-scaled to
25% standalone per instrument, **funding charged on the position** (longs pay,
shorts receive), 5 bps one-way (Binance taker) with 10/20 bps stress, 365-day
annualization.

- *Audit*: vols 57% (BTC) → 201% (DOGE); mean cross-corr 0.56, corr-to-BTC
  0.63; daily return autocorr ≈ 0 at all lags; volume is where the structure
  lives — |ret| vs log-volume corr +0.41, log-volume autocorr +0.76,
  taker-buy imbalance autocorr +0.31 and corr(imbalance, ret) +0.38; funding
  averages +9.5%/yr (77% of days positive).
- *Grid (21 variants)*: plain trend mediocre (Donchian 55/20 Sharpe 0.27,
  mom_30 0.63, mom_90 −0.08, SMA crosses negative); taker-flow dead
  (taker_mom −0.18, taker_fade −0.73); 1-day reversal dead (−0.25). **The
  volume×momentum family dominates**: `obv_30` (30-day OBV trend) Sharpe 0.87,
  PF 1.42, 2045 trades, MDD −19%, 10/10 symbols positive; `vwr_30`
  (volume-weighted return momentum) 0.91, PF 1.50; mom_30 gated on
  high-volume days 0.73; volume-confirmed Donchian 0.36 vs 0.27 plain. Volume
  *presence* confirms monthly momentum; the taker buy/sell split adds nothing.
  B&H all-long: 0.55 (funding drag −0.68 return units over the sample).
- *Kill tests (obv_30 / vwr_30)*: random gate p=0.005 both (actual +1.28/+1.43
  vs null medians −0.39/−0.40); leave-one-out min +0.73/+0.82 (even dropping
  any single symbol); cost stress 10/20 bps → 0.80/0.66 and 0.86/0.75; 7/8
  years positive; sample thirds decay +1.54/+0.71/+0.51 (obv_30). Long-only is
  strongest: obv_30_lo Sharpe 0.98 PF 1.82 (802 trades), vwr_30_lo 0.97 PF
  1.90. Horizon scan: broad plateau 20-35 days (long-only 0.79-0.98), peak at
  30 — neighborhood positive, not knife-edge.
- *Honest caveats*: 30-day peak and recent-12m softening (long-only flat over
  the last year, long/short still positive) — this is the best candidate the
  program has produced, but it is a forward-validation candidate, not a
  proven edge. Execution is daily close → next-day market orders; realistic on
  Binance perps at the majors' liquidity.

Verdict: **30-day volume-weighted momentum is the strongest surviving
candidate of the crypto lab** and pairs naturally with the funding carry
(long the trend when it's on, harvest funding when flat). H1 perp data
(53-62k bars/symbol) is staged for the intraday refinement phase.
