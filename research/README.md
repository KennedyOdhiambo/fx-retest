# Research — structural variants vs the retest book

Ad-hoc (but reproducible) research tooling used to stress-test the retest
strategy and two structural rewrites on cached H1/D data, across two
universes: the 11-pair FX/metal/BTC book and six index CFDs, plus (2026-10)
cross-asset breadth studies — trend/momentum and mean reversion — over 71
instruments in six asset classes. Run everything **from the repo root** (data
loading uses relative cache paths).

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

After the retest book was shelved, two *different* families were tested
across every asset class with cached history — trend/momentum and mean
reversion — on daily closes, next-bar execution, equal 1/N weights,
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


