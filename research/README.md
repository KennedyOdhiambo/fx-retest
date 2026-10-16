# Research — structural variants vs the retest book

Ad-hoc (but reproducible) research tooling used to stress-test the retest
strategy and two structural rewrites on cached H1/D data, across two
universes: the 11-pair FX/metal/BTC book and six index CFDs. Run everything
**from the repo root** (data loading uses relative cache paths).

## Scripts

| script | what it does |
|---|---|
| `alpha_sweep.py` | one-knob + combo sweeps for the daily-level **retest** book (single|combo modes) |
| `alpha_sweep2.py` | smoke + knob sweeps for the structural variants: MTF retest, **breakout-continuation**, **fade** (`smoke|sweep|fade|h1idx` modes; `SWEEP_UNIVERSE=SYM1,SYM2,...` overrides the universe) |
| `sweep_final.py` | kill-test for a retest config: full sample, OOS last 25%, walk-forward windows, random-entry gate |
| `sweep_final2.py` | same kill-test for the variants: `python research/sweep_final2.py <strat> '<r_over json>' ['<bt_over json>']` |
| `sweep_daily.py` | daily-history long-run test (2002-2026): breakout + retest on daily bars, session off, one signal pass per symbol, yearly/quarterly regime view (`quick|full|costs` modes) |

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

Raw evidence: `results/` (CSV of every config tried, appended; the `universe`
column tags the source book — rows before 2026-10-16 are FX) and `logs/`
(one log per sweep/validation run). Note: `logs/final2_fade.log` was produced
by a spawn-worker bug that ignored the strategy argument and re-ran breakout_d;
`logs/final2_fade_real.log` is the correct fade validation.
