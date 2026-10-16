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

Raw evidence: `results/` (CSV of every config tried, appended; the `universe`
column tags the source book — rows before 2026-10-16 are FX) and `logs/`
(one log per sweep/validation run). Note: `logs/final2_fade.log` was produced
by a spawn-worker bug that ignored the strategy argument and re-ran breakout_d;
`logs/final2_fade_real.log` is the correct fade validation.
