# Research — structural variants vs the retest book

Ad-hoc (but reproducible) research tooling used to stress-test the retest
strategy and two structural rewrites on the cached H1/D data. Run everything
**from the repo root** (data loading uses relative cache paths).

## Scripts

| script | what it does |
|---|---|
| `alpha_sweep.py` | one-knob + combo sweeps for the daily-level **retest** book (single|combo modes) |
| `alpha_sweep2.py` | smoke + knob sweeps for the structural variants: MTF retest, **breakout-continuation**, **fade** (`smoke|sweep|fade` modes) |
| `sweep_final.py` | kill-test for a retest config: full sample, OOS last 25%, walk-forward windows, random-entry gate |
| `sweep_final2.py` | same kill-test for the variants: `python research/sweep_final2.py <strat> '<r_over json>' ['<bt_over json>']` |

Variant strategies themselves live in `src/retest/strategies/`
(`mtf.py`, `breakout.py`, `fade.py`), tested in `tests/test_strategies.py`.

## Results (2026-10, ~1.8y H1 history, 11-pair universe)

- All in-sample darlings die out-of-sample, again. Best retest config
  (+1703, sharpe 1.07 in-sample) went to walk-forward −686.
- Fine-timeframe levels (H1/H2/H4) are noise in every family.
- **Breakout-continuation on daily levels** is the best raw material found:
  165→172 trades (6x the retest book), +1452 in-sample, random gate passes
  11/11 — but its walk-forward is one regime (Q4-2025) wearing a trench coat
  (strip that window and it loses).
- Fade at level touches: negative across almost all 22 knob settings.
- Nothing here earned adoption into `config.yaml`.

Raw evidence: `results/` (CSV of every config tried, appended) and `logs/`
(one log per sweep/validation run). Note: `logs/final2_fade.log` was produced
by a spawn-worker bug that ignored the strategy argument and re-ran breakout_d;
`logs/final2_fade_real.log` is the correct fade validation.
