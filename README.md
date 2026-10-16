# Retest — FX breakout-then-return research

Systematic version of a discretionarily-traded FX strategy: when price breaks a
key support/resistance level and comes straight back, trade the return. Touch
count grades the setup; young levels need daily confirmations. Daily decides
*where* (levels, targets, context); hourly decides *when* (break, return,
entries). Price data: TradingView websocket feed (OANDA venue), scraped
directly with history paging.

**Verdict (2026-10, ~1.8y of H1 history, 11-pair universe): no edge that
survives out-of-sample.** The book trades too rarely to prove anything (29
trades), every tuned configuration that looked strong in-sample failed the
walk-forward, and two full structural rewrites (breakout-continuation, level
fade, plus MTF-level retests) confirmed the finding. Details and numbers under
[Findings](#findings). The repo remains as an honest, fully-tested research
base — see [Way forward](#way-forward).

## Pipeline

```
TradingView feed → cache → daily levels (fractal swings, clustered)
  → H1 break detect → first-return scan (touch grading + daily confirmations)
  → limit orders (level/fib, stop, target) → backtest engine (limit fills
     with expiry, stop-first exits, warning exit, costs, 1%-risk sizing)
  → falsification gate → walk-forward / significance / sensitivity → reports
```

No lookahead anywhere: every output at bar `t` uses only data available at or
before `t`. ATR regimes are measured through the prior bar so violent bars
never set their own hurdles. The random-entry falsification gate passes on
every configuration tested (11/11 pairs), so engine results are trustworthy.

## Project structure

```
config.yaml                  # every tunable number (strategy never hardcodes)
philosophy.md                # background + principles (why)
strategy.md                  # retest spec — the tradable rules
src/retest/
  config.py                  # typed config reader
  data/                      # OHLC schema + TV websocket scraper + parquet cache
  levels/                    # fractal swings, clustered levels, quality filters
  confirmation/              # SFP detector (daily-bias pattern input)
  signals/                   # retest: daily levels + H1 break/return → orders
  strategies/                # structural variants (research): MTF retest,
                             #   breakout continuation, level fade
  backtest/                  # limit fills w/ expiry, stop-first exits, costs, sizing
  validation/                # random_baseline, walk_forward, significance, sensitivity
  report/                    # metrics table + equity-curve plot
  cli.py                     # fetch / signals / backtest / validate / portfolio
tests/                       # one file per module (90 tests)
research/                    # sweep + OOS kill-test harnesses, results, logs
data/cache/                  # local parquet (ignored by git, re-fetchable)
reports/                     # generated plots (ignored by git)
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev     # install runtime + dev deps (pytest, ruff)
```

## Run

```bash
# pull data (daily + hourly; re-runs only fetch what's new)
uv run retest fetch EUR_USD
uv run retest fetch EUR_USD --granularity H1

# generate signals
uv run retest signals EUR_USD --start 2025-01-01 --end 2025-06-30 --out signals.csv

# backtest + falsification gate + equity plot
uv run retest backtest EUR_USD

# walk-forward, or sweep one parameter (robust vs knife-edge verdict)
uv run retest validate EUR_USD
uv run retest validate EUR_USD --parameter fib_proximity_atr --values 0.1 0.2 0.3

# whole universe at once
uv run retest portfolio
```

Any TradingView-covered pair works (`EUR_USD`, `GBP_USD`, `USD_JPY`, …).
Each pair/timeframe caches separately.

The structural variants are exercised through the research harnesses (run from
the repo root; see `research/README.md` for modes and caveats):

```bash
uv run python research/alpha_sweep.py single        # retest knob sweep (batched)
uv run python research/alpha_sweep2.py smoke        # all variants, trade counts
uv run python research/sweep_final.py  <label> '<r_over json>'
uv run python research/sweep_final2.py <strat> '<r_over json>'
```

## Test

```bash
uv run pytest          # 90 tests
uvx ruff check src tests
uvx ruff format --check src tests
```

## Configuration

All strategy numbers live in `config.yaml` — tune config, never code. Key
sections: `levels` (swings, tolerance, quality), `retest` (return window,
touch gates, daily confirmations, stops/targets), `backtest` (costs incl.
per-pair pips, risk %), `validation` (windows, trials). Changed during the
research effort: nothing — every candidate that improved in-sample failed
out-of-sample, so the config stayed as-is.

## Findings

In-sample full sample / out-of-sample (rolling 9mo-IS → 3mo-OOS walk-forward,
plus a last-25% time split) on the 11-pair universe, honest costs throughout:

| candidate | trades | in-sample | out-of-sample verdict |
|---|---|---|---|
| retest book (current config) | 29 | +940 (+9.4%), Sharpe 0.64 | inconsistent windows, no significance |
| best tuned retest (~190 configs) | 21 | +1703 (+17%), Sharpe 1.07 | WF pooled −686, Sharpe −1.55 |
| MTF retest (H1/H2/H4 levels) | 142–152 | −2.9k to −5.0k | dead — fine-timeframe levels are noise |
| breakout-continuation on daily levels | 165–172 | +566 to +1452, Sharpe ≤ 0.54 | WF pooled positive but one regime (Q4-2025); strip it and it loses |
| fade at level touches | 36–142 | negative in 19/22 settings | dead |

The random-entry gate passes everywhere (engine is honest), and no
configuration both (a) produces enough trades to judge and (b) survives
out-of-sample. Raw evidence: `research/results/` (every config tried) and
`research/logs/` (one log per run).

## Way forward

Only three levers aren't curve-fitting:

1. **More data** — H1 history is ~1.8 years; deeper H4/M5 pulls would let the
   breakout family be judged across several regimes.
2. **Different instruments** — current universe is FX majors + metals + BTC on
   one venue.
3. **A new hypothesis, not another knob** — e.g. whether an observable regime
   condition separates the trending Q4-2025-style windows from the chop,
   tested with the same out-of-sample discipline.

Absent one of those, shelving the strategy is the rational move.
