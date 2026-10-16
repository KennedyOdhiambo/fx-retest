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
fade, plus MTF-level retests) confirmed the finding. Deeper history (24y
daily) and a follow-up breadth program — trend/momentum, mean reversion,
funding carry across 71 instruments in six asset classes — found no tradable
directional edge either; the only surviving candidate is **delta-neutral
crypto funding carry** (details in `research/README.md`). See
[Findings](#findings) and [Way forward](#way-forward).

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

The obvious levers have now been pulled: deeper history (24y daily), a much
broader universe (71 instruments across six asset classes), and different
hypothesis families (trend/momentum, mean reversion, XAU/XAG spread, funding
carry) — all tested with the same discipline. No directional price edge
survived. Full numbers: `research/README.md`.

The one candidate with persistent positive expectancy is **crypto funding
carry** — hedged spot+perp harvester, ~8-10%/yr net of fees, Sharpe 6-8,
positive in both split halves — a yield harvest rather than a price strategy,
with real venue, margin and execution risks. It is the only thing found so
far worth building around.

2026-10 crypto lab (10 Binance perps, full perp history, volume + taker-flow
data, funding charged on positions): plain trend and taker-flow signals are
dead, but **30-day volume-weighted momentum** (OBV trend / volume-weighted
return momentum) passed every kill test — Sharpe 0.87-0.97, 10/10 symbols
positive, gate p=0.005, leave-one-out robust, cost-stressed, positive in 7/8
years — with two honest caveats (30-day horizon is the peak of a broad
plateau; recent 12 months softened). A forward-validation candidate that
pairs with the funding carry: participate long when the volume trend is on,
harvest funding when flat. Full numbers: `research/README.md` (Study D).
