# Retest — FX Breakout-then-Return Algorithmic Trading System

Systematic version of a discretionarily-traded FX strategy: when price breaks a
key support/resistance level and comes straight back, trade the return. Touch
count grades the setup; young levels need daily confirmations. Daily decides
*where* (levels, targets, context); hourly decides *when* (break, return,
entries). Price data: TradingView websocket feed (OANDA venue), scraped
directly with history paging.

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
never set their own hurdles.

## Project structure

```
config.yaml                  # every tunable number (strategy never hardcodes)
philosophy.md                # background + principles (why)
strategy.md                  # retest spec — the tradable rules
src/sfp_reversion/
  config.py                  # typed config reader
  data/                      # OHLC schema + TV websocket scraper + parquet cache
  levels/                    # fractal swings, clustered levels, quality filters
  confirmation/              # SFP detector (daily-bias pattern input)
  signals/                   # retest: daily levels + H1 break/return → orders
  backtest/                  # limit fills w/ expiry, stop-first exits, costs, sizing
  validation/                # random_baseline, walk_forward, significance, sensitivity
  report/                    # metrics table + equity-curve plot
  cli.py                     # fetch / signals / backtest / validate / portfolio
tests/                       # one file per module
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

## Test

```bash
uv run pytest
uvx ruff check src tests
uvx ruff format --check src tests
```

## Configuration

All strategy numbers live in `config.yaml` — tune config, never code. Key
sections: `levels` (swings, tolerance, quality), `retest` (return window,
touch gates, daily confirmations, stops/targets), `backtest` (costs incl.
per-pair pips, risk %), `validation` (windows, trials).

## Current status (honest)

System rebuilt end to end with honest fills (exit slippage, min-stop filter,
per-pair costs). Walk-forward out-of-sample is the only verdict that counts —
in-sample green adopts nothing.
