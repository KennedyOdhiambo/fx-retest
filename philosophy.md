# Philosophy — Trader Dante, as we understand him

> Background and principles. Not a spec. The tradable spec lives in `strategy.md` (breakout-then-return). Everything in code is trial/experimentation until walk-forward + significance say otherwise. Education only, not financial advice.

## 1. Who he is (public record)
- Tom Dante (@Trader_Dante), London. Trading since 1999. ~7 years struggling, then profitable, then a stint at a London prop futures firm, then independent.
- Still posting 2007 → 2026. Blunt style. No audited public track record found — critics flag this, supporters point to longevity + prop background. Treat all performance claims (including a NexusFi user's "~55% / 1.7R over 18 months") as unverified anecdotes.

## 2. What we trade here
- **Breakout-then-return, touch-graded.** Fresh break, first return, graded 3 touches = take / 2 touches = 1 confirmation / 1 touch = 2 confirmations (confirmations: daily pin/engulfing/SFP, fib 38/50/61, ATR stretch). This is the method traded profitably by hand in 2022–2023, now systematized.
- Out of scope: SFP fading, the short-term Bund method (agenda titles only are public), MACD, gaps/correlations.

## 3. Core beliefs behind it
1. **Obviousness is liquidity.** Trade levels everyone sees (prior swings, equal highs/lows, day/session extremes). If you squint, there is no fuel.
2. **The close decides.** A close beyond the level is a genuine break — trade its return. Drift that never breaks is not this trade.
3. **Fresh beats exhausted.** Proven levels (3 touches) trade clean; young levels must prove themselves with confirmations.
4. **Higher timeframe chooses location, lower timeframe delivers trigger.** Daily structure for levels/targets/confirmations, H1 for break and return.
5. **Geometry first, prediction never.** Limit at the level, stop beyond the return extreme + ATR, target the next opposing pool. Skip when the math doesn't pay.
6. **Process is half the edge.** Daily plan sheet, pre-marked areas, stupid-trade scoring (Demon Finder), weekly review. No setup survives without this.

## 4. What is public vs what we invent
- Public: concepts above, touch grading 3/2/1, fib ratios 38/50/61, 20-period ATR mention, warning-sign exit (first H1 close back through the extreme), ATR-buffered stops.
- NOT public: exact swing lookback, touch tolerance, return window, pin thresholds, fib proximity, stop-buffer ATR multiple, min R:R. Every number in `config.yaml` is our operational choice, tunable via walk-forward — never presented as "Dante's numbers".

## 5. How to use these docs
- `philosophy.md` (this file) — why we trade this way. Changes rarely.
- `strategy.md` — exact retest rules, flowchart, parameters.
- Code is guilty until proven innocent: random-baseline gate, walk-forward OOS, Sharpe CI + deflated Sharpe, sensitivity hills-not-spikes.
