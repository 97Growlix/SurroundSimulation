# prediction-arb: handoff notes for Claude

Read this first. `README.md` has the full write-up and every result table.

## What this is and the owner's preferences

An automatic prediction-market arbitrage bot across Kalshi, Polymarket and
PredictIt. The owner sets a minimum return (% per year), and the bot finds
bundles that pay ≥ $1 in every outcome for less than $1.

* **Research and paper trading only. Never spend real money.** There is no
  live order code: `arb/execution.py` `LiveExecutor` raises on construction.
  Don't add live trading, API keys or wallets until the owner explicitly asks,
  and only after venue access and legality are settled (Polymarket
  international blocks US persons; Kalshi needs KYC).
* The owner accepts small, *quantified* risk, but wants low-risk options when
  expected return is similar. They like fast-resolving, release-day trades.
* Explain things plainly. The owner is not a finance specialist (e.g. asked
  "what's a CPI?").

## Layout

```
arb/sources/{kalshi,polymarket,predictit}.py  public market data (no auth)
arb/fees.py         per-venue fee models (Polymarket formula not yet checked against docs)
arb/engine.py       scenario-based worst-case evaluation + order-book sizing
arb/strategies.py   no_basket, yes_basket, cross_pair, ladder (implication) strategies
arb/numeric.py      range-covering bundles on official numbers: Kalshi strikes and
                    Polymarket buckets -> intervals of printed values; MILP cover;
                    choose_cover = top-K covers + implied distribution + $ risk score
arb/underlyings.py  REGISTRY of verified underlyings (read both venues' rules before adding)
arb/policy.py       capital rules (APY hurdle, hold-APY, switch, sell-now)
arb/watch.py        `releases` calendar + `watch` release-focused paper trader
arb/settle.py       resolve paper trades against actual venue outcomes
arb/sports.py       live in-game cross-venue recorder (game winners)
arb/backtest/       collect.py (history download), sim.py (hourly simulator), run.py (scenarios)
tests/              pytest suite; keep it green
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m arb scan --min-apy 15            # one-off scan of all venues (~1.5 min)
python -m arb releases                     # upcoming releases for registry underlyings
python -m arb watch --bankroll 100         # paper trader; state in data/paper_portfolio.json
python -m arb settle                       # score paper_ledger.jsonl trades
python -m arb.backtest.collect 2026-05-08 2026-11-08   # tops up data/backtest/*.json
python -m arb.backtest.run                 # all scenarios -> data/backtest/results.json
python -m arb.sports 60                    # record live games for 60 minutes
```

`data/` outputs are gitignored, and cloud containers are ephemeral, so anything
downloaded must be re-collected in a new session (Kalshi part ~1-2 h with gas).

## Environment notes (cloud sessions)

* Network allowlist needed: gamma-api.polymarket.com, clob.polymarket.com,
  api.elections.kalshi.com, www.predictit.org (the personal environment
  "SurroundSimulation (prediction market data)" has these).
* Kalshi rate-limits hard. `arb/http.py` paces requests per host, and the
  collector pauses 60s when throttled. Don't run the collector and live
  pollers at the same time.
* Kalshi markets settled before its historical cutoff (~2026-09-09) come back
  without nested markets: fetch `/historical/markets?event_ticker=...` and
  `/historical/markets/{ticker}/candlesticks`.
* Polymarket `prices-history` rejects windows longer than ~14 days (chunk the
  requests). It has one price per hour and no bid/ask.

## Hard-won gotchas (each has a regression test)

* Data periods: take the reference month from the **title**. Kalshi PPI/JOLTS
  tickers are dated by release date. Match whole month words only
  ("decision" must not read as "dec").
* A "by December 31" deadline with no year must be inferred from the end time;
  Polymarket end times are 04:59 UTC on Jan 1.
* Ladders: date deadlines nest the opposite way to "above X" thresholds.
* Polymarket duplicates or abandoned events (closedTime far before endDate) have
  fake resolutions. Drop them, and stop trading at closedTime.
* Stale Polymarket bucket prices create phantom arbs. Only trust an event when
  its bucket prices sum to about 1.
* Excluded underlyings, with reasons in `underlyings.py`: city temperatures
  (different weather stations), crypto (different index and time), Bank of
  Japan (10-15bp moves vs labels), ECB (deposit vs primary rate), payrolls
  (Polymarket series irregular).

## Findings so far

* Arbitrage within a single venue (NO baskets) is essentially gone. Wording-
  matched cross-venue pairs are mostly *different* propositions; a big price
  gap usually means mismatch risk, not free money.
* Real edge is in **range-covering bundles on official releases** (CPI, PCE,
  PPI, GDP, unemployment, JOLTS, central-bank decisions, daily gas), including
  Kalshi's threshold vs exact-value ladders.
* Six-month backtest (May 8 - Nov 8, 2026, original 7 underlyings), v1
  (cheapest cover, hold to settlement): $100 -> ~$150 realistic, with
  guaranteed returns of 27-55% over 6 months depending on liquidity
  assumptions. Returns shrink with size: about 8% at $100k. No bundle ever paid
  below its guaranteed minimum against actual outcomes.
* v2 (50% APY hurdle + early exit/switch + risk chooser + max 50% per trade) on
  the same data: $100 -> $203.75. The full 14-scenario run on the expanded
  registry (250 groups) incl. the maker option was **in progress at handoff**.
  Check `data/backtest/run.log`, or re-run it.
* Live in-game sports (13 games, 2,932 snapshots): median bundle cost $1.022
  after fees, under $1 only 0.3% of the time, gaps gone within 15 s. Low
  priority.
* Paper trades open for the Nov 10, 2026 CPI release: run `python -m arb settle`
  and check `data/paper_portfolio.json`, if the container still exists.

## Caveats to keep in mind

Hourly history (fills assumed at the quoted price), thin Kalshi exact-value
markets, unverifiable Polymarket depth in the past, a 6-month sample, and
path dependence with small bankrolls. Treat backtest numbers as upper-ish
estimates.

## Next steps the owner agreed to or was offered

1. Report the full backtest (v1 vs v2 vs maker variants; contribution of the new
   underlyings).
2. Run `watch` through several real releases and compare paper fills with the
   simulation.
3. Websocket feeds for release windows (gaps close in seconds to minutes).
4. Only if the owner asks: live execution (Kalshi RSA-PSS signed REST, Polymarket
   CLOB client or Polymarket US), with fill-or-kill orders, the thinnest leg
   first, auto-unwind, capital caps and a kill switch.
