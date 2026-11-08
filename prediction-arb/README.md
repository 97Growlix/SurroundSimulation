# prediction-arb

A scanner for **risk-free (arbitrage) opportunities** on prediction markets
(Kalshi, Polymarket, PredictIt). You set a minimum annualised return. It checks
every open market, confirms each opportunity against the live order books after
fees, and logs the trades it *would* make.

**Status: research and paper trading only. There is no code that can spend money.**
(`LiveExecutor` raises on construction.)

## Quick start

```bash
pip install -r requirements.txt
python -m arb scan  --min-apy 15 --min-profit 2 --max-capital 500   # find opportunities
python -m arb paper --min-apy 15                                     # scan + record dry-run trades of tier-A opps
python -m arb match                                                  # suggest cross-venue pairs to review
python -m pytest -q tests
```

Useful flags: `--platforms kalshi,polymarket`, `--max-days 120` (skip long-dated
markets), `--include-unverified` (also show tier-C ideas), `--cache-minutes 30`
(reuse the last snapshot while developing). A full scan of all three venues takes
about 1.5 minutes: around 12k Kalshi events, 19k Polymarket events and
360k markets.

## How it works

1. **Fetch.** Pull public market data (no API keys needed) and convert every contract
   into a `BinaryMarket` with its best YES/NO ask, close date, rules and fee model.
2. **Generate structural candidates.** For each one, list *every* way it can
   resolve and which legs pay out in each case (`arb/strategies.py`):
   * `no_basket`: in a **mutually exclusive** event (Kalshi `mutually_exclusive`,
     Polymarket `negRisk`), buy NO on a set of outcomes. At most one can win, so
     at least N-1 legs pay. This does **not** require the outcome list to be complete.
   * `yes_basket`: buy YES on every outcome. This is only safe when the venue guarantees
     the list is exhaustive. It is off by default; see the findings below for why.
   * `cross_pair`: the same proposition on two venues; buy YES on one and NO on the other.
   * `numeric` (range covering): for official numbers listed on several venues
     (CPI, PCE, GDP, Fed decisions; see `arb/underlyings.py`), each market becomes
     the set of printed values it pays on. An integer program picks the cheapest
     set of YES/NO positions that pays at least $1 for *every* possible print, for
     example Kalshi "core CPI above 0.1%" + Polymarket "≤0.0%" + "0.1%". Works
     across venues and within Kalshi (threshold vs exact-value ladders).
   * `ladder`: nested markets within one event, such as "over 10.5" ⊂ "over 7.5" or
     "by Mar 31" ⊂ "by Jun 30". If the stricter one is priced above the looser one,
     buy YES on the looser one and NO on the stricter one; the worst case is $1.
3. **Screen** at the top of the book. Then **verify** against the full order books
   (Kalshi `/orderbook`, Polymarket CLOB `/books`). Choose the quantity that
   maximises *worst-case* profit while still meeting your minimum APY, minimum
   profit and maximum capital (`arb/engine.py`).
4. **Annualise.** APY = (1+ROI)^(365/days) - 1, where days runs to the *latest*
   leg's resolution, plus a 2-day settlement buffer. Capital is locked until then,
   so a 2% edge that takes a year to settle is a 2% APY.

### Risk tiers

| Tier | Meaning | Auto-traded in paper mode |
|---|---|---|
| A | Venue rules guarantee the payout (mutually exclusive event, same venue) | yes |
| B | Cross-venue pair **you** confirmed in `data/pairs.json` | no (planned) |
| C | Auto-matched pair or assumed-exhaustive basket: an *idea*, not an arb | never |

## Findings from the first live scan (2026-11-08)

* **Arbitrage within a single venue is essentially gone.** All 6,172 mutually exclusive
  events on Polymarket and PredictIt, and all Kalshi events, were priced so that
  no NO-basket cleared fees. The best case (Polymarket NHL Hart Trophy, 90 legs)
  made **$0.002 on $89**, and it turned negative at any real size. Each venue's
  YES and NO books are linked, so YES+NO on one contract never costs less than $1.
* **Buying YES on every outcome is a trap.** Many Polymarket events showed YES prices
  summing to 0.11-0.30. That looks like free money, but those events have outcomes
  missing from the listing (unlisted, inactive or "Other" markets). Buying the
  listed outcomes would most likely lose everything.
* **The real gaps are between venues, and almost all are false matches.**
  With `--include-unverified`, the scanner found 46 cross-venue "opportunities",
  some showing ROI above 100%. Spot checks showed the large ones are different
  propositions:
  * Kalshi "Core CPI rises **above** 0.1%" vs Polymarket "Core CPI **is** 0.1%"
  * Kalshi Streamer Awards "**Gamer** of the Year" vs Polymarket "**Streamer** of the Year"
  * Kalshi "Mahomes 2+ TDs (**Protected**)" vs a plain Polymarket prop
  * PredictIt markets that are stale after the outcome is effectively known

  The plausible real ones (same question, small gap) yielded roughly 0.1-1% ROI
  on events months away. That works out to around 0-8% APY after fees, often on
  a few hundred dollars of depth.
* **Conclusion so far:** a fully automatic "find and buy" bot will mostly find
  wording mismatches. Real, low-risk edges need a **human-verified pair list**
  (tier B), and they are thin. That is the main design decision for the next phase.

## Accepting some risk: near-equivalent pairs (analysis 2026-11-08)

When two venues list "almost the same" question, the price gap is roughly the
market's estimate of the chance they resolve differently. If the bad mismatch
happens, you lose **all** the capital in the pair. So a pair with ROI *r* is +EV
only if P(bad mismatch) < r/(1+r). The headline ROI is the price of that risk.

| Bucket (30 live pairs, ≥$0.50 profit) | Per-trade ROI | Horizon | What it actually was |
|---|---|---|---|
| Wording/threshold mismatch | 28–111% | 2–7 d | "above 0.1%" vs "= 0.1%" CPI, Gamer vs Streamer of the Year, 2+ TD props with different void rules |
| Stale/thin PredictIt books | 14–68% | unknown | PredictIt contracts far from Polymarket (post-election "closest race", French election) |
| Genuinely identical wording | 0.2–1.5% | 2–90 d | TIME Person of the Year, arrests, Asian Cup, UN Sec-Gen: about 2–20% APY |

Within-venue `ladder` arbitrage (risk-free, tier A) found about 15 live cases, at
0.1–1.7% per trade (about 1–12% APY), mostly under $400 of depth.

Takeaway: the only route to large ROI is taking resolution-mismatch risk at
roughly fair odds. The edge appears only where you know the rules better than
the market (void rules, deadlines, sources). Encode that knowledge as explicit
scenarios and probabilities, then size so one mismatch can't wipe out many wins.

Two bugs found while building `ladder` are now regression-tested. Both would
have turned a "risk-free" trade into a directional bet:
(1) date deadlines treated as numbers with the wrong nesting direction;
(2) "by December 31" read as 2027 because Polymarket end times are 04:59 UTC on Jan 1.

## Range-covering bundles (built 2026-11-08)

`arb/underlyings.py` lists the official numbers that both venues settle on, each
checked against both rule texts. Live results on the first run (after full
order books and fees):

| Bundle | Legs | Guaranteed profit | ROI | Settles |
|---|---|---|---|---|
| Core CPI MoM, Oct 2026 | Kalshi YES ">0.1%" + Polymarket YES "≤0.0%", "0.1%" | $3.65 on $146 | 2.5% | ~2-4 days |
| Core CPI YoY, Nov 2026 | Kalshi exact "2.8%" + Kalshi NO ">2.7%" + YES ">2.8%" | $0.48 on $100 | 0.5% | 34 days |
| Fed decision, Apr 2027 | Kalshi NO "cut 25" + Polymarket YES "25bp decrease" | $0.61 on $49 | 1.2% | 173 days |
| Fed decision, Dec 2026 / Jan 2027 | Kalshi NO "hike >25" + Polymarket YES "50+ increase" | $0.70-3.30 on $1000 | 0.1-0.3% | 1-3 months |

Excluded after reading the rules: city temperatures (Kalshi NYC = Central Park,
Polymarket NYC = LaGuardia), crypto prices (different reference time and index),
and Fed funds *level* vs *decision* (linking them needs the pre-meeting rate).

Residual risks for these bundles: different clauses if the data release is
delayed (e.g. a government shutdown), Polymarket's UMA dispute process, and
legging (2-3 orders on 2 venues). Paper-trade them with `python -m arb paper --paper-tiers A,B`.

To add an underlying: read both venues' rules, then add an `Underlying` with the
Kalshi series tickers and a Polymarket title regex. Events whose bucket labels
can't be parsed cleanly, or whose parsed buckets overlap, are skipped rather
than guessed.

## Risks that "risk-free" does not remove

* **Resolution mismatch** across venues: different sources, deadlines, time
  zones, tie rules, "Other" or 50/50 fallback (`equal_split`). This is the #1 risk.
* **Legging risk:** orders on two venues are not atomic. One leg can fill while
  the other moves away.
* **Cancellation and void rules:** a void can refund at a price that breaks the hedge.
* **Capital lock-up and venue risk:** funds sit on two platforms until both settle.
  Withdrawal fees apply (PredictIt 5%; on Polymarket, USDC on- and off-ramp costs).
* **Fee schedule drift:** check `arb/fees.py` against current docs before going live.
  The Polymarket formula matches published peak rates but has not been checked
  against the docs.
* **Access and legal:** international Polymarket blocks US persons. Polymarket US
  is a separate, regulated venue with its own API. Kalshi requires KYC.
  PredictIt caps positions at $850 per contract. Confirm what you can legally use
  before adding any execution code.

## Roadmap to live trading (not started)

1. Run `paper` on a schedule for a few weeks and check settled paper trades
   against actual resolutions (add a `settle` command).
2. Build and maintain `data/pairs.json` (human-verified equivalences). Promote
   tier B to paper trading.
3. Add live executors: Kalshi signed REST (RSA-PSS) and Polymarket
   `py-clob-client` (or the Polymarket US API). Use fill-or-kill orders, thinnest leg
   first, auto-unwind on a partial fill, a re-check of the book just before sending,
   per-trade and daily capital caps, and a kill switch.
4. Switch to websocket order-book feeds. Real cross-venue gaps close in
   seconds to minutes, so polling every 1.5 minutes will miss most of them.

## Layout

```
arb/sources/   kalshi.py, polymarket.py, predictit.py   venue adapters (public data)
arb/fees.py    per-venue fee models
arb/engine.py  scenario-based worst-case evaluation + order-book sizing
arb/strategies.py  no_basket / yes_basket / cross_pair generators
arb/matching.py    cross-venue pair suggestions
arb/scan.py    fetch -> screen -> verify pipeline
arb/execution.py   DryRunExecutor (paper ledger); LiveExecutor deliberately disabled
data/pairs.json    human-verified cross-venue pairs (empty)
```
