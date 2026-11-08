"""End-to-end scan: fetch -> screen at top of book -> verify against full order books."""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import numeric, strategies
from .engine import Opportunity, evaluate, size
from .http import log
from .matching import load_confirmed, suggest_pairs
from .models import Event
from .sources import kalshi, polymarket, predictit

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ScanConfig:
    min_apy: float = 0.10                # 10%/yr
    min_profit: float = 1.00             # dollars of guaranteed profit
    max_capital: float = 1000.0          # dollars per opportunity
    max_days: float = 400.0              # ignore markets resolving later than this
    settle_buffer_days: float = 2.0      # resolution / payout lag
    unknown_days: float = 365.0          # assumed horizon when a venue gives no date
    platforms: tuple[str, ...] = ("kalshi", "polymarket", "predictit")
    include_unverified: bool = False     # report tier-C (auto-matched / assumed) ideas
    assume_exhaustive: bool = False
    cache_minutes: float = 0.0           # reuse a recent snapshot (dev only)


def fetch_all(cfg: ScanConfig) -> dict[str, list[Event]]:
    DATA.joinpath("cache").mkdir(parents=True, exist_ok=True)
    max_close = datetime.now(timezone.utc) + timedelta(days=cfg.max_days)
    fetchers = {
        "kalshi": lambda: kalshi.fetch_events(max_close),
        "polymarket": lambda: polymarket.fetch_events(max_close),
        "predictit": predictit.fetch_events,
    }
    out = {}

    def run(p):
        cache = DATA / "cache" / f"{p}.pkl"
        if cfg.cache_minutes and cache.exists() and time.time() - cache.stat().st_mtime < cfg.cache_minutes * 60:
            return p, pickle.loads(cache.read_bytes())
        t = time.time()
        log(f"fetching {p} ...")
        evs = fetchers[p]()
        log(f"  {p}: {len(evs)} events, {sum(len(e.markets) for e in evs)} markets in {time.time() - t:.0f}s")
        cache.write_bytes(pickle.dumps(evs))
        return p, evs

    with ThreadPoolExecutor(3) as ex:
        for p, evs in ex.map(run, cfg.platforms):
            out[p] = evs
    return out


def generate(events: dict[str, list[Event]], cfg: ScanConfig) -> list[Opportunity]:
    opps: list[Opportunity] = []
    for evs in events.values():
        for ev in evs:
            if (o := strategies.no_basket(ev)):
                opps.append(o)
            if (o := strategies.yes_basket(ev, cfg.assume_exhaustive)):
                opps.append(o)
            opps += strategies.ladder_pairs(ev)
    # range-covering bundles on registry-verified numeric underlyings
    opps += numeric.numeric_bundles(events)
    # cross-venue pairs: confirmed by a human, plus (optionally) auto-suggested
    by_id = {(m.platform, m.market_id): m for evs in events.values() for ev in evs for m in ev.markets}
    pairs = [(p, True) for p in load_confirmed(DATA / "pairs.json")]
    if cfg.include_unverified and len(events) > 1:
        pairs += [(p, False) for p in suggest_pairs(events)]
    for p, confirmed in pairs:
        a, b = by_id.get((p["a_platform"], p["a_id"])), by_id.get((p["b_platform"], p["b_id"]))
        if a and b:
            opps += strategies.cross_pair(a, b, invert=p.get("invert", False),
                                          confirmed=confirmed, score=p.get("score"))
    if not cfg.include_unverified:
        opps = [o for o in opps if o.risk_tier != "C"]
    return opps


def attach_books(opps: list[Opportunity]) -> None:
    k_tickers = {l.market.book_ref["ticker"] for o in opps for l in o.legs if l.market.platform == "kalshi"}
    p_tokens = {l.market.book_ref[l.side] for o in opps for l in o.legs if l.market.platform == "polymarket"}
    k_books = {}
    with ThreadPoolExecutor(4) as ex:
        for t, b in zip(k_tickers, ex.map(kalshi.orderbook, k_tickers)):
            k_books[t] = b
    p_books = polymarket.orderbooks(sorted(p_tokens)) if p_tokens else {}
    for o in opps:
        for l in o.legs:
            m = l.market
            if m.platform == "kalshi":
                l.ladder = k_books.get(m.book_ref["ticker"], {}).get(l.side, [])
            elif m.platform == "polymarket":
                l.ladder = p_books.get(m.book_ref[l.side], [])
            elif m.platform == "predictit":
                price = m.ask(l.side)
                # No depth API; cap at PredictIt's $850 per-contract limit.
                l.ladder = [(price, int(predictit.MAX_PER_CONTRACT_USD / price))] if price else []
                if "PredictIt depth unknown (top-of-book only; $850/contract cap)" not in o.notes:
                    o.notes.append("PredictIt depth unknown (top-of-book only; $850/contract cap)")
        o.depth_checked = True


def run_scan(cfg: ScanConfig, events: dict[str, list[Event]] | None = None):
    now = datetime.now(timezone.utc)
    events = events or fetch_all(cfg)
    raw = generate(events, cfg)
    # Screen at top of book with 1 unit, unrounded fees.
    screened = []
    for o in raw:
        e = evaluate(o, 1, now=now, use_depth=False, round_fees=False,
                     settle_buffer_days=cfg.settle_buffer_days, unknown_days=cfg.unknown_days)
        if e and e.profit > 0 and e.apy >= cfg.min_apy:
            screened.append(e)
    log(f"{len(raw)} structural candidates, {len(screened)} pass top-of-book screen")
    attach_books(screened)
    final = []
    for o in screened:
        s = size(o, now=now, min_apy=cfg.min_apy, min_profit=cfg.min_profit,
                 max_capital=cfg.max_capital, settle_buffer_days=cfg.settle_buffer_days,
                 unknown_days=cfg.unknown_days)
        if s:
            final.append(s)
    final.sort(key=lambda o: (o.risk_tier, -o.apy))
    return final, screened, events
