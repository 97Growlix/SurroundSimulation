"""Kalshi public market data (no auth needed for reads)."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from .. import fees
from ..http import get_json, log
from ..models import BinaryMarket, Event, parse_ts

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _market(m: dict, ev: dict) -> BinaryMarket | None:
    if m.get("status") not in ("active", "open"):
        return None
    if m.get("market_type", "binary") != "binary":
        return None
    yes_ask, no_ask = _f(m.get("yes_ask_dollars")), _f(m.get("no_ask_dollars"))
    # An ask of 0 or 1 means "no offers"; treat as unavailable.
    yes_ask = yes_ask if yes_ask and 0 < yes_ask < 1 else None
    no_ask = no_ask if no_ask and 0 < no_ask < 1 else None
    close = parse_ts(m.get("expected_expiration_time")) or parse_ts(m.get("close_time"))
    return BinaryMarket(
        platform="kalshi",
        market_id=m["ticker"],
        event_id=m.get("event_ticker", ev.get("event_ticker", "")),
        title=m.get("title") or ev.get("title", ""),
        outcome_label=m.get("yes_sub_title") or "",
        yes_ask=yes_ask,
        no_ask=no_ask,
        yes_bid=_f(m.get("yes_bid_dollars")),
        no_bid=_f(m.get("no_bid_dollars")),
        yes_ask_size=_f(m.get("no_bid_size_fp")),   # YES offers are NO bids
        no_ask_size=_f(m.get("yes_bid_size_fp")),   # NO offers are YES bids
        close_time=close,
        url=f"https://kalshi.com/markets/{ev.get('series_ticker', '').lower()}",
        rules=(m.get("rules_primary") or "") + " " + (m.get("rules_secondary") or ""),
        fee=fees.KALSHI,
        book_ref={"ticker": m["ticker"]},
        volume=_f(m.get("volume_fp")) or 0.0,
    )


def fetch_events(max_close: datetime | None = None, max_pages: int = 1000) -> list[Event]:
    events: list[Event] = []
    cursor = None
    pages = 0
    now_ts = int(time.time())
    while pages < max_pages:
        params = {"limit": 200, "status": "open", "with_nested_markets": "true",
                  "min_close_ts": now_ts}
        if cursor:
            params["cursor"] = cursor
        d = get_json(f"{BASE}/events", params)
        pages += 1
        for ev in d.get("events", []):
            mkts = [bm for m in ev.get("markets", []) if (bm := _market(m, ev))]
            if max_close:
                mkts = [m for m in mkts if m.close_time is None or m.close_time <= max_close]
            if not mkts:
                continue
            events.append(Event(
                platform="kalshi",
                event_id=ev["event_ticker"],
                title=ev.get("title", ""),
                markets=mkts,
                mutually_exclusive=bool(ev.get("mutually_exclusive")),
                exhaustive=None,  # Kalshi does not publish an exhaustiveness flag
                me_source="kalshi mutually_exclusive flag" if ev.get("mutually_exclusive") else "",
                url=f"https://kalshi.com/markets/{ev.get('series_ticker', '').lower()}",
            ))
        cursor = d.get("cursor")
        if pages % 25 == 0:
            log(f"  kalshi: {pages} pages, {len(events)} events")
        if not cursor:
            break
    return events


def orderbook(ticker: str) -> dict[str, list[tuple[float, float]]]:
    """Return ask ladders {'yes': [(price, size)...], 'no': [...]} sorted cheapest first.

    Kalshi only publishes bids; a NO bid at p is a YES offer at 1-p and vice versa.
    """
    d = get_json(f"{BASE}/markets/{ticker}/orderbook")
    ob = d.get("orderbook_fp") or {}
    yes_bids = [(float(p), float(s)) for p, s in ob.get("yes_dollars") or []]
    no_bids = [(float(p), float(s)) for p, s in ob.get("no_dollars") or []]
    return {
        "yes": sorted(((round(1 - p, 4), s) for p, s in no_bids), key=lambda x: x[0]),
        "no": sorted(((round(1 - p, 4), s) for p, s in yes_bids), key=lambda x: x[0]),
    }
