"""Polymarket (international CLOB) public market data via the Gamma + CLOB APIs."""
from __future__ import annotations

import json
from datetime import datetime

from .. import fees
from ..http import get_json, log
from ..models import BinaryMarket, Event, parse_ts

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _jl(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x or "[]")
    except (TypeError, ValueError):
        return []


def _market(m: dict, ev: dict) -> BinaryMarket | None:
    if m.get("closed") or not m.get("active") or not m.get("acceptingOrders", True):
        return None
    if not m.get("enableOrderBook"):
        return None
    tokens = _jl(m.get("clobTokenIds"))
    outcomes = _jl(m.get("outcomes"))
    if len(tokens) != 2:
        return None
    best_ask, best_bid = _f(m.get("bestAsk")), _f(m.get("bestBid"))
    yes_ask = best_ask if best_ask and 0 < best_ask < 1 else None
    # NO token's best ask mirrors the YES best bid in Polymarket's unified book.
    no_ask = round(1 - best_bid, 6) if best_bid and 0 < best_bid < 1 else None
    label = m.get("groupItemTitle") or ""
    if not label and outcomes and outcomes[0].lower() != "yes":
        label = f"{outcomes[0]} (vs {outcomes[1]})" if len(outcomes) > 1 else outcomes[0]
    return BinaryMarket(
        platform="polymarket",
        market_id=m.get("conditionId") or m["id"],
        event_id=str(ev.get("id", "")),
        title=m.get("question") or ev.get("title", ""),
        outcome_label=label,
        yes_ask=yes_ask,
        no_ask=no_ask,
        yes_bid=best_bid,
        no_bid=round(1 - best_ask, 6) if best_ask else None,
        close_time=parse_ts(m.get("endDate")) or parse_ts(ev.get("endDate")),
        url=f"https://polymarket.com/event/{ev.get('slug', '')}",
        rules=m.get("description") or "",
        fee=fees.polymarket(bool(m.get("feesEnabled")), m.get("feeSchedule")),
        book_ref={"yes": tokens[0], "no": tokens[1], "neg_risk": bool(m.get("negRisk"))},
        volume=_f(m.get("volumeNum")) or 0.0,
    )


def fetch_events(max_close: datetime | None = None, max_pages: int = 400) -> list[Event]:
    events: list[Event] = []
    cursor = None
    pages = 0
    params_base = {"limit": 500, "active": "true", "closed": "false"}
    if max_close:
        params_base["end_date_max"] = max_close.strftime("%Y-%m-%dT%H:%M:%SZ")
    while pages < max_pages:
        params = dict(params_base)
        if cursor:
            params["after_cursor"] = cursor
        d = get_json(f"{GAMMA}/events/keyset", params)
        pages += 1
        for ev in d.get("events", []):
            mkts = [bm for m in ev.get("markets", []) if (bm := _market(m, ev))]
            if max_close:
                mkts = [m for m in mkts if m.close_time is None or m.close_time <= max_close]
            if not mkts:
                continue
            neg = bool(ev.get("negRisk") or ev.get("enableNegRisk"))
            events.append(Event(
                platform="polymarket",
                event_id=str(ev["id"]),
                title=ev.get("title", ""),
                markets=mkts,
                mutually_exclusive=neg,
                # negRiskAugmented events can gain new outcomes later => never exhaustive.
                exhaustive=None,
                me_source="polymarket negRisk (on-chain mutually exclusive)" if neg else "",
                url=f"https://polymarket.com/event/{ev.get('slug', '')}",
            ))
        cursor = d.get("next_cursor")
        if pages % 20 == 0:
            log(f"  polymarket: {pages} pages, {len(events)} events")
        if not cursor or not d.get("events"):
            break
    return events


def orderbooks(token_ids: list[str]) -> dict[str, list[tuple[float, float]]]:
    """Ask ladders per token id, cheapest first."""
    out: dict[str, list[tuple[float, float]]] = {}
    for i in range(0, len(token_ids), 50):
        chunk = token_ids[i:i + 50]
        d = get_json(f"{CLOB}/books", method="POST", json_body=[{"token_id": t} for t in chunk])
        for book in d or []:
            asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
            out[book.get("asset_id")] = sorted(asks, key=lambda x: x[0])
    return out


def books_full(token_ids: list[str]) -> dict[str, dict]:
    """{token: {'asks': [...cheapest first], 'bids': [...best first]}}"""
    out: dict[str, dict] = {}
    for i in range(0, len(token_ids), 50):
        chunk = token_ids[i:i + 50]
        d = get_json(f"{CLOB}/books", method="POST", json_body=[{"token_id": t} for t in chunk])
        for book in d or []:
            out[book.get("asset_id")] = {
                "asks": sorted(((float(a["price"]), float(a["size"])) for a in book.get("asks", [])),
                               key=lambda x: x[0]),
                "bids": sorted(((float(b["price"]), float(b["size"])) for b in book.get("bids", [])),
                               key=lambda x: -x[0])}
    return out
