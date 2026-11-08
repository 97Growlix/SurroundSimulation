"""PredictIt public market data. Top-of-book only; no depth or end dates in the API."""
from __future__ import annotations

from .. import fees
from ..http import get_json
from ..models import BinaryMarket, Event, parse_ts

URL = "https://www.predictit.org/api/marketdata/all/"
MAX_PER_CONTRACT_USD = 850.0  # PredictIt per-contract position cap


def fetch_events() -> list[Event]:
    d = get_json(URL)
    events = []
    for mk in d.get("markets", []):
        contracts = [c for c in mk.get("contracts", []) if c.get("status") == "Open"]
        bms = []
        for c in contracts:
            ya, na = c.get("bestBuyYesCost"), c.get("bestBuyNoCost")
            bms.append(BinaryMarket(
                platform="predictit",
                market_id=str(c["id"]),
                event_id=str(mk["id"]),
                title=mk.get("name", "") if len(contracts) == 1 else f"{mk.get('name', '')} — {c.get('name', '')}",
                outcome_label=c.get("name", ""),
                yes_ask=ya if ya and 0 < ya < 1 else None,
                no_ask=na if na and 0 < na < 1 else None,
                yes_bid=c.get("bestSellYesCost"),
                no_bid=c.get("bestSellNoCost"),
                close_time=parse_ts(c.get("dateEnd")),
                url=mk.get("url", ""),
                fee=fees.PREDICTIT,
                book_ref={"contract_id": c["id"]},
            ))
        if not bms:
            continue
        events.append(Event(
            platform="predictit",
            event_id=str(mk["id"]),
            title=mk.get("name", ""),
            markets=bms,
            # PredictIt multi-contract markets are *usually* "who will win" style,
            # but the API has no flag, so this must be human-verified.
            mutually_exclusive=len(bms) > 1,
            exhaustive=None,
            me_source="ASSUMED (predictit multi-contract market; verify rules)" if len(bms) > 1 else "",
            url=mk.get("url", ""),
        ))
    return events
