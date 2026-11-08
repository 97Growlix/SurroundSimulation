from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .fees import FeeModel, NO_FEES


def parse_ts(s) -> datetime | None:
    if not s or s == "NA":
        return None
    try:
        s = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass
class BinaryMarket:
    """One YES/NO contract paying $1. Prices are best asks (cost to buy) in dollars."""
    platform: str
    market_id: str
    event_id: str
    title: str                      # full question
    outcome_label: str = ""        # e.g. candidate name inside a multi-outcome event
    yes_ask: float | None = None
    no_ask: float | None = None
    yes_bid: float | None = None
    no_bid: float | None = None
    yes_ask_size: float | None = None
    no_ask_size: float | None = None
    close_time: datetime | None = None
    url: str = ""
    rules: str = ""
    fee: FeeModel = NO_FEES
    book_ref: dict = field(default_factory=dict)
    volume: float = 0.0
    strike: dict = field(default_factory=dict)   # Kalshi strike_type/floor_strike/cap_strike

    def ask(self, side: str) -> float | None:
        return self.yes_ask if side == "yes" else self.no_ask

    def ask_size(self, side: str) -> float | None:
        return self.yes_ask_size if side == "yes" else self.no_ask_size


@dataclass
class Event:
    platform: str
    event_id: str
    title: str
    markets: list[BinaryMarket]
    mutually_exclusive: bool = False
    # True only when the platform guarantees exactly one market resolves YES.
    exhaustive: bool | None = None
    me_source: str = ""             # why we believe it is mutually exclusive
    url: str = ""
