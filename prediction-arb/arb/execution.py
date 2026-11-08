"""Order execution.  Only the dry-run (paper) executor is implemented.

Live trading is intentionally NOT implemented yet.  Design notes for when it is:
  * Kalshi: REST v2 with RSA-PSS signed requests (API key id + private key);
    use limit orders at the verified price with time_in_force=immediate_or_cancel.
  * Polymarket: py-clob-client with an EOA/proxy wallet key, FOK limit orders;
    collateral is USDC on Polygon.  Note geographic restrictions.
  * Legging risk: orders on different legs are not atomic.  Submit the
    thinnest / most volatile leg first as fill-or-kill, then the rest; if a later
    leg fails, immediately unwind the filled legs (and account for the loss).
  * Re-check the book immediately before sending and abort if the edge vanished.
  * Hard caps: per-trade capital, daily capital, kill switch file.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .engine import Opportunity


class DryRunExecutor:
    """Records what *would* have been bought at verified order-book prices."""

    def __init__(self, ledger: Path):
        self.ledger = ledger
        ledger.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, o: Opportunity) -> dict:
        rec = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": o.kind, "risk_tier": o.risk_tier, "title": o.title,
            "qty": o.qty, "cost": round(o.cost, 4), "fees": round(o.fees, 4),
            "worst_payout": round(o.worst_payout, 4), "profit": round(o.profit, 4),
            "roi": o.roi, "apy": o.apy, "days": o.days,
            "legs": [{"platform": l.market.platform, "market_id": l.market.market_id,
                      "event_id": l.market.event_id, "outcome": l.market.outcome_label,
                      "side": l.side, "avg_price": round(f.avg_price, 5),
                      "fee": round(f.fee, 4), "close_time": l.market.close_time.isoformat()
                      if l.market.close_time else None}
                     for l, f in zip(o.legs, o.fills)],
            "notes": o.notes, "status": "open",
        }
        with self.ledger.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec


class LiveExecutor:
    def __init__(self, *_, **__):
        raise NotImplementedError("Live trading is disabled in this build (research/paper mode only).")
