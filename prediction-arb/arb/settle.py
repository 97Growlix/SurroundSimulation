"""Resolve paper trades against what actually happened on each venue."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .http import get_json
from .sources.kalshi import BASE as KALSHI
from .sources.polymarket import GAMMA

_cache: dict[tuple[str, str], float | None] = {}


def kalshi_yes_value(ticker: str) -> float | None:
    """1.0 if YES won, 0.0 if NO won, fractional for scalar/void settlements, None if unresolved."""
    for path in (f"/markets/{ticker}", f"/historical/markets/{ticker}"):
        try:
            m = get_json(KALSHI + path).get("market") or {}
        except RuntimeError:
            continue
        res = (m.get("result") or "").lower()
        if res in ("yes", "no"):
            return 1.0 if res == "yes" else 0.0
        if m.get("status") in ("settled", "finalized") and m.get("settlement_value_dollars") is not None:
            return float(m["settlement_value_dollars"])
        if m:
            return None
    return None


def polymarket_yes_value(condition_id: str) -> float | None:
    d = get_json(f"{GAMMA}/markets", {"condition_ids": condition_id, "closed": "true"})
    if not d:
        return None
    m = d[0]
    if not m.get("closed"):
        return None
    prices = json.loads(m.get("outcomePrices") or "[]")
    if not prices:
        return None
    p = float(prices[0])
    # Only treat as final once prices are pinned (resolved), else unresolved.
    return p if p in (0.0, 0.5, 1.0) or m.get("umaResolutionStatus") == "resolved" else None


def yes_value(platform: str, market_id: str) -> float | None:
    key = (platform, market_id)
    if key not in _cache:
        if platform == "kalshi":
            _cache[key] = kalshi_yes_value(market_id)
        elif platform == "polymarket":
            _cache[key] = polymarket_yes_value(market_id)
        else:
            _cache[key] = None     # PredictIt has no settlement API
    return _cache[key]


def realized_payout(rec: dict, values: list[float]) -> float:
    qty = rec["qty"]
    total = 0.0
    for leg, v in zip(rec["legs"], values):
        frac = v if leg["side"] == "yes" else 1 - v
        total += qty * frac
    return total


def settle_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for rec in recs:
        if rec.get("status") != "open":
            continue
        vals = [yes_value(l["platform"], l["market_id"]) for l in rec["legs"]]
        if any(v is None for v in vals):
            continue
        payout = realized_payout(rec, vals)
        rec.update(status="settled", settled_ts=datetime.now(timezone.utc).isoformat(),
                   payout=round(payout, 4),
                   realized_profit=round(payout - rec["cost"] - rec["fees"], 4),
                   matched_expectation=payout >= rec["worst_payout"] - 1e-6)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return recs
