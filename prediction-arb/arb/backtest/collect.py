"""Download settled markets + hourly price history for registry underlyings.

Kalshi: real hourly best bid / best ask / volume candles (recent markets under
/series/.../candlesticks, markets settled before the historical cutoff under
/historical/markets/.../candlesticks).
Polymarket: hourly price history (a single price per hour - no bid/ask), final
outcome prices, fee schedule.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..http import get_json, log
from ..models import parse_ts
from ..sources.kalshi import BASE as K
from ..sources.polymarket import CLOB, GAMMA
from ..underlyings import REGISTRY

OUT = Path(__file__).resolve().parents[2] / "data" / "backtest"
PM_QUERIES = ["Core CPI MoM", "CPI MoM", "Core CPI YoY", "CPI YoY", "Core PCE MoM",
              "US GDP growth in Q", "Fed decision in", "Inflation US - Monthly", "Inflation US - Annual",
              "Unemployment Rate", "PPI YoY", "Producer Price Index", "JOLTS Job Openings",
              "Bank of England decision", "Bank of Canada Decision", "Reserve Bank of Australia Decision",
              "Gas prices on"]


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _patient(fn, tries: int = 5, pause: float = 60.0):
    """Retry through sustained rate limiting (pause a minute between bursts)."""
    import time
    for i in range(tries):
        try:
            return fn()
        except RuntimeError as ex:
            if i == tries - 1:
                raise
            log(f"    throttled ({ex}); pausing {pause:.0f}s")
            time.sleep(pause)


def kalshi_candles(series: str, ticker: str, start: int, end: int, historical: bool) -> list[dict]:
    path = f"/historical/markets/{ticker}/candlesticks" if historical else \
        f"/series/{series}/markets/{ticker}/candlesticks"
    out = []
    s = start
    while s < end:
        e = min(end, s + 3600 * 4000)
        try:
            d = _patient(lambda: get_json(K + path, {"start_ts": s, "end_ts": e, "period_interval": 60}))
        except RuntimeError as ex:
            log(f"    candles failed {ticker}: {ex}")
            break
        for c in d.get("candlesticks", []):
            ya, yb = c.get("yes_ask", {}), c.get("yes_bid", {})
            out.append({
                "t": c["end_period_ts"],
                "ask": _f(ya.get("close_dollars", ya.get("close"))),
                "bid": _f(yb.get("close_dollars", yb.get("close"))),
                "vol": _f(c.get("volume_fp", c.get("volume"))) or 0.0,
            })
        s = e
    return out


def collect_kalshi(start: datetime, end: datetime, skip_series: set | None = None) -> list[dict]:
    cutoff = parse_ts(get_json(K + "/historical/cutoff").get("market_settled_ts"))
    events = []
    for u in REGISTRY:
        for series in u.kalshi_series:
            if skip_series and series in skip_series:
                continue
            cursor = None
            while True:
                p = {"series_ticker": series, "status": "settled", "with_nested_markets": "true", "limit": 100}
                if cursor:
                    p["cursor"] = cursor
                d = _patient(lambda: get_json(K + "/events", p))
                for ev in d.get("events", []):
                    mk = ev.get("markets") or []
                    if not mk:
                        # markets settled before the historical cutoff live in a separate archive
                        mk = _patient(lambda: get_json(K + "/historical/markets", {
                            "event_ticker": ev["event_ticker"], "limit": 200}).get("markets", []))
                    closes = [parse_ts(m.get("close_time")) for m in mk if m.get("close_time")]
                    if not closes or not (start <= max(closes) <= end):
                        continue
                    log(f"  kalshi {ev['event_ticker']} ({len(mk)} markets)")
                    rows = []
                    for m in mk:
                        op, cl = parse_ts(m.get("open_time")), parse_ts(m.get("close_time"))
                        hist = bool(cutoff and cl and cl < cutoff)
                        rows.append({
                            "ticker": m["ticker"], "label": m.get("yes_sub_title") or "",
                            "title": m.get("title") or ev.get("title"),
                            "strike": {"type": m.get("strike_type"), "floor": m.get("floor_strike"),
                                       "cap": m.get("cap_strike")},
                            "result": m.get("result"), "open": m.get("open_time"), "close": m.get("close_time"),
                            "candles": kalshi_candles(series, m["ticker"], _ts(max(op, start)), _ts(cl) + 3600, hist),
                        })
                    events.append({"platform": "kalshi", "event_id": ev["event_ticker"], "series": series,
                                   "title": ev.get("title", ""), "mutually_exclusive": ev.get("mutually_exclusive"),
                                   "markets": rows})
                cursor = d.get("cursor")
                if not cursor or not d.get("events"):
                    break
    return events


def collect_polymarket(start: datetime, end: datetime, skip_ids: set | None = None) -> list[dict]:
    seen, events = set(), []
    pages_max = 12
    for q in PM_QUERIES:
        for page in range(1, pages_max + 1):
            d = get_json(f"{GAMMA}/public-search", {"q": q, "events_status": "closed", "limit_per_type": 50,
                                                     "page": page})
            hits = d.get("events", []) if isinstance(d, dict) else []
            if not hits:
                break
            for h in hits:
                if h.get("slug") in seen:
                    continue
                seen.add(h.get("slug"))
                if not any(u.matches_polymarket(h.get("title", "")) for u in REGISTRY):
                    continue
                endd = parse_ts(h.get("endDate"))
                if not endd or not (start <= endd <= end):
                    continue
                if skip_ids and str(h.get("id")) in skip_ids:
                    continue
                full = get_json(f"{GAMMA}/events", {"slug": h["slug"]})
                if not full:
                    continue
                ev = full[0]
                log(f"  polymarket {ev['title']} ({len(ev.get('markets', []))} markets)")
                rows = []
                for m in ev.get("markets", []):
                    toks = json.loads(m.get("clobTokenIds") or "[]")
                    if len(toks) != 2:
                        continue
                    created = parse_ts(m.get("startDate") or m.get("createdAt")) or start
                    s, e = _ts(max(created, start)), _ts(parse_ts(m.get("endDate")) or endd) + 86400
                    hist = []
                    for cs in range(s, e, 14 * 86400):   # API rejects windows longer than ~2 weeks
                        try:
                            hist += get_json(f"{CLOB}/prices-history", {
                                "market": toks[0], "startTs": cs, "endTs": min(e, cs + 14 * 86400),
                                "fidelity": 60}).get("history", [])
                        except Exception as ex:      # noqa: BLE001 - keep going, log the gap
                            log(f"    history gap {m.get('groupItemTitle')}: {ex}")
                    rows.append({
                        "condition_id": m.get("conditionId"), "label": m.get("groupItemTitle") or "",
                        "title": m.get("question"), "outcome_prices": json.loads(m.get("outcomePrices") or "[]"),
                        "fees_enabled": bool(m.get("feesEnabled")), "fee_schedule": m.get("feeSchedule"),
                        "end": m.get("endDate") or ev.get("endDate"), "rules": m.get("description") or "",
                        "closed_time": m.get("closedTime"),
                        "history": [{"t": int(p["t"]), "p": float(p["p"])} for p in hist],
                    })
                events.append({"platform": "polymarket", "event_id": str(ev["id"]), "title": ev["title"],
                               "neg_risk": bool(ev.get("negRisk")), "markets": rows})
    return events


def _cached(name: str, fn):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    d = fn()
    f.write_text(json.dumps(d))
    return d


def _topup(name: str, fn):
    """Load cached events and fetch only what's missing (new registry underlyings)."""
    f = OUT / f"{name}.json"
    have = json.loads(f.read_text()) if f.exists() else []
    new = fn(have)
    if new:
        have += new
        f.write_text(json.dumps(have))
    log(f"{name}: {len(have)} events ({len(new)} new)")
    return have


def main(start: datetime, end: datetime) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"start": start.isoformat(), "end": end.isoformat(),
            "kalshi": _topup("kalshi", lambda have: collect_kalshi(
                start, end, skip_series={e["series"] for e in have})),
            "polymarket": _topup("polymarket", lambda have: collect_polymarket(
                start, end, skip_ids={e["event_id"] for e in have}))}
    path = OUT / "raw.json"
    path.write_text(json.dumps(data))
    return path


if __name__ == "__main__":
    import sys
    s = datetime.fromisoformat(sys.argv[1]).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
    print(main(s, e))
