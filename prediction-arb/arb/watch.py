"""Release-focused live paper trader for range-covering bundles.

  python -m arb releases                 # upcoming release calendar for registry underlyings
  python -m arb watch --bankroll 100     # paper-trade around releases (Ctrl-C to stop)

Only registry underlyings are fetched (a few dozen API calls, not the whole
exchange), so a cycle takes seconds.  Cadence follows the calendar: every 15s in
the 2 hours before a release, every 60s inside 48 hours, otherwise every 10 min.

Capital rules (arb/policy.py):
  * a new bundle must clear --min-apy (default 50%/yr) after fees at full book depth
  * one bundle may use at most --max-trade-frac of equity
  * held bundles are sold at live bids when (a) they can be sold for >= their
    guaranteed payout, or (b) cash is needed for a new bundle whose APY beats the
    held bundle's remaining APY by --switch-margin
Nothing here can place a real order: positions are written to data/paper_portfolio.json.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import numeric, policy
from .engine import Opportunity, evaluate, size
from .http import get_json, log
from .models import Event, parse_ts
from .settle import yes_value
from .sources import kalshi, polymarket
from .underlyings import REGISTRY

DATA = Path(__file__).resolve().parent.parent / "data"
PM_QUERIES = ["Core CPI MoM", "CPI MoM", "Core CPI YoY", "CPI YoY", "Core PCE MoM", "US GDP growth in Q",
              "Fed decision in", "Inflation US - Monthly", "Inflation US - Annual", "Unemployment Rate",
              "PPI YoY", "JOLTS Job Openings", "Bank of England decision", "Bank of Canada Decision",
              "Reserve Bank of Australia Decision", "Gas prices on"]


# ------------------------------------------------------------------ data

def fetch_registry_events() -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = {"kalshi": [], "polymarket": []}
    for u in REGISTRY:
        for series in u.kalshi_series:
            d = get_json(f"{kalshi.BASE}/events", {"series_ticker": series, "status": "open",
                                                   "with_nested_markets": "true", "limit": 50})
            for ev in d.get("events", []):
                ms = [bm for m in ev.get("markets", []) if (bm := kalshi._market(m, ev))]
                if ms:
                    out["kalshi"].append(Event("kalshi", ev["event_ticker"], ev.get("title", ""), ms,
                                               mutually_exclusive=bool(ev.get("mutually_exclusive"))))
    seen = set()
    for q in PM_QUERIES:
        d = get_json(f"{polymarket.GAMMA}/public-search", {"q": q, "events_status": "active",
                                                           "limit_per_type": 50})
        for h in (d.get("events") or []) if isinstance(d, dict) else []:
            if h.get("slug") in seen or not any(u.matches_polymarket(h.get("title", "")) for u in REGISTRY):
                continue
            seen.add(h.get("slug"))
            full = get_json(f"{polymarket.GAMMA}/events", {"slug": h["slug"]})
            if not full:
                continue
            ev = full[0]
            ms = [bm for m in ev.get("markets", []) if (bm := polymarket._market(m, ev))]
            if ms:
                out["polymarket"].append(Event("polymarket", str(ev["id"]), ev.get("title", ""), ms,
                                               mutually_exclusive=bool(ev.get("negRisk"))))
    return out


def release_calendar(events: dict[str, list[Event]]) -> list[dict]:
    rows = []
    for (uid, per), items in numeric.build_groups(events).items():
        k_close = [parse_ts(it.market.book_ref.get("trading_close")) or it.market.close_time
                   for it in items if it.market.platform == "kalshi"]
        k_close = [c for c in k_close if c]
        p_close = [it.market.close_time for it in items if it.market.platform == "polymarket" and it.market.close_time]
        when = min(k_close) if k_close else (min(p_close) if p_close else None)
        if when is None:
            continue
        rows.append({"release": when, "underlying": uid, "period": per,
                     "venues": sorted({it.market.platform for it in items}), "markets": len(items)})
    return sorted(rows, key=lambda r: r["release"])


def cadence_seconds(calendar: list[dict], now: datetime) -> int:
    upcoming = [r["release"] for r in calendar if r["release"] > now]
    if not upcoming:
        return 600
    dt = (min(upcoming) - now).total_seconds()
    return 15 if dt <= 2 * 3600 else 60 if dt <= 48 * 3600 else 600


# ------------------------------------------------------------------ paper portfolio

class PaperPortfolio:
    def __init__(self, path: Path, bankroll: float):
        self.path = path
        if path.exists():
            self.state = json.loads(path.read_text())
        else:
            self.state = {"start": datetime.now(timezone.utc).isoformat(), "bankroll": bankroll,
                          "cash": bankroll, "positions": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1, default=str))

    @property
    def cash(self) -> float:
        return self.state["cash"]

    def open_positions(self) -> list[dict]:
        return [p for p in self.state["positions"] if p["status"] == "open"]

    def equity_at_cost(self) -> float:
        return self.cash + sum(p["cost"] + p["fees"] for p in self.open_positions())

    def buy(self, o: Opportunity) -> dict:
        settle = max(l.market.close_time for l in o.legs) + timedelta(days=1)
        rec = {"id": uuid.uuid4().hex[:10], "opened": datetime.now(timezone.utc).isoformat(),
               "title": o.title, "tier": o.risk_tier, "qty": o.qty, "cost": o.cost, "fees": o.fees,
               "guaranteed": o.worst_payout, "apy": o.apy, "settle_after": settle.isoformat(),
               "risk": o.meta.get("risk"), "status": "open",
               "legs": [{"platform": l.market.platform, "market_id": l.market.market_id, "side": l.side,
                         "outcome": l.market.outcome_label, "avg_price": f.avg_price,
                         "book_ref": l.market.book_ref, "fee_kind": l.market.fee.kind,
                         "fee_rate": l.market.fee.rate, "fee_exp": l.market.fee.exponent}
                        for l, f in zip(o.legs, o.fills)]}
        self.state["cash"] -= o.cost + o.fees
        self.state["positions"].append(rec)
        return rec

    def close(self, p: dict, proceeds: float, how: str):
        p.update(status=how, closed=datetime.now(timezone.utc).isoformat(), proceeds=round(proceeds, 4),
                 pnl=round(proceeds - p["cost"] - p["fees"], 4))
        self.state["cash"] += proceeds


def _walk(ladder, qty):
    rem, total = qty, 0.0
    for price, sz in ladder or []:
        take = min(rem, sz)
        total += take * price
        rem -= take
        if rem <= 1e-9:
            return total
    return None


def exit_value(p: dict) -> float | None:
    """What the bundle could be sold for right now at live bids (after taker fees)."""
    from .fees import FeeModel
    total = 0.0
    pm_tokens = [l["book_ref"][l["side"]] for l in p["legs"] if l["platform"] == "polymarket"]
    pm_books = polymarket.books_full(pm_tokens) if pm_tokens else {}
    for l in p["legs"]:
        if l["platform"] == "kalshi":
            ladder = kalshi.orderbook_full(l["market_id"])[l["side"] + "_bid"]
        else:
            ladder = pm_books.get(l["book_ref"][l["side"]], {}).get("bids")
        gross = _walk(ladder, p["qty"])
        if gross is None:
            return None
        fee = FeeModel(l["fee_kind"], rate=l["fee_rate"], exponent=l["fee_exp"]).taker_fee(gross / p["qty"], p["qty"])
        total += gross - fee
    return total


# ------------------------------------------------------------------ loop

def cycle(pf: PaperPortfolio, *, min_apy: float, max_trade_frac: float, switch_margin: float,
          min_profit: float, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    msgs = []
    # 1. settle anything past its settlement time
    for p in pf.open_positions():
        if datetime.fromisoformat(p["settle_after"]) > now:
            continue
        vals = [yes_value(l["platform"], l["market_id"]) for l in p["legs"]]
        if all(v is not None for v in vals):
            payout = sum(p["qty"] * (v if l["side"] == "yes" else 1 - v) for l, v in zip(p["legs"], vals))
            pf.close(p, payout, "settled")
            msgs.append(f"SETTLED {p['title']}: ${payout:.2f} (P&L ${p['pnl']:+.2f})")
    events = fetch_registry_events()
    cal = release_calendar(events)
    # 2. sell bundles worth >= their guaranteed payout
    held = []
    for p in pf.open_positions():
        ev = exit_value(p)
        if ev is None:
            continue
        if policy.should_sell_now(ev, p["guaranteed"]):
            pf.close(p, ev, "exited")
            msgs.append(f"EXIT (worth >= payout) {p['title']}: ${ev:.2f}")
            continue
        days_left = max((datetime.fromisoformat(p["settle_after"]) - now).total_seconds() / 86400, 1 / 24)
        held.append((policy.hold_apy(ev, p["guaranteed"], days_left), ev, p))
    # 3. new bundles, best APY first
    opps = []
    for o in numeric.numeric_bundles(events):
        e = evaluate(o, 1, now=now, use_depth=False, round_fees=False)
        if e and e.profit > 0 and e.apy >= min_apy:
            opps.append(e)
    if opps:
        from .scan import attach_books
        attach_books(opps)
    for o in sorted(opps, key=lambda o: -o.apy):
        budget = pf.equity_at_cost() * max_trade_frac
        for hold, ev, p in sorted(held, key=lambda x: x[0]):
            if pf.cash >= budget or p["status"] != "open":
                continue
            if policy.should_switch(o.apy, hold, switch_margin):
                pf.close(p, ev, "switched")
                msgs.append(f"SWITCH out of {p['title']} (held {hold * 100:.0f}%/yr) -> {o.title} ({o.apy * 100:.0f}%/yr)")
        cap = min(pf.cash, budget)
        if cap < 1:
            continue
        s = size(o, now=now, min_apy=min_apy, min_profit=min_profit, max_capital=cap)
        if s:
            rec = pf.buy(s)
            msgs.append(f"BUY {s.title}: {s.qty:g} bundles, ${s.cost + s.fees:.2f} -> guaranteed ${s.worst_payout:.2f} "
                        f"({s.apy * 100:.0f}%/yr, risk ${rec['risk'] or 0:.3f})")
    pf.save()
    return msgs, cal


def watch(bankroll: float = 100.0, min_apy: float = 0.5, max_trade_frac: float = 0.5,
          switch_margin: float = 0.5, min_profit: float = 0.25, max_cycles: int | None = None):
    pf = PaperPortfolio(DATA / "paper_portfolio.json", bankroll)
    n = 0
    while max_cycles is None or n < max_cycles:
        n += 1
        try:
            msgs, cal = cycle(pf, min_apy=min_apy, max_trade_frac=max_trade_frac,
                              switch_margin=switch_margin, min_profit=min_profit)
        except Exception as ex:   # noqa: BLE001 - keep the watcher alive
            log(f"cycle failed: {ex}")
            msgs, cal = [], []
        now = datetime.now(timezone.utc)
        for m in msgs:
            log(m)
        wait = cadence_seconds(cal, now)
        nxt = next((r for r in cal if r["release"] > now), None)
        log(f"[{now:%H:%M:%S}] cash ${pf.cash:.2f}, {len(pf.open_positions())} open; next release "
            f"{nxt['underlying'] + ' ' + nxt['release'].strftime('%b %d %H:%M UTC') if nxt else '-'}; sleeping {wait}s")
        if max_cycles is not None and n >= max_cycles:
            break
        time.sleep(wait)
