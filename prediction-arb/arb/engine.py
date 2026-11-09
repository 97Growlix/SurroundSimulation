"""Worst-case payoff evaluation and order-book-aware sizing.

An opportunity is a set of legs (buy `side` of `market`, same quantity Q on every
leg) plus an explicit list of *every* possible resolution scenario.  For each
scenario we know which legs pay out $1.  The guaranteed profit is the *minimum*
payout across scenarios minus total cost and fees.  If that minimum is positive
in every scenario, the position is an arbitrage (given the scenario list is
complete - which is exactly where the real-world risk lives, see risk_tier).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import BinaryMarket

INF = float("inf")


@dataclass
class Leg:
    market: BinaryMarket
    side: str                                   # 'yes' | 'no'
    ladder: list[tuple[float, float]] | None = None   # ask levels (price, size), cheapest first

    def top_ladder(self) -> list[tuple[float, float]]:
        price = self.market.ask(self.side)
        size = self.market.ask_size(self.side)
        return [(price, size if size else INF)] if price is not None else []

    def describe(self) -> str:
        m = self.market
        label = f" [{m.outcome_label}]" if m.outcome_label else ""
        return f"{m.platform}:{m.market_id}{label} BUY {self.side.upper()}"


@dataclass
class Fill:
    qty: float
    cost: float
    fee: float

    @property
    def avg_price(self) -> float:
        return self.cost / self.qty if self.qty else 0.0


def fill(leg: Leg, qty: float, *, use_depth: bool = True, round_fees: bool = True) -> Fill | None:
    """Cost of buying `qty` on this leg by sweeping the ask ladder. None if not enough depth."""
    ladder = leg.ladder if (use_depth and leg.ladder is not None) else leg.top_ladder()
    remaining, cost, fee = qty, 0.0, 0.0
    for price, size in ladder:
        take = min(remaining, size)
        if take <= 0:
            continue
        cost += take * price
        fee += leg.market.fee.taker_fee(price, take, round_up=round_fees)
        remaining -= take
        if remaining <= 1e-9:
            return Fill(qty, cost, fee)
    return None


@dataclass
class Opportunity:
    kind: str
    title: str
    legs: list[Leg]
    scenarios: list[tuple[str, frozenset[int]]]    # (description, indices of winning legs)
    risk_tier: str                                  # A=structural, B=needs human check, C=unverified
    notes: list[str] = field(default_factory=list)
    # filled in by evaluate()
    qty: float = 0.0
    cost: float = 0.0
    fees: float = 0.0
    worst_payout: float = 0.0
    worst_scenario: str = ""
    profit: float = 0.0
    roi: float = 0.0
    days: float = 0.0
    apy: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    depth_checked: bool = False
    meta: dict = field(default_factory=dict)       # e.g. risk score, expected profit

    def key(self) -> str:
        return self.kind + "|" + "|".join(sorted(l.describe() for l in self.legs))


def horizon_days(legs: list[Leg], now: datetime, settle_buffer_days: float,
                 unknown_days: float) -> tuple[float, bool]:
    closes = [l.market.close_time for l in legs]
    if any(c is None for c in closes):
        return unknown_days, False
    last = max(closes)
    return max((last - now).total_seconds() / 86400 + settle_buffer_days, 1.0), True


def annualize(roi: float, days: float) -> float:
    if roi <= -1:
        return -1.0
    try:
        return (1 + roi) ** (365.0 / days) - 1
    except OverflowError:
        return INF


def evaluate(opp: Opportunity, qty: float, *, now: datetime, use_depth: bool,
             round_fees: bool = True, settle_buffer_days: float = 2.0,
             unknown_days: float = 365.0) -> Opportunity | None:
    fills = []
    for leg in opp.legs:
        f = fill(leg, qty, use_depth=use_depth, round_fees=round_fees)
        if f is None:
            return None
        fills.append(f)
    cost = sum(f.cost for f in fills)
    fees_ = sum(f.fee for f in fills)
    worst, worst_label = INF, ""
    for label, winners in opp.scenarios:
        payout = 0.0
        for i in winners:
            payout += qty - opp.legs[i].market.fee.profit_fee(fills[i].avg_price, qty)
        if payout < worst:
            worst, worst_label = payout, label
    days, known = horizon_days(opp.legs, now, settle_buffer_days, unknown_days)
    opp.qty, opp.cost, opp.fees, opp.fills = qty, cost, fees_, fills
    opp.worst_payout, opp.worst_scenario = worst, worst_label
    opp.profit = worst - cost - fees_
    opp.roi = opp.profit / (cost + fees_) if cost + fees_ > 0 else 0.0
    opp.days = days
    opp.apy = annualize(opp.roi, days)
    if not known and f"unknown resolution date; assumed {unknown_days:.0f} days" not in opp.notes:
        opp.notes.append(f"unknown resolution date; assumed {unknown_days:.0f} days")
    return opp


def no_basket_at(opp: Opportunity, qty: float) -> Opportunity | None:
    """Re-select the legs of a NO basket for a given quantity.

    Any subset of NO legs in a mutually-exclusive event is valid (worst case is
    |S|-1 winners), so keep only legs with enough depth and positive edge at qty.
    """
    keep = []
    for leg in opp.legs:
        f = fill(leg, qty)
        if f is None:
            continue
        edge = qty - f.cost - f.fee - leg.market.fee.profit_fee(f.avg_price, qty)
        if edge > 0:
            keep.append(leg)
    if len(keep) < 2:
        return None
    scen = [(f"'{l.market.outcome_label or l.market.title}' wins",
             frozenset(j for j in range(len(keep)) if j != i)) for i, l in enumerate(keep)]
    scen.append(("none of the basket's outcomes wins", frozenset(range(len(keep)))))
    return Opportunity(opp.kind, opp.title, keep, scen, opp.risk_tier, list(opp.notes),
                       depth_checked=opp.depth_checked)


def size(opp: Opportunity, *, now: datetime, min_apy: float, min_profit: float,
         max_capital: float, settle_buffer_days: float = 2.0,
         unknown_days: float = 365.0) -> Opportunity | None:
    """Pick the quantity that maximises guaranteed profit while APY >= min_apy.

    Cost is convex piecewise-linear in Q (we sweep progressively worse prices), so
    profit is concave and its optimum lies on a ladder breakpoint; we evaluate all
    breakpoints plus the capital cap.
    """
    ladders = [l.ladder if l.ladder is not None else l.top_ladder() for l in opp.legs]
    flexible = opp.kind == "no_basket"
    depths = sorted(sum(s for _, s in lad) for lad in ladders)
    if not depths:
        return None
    # A NO basket only needs two legs to survive, so depth is the 2nd-largest leg.
    depth = depths[-2] if flexible and len(depths) >= 2 else depths[0]
    if depth <= 0:
        return None
    qs = set()
    for lad in ladders:
        cum = 0.0
        for _, s in lad:
            cum += s
            if cum == INF:
                break
            qs.add(math.floor(cum))
    tops = sorted(lad[0][0] for lad in ladders if lad)
    unit_cost = sum(tops[:2]) if flexible else sum(tops)   # cost bound checked again below
    cap_q = math.floor(max_capital / unit_cost) if unit_cost > 0 else 0
    qs.add(cap_q)
    qs.update(int(x) for x in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000))
    qs = sorted(q for q in qs if 1 <= q <= min(depth, cap_q))
    best = None
    kw = dict(now=now, use_depth=True, settle_buffer_days=settle_buffer_days,
              unknown_days=unknown_days)
    def at(q):
        base = no_basket_at(opp, q) if flexible else opp
        return evaluate(base, q, **kw) if base else None

    for q in qs:
        o = at(q)
        if o is None or o.cost + o.fees > max_capital * 1.0001:
            continue
        if o.profit > 0 and o.apy >= min_apy and (best is None or o.profit > best[1]):
            best = (q, o.profit)
    if best is None:
        return None
    o = at(best[0])
    if o.profit < min_profit:
        return None
    return o
