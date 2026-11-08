"""Opportunity generators.  Each returns un-evaluated Opportunity objects."""
from __future__ import annotations

from .engine import Leg, Opportunity
from .models import BinaryMarket, Event


def _unit_edge_no(m: BinaryMarket) -> float:
    p = m.no_ask
    return 1 - p - m.fee.taker_fee(p, 1, round_up=False) - m.fee.profit_fee(p, 1)


def no_basket(ev: Event) -> Opportunity | None:
    """Mutually-exclusive event: buy NO on a set S of outcomes.

    At most one outcome can win, so at least |S|-1 of the NO legs pay out.  Only
    outcomes whose NO is cheap enough to add positive edge are included.  This
    does NOT require the outcome list to be exhaustive.
    """
    if not ev.mutually_exclusive or len(ev.markets) < 2:
        return None
    mk = [m for m in ev.markets if m.no_ask is not None and _unit_edge_no(m) > 0]
    if len(mk) < 2:
        return None
    if sum(_unit_edge_no(m) for m in mk) <= 1:   # quick screen; exact check in engine
        return None
    legs = [Leg(m, "no") for m in mk]
    scen = [(f"'{m.outcome_label or m.title}' wins", frozenset(j for j in range(len(legs)) if j != i))
            for i, m in enumerate(mk)]
    scen.append(("none of the basket's outcomes wins", frozenset(range(len(legs)))))
    tier = "A" if ev.platform in ("kalshi", "polymarket") else "B"
    notes = [f"mutual exclusivity: {ev.me_source}"]
    if ev.platform == "polymarket":
        notes.append("check fallbackResolutionType: an 'equal_split' 50/50 resolution of one leg reduces payout")
    return Opportunity("no_basket", ev.title, legs, scen, tier, notes)


def yes_basket(ev: Event, assume_exhaustive: bool = False) -> Opportunity | None:
    """Buy YES on every outcome of a mutually-exclusive AND exhaustive event."""
    if not ev.mutually_exclusive or len(ev.markets) < 2:
        return None
    if any(m.yes_ask is None for m in ev.markets):
        return None
    if sum(m.yes_ask for m in ev.markets) >= 1:
        return None
    if ev.exhaustive is not True and not assume_exhaustive:
        return None
    legs = [Leg(m, "yes") for m in ev.markets]
    scen = [(f"'{m.outcome_label or m.title}' wins", frozenset([i])) for i, m in enumerate(ev.markets)]
    tier = "A" if ev.exhaustive is True else "C"
    notes = ["ASSUMES the listed outcomes are exhaustive - a single unlisted winner loses everything"] \
        if ev.exhaustive is not True else []
    return Opportunity("yes_basket", ev.title, legs, scen, tier, notes)


def cross_pair(a: BinaryMarket, b: BinaryMarket, *, invert: bool = False,
               confirmed: bool = False, score: float | None = None) -> list[Opportunity]:
    """Same proposition on two venues: buy YES on one and NO on the other.

    invert=True means a.YES is equivalent to b.NO (questions phrased oppositely).
    """
    out = []
    b_same, b_opp = ("no", "yes") if invert else ("yes", "no")
    for a_side, b_side in (("yes", b_opp), ("no", b_same)):
        pa, pb = a.ask(a_side), b.ask(b_side)
        if pa is None or pb is None or pa + pb >= 1:
            continue
        legs = [Leg(a, a_side), Leg(b, b_side)]
        scen = [("proposition true", frozenset([0 if a_side == "yes" else 1])),
                ("proposition false", frozenset([1 if a_side == "yes" else 0]))]
        notes = ["cross-venue: resolution sources/wording/deadlines must be identical",
                 f"A: {a.title} [{a.outcome_label}]", f"B: {b.title} [{b.outcome_label}]"]
        if score is not None:
            notes.append(f"match score {score:.0f}")
        gap = 1 - pa - pb
        if gap > 0.10:
            notes.append(f"WARNING: venues disagree by {gap * 100:.0f} cents - large gaps are almost always "
                         "different propositions (thresholds, categories, dates), stale books, or "
                         "an outcome already known on one venue")
        out.append(Opportunity("cross_pair", a.title, legs, scen,
                               "B" if confirmed else "C", notes))
    return out
