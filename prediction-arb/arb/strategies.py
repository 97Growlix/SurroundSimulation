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


# ---------------------------------------------------------------------------
# Implication ("ladder") arbitrage.
#
# Markets that differ only by a threshold are nested: "over 10.5" implies
# "over 7.5".  If the narrower one (subset) is priced ABOVE the broader one
# (superset), buy YES superset + NO subset:
#   subset true  -> superset true too: YES pays, NO loses   -> $1
#   only superset true:               both pay              -> $2
#   neither:                          NO pays               -> $1
# Worst case $1 for a cost of yes_ask(super) + no_ask(sub) < $1.
# ---------------------------------------------------------------------------
import re as _re

_NUM = _re.compile(r"(?<![\w.])[$]?(\d[\d,]*(?:\.\d+)?)\s*([kKmMbB%]?)(?![\w.])")
_UP = _re.compile(r"\b(over|above|more than|greater than|at least|or more|or higher|exceed\w*|higher than|reach\w*)\b|\d\+")
_DOWN = _re.compile(r"\b(under|below|less than|fewer than|or less|or lower|lower than)\b")
_UNSAFE = _re.compile(r"\b(between|exactly|range|spread|margin|by more than|wins by|to)\b|[-–]\s*\d|\(\s*[+-]")
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9}


_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_DATE = _re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d\d))?\b"
                    r"|\b(?:end of |before |by )(20\d\d)\b")
_DEADLINE = _re.compile(r"\b(by|before|prior to|no later than)\b")


def _date_value(text: str, close):
    """Parse a single deadline date. Missing years are inferred as the year whose
    date is closest to the market's own end time (end times are often 04:59 UTC
    on the following day, so naive close_time.year is wrong for Dec 31)."""
    from datetime import date, timedelta
    found = _DATE.findall(text)
    if len(found) != 1:
        return None
    mon, day, yr, yr_only = found[0]
    if yr_only:
        return date(int(yr_only), 1, 1)
    mon_i, day_i = _MONTHS.index(mon[:3]) + 1, int(day)
    try:
        if yr:
            return date(int(yr), mon_i, day_i)
        if close is None:
            return None
        ref = (close - timedelta(hours=12)).date()
        cands = [date(y, mon_i, day_i) for y in (ref.year - 1, ref.year, ref.year + 1)]
    except ValueError:
        return None
    return min(cands, key=lambda d: abs((d - ref).days))


def _ladder_key(m: BinaryMarket):
    """-> (template, value, direction) where direction=+1 means larger value is the
    STRICTER (subset) proposition, -1 means smaller value is stricter."""
    title, label = m.title.lower(), (m.outcome_label or "").lower()
    text = f"{title} || {label}"
    if _UNSAFE.search(text):
        return None
    # Deadline ladders: "by Nov 11" is stricter than "by Nov 30".
    if _DATE.search(text):
        if not _DEADLINE.search(text):
            return None          # "in November" style buckets are not nested
        # prefer whichever text carries an explicit year
        cands = [t for t in (label, title) if _re.search(r"20\d\d", t)] + [title, label]
        v = next((d for t in cands if (d := _date_value(t, m.close_time))), None)
        if v is None:
            return None
        return ("date|" + _DATE.sub("@", title) + "|" + _DATE.sub("@", label), {v}, -1)
    nums = _NUM.findall(title) or _NUM.findall(label)
    vals = {float(n.replace(",", "")) * _SCALE.get(sfx.lower(), 1) for n, sfx in nums}
    up, down = bool(_UP.search(text)), bool(_DOWN.search(text))
    if up == down or not vals:
        return None
    # label must match too (e.g. "India (2+ times)" vs "Club Wingstop (3+ times)" are different words)
    return ("num|" + _NUM.sub("#", title) + "|" + _NUM.sub("#", label), vals, 1 if up else -1)


def ladder_pairs(ev: Event) -> list[Opportunity]:
    if ev.mutually_exclusive or len(ev.markets) < 2:
        return []
    groups: dict[tuple[str, int], list[tuple[float, BinaryMarket]]] = {}
    keyed = [(m, _ladder_key(m)) for m in ev.markets]
    templates: dict[str, list] = {}
    for m, k in keyed:
        if k:
            templates.setdefault(k[0], []).append((m, k))
    for tmpl, items in templates.items():
        # numbers shared by every market in the group are part of the template (e.g. years)
        common = set.intersection(*(set(k[1]) for _, k in items))
        for m, (_, vals, d) in items:
            v = set(vals) - common
            if len(v) == 1:
                groups.setdefault((tmpl, d), []).append((v.pop(), m))
    out = []
    for (tmpl, d), items in groups.items():
        if len({v for v, _ in items}) < 2:
            continue
        for v1, m1 in items:
            for v2, m2 in items:
                # subset = stricter threshold
                if not ((d == 1 and v2 > v1) or (d == -1 and v2 < v1)):
                    continue
                sup, sub = m1, m2
                if sup.yes_ask is None or sub.no_ask is None or sup.yes_ask + sub.no_ask >= 1:
                    continue
                legs = [Leg(sup, "yes"), Leg(sub, "no")]
                scen = [(f"stricter '{sub.outcome_label or sub.title}' true", frozenset([0])),
                        ("only looser threshold true", frozenset([0, 1])),
                        ("neither true", frozenset([1]))]
                out.append(Opportunity(
                    "ladder", ev.title, legs, scen, "A",
                    [f"implication: '{sub.title}' => '{sup.title}' (same venue/event, threshold {v2} vs {v1})",
                     "verify both markets use the same underlying, source and measurement time"]))
    return out
