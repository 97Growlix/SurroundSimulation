"""Combinatorial (range-covering) arbitrage on numeric underlyings.

Every market on a registered underlying is translated into the set of printed
values for which it pays YES, expressed as an integer interval of "ticks"
(tick = value / step).  A NO position pays on the complement.  We then pick the
cheapest set of positions (one contract each) such that EVERY possible printed
value is covered by at least one winning position - an integer program solved
exactly with scipy's MILP solver.  If that cost is below $1, the bundle pays at
least $1 whatever the number turns out to be.

This finds, e.g., Kalshi "core CPI above 0.1%" + Polymarket buckets "<=0.0%" and
"0.1%" (exactly one pays in every world), as well as within-venue mispricings
between Kalshi's threshold and exact-value ladders.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, milp

from .engine import Leg, Opportunity
from .models import BinaryMarket, Event
from .underlyings import REGISTRY, Underlying

INF = math.inf
_MON = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


@dataclass(frozen=True)
class Interval:
    lo: float   # inclusive tick, -inf allowed
    hi: float   # inclusive tick, +inf allowed

    def contains(self, t: int) -> bool:
        return self.lo <= t <= self.hi


def _tick(v: float, step: float) -> int:
    return int(round(float(v) / step))


# ---------------------------------------------------------------- parsing

_FED = [   # order matters: "more than 25" before plain "25"
    (re.compile(r"cut\s*(>|more than)\s*25|50\+?\s*bps?\s*decrease"), -INF, -2),
    (re.compile(r"cut\s*(1\s*-\s*)?25|^25\s*bps?\s*decrease"), -1, -1),
    (re.compile(r"maintain|no change|hold"), 0, 0),
    (re.compile(r"hike\s*(>|more than)\s*25|50\+?\s*bps?\s*increase"), 2, INF),
    (re.compile(r"hike\s*(1\s*-\s*)?25|^25\+?\s*bps?\s*increase"), 1, 1),
]


def _fed_interval(label: str, step: float) -> Interval | None:
    l = label.lower().strip()
    if step != 25:
        return None
    for pat, lo, hi in _FED:
        if pat.search(l):
            return Interval(lo, hi)
    return None


def kalshi_interval(m: BinaryMarket, u: Underlying) -> Interval | None:
    st = m.strike or {}
    typ, fl, cap = st.get("type"), st.get("floor"), st.get("cap")
    s = u.step
    if typ == "greater" and fl is not None:
        return Interval(_tick(fl, s) + 1, INF)
    if typ == "greater_or_equal" and fl is not None:
        return Interval(_tick(fl, s), INF)
    if typ == "less" and cap is not None:
        return Interval(-INF, _tick(cap, s) - 1)
    if typ == "less_or_equal" and cap is not None:
        return Interval(-INF, _tick(cap, s))
    if typ == "between" and fl is not None and cap is not None:
        return Interval(_tick(fl, s), _tick(cap, s))
    if typ == "custom":
        lab = (m.outcome_label or "").lower()
        if (fi := _fed_interval(lab, s)):
            return fi
        mm = re.fullmatch(r"exactly\s*(-?\d+(?:\.\d+)?)\s*%?", lab.strip())
        if mm:
            t = _tick(mm.group(1), s)
            return Interval(t, t)
    return None


_N = r"(-?\d+(?:\.\d+)?)"


def polymarket_interval(label: str, u: Underlying, half_open_upper: bool) -> Interval | None:
    """Parse bucket labels like '<=0.0%', '0.1%', '0.6%+', '2.0-2.5%', '<1.5%', '>2.5%'."""
    l = label.lower().replace("−", "-").replace("–", "-").replace("—", "-").replace(" ", "")
    l = l.replace("$", "").replace(",", "")
    if u.step < 0.01:                       # JOLTS-style "7.2M" labels
        l = re.sub(r"(\d)m\b", r"\1", l).replace("m+", "+").replace("m", "")
    s = u.step
    if (fi := _fed_interval(label, s)):
        return fi
    if (mm := re.fullmatch(rf"(?:≤|<=){_N}%?", l)) or (mm := re.fullmatch(rf"{_N}%?(?:orbelow|orless|orlower)", l)):
        return Interval(-INF, _tick(mm.group(1), s))
    if (mm := re.fullmatch(rf"<{_N}%?", l)):
        return Interval(-INF, _tick(mm.group(1), s) - 1)
    if (mm := re.fullmatch(rf"(?:≥|>=){_N}%?", l)) or (mm := re.fullmatch(rf"{_N}%?\+", l)) \
            or (mm := re.fullmatch(rf"{_N}%?(?:orabove|ormore|orhigher)", l)):
        return Interval(_tick(mm.group(1), s), INF)
    if (mm := re.fullmatch(rf">{_N}%?", l)):
        return Interval(_tick(mm.group(1), s) + 1, INF)
    if (mm := re.fullmatch(rf"{_N}%?(?:-|to){_N}%?", l)):
        lo_v, hi_v = float(mm.group(1)), float(mm.group(2))
        lo = math.ceil(lo_v / s - 1e-6)
        if half_open_upper:
            hi = math.ceil(hi_v / s - 1e-6) - 1      # [lo, hi)
        else:
            hi = math.floor(hi_v / s + 1e-6)         # [lo, hi], e.g. $4.0000-4.0099 -> 4.000..4.009
        return Interval(lo, hi)
    if (mm := re.fullmatch(rf"{_N}%?", l)):
        t = _tick(mm.group(1), s)
        return Interval(t, t)
    return None


def _validate_buckets(ivs: list[Interval]) -> bool:
    """Polymarket bucket sets must not overlap (sanity check on our parsing)."""
    srt = sorted(ivs, key=lambda i: i.lo)
    return all(a.hi < b.lo for a, b in zip(srt, srt[1:]))


# ---------------------------------------------------------------- periods

_FULLMON = r"(january|february|march|april|may|june|july|august|september|october|november|december)"


def period_of(ev: Event, m: BinaryMarket, gran: str = "M") -> tuple | None:
    t = ev.title.lower()
    if gran == "D":
        if ev.platform == "kalshi" and (k := re.search(r"-(\d\d)([A-Z]{3})(\d\d)$", ev.event_id)):
            if k.group(2).lower() in _MON:
                return ("D", 2000 + int(k.group(1)), _MON[k.group(2).lower()], int(k.group(3)))
        if (mm := re.search(_FULLMON + r"\s+(\d{1,2})\b(?:,?\s*(20\d\d))?", t)) and m.close_time:
            mon, day = _MON[mm.group(1)[:3]], int(mm.group(2))
            y = int(mm.group(3)) if mm.group(3) else m.close_time.year
            if not mm.group(3) and mon == 12 and m.close_time.month == 1:
                y -= 1
            return ("D", y, mon, day)
        return None
    if (q := re.search(r"\bq([1-4])\s*(20\d\d)", t)):
        return ("Q", int(q.group(2)), int(q.group(1)))
    # Reference month from the TITLE first: Kalshi tickers are sometimes dated by the
    # release date (PPI "...-26NOV13" is October data), titles name the data month.
    # whole month words only ("decision" must not read as "dec")
    mm = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
                   r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b\.?(?:\s+(20\d\d))?", t)
    if mm:
        mon = _MON[mm.group(1)[:3]]
        if mm.group(2):
            return ("M", int(mm.group(2)), mon)
        if m.close_time:
            # the data month is at or before the release month
            y = m.close_time.year if mon <= m.close_time.month else m.close_time.year - 1
            return ("M", y, mon)
    if ev.platform == "kalshi" and (k := re.search(r"-(\d\d)([A-Z]{3})(?:\d\d)?$", ev.event_id)):
        if k.group(2).lower() in _MON:
            return ("M", 2000 + int(k.group(1)), _MON[k.group(2).lower()])
    return None


# ---------------------------------------------------------------- grouping

@dataclass
class Priced:
    market: BinaryMarket
    interval: Interval


def build_groups(events: dict[str, list[Event]]) -> dict[tuple, list[Priced]]:
    groups: dict[tuple, list[Priced]] = defaultdict(list)
    for u in REGISTRY:
        for ev in events.get("kalshi", []):
            if ev.event_id.split("-")[0] not in u.kalshi_series:
                continue
            for m in ev.markets:
                iv = kalshi_interval(m, u)
                per = period_of(ev, m, u.period)
                if iv and per:
                    groups[(u.id, per)].append(Priced(m, iv))
        for ev in events.get("polymarket", []):
            if not u.matches_polymarket(ev.title):
                continue
            rules = " ".join(m.rules for m in ev.markets[:1]).lower()
            half_open = bool(re.search(r"(higher|upper) (range )?bracket|falls exactly .{0,80}higher", rules))
            parsed = [(m, polymarket_interval(m.outcome_label or "", u, half_open)) for m in ev.markets]
            if any(iv is None for _, iv in parsed) or not _validate_buckets([iv for _, iv in parsed]):
                continue        # refuse to guess on events we can't parse cleanly
            for m, iv in parsed:
                per = period_of(ev, m, u.period)
                if per:
                    groups[(u.id, per)].append(Priced(m, iv))
    return groups


# ---------------------------------------------------------------- solver

def _unit_cost(m: BinaryMarket, side: str) -> float | None:
    p = m.ask(side)
    if p is None:
        return None
    return p + m.fee.taker_fee(p, 1, round_up=False)


def solve_cover(items: list[Priced]) -> tuple[list[tuple[Priced, str]], list[int], float] | None:
    """Cheapest set of legs covering every possible printed value. Returns (legs, outcome ticks, cost)."""
    bounds = [b for it in items for b in (it.interval.lo, it.interval.hi) if math.isfinite(b)]
    if not bounds:
        return None
    outcomes = list(range(int(min(bounds)) - 1, int(max(bounds)) + 2))
    cands = []
    for it in items:
        for side in ("yes", "no"):
            c = _unit_cost(it.market, side)
            if c is not None:
                cands.append((it, side, c))
    if len(cands) < 2:
        return None
    A = np.zeros((len(outcomes), len(cands)))
    for j, (it, side, _) in enumerate(cands):
        for i, t in enumerate(outcomes):
            inside = it.interval.contains(t)
            A[i, j] = 1.0 if inside == (side == "yes") else 0.0
    cost = np.array([c for _, _, c in cands])
    res = milp(cost, constraints=LinearConstraint(A, lb=np.ones(len(outcomes)), ub=np.inf),
               integrality=np.ones(len(cands)), bounds=(0, 1))
    if not res.success:
        return None
    pick = [j for j, x in enumerate(res.x) if x > 0.5]
    return [(cands[j][0], cands[j][1]) for j in pick], outcomes, float(cost[pick].sum())


# ---------------------------------------------------------------- risk-aware choice

def _mid(m: BinaryMarket) -> float | None:
    if m.yes_ask is not None and m.no_ask is not None:
        return (m.yes_ask + 1 - m.no_ask) / 2
    return None


def implied_distribution(items: list[Priced], outcomes: list[int]) -> np.ndarray:
    """Market-implied probability of each printed value: non-negative least squares fit
    of every two-sided market's mid price, constrained to sum to 1."""
    from scipy.optimize import nnls
    rows, target = [], []
    for it in items:
        mid = _mid(it.market)
        if mid is None:
            continue
        rows.append([1.0 if it.interval.contains(t) else 0.0 for t in outcomes])
        target.append(mid)
    n = len(outcomes)
    if not rows:
        return np.full(n, 1 / n)
    big = 100.0
    A = np.vstack([np.array(rows), big * np.ones((1, n))])
    b = np.concatenate([np.array(target), [big]])
    p, _ = nnls(A, b)
    tot = p.sum()
    return p / tot if tot > 0 else np.full(n, 1 / n)


@dataclass
class CoverChoice:
    legs: list            # [(Priced, side)]
    cost: float           # per $1 bundle incl. unit fees
    expected_payout: float
    risk: float
    risk_parts: dict

    @property
    def guaranteed(self) -> float:
        return 1 - self.cost

    @property
    def expected_profit(self) -> float:
        return self.expected_payout - self.cost


def risk_of(legs, u: Underlying | None) -> tuple[float, dict]:
    """Rough $-per-bundle penalty for things the worst-case math can't see.

    legging     - each extra leg is another order that can fail / move before it fills
    cross_venue - two platforms: non-atomic execution, capital on both, two settlement processes
    rules       - residual rule differences for this underlying (registry rule_risk)
    """
    venues = {it.market.platform for it, _ in legs}
    cross = len(venues) > 1
    parts = {"legging": 0.002 * max(len(legs) - 1, 0),
             "cross_venue": 0.004 if cross else 0.0,
             "rules": (u.rule_risk if (u and cross) else 0.0)}
    return sum(parts.values()), parts


def candidate_covers(items: list[Priced], k: int = 6):
    """Up to k distinct cheapest covers (MILP with no-good cuts), each still < $1."""
    bounds = [b for it in items for b in (it.interval.lo, it.interval.hi) if math.isfinite(b)]
    if not bounds:
        return [], []
    outcomes = list(range(int(min(bounds)) - 1, int(max(bounds)) + 2))
    cands = []
    for it in items:
        for side in ("yes", "no"):
            c = _unit_cost(it.market, side)
            if c is not None:
                cands.append((it, side, c))
    if len(cands) < 2:
        return [], outcomes
    A = np.zeros((len(outcomes), len(cands)))
    for j, (it, side, _) in enumerate(cands):
        for i, t in enumerate(outcomes):
            A[i, j] = 1.0 if it.interval.contains(t) == (side == "yes") else 0.0
    cost = np.array([c for _, _, c in cands])
    cons = [LinearConstraint(A, lb=np.ones(len(outcomes)), ub=np.inf)]
    found = []
    for _ in range(k):
        res = milp(cost, constraints=cons, integrality=np.ones(len(cands)), bounds=(0, 1))
        if not res.success:
            break
        pick = [j for j, x in enumerate(res.x) if x > 0.5]
        c = float(cost[pick].sum())
        if c >= 1:
            break
        found.append(([(cands[j][0], cands[j][1]) for j in pick], c, A[:, pick].sum(axis=1)))
        row = np.zeros(len(cands)); row[pick] = 1
        cons.append(LinearConstraint(row.reshape(1, -1), lb=-np.inf, ub=len(pick) - 1))
    return found, outcomes


def choose_cover(items: list[Priced], u: Underlying | None = None, *, k: int = 6,
                 similar_tol: float = 0.003, risk_aversion: float = 1.0
                 ) -> tuple[CoverChoice | None, list[CoverChoice], list[int]]:
    """Score = expected profit - risk_aversion * risk ($ per $1 bundle).  Among covers
    within `similar_tol` of the best score, pick the lowest-risk one."""
    found, outcomes = candidate_covers(items, k)
    if not found:
        return None, [], outcomes
    p = implied_distribution(items, outcomes)
    choices = []
    for legs, c, payout_vec in found:
        r, parts = risk_of(legs, u)
        choices.append(CoverChoice(legs, c, float(p @ payout_vec), r, parts))
    score = lambda ch: ch.expected_profit - risk_aversion * ch.risk  # noqa: E731
    best = max(score(ch) for ch in choices)
    near = [ch for ch in choices if score(ch) >= best - similar_tol]
    pick = min(near, key=lambda ch: (round(ch.risk, 6), -ch.expected_profit))
    return pick, choices, outcomes


def _fmt_tick(t: int, step: float) -> str:
    v = t * step
    return f"{v:g}"


def numeric_bundles(events: dict[str, list[Event]]) -> list[Opportunity]:
    out = []
    reg = {u.id: u for u in REGISTRY}
    for (uid, per), items in build_groups(events).items():
        u = reg[uid]
        pick, alts, outcomes = choose_cover(items, u)
        if pick is None:
            continue
        chosen = pick.legs
        legs = [Leg(it.market, side) for it, side in chosen]
        lo, hi = outcomes[0], outcomes[-1]
        scen = []
        for t in outcomes:
            label = (f"print <= {_fmt_tick(t, u.step)}" if t == lo else
                     f"print >= {_fmt_tick(t, u.step)}" if t == hi else f"print = {_fmt_tick(t, u.step)}")
            winners = frozenset(k for k, (it, side) in enumerate(chosen)
                                if it.interval.contains(t) == (side == "yes"))
            scen.append((label, winners))
        venues = {l.market.platform for l in legs}
        tier = "A" if len(venues) == 1 else "B"
        title = f"{uid} {per[1]}-{'Q' if per[0] == 'Q' else ''}{per[2]:02d}" + (f"-{per[3]:02d}" if len(per) > 3 else "")
        notes = [f"settles on: {u.source}"]
        if u.caveats:
            notes.append(f"caveat: {u.caveats}")
        if tier == "B":
            notes.append("cross-venue bundle on a registry-verified underlying; legs are not atomic")
        notes.append(f"risk ${pick.risk:.3f}/bundle {pick.risk_parts}; guaranteed ${pick.guaranteed:.3f}, "
                     f"expected ${pick.expected_profit:.3f} per $1 bundle "
                     f"(lowest-risk of {len(alts)} alternatives within $0.003 of the best risk-adjusted score)")
        out.append(Opportunity("numeric", title, legs, scen, tier, notes,
                               meta={"risk": pick.risk, "risk_parts": pick.risk_parts,
                                     "expected_profit_unit": pick.expected_profit,
                                     "guaranteed_unit": pick.guaranteed, "alternatives": len(alts),
                                     "underlying": uid}))
    return out
