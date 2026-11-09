"""Event-driven backtest of the range-covering strategy with a capital constraint.

Execution realism knobs (SimConfig):
  * pm_half_spread - Polymarket history is one price per hour (no bid/ask); we
    pay that price + this amount.  Kalshi candles carry real best bid/ask.
  * depth_kalshi / depth_pm - contracts available per 1c price level.  Buying more
    walks up the book 1c per level (price impact).  Calibrated from live books.
  * replenish_hours - liquidity we consumed comes back linearly over this time,
    so repeatedly hitting the same market gets progressively more expensive
    (this is "moving the line").
  * latency_hours - we detect at hour t but execute at hour t+latency prices, and
    only if the bundle is still profitable then.
Payouts use the venues' ACTUAL settlement results, so any bundle that was not
really riskless would show up as a loss.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .. import fees as feemod
from ..models import BinaryMarket, Event, parse_ts
from ..numeric import build_groups, solve_cover

H = 3600


@dataclass
class SimConfig:
    bankroll: float = 100.0
    pm_half_spread: float = 0.01
    depth_kalshi: float = 200.0
    depth_pm: float = 150.0
    replenish_hours: float = 6.0
    latency_hours: int = 1
    min_edge: float = 0.005          # min profit per $1 bundle, after fees
    min_apy: float = 0.0             # opportunity-cost hurdle for locking capital
    settle_delay_kalshi_h: float = 3
    settle_delay_pm_h: float = 24
    max_staleness_h: float = 3
    # Polymarket bucket events: if the YES prices of an event's buckets don't sum to
    # ~1 at time t, some prices are stale/untraded - don't trust that event at t.
    pm_sum_tolerance: float = 0.05
    # Polymarket events closed this long before their scheduled end were abandoned /
    # duplicates; their "resolution" is not a real outcome.
    pm_early_close_days: float = 2.0
    unlimited_depth: bool = False
    platforms: tuple = ("kalshi", "polymarket")
    label: str = ""


@dataclass
class SimMarket:
    bm: BinaryMarket
    yes_value: float | None
    close_ts: int
    times: list[int] = field(default_factory=list)
    a: list = field(default_factory=list)   # kalshi yes ask | pm price
    b: list = field(default_factory=list)   # kalshi yes bid | None


def load(raw_path: Path):
    raw = json.loads(raw_path.read_text())
    events: dict[str, list[Event]] = {"kalshi": [], "polymarket": []}
    sims: dict[tuple, SimMarket] = {}
    for ev in raw["kalshi"]:
        ms = []
        for m in ev["markets"]:
            bm = BinaryMarket("kalshi", m["ticker"], ev["event_id"], m["title"], m["label"],
                              close_time=parse_ts(m["close"]), fee=feemod.KALSHI, strike=m["strike"])
            res = (m.get("result") or "").lower()
            sm = SimMarket(bm, 1.0 if res == "yes" else 0.0 if res == "no" else None,
                           int(parse_ts(m["close"]).timestamp()))
            for c in sorted(m["candles"], key=lambda c: c["t"]):
                sm.times.append(c["t"]); sm.a.append(c["ask"]); sm.b.append(c["bid"])
            sims[("kalshi", bm.market_id)] = sm
            ms.append(bm)
        events["kalshi"].append(Event("kalshi", ev["event_id"], ev["title"], ms,
                                      mutually_exclusive=bool(ev.get("mutually_exclusive"))))
    for ev in raw["polymarket"]:
        ms = []
        for m in ev["markets"]:
            fee = feemod.polymarket(m.get("fees_enabled"), m.get("fee_schedule"))
            bm = BinaryMarket("polymarket", m["condition_id"], ev["event_id"], m["title"], m["label"],
                              close_time=parse_ts(m["end"]), fee=fee, rules=m.get("rules", ""))
            op = m.get("outcome_prices") or []
            yv = float(op[0]) if op else None
            end_ts = int(parse_ts(m["end"]).timestamp())
            closed = parse_ts(m.get("closed_time") or "")
            if closed and closed.timestamp() < end_ts - SimConfig.pm_early_close_days * 86400:
                continue          # abandoned / duplicate market
            close_ts = min(end_ts, int(closed.timestamp())) if closed else end_ts
            sm = SimMarket(bm, yv if yv in (0.0, 0.5, 1.0) else None, close_ts)
            for p in sorted(m["history"], key=lambda p: p["t"]):
                sm.times.append(p["t"]); sm.a.append(p["p"]); sm.b.append(None)
            sims[("polymarket", bm.market_id)] = sm
            ms.append(bm)
        if ms:
            events["polymarket"].append(Event("polymarket", ev["event_id"], ev["title"], ms,
                                              mutually_exclusive=bool(ev.get("neg_risk"))))
    groups = build_groups(events)
    return raw, groups, sims


def _last_price(sm: SimMarket, t: int, cfg: SimConfig):
    if t >= sm.close_ts or not sm.times:
        return None
    i = bisect.bisect_right(sm.times, t) - 1
    if i < 0 or t - sm.times[i] > cfg.max_staleness_h * H:
        return None
    return sm.a[i]


def pm_event_consistent(members: list[SimMarket], t: int, cfg: SimConfig) -> bool:
    ps = [_last_price(m, t, cfg) for m in members]
    if any(p is None for p in ps):
        return False
    return abs(sum(ps) - 1) <= cfg.pm_sum_tolerance


def quote(sm: SimMarket, t: int, cfg: SimConfig):
    """(yes_ask, no_ask) at time t, or None if not tradable."""
    if t >= sm.close_ts or not sm.times:
        return None
    i = bisect.bisect_right(sm.times, t) - 1
    if i < 0 or t - sm.times[i] > cfg.max_staleness_h * H:
        return None
    if sm.bm.platform == "kalshi":
        ask, bid = sm.a[i], sm.b[i]
        ya = ask if ask is not None and 0 < ask < 1 else None
        na = round(1 - bid, 4) if bid is not None and 0 < bid < 1 else None
        return ya, na
    p = sm.a[i]
    if p is None:
        return None
    ya = min(0.999, max(0.001, p + cfg.pm_half_spread))
    na = min(0.999, max(0.001, 1 - p + cfg.pm_half_spread))
    return ya, na


@dataclass
class Position:
    group: tuple
    opened: int
    settles: int
    legs: list          # (SimMarket, side, qty, cost, fee)
    qty: int
    cost: float
    fees: float
    payout: float = 0.0
    closed: bool = False


def _leg_marginal(sm: SimMarket, side: str, ask: float, consumed: float, k: int, cfg: SimConfig) -> float:
    if cfg.unlimited_depth:
        return ask
    depth = cfg.depth_kalshi if sm.bm.platform == "kalshi" else cfg.depth_pm
    return ask + 0.01 * math.floor((consumed + k) / depth)


def run(groups, sims, start: datetime, end: datetime, cfg: SimConfig) -> dict:
    t0, t1 = int(start.timestamp()), int(end.timestamp())
    solve_cache: dict = {}
    pm_events: dict[str, list[SimMarket]] = {}
    for (plat, _), sm in sims.items():
        if plat == "polymarket":
            pm_events.setdefault(sm.bm.event_id, []).append(sm)
    consistent_cache: dict = {}

    def pm_ok(sm, t):
        if sm.bm.platform != "polymarket" or len(pm_events.get(sm.bm.event_id, [])) < 2:
            return True
        k = (sm.bm.event_id, t)
        if k not in consistent_cache:
            consistent_cache[k] = pm_event_consistent(pm_events[sm.bm.event_id], t, cfg)
        return consistent_cache[k]
    # each group is only tradable between its first price and its last close
    windows = {}
    for gkey, items in groups.items():
        sm = [sims[(it.market.platform, it.market.market_id)] for it in items
              if it.market.platform in cfg.platforms and (it.market.platform, it.market.market_id) in sims]
        ts = [x.times[0] for x in sm if x.times]
        if len(ts) >= 2:
            windows[gkey] = (min(ts), max(x.close_ts for x in sm))
    cash = cfg.bankroll
    positions: list[Position] = []
    consumed: dict[tuple, tuple[float, int]] = {}   # (platform,id,side) -> (amount, ts)
    equity = []
    attempts = executed = vanished = 0

    def used(key, t):
        amt, ts = consumed.get(key, (0.0, t))
        if cfg.replenish_hours <= 0:
            return 0.0
        return max(0.0, amt * (1 - (t - ts) / (cfg.replenish_hours * H)))

    for t in range(t0 - t0 % H, t1, H):
        for p in positions:
            if not p.closed and p.settles <= t:
                cash += p.payout
                p.closed = True
        # find candidate bundles at time t
        cands = []
        for gkey, items in groups.items():
            w = windows.get(gkey)
            if not w or not (w[0] <= t < w[1]):
                continue
            live = []
            for it in items:
                if it.market.platform not in cfg.platforms:
                    continue
                sm = sims.get((it.market.platform, it.market.market_id))
                if sm is None or not pm_ok(sm, t):
                    continue
                q = quote(sm, t, cfg)
                if q is None:
                    continue
                bm = replace(sm.bm, yes_ask=q[0], no_ask=q[1])
                live.append(type(it)(bm, it.interval))
            if len(live) < 2:
                continue
            ck = (gkey, tuple((it.market.market_id, it.market.yes_ask, it.market.no_ask) for it in live))
            if ck not in solve_cache:
                solve_cache[ck] = solve_cover(live)
            sol = solve_cache[ck]
            if not sol or sol[2] > 1 - cfg.min_edge:
                continue
            legs, outcomes, unit = sol
            closes = [sims[(it.market.platform, it.market.market_id)].close_ts for it, _ in legs]
            settle = max(c + (cfg.settle_delay_kalshi_h if it.market.platform == "kalshi"
                              else cfg.settle_delay_pm_h) * H for c, (it, _) in zip(closes, legs))
            days = max((settle - t) / 86400, 1 / 24)
            roi = (1 - unit) / unit
            apy = (1 + roi) ** (365 / max(days, 1)) - 1
            if apy >= cfg.min_apy:
                cands.append((apy, gkey, legs, settle))
        cands.sort(key=lambda c: -c[0])
        for apy, gkey, legs, settle in cands:
            if cash < 1:
                break
            attempts += 1
            te = t + cfg.latency_hours * H
            # re-quote chosen legs at execution time
            exec_legs = []
            for it, side in legs:
                sm = sims[(it.market.platform, it.market.market_id)]
                q = quote(sm, te, cfg) if pm_ok(sm, te) else None
                ask = None if q is None else (q[0] if side == "yes" else q[1])
                if ask is None:
                    break
                key = (it.market.platform, it.market.market_id, side)
                exec_legs.append((sm, side, ask, used(key, te), key))
            if len(exec_legs) != len(legs):
                vanished += 1
                continue
            # add contracts one at a time while the marginal bundle stays profitable
            qty, spent, legcost = 0, 0.0, [0.0] * len(exec_legs)
            while True:
                marg = [_leg_marginal(sm, side, ask, c0, qty, cfg) for sm, side, ask, c0, _ in exec_legs]
                mfee = [sm.bm.fee.taker_fee(p, 1, round_up=False) for (sm, *_), p in zip(exec_legs, marg)]
                unit = sum(marg) + sum(mfee)
                if unit > 1 - cfg.min_edge or spent + unit > cash or any(p >= 1 for p in marg):
                    break
                qty += 1
                spent += unit
                for i, p in enumerate(marg):
                    legcost[i] += p
            if qty == 0:
                vanished += 1
                continue
            leg_rows, fees_total = [], 0.0
            for (sm, side, ask, c0, key), c in zip(exec_legs, legcost):
                f = sm.bm.fee.taker_fee(c / qty, qty)           # exact (rounded) fee on the order
                fees_total += f
                leg_rows.append((sm, side, qty, c, f))
                consumed[key] = (used(key, te) + qty, te)
            cost = sum(c for *_, c, _ in leg_rows)
            if cost + fees_total > cash:
                continue
            payout = 0.0
            for sm, side, q, *_ in leg_rows:
                v = sm.yes_value
                if v is None:
                    v = 0.0        # unresolved/unknown -> assume the worst
                payout += q * (v if side == "yes" else 1 - v)
            cash -= cost + fees_total
            positions.append(Position(gkey, te, settle, leg_rows, qty, cost, fees_total, payout))
            executed += 1
        locked = sum(p.cost + p.fees for p in positions if not p.closed)
        equity.append((t, cash + locked))
    for p in positions:
        if not p.closed:
            cash += p.payout
            p.closed = True
    trades = [{
        "group": f"{p.group[0]} {p.group[1]}",
        "opened": datetime.fromtimestamp(p.opened, timezone.utc).isoformat(),
        "settles": datetime.fromtimestamp(p.settles, timezone.utc).isoformat(),
        "qty": p.qty, "cost": round(p.cost + p.fees, 2), "fees": round(p.fees, 2),
        "payout": round(p.payout, 2), "pnl": round(p.payout - p.cost - p.fees, 2),
        "guaranteed_min": p.qty, "worst_case_held": p.payout >= p.qty - 1e-9,
        "legs": [f"{sm.bm.platform}:{sm.bm.outcome_label or sm.bm.market_id} {side.upper()} {q}@{c / q:.3f}"
                 for sm, side, q, c, f in p.legs],
    } for p in positions]
    final = cash
    days = (t1 - t0) / 86400
    return {"config": cfg.__dict__, "final": final, "pnl": final - cfg.bankroll,
            "return": final / cfg.bankroll - 1,
            "annualized": (final / cfg.bankroll) ** (365 / days) - 1,
            "trades": trades, "attempts": attempts, "executed": executed, "vanished": vanished,
            "equity": equity, "mismatches": [x for x in trades if not x["worst_case_held"]]}
