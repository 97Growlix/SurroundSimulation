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
from ..numeric import build_groups, choose_cover, solve_cover
from ..underlyings import REGISTRY

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
    # cap each Kalshi leg at this fraction of the contracts that actually traded in
    # that market over the previous 24h (None = no cap). Polymarket history has no
    # volume, so its legs stay depth-model only.
    max_participation: float | None = None
    # --- capital management (added 2026-11-09) ---
    max_trade_frac: float = 1.0      # max share of current equity one new bundle may use
    early_exit: bool = False         # sell held bundles early (at bids) when it pays
    switch_margin: float = 0.5       # new APY must beat a held bundle's remaining APY by this much
    risk_chooser: bool = False       # pick lowest-risk cover among similar risk-adjusted EV
    # --- maker (resting limit order) option, Kalshi legs only ---
    maker: bool = False
    maker_window: float = 0.03       # consider bundles whose taker cost is up to $1.03
    maker_min_edge: float = 0.01     # required edge if the resting bid fills (buffer vs adverse selection)
    maker_fee_rate: float = 0.0175   # Kalshi maker fee coefficient (1/4 of taker), conservative
    maker_ttl_hours: float = 24
    maker_blackout_hours: float = 1  # pull resting bids this long before the release/close
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
    v: list = field(default_factory=list)   # kalshi hourly volume
    cumv: list = field(default_factory=list)

    def volume_24h(self, t: int) -> float:
        if not self.cumv:
            return 0.0
        i = bisect.bisect_right(self.times, t) - 1
        j = bisect.bisect_right(self.times, t - 86400) - 1
        return (self.cumv[i] if i >= 0 else 0.0) - (self.cumv[j] if j >= 0 else 0.0)


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
            run_v = 0.0
            for c in sorted(m["candles"], key=lambda c: c["t"]):
                sm.times.append(c["t"]); sm.a.append(c["ask"]); sm.b.append(c["bid"])
                run_v += c.get("vol") or 0.0
                sm.v.append(c.get("vol") or 0.0); sm.cumv.append(run_v)
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


def _annualize(growth: float, days: float) -> float:
    try:
        return growth ** (365 / days) - 1
    except OverflowError:
        return math.inf


def quote_bid(sm: SimMarket, t: int, cfg: SimConfig):
    """(yes_bid, no_bid) at time t - what we could SELL at."""
    if t >= sm.close_ts or not sm.times:
        return None
    i = bisect.bisect_right(sm.times, t) - 1
    if i < 0 or t - sm.times[i] > cfg.max_staleness_h * H:
        return None
    if sm.bm.platform == "kalshi":
        ask, bid = sm.a[i], sm.b[i]
        yb = bid if bid is not None and 0 < bid < 1 else None
        nb = round(1 - ask, 4) if ask is not None and 0 < ask < 1 else None
        return yb, nb
    p = sm.a[i]
    if p is None:
        return None
    return max(0.0, p - cfg.pm_half_spread), max(0.0, 1 - p - cfg.pm_half_spread)


def staircase_total(price: float, qty: int, depth: float, consumed: float, sign: int) -> float:
    """Total $ for qty contracts when each `depth` contracts move the price 1c
    (sign=+1 buying walks up, -1 selling walks down)."""
    total, k = 0.0, 0
    while k < qty:
        level = math.floor((consumed + k) / depth)
        room = min(qty - k, (level + 1) * depth - (consumed + k))
        room = max(1, int(math.ceil(room)))
        room = min(room, qty - k)
        total += room * max(0.0, price + sign * 0.01 * level)
        k += room
    return total


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
    exited_early: bool = False
    exit_proceeds: float = 0.0


def _leg_marginal(sm: SimMarket, side: str, ask: float, consumed: float, k: int, cfg: SimConfig) -> float:
    if cfg.unlimited_depth:
        return ask
    depth = cfg.depth_kalshi if sm.bm.platform == "kalshi" else cfg.depth_pm
    return ask + 0.01 * math.floor((consumed + k) / depth)


def run(groups, sims, start: datetime, end: datetime, cfg: SimConfig) -> dict:
    t0, t1 = int(start.timestamp()), int(end.timestamp())
    solve_cache: dict = {}
    reg = {u.id: u for u in REGISTRY}
    pm_events: dict[str, list[SimMarket]] = {}
    for (plat, _), sm in sims.items():
        if plat == "polymarket":
            pm_events.setdefault(sm.bm.event_id, []).append(sm)
    consistent_cache: dict = {}
    resting: list[dict] = []
    mk_stats = {"posted": 0, "filled": 0, "completed": 0, "completed_at_loss": 0, "unwound": 0,
                "cancelled": 0, "hedge_slippage": []}
    maker_fee = feemod.FeeModel("kalshi", rate=cfg.maker_fee_rate)

    def side_quote(sm, side, at):
        q = quote(sm, at, cfg)
        return None if q is None else (q[0] if side == "yes" else q[1])

    def side_bid(sm, side, at):
        q = quote_bid(sm, at, cfg)
        return None if q is None else (q[0] if side == "yes" else q[1])

    def hour_volume(sm, at):
        if not sm.times:
            return 0.0
        i = bisect.bisect_right(sm.times, at) - 1
        return sm.v[i] if i >= 0 and sm.times[i] > at - H and sm.v else 0.0

    def reserved():
        return sum(r["reserve"] for r in resting)

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
        maker_cands = []
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
                if cfg.risk_chooser:
                    pick, _, outc = choose_cover(live, reg.get(gkey[0]))
                    solve_cache[ck] = (pick.legs, outc, pick.cost) if pick else None
                else:
                    solve_cache[ck] = solve_cover(live)
            sol = solve_cache[ck]
            near_miss = False
            if cfg.maker and not any(r["gkey"] == gkey for r in resting):
                tkey = ("tk",) + ck
                if tkey not in solve_cache:
                    solve_cache[tkey] = solve_cover(live) if cfg.risk_chooser else sol
                tc = solve_cache[tkey]
                near_miss = bool(tc) and 1 - cfg.min_edge < tc[2] <= 1 + cfg.maker_window
            if near_miss:
                # One resting bid per bundle: try the widest-spread Kalshi legs one at a time,
                # pricing only that leg at bid+1c and everything else at the ask.
                spreads = []
                for idx, it in enumerate(live):
                    if it.market.platform != "kalshi":
                        continue
                    qb = quote_bid(sims[("kalshi", it.market.market_id)], t, cfg)
                    if not qb:
                        continue
                    for sd, ask, bid in (("yes", it.market.yes_ask, qb[0]), ("no", it.market.no_ask, qb[1])):
                        if ask is not None and bid and ask - bid >= 0.02:
                            spreads.append((ask - bid, idx, sd, round(bid + 0.01, 2)))
                taker_cost = tc[2]
                best_m = None
                for _, idx, sd, mprice in sorted(spreads, reverse=True)[:3]:
                    mview = list(live)
                    bm = live[idx].market
                    bm = replace(bm, yes_ask=mprice if sd == "yes" else bm.yes_ask,
                                 no_ask=mprice if sd == "no" else bm.no_ask)
                    mview[idx] = type(live[idx])(bm, live[idx].interval)
                    mkey = ("mk", ck, idx, sd, mprice)
                    if mkey not in solve_cache:
                        solve_cache[mkey] = solve_cover(mview)
                    ms = solve_cache[mkey]
                    if not ms or not any(it.market.market_id == bm.market_id and s2 == sd for it, s2 in ms[0]):
                        continue
                    if ms[2] <= 1 - cfg.maker_min_edge and ms[2] < taker_cost - 0.005 \
                            and (best_m is None or ms[2] < best_m[0]):
                        best_m = (ms[2], ms[0], bm.market_id, sd)
                if best_m:
                    maker_cands.append((gkey, best_m[1], best_m[2], best_m[3]))
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

        def exit_value(p: Position, at: int) -> float | None:
            total = 0.0
            for sm, side, q, c, f in p.legs:
                qb = quote_bid(sm, at, cfg)
                bid = None if qb is None else (qb[0] if side == "yes" else qb[1])
                if bid is None or bid <= 0:
                    return None
                depth = cfg.depth_kalshi if sm.bm.platform == "kalshi" else cfg.depth_pm
                key = (sm.bm.platform, sm.bm.market_id, side + "_bid")
                gross = q * bid if cfg.unlimited_depth else staircase_total(bid, q, depth, used(key, at), -1)
                total += gross - sm.bm.fee.taker_fee(gross / q, q)
            return total

        def sell(p: Position, at: int, proceeds: float):
            nonlocal cash
            for sm, side, q, *_ in p.legs:
                key = (sm.bm.platform, sm.bm.market_id, side + "_bid")
                consumed[key] = (used(key, at) + q, at)
            cash += proceeds
            p.closed, p.exited_early, p.exit_proceeds = True, True, proceeds

        open_pos = [p for p in positions if not p.closed]
        if cfg.early_exit:
            # (a) a bundle that can be sold for >= its guaranteed payout: take it now
            for p in open_pos:
                ev = exit_value(p, t)
                if ev is not None and ev >= p.qty:
                    sell(p, t, ev)
        equity_now = cash + reserved() + sum(p.cost + p.fees for p in positions if not p.closed)
        for apy, gkey, legs, settle in cands:
            budget = equity_now * cfg.max_trade_frac
            if cfg.early_exit and cash < budget:
                # (b) switch: sell held bundles whose remaining return is much worse
                held = []
                for p in positions:
                    if p.closed or p.group == gkey:
                        continue
                    ev = exit_value(p, t)
                    if ev is None or ev <= 0:
                        continue
                    days_left = max((p.settles - t) / 86400, 1 / 24)
                    hold_apy = (p.qty / ev) ** (365 / max(days_left, 1)) - 1
                    held.append((hold_apy, ev, p))
                for hold_apy, ev, p in sorted(held, key=lambda x: x[0]):
                    if cash >= budget or hold_apy > apy - cfg.switch_margin:
                        break
                    sell(p, t, ev)
            spend_cap = min(cash, budget)
            if spend_cap < 1:
                continue
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
            qcap = math.inf
            if cfg.max_participation is not None:
                for sm, *_ in exec_legs:
                    if sm.bm.platform == "kalshi":
                        qcap = min(qcap, math.floor(cfg.max_participation * sm.volume_24h(te)))
            while True:
                marg = [_leg_marginal(sm, side, ask, c0, qty, cfg) for sm, side, ask, c0, _ in exec_legs]
                mfee = [sm.bm.fee.taker_fee(p, 1, round_up=False) for (sm, *_), p in zip(exec_legs, marg)]
                unit = sum(marg) + sum(mfee)
                if unit > 1 - cfg.min_edge or spent + unit > spend_cap or any(p >= 1 for p in marg) or qty >= qcap:
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
            equity_now = cash + reserved() + sum(p.cost + p.fees for p in positions if not p.closed)
            payout = 0.0
            for sm, side, q, *_ in leg_rows:
                v = sm.yes_value
                if v is None:
                    v = 0.0        # unresolved/unknown -> assume the worst
                payout += q * (v if side == "yes" else 1 - v)
            cash -= cost + fees_total
            positions.append(Position(gkey, te, settle, leg_rows, qty, cost, fees_total, payout))
            executed += 1
        if cfg.maker:
            # ---- 1. manage resting bids: cancel, or fill + hedge at the NEXT hour's prices
            for r in list(resting):
                msm, mside, price = r["maker"][0], r["maker"][1], r["price"]
                hedge_now = [side_quote(sm, sd, t) for sm, sd in r["hedges"]]
                est = None if any(h is None for h in hedge_now) else \
                    price + maker_fee.taker_fee(price, 1, round_up=False) + sum(
                        h + sm.bm.fee.taker_fee(h, 1, round_up=False) for (sm, _), h in zip(r["hedges"], hedge_now))
                if t >= r["expires"] or t >= r["close_min"] - cfg.maker_blackout_hours * H \
                        or est is None or est > 1 - cfg.maker_min_edge:
                    cash += r["reserve"]; resting.remove(r); mk_stats["cancelled"] += 1
                    continue
                ask_now = side_quote(msm, mside, t)
                vol = hour_volume(msm, t)
                if ask_now is None or ask_now > price or vol <= 0:
                    continue
                q = int(min(r["qty"], math.floor(vol)))
                if q < 1:
                    continue
                cash += r["reserve"]; resting.remove(r); mk_stats["filled"] += 1
                cost_m = price * q + maker_fee.taker_fee(price, q)
                te = t + cfg.latency_hours * H
                hedge_rows, hedge_cost, ok = [], 0.0, True
                for sm, sd in r["hedges"]:
                    a = side_quote(sm, sd, te)
                    if a is None:
                        ok = False
                        break
                    depth = cfg.depth_kalshi if sm.bm.platform == "kalshi" else cfg.depth_pm
                    key = (sm.bm.platform, sm.bm.market_id, sd)
                    gross = q * a if cfg.unlimited_depth else staircase_total(a, q, depth, used(key, te), +1)
                    f = sm.bm.fee.taker_fee(gross / q, q)
                    hedge_rows.append((sm, sd, q, gross, f, key))
                    hedge_cost += gross + f
                if ok:
                    mk_stats["hedge_slippage"].append(hedge_cost / q - (est - price - maker_fee.taker_fee(price, 1, round_up=False)))
                b = side_bid(msm, mside, te)
                proceeds = 0.0
                if b:
                    depth = cfg.depth_kalshi
                    proceeds = staircase_total(b, q, depth, used((msm.bm.platform, msm.bm.market_id, mside + "_bid"), te), -1)
                    proceeds -= msm.bm.fee.taker_fee(proceeds / q, q) if proceeds > 0 else 0
                loss_complete = cost_m + hedge_cost - q if ok else math.inf
                loss_unwind = cost_m - proceeds
                if ok and (loss_complete <= 0 or loss_complete <= loss_unwind) and cash >= cost_m + hedge_cost:
                    rows = [(msm, mside, q, price * q, maker_fee.taker_fee(price, q))] + \
                           [(sm, sd, qq, g, f) for sm, sd, qq, g, f, _ in hedge_rows]
                    for *_, key in hedge_rows:
                        consumed[key] = (used(key, te) + q, te)
                    payout = sum(qq * ((sm.yes_value or 0.0) if sd == "yes" else 1 - (sm.yes_value or 0.0))
                                 for sm, sd, qq, *_ in rows)
                    settle = max(sm.close_ts + (cfg.settle_delay_kalshi_h if sm.bm.platform == "kalshi"
                                                else cfg.settle_delay_pm_h) * H for sm, *_ in rows)
                    cost = sum(g for *_, g, _ in rows); fees_t = sum(f for *_, f in rows)
                    cash -= cost + fees_t
                    positions.append(Position(r["gkey"], te, settle, rows, q, cost, fees_t, payout))
                    mk_stats["completed"] += 1
                    mk_stats["completed_at_loss"] += loss_complete > 0
                else:
                    cash += proceeds - cost_m
                    pos = Position(r["gkey"], te, te, [(msm, mside, q, price * q, maker_fee.taker_fee(price, q))],
                                   q, price * q, maker_fee.taker_fee(price, q), 0.0)
                    pos.closed, pos.exited_early, pos.exit_proceeds = True, True, proceeds
                    positions.append(pos)
                    mk_stats["unwound"] += 1
            # ---- 2. post new resting bids on near-miss bundles
            for gkey, legs, mid, mside in maker_cands:
                if any(r["gkey"] == gkey for r in resting):
                    continue
                j = next(k for k, (it, sd) in enumerate(legs) if it.market.market_id == mid and sd == mside)
                msm = sims[("kalshi", mid)]
                b = side_bid(msm, mside, t)
                if b is None:
                    continue
                price = round(b + 0.01, 2)
                hedges = [(sims[(it.market.platform, it.market.market_id)], sd) for k, (it, sd) in enumerate(legs) if k != j]
                hq = [side_quote(sm, sd, t) for sm, sd in hedges]
                if any(h is None for h in hq):
                    continue
                unit = price + maker_fee.taker_fee(price, 1, round_up=False) + sum(
                    h + sm.bm.fee.taker_fee(h, 1, round_up=False) for (sm, _), h in zip(hedges, hq))
                if unit > 1 - cfg.maker_min_edge:
                    continue
                closes = [msm.close_ts] + [sm.close_ts for sm, _ in hedges]
                settle = max(closes) + cfg.settle_delay_pm_h * H
                apy = (1 / unit) ** (365 / max((settle - t) / 86400, 1)) - 1
                if apy < cfg.min_apy:
                    continue
                equity_now = cash + reserved() + sum(p.cost + p.fees for p in positions if not p.closed)
                cap = min(cash, equity_now * cfg.max_trade_frac)
                qty = math.floor(cap / unit)
                qty = min(qty, int(min(cfg.depth_kalshi if sm.bm.platform == "kalshi" else cfg.depth_pm
                                       for sm, _ in hedges)))
                if cfg.max_participation is not None:
                    for sm, _ in hedges + [(msm, mside)]:
                        if sm.bm.platform == "kalshi":
                            qty = min(qty, math.floor(cfg.max_participation * sm.volume_24h(t)))
                if qty < 1:
                    continue
                reserve = qty * unit
                cash -= reserve
                resting.append({"gkey": gkey, "maker": (msm, mside), "price": price, "qty": qty, "hedges": hedges,
                                "expires": t + int(cfg.maker_ttl_hours * H), "close_min": min(closes),
                                "reserve": reserve})
                mk_stats["posted"] += 1
        locked = sum(p.cost + p.fees for p in positions if not p.closed)
        equity.append((t, cash + locked + reserved()))
    for r in resting:
        cash += r["reserve"]
    for p in positions:
        if not p.closed:
            cash += p.payout
            p.closed = True
    trades = [{
        "group": f"{p.group[0]} {p.group[1]}",
        "opened": datetime.fromtimestamp(p.opened, timezone.utc).isoformat(),
        "settles": datetime.fromtimestamp(p.settles, timezone.utc).isoformat(),
        "qty": p.qty, "cost": round(p.cost + p.fees, 2), "fees": round(p.fees, 2),
        "payout": round(p.exit_proceeds if p.exited_early else p.payout, 2),
        "pnl": round((p.exit_proceeds if p.exited_early else p.payout) - p.cost - p.fees, 2),
        "exited_early": p.exited_early,
        "guaranteed_min": p.qty, "worst_case_held": p.exited_early or p.payout >= p.qty - 1e-9,
        "legs": [f"{sm.bm.platform}:{sm.bm.outcome_label or sm.bm.market_id} {side.upper()} {q}@{c / q:.3f}"
                 for sm, side, q, c, f in p.legs],
    } for p in positions]
    final = cash
    days = (t1 - t0) / 86400
    return {"config": cfg.__dict__, "final": final, "pnl": final - cfg.bankroll,
            "return": final / cfg.bankroll - 1,
            "annualized": _annualize(final / cfg.bankroll, days),
            "trades": trades, "attempts": attempts, "executed": executed, "vanished": vanished,
            "equity": equity, "mismatches": [x for x in trades if not x["worst_case_held"]],
            "maker": {k: (v if k != "hedge_slippage" else
                          {"n": len(v), "mean": (sum(v) / len(v)) if v else 0.0,
                           "worse_than_planned": sum(x > 0.001 for x in v)})
                      for k, v in mk_stats.items()}}
