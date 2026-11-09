from datetime import datetime, timezone

import pytest

from arb import fees
from arb.backtest.sim import SimConfig, SimMarket, run
from arb.models import BinaryMarket
from arb.numeric import Interval, Priced

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 6, 3, tzinfo=timezone.utc)
H = 3600
INF = float("inf")


def setup(k_ask=0.85, pm_low=0.02, pm_mid=0.09, result_print=2):
    """Kalshi '>0.1' + Polymarket '<=0.0', '0.1' ; printed value = result_print ticks."""
    t0, close = int(T0.timestamp()), int(T0.timestamp()) + 30 * H
    times = list(range(t0, close, H))
    k = SimMarket(BinaryMarket("kalshi", "K", "E", "k", "Above 0.1%", fee=fees.NO_FEES), 
                  1.0 if result_print >= 2 else 0.0, close, times, [k_ask] * len(times), [k_ask - 0.02] * len(times))
    p0 = SimMarket(BinaryMarket("polymarket", "P0", "E2", "p", "≤0.0%", fee=fees.NO_FEES),
                   1.0 if result_print <= 0 else 0.0, close, times, [pm_low] * len(times), [None] * len(times))
    p1 = SimMarket(BinaryMarket("polymarket", "P1", "E2", "p", "0.1%", fee=fees.NO_FEES),
                   1.0 if result_print == 1 else 0.0, close, times, [pm_mid] * len(times), [None] * len(times))
    groups = {("cpi", ("M", 2026, 5)): [Priced(k.bm, Interval(2, INF)), Priced(p0.bm, Interval(-INF, 0)),
                                         Priced(p1.bm, Interval(1, 1))]}
    sims = {("kalshi", "K"): k, ("polymarket", "P0"): p0, ("polymarket", "P1"): p1}
    return groups, sims


def test_frictionless_profit_matches_arithmetic():
    groups, sims = setup()
    cfg = SimConfig(pm_sum_tolerance=10, bankroll=100, pm_half_spread=0, unlimited_depth=True, latency_hours=0,
                    replenish_hours=0, min_edge=0.001)
    r = run(groups, sims, T0, T1, cfg)
    t = r["trades"][0]
    assert t["qty"] == 104                      # floor(100 / 0.96)
    assert t["pnl"] == pytest.approx(104 * 0.04, abs=0.01)
    assert not r["mismatches"]


def test_spread_kills_marginal_edge():
    groups, sims = setup(k_ask=0.88)            # 0.88 + 0.02 + 0.09 = 0.99 at mid
    cfg = SimConfig(pm_sum_tolerance=10, bankroll=100, pm_half_spread=0.01, latency_hours=0)
    r = run(groups, sims, T0, T1, cfg)
    assert r["executed"] == 0                    # +2c of Polymarket spread makes it 1.01


def test_depth_caps_size_and_refill_limits_repeat_trading():
    groups, sims = setup()
    cfg = SimConfig(pm_sum_tolerance=10, bankroll=10_000, pm_half_spread=0, depth_kalshi=50, depth_pm=50, latency_hours=0,
                    replenish_hours=1000)
    r = run(groups, sims, T0, T1, cfg)
    total_qty = sum(t["qty"] for t in r["trades"])
    # 4c edge, 1c impact per 50 contracts per leg (x3 legs): stops well before bankroll is used
    assert 0 < total_qty < 200
    assert r["final"] > 10_000


def test_payout_uses_actual_results_whatever_the_print():
    for printed in (0, 1, 2, 5):
        groups, sims = setup(result_print=printed)
        r = run(groups, sims, T0, T1, SimConfig(pm_sum_tolerance=10, pm_half_spread=0, latency_hours=0))
        assert all(t["worst_case_held"] for t in r["trades"])


def test_stale_polymarket_prices_are_not_traded():
    """Buckets whose prices sum far from 1 indicate stale history: skip them."""
    groups, sims = setup(pm_low=0.02, pm_mid=0.09)
    # P0+P1 are only two buckets of the event, summing to 0.11 -> inconsistent
    cfg = SimConfig(pm_half_spread=0, unlimited_depth=True, latency_hours=0, pm_sum_tolerance=0.05)
    assert run(groups, sims, T0, T1, cfg)["executed"] == 0
    cfg = SimConfig(pm_half_spread=0, unlimited_depth=True, latency_hours=0, pm_sum_tolerance=10)
    assert run(groups, sims, T0, T1, cfg)["executed"] >= 1


def test_volume_cap_limits_kalshi_leg():
    groups, sims = setup()
    k = sims[("kalshi", "K")]
    k.v = [1.0] * len(k.times)               # one contract per hour traded
    k.cumv = [float(i + 1) for i in range(len(k.times))]
    cfg = SimConfig(pm_sum_tolerance=10, bankroll=10_000, pm_half_spread=0, unlimited_depth=True,
                    latency_hours=0, replenish_hours=0, max_participation=0.5)
    r = run(groups, sims, T0, T1, cfg)
    assert all(t["qty"] <= 12 for t in r["trades"])   # <= 50% of <=24 contracts/day
