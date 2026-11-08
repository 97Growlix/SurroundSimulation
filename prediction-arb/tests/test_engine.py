from datetime import datetime, timedelta, timezone

import pytest

from arb import fees
from arb.engine import Leg, annualize, evaluate, fill, size
from arb.models import BinaryMarket, Event
from arb.strategies import cross_pair, no_basket, yes_basket

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mk(mid, yes=None, no=None, fee=fees.NO_FEES, days=30, label="", platform="kalshi"):
    return BinaryMarket(platform=platform, market_id=mid, event_id="E", title=f"Q {mid}",
                        outcome_label=label or mid, yes_ask=yes, no_ask=no, fee=fee,
                        close_time=NOW + timedelta(days=days))


def test_kalshi_fee_rounds_up_to_cent():
    assert fees.KALSHI.taker_fee(0.5, 1) == 0.02          # 0.0175 -> 0.02
    assert fees.KALSHI.taker_fee(0.5, 100) == 1.75
    assert fees.KALSHI.taker_fee(0.5, 1, round_up=False) == pytest.approx(0.0175)


def test_polymarket_fee_peak_rate():
    f = fees.polymarket(True, {"rate": 0.25, "exponent": 2})
    # Effective rate at p=0.5 is 0.25 * 0.25**2 = 1.5625% of notional
    assert f.taker_fee(0.5, 100) / (0.5 * 100) == pytest.approx(0.015625)
    assert fees.polymarket(False, {"rate": 0.25}).taker_fee(0.5, 100) == 0


def test_predictit_profit_fee():
    assert fees.PREDICTIT.profit_fee(0.6, 10) == pytest.approx(0.4)


def test_no_basket_worst_case_is_n_minus_one():
    ev = Event("kalshi", "E", "who wins", [mk("a", no=0.60), mk("b", no=0.70), mk("c", no=0.65)],
               mutually_exclusive=True)
    o = no_basket(ev)
    assert o is not None
    e = evaluate(o, 10, now=NOW, use_depth=False)
    assert e.cost == pytest.approx(19.5)
    assert e.worst_payout == pytest.approx(20)    # two of three NOs pay
    assert e.profit == pytest.approx(0.5)


def test_no_basket_requires_mutual_exclusivity():
    ev = Event("kalshi", "E", "x", [mk("a", no=0.1), mk("b", no=0.1)], mutually_exclusive=False)
    assert no_basket(ev) is None


def test_no_basket_skips_unprofitable_legs():
    ev = Event("kalshi", "E", "x", [mk("a", no=0.40), mk("b", no=0.45), mk("c", no=0.999)],
               mutually_exclusive=True)
    o = no_basket(ev)
    assert [l.market.market_id for l in o.legs] == ["a", "b", "c"]  # 0.999 still +edge w/o fees
    ev.markets[2].fee = fees.FeeModel("polymarket", rate=0.01, exponent=0)  # 1% of notional > 0.1c edge
    o = no_basket(ev)
    assert [l.market.market_id for l in o.legs] == ["a", "b"]


def test_yes_basket_needs_exhaustive():
    ms = [mk("a", yes=0.3), mk("b", yes=0.3), mk("c", yes=0.3)]
    ev = Event("kalshi", "E", "x", ms, mutually_exclusive=True, exhaustive=None)
    assert yes_basket(ev) is None
    ev.exhaustive = True
    o = yes_basket(ev)
    e = evaluate(o, 10, now=NOW, use_depth=False)
    assert e.profit == pytest.approx(1.0)
    assert o.risk_tier == "A"


def test_cross_pair_scenarios_and_profit():
    a = mk("a", yes=0.40, no=0.62, platform="kalshi")
    b = mk("b", yes=0.55, no=0.50, platform="polymarket")
    opps = cross_pair(a, b)
    assert len(opps) == 1                          # only A.yes + B.no (0.90) < 1
    e = evaluate(opps[0], 100, now=NOW, use_depth=False)
    assert e.profit == pytest.approx(10.0)
    assert {lbl for lbl, _ in opps[0].scenarios} == {"proposition true", "proposition false"}
    assert opps[0].risk_tier == "C"


def test_cross_pair_inverted():
    a = mk("a", yes=0.40, no=0.62)
    b = mk("b", yes=0.50, no=0.55)                 # b.NO means the same as a.YES
    opps = cross_pair(a, b, invert=True, confirmed=True)
    sides = {(o.legs[0].side, o.legs[1].side) for o in opps}
    assert ("yes", "yes") in sides and opps[0].risk_tier == "B"


def test_fill_walks_the_book_and_reports_shortfall():
    leg = Leg(mk("a", no=0.5), "no", ladder=[(0.5, 10), (0.6, 10)])
    f = fill(leg, 15)
    assert f.cost == pytest.approx(5 + 3)
    assert fill(leg, 25) is None


def test_size_stops_where_edge_runs_out():
    ev = Event("kalshi", "E", "x", [mk("a", no=0.4), mk("b", no=0.4)], mutually_exclusive=True)
    o = no_basket(ev)
    o.legs[0].ladder = [(0.40, 50), (0.65, 1000)]
    o.legs[1].ladder = [(0.40, 50), (0.65, 1000)]
    s = size(o, now=NOW, min_apy=0.0, min_profit=0.01, max_capital=10_000)
    assert s.qty == 50                              # beyond 50 the basket costs > 1/unit
    assert s.profit == pytest.approx(50 - 40)


def test_annualize():
    assert annualize(0.01, 365) == pytest.approx(0.01)
    assert annualize(0.01, 36.5) == pytest.approx(1.01 ** 10 - 1)
