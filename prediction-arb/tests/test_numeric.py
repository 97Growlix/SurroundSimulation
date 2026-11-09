from datetime import datetime, timezone

import pytest

from arb import fees
from arb.engine import evaluate
from arb.models import BinaryMarket, Event
from arb.numeric import (Interval, kalshi_interval, numeric_bundles, period_of,
                         polymarket_interval, solve_cover, Priced)
from arb.underlyings import REGISTRY

CPI = next(u for u in REGISTRY if u.id == "us_core_cpi_mom")
FED = next(u for u in REGISTRY if u.id == "fed_decision_bps")
GDP = next(u for u in REGISTRY if u.id == "us_gdp_q_saar")
INF = float("inf")
NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)
CLOSE = datetime(2026, 11, 11, 4, 59, tzinfo=timezone.utc)


def km(ticker, typ, floor=None, cap=None, label="", yes=None, no=None):
    return BinaryMarket("kalshi", ticker, ticker.rsplit("-", 1)[0], label, label, yes_ask=yes, no_ask=no,
                        close_time=CLOSE, fee=fees.NO_FEES, strike={"type": typ, "floor": floor, "cap": cap})


def pm(label, yes=None, no=None):
    return BinaryMarket("polymarket", "pm" + label, "E", label, label, yes_ask=yes, no_ask=no,
                        close_time=CLOSE, fee=fees.NO_FEES)


def test_kalshi_strike_semantics():
    assert kalshi_interval(km("A-T0.1", "greater", 0.1), CPI) == Interval(2, INF)      # >0.1 => 0.2+
    assert kalshi_interval(km("A-T0.5", "less", cap=0.5), CPI) == Interval(-INF, 4)    # <0.5 => <=0.4
    assert kalshi_interval(km("A-B", "between", 0.2, 0.3), CPI) == Interval(2, 3)
    assert kalshi_interval(km("A-X", "custom", label="Exactly -0.1%"), CPI) == Interval(-1, -1)
    assert kalshi_interval(km("A-C26", "custom", label="Cut >25bps"), FED) == Interval(-INF, -2)
    assert kalshi_interval(km("A-H0", "custom", label="Fed maintains rate"), FED) == Interval(0, 0)


def test_polymarket_bucket_parsing():
    assert polymarket_interval("≤0.0%", CPI, False) == Interval(-INF, 0)
    assert polymarket_interval("0.1%", CPI, False) == Interval(1, 1)
    assert polymarket_interval("0.6%+", CPI, False) == Interval(6, INF)
    assert polymarket_interval("≥3.1%", CPI, False) == Interval(31, INF)
    # GDP brackets are [low, high) because ties go to the higher bracket
    assert polymarket_interval("1.5% to 2.0%", GDP, True) == Interval(15, 19)
    assert polymarket_interval("<1.5%", GDP, True) == Interval(-INF, 14)
    assert polymarket_interval("50+ bps decrease", FED, False) == Interval(-INF, -2)
    assert polymarket_interval("No change", FED, False) == Interval(0, 0)
    assert polymarket_interval("Negative", CPI, False) is None


def test_period_parsing():
    k = Event("kalshi", "KXCPICORE-26OCT", "Core CPI in October", [])
    p = Event("polymarket", "1", "Core CPI MoM - October 2026", [])
    f = Event("polymarket", "2", "Fed Decision in December?", [])
    m = pm("x")
    m.close_time = datetime(2026, 12, 9, tzinfo=timezone.utc)
    assert period_of(k, m) == period_of(p, m) == ("M", 2026, 10)
    assert period_of(f, m) == ("M", 2026, 12)


def test_cpi_example_from_the_docs():
    """Kalshi YES '>0.1%' @0.87 + Polymarket '<=0.0%' @0.02 + '0.1%' @0.09 = 0.98 covers everything."""
    items = [Priced(km("K-T0.1", "greater", 0.1, label="Above 0.1%", yes=0.87, no=0.14), Interval(2, INF))]
    for lab, y, n in [("≤0.0%", .02, .995), ("0.1%", .09, .97), ("0.2%", .48, .54), ("0.3%", .35, .67),
                      ("0.4%", .09, .93), ("0.5%", .01, .997), ("0.6%+", .003, .999)]:
        items.append(Priced(pm(lab, y, n), polymarket_interval(lab, CPI, False)))
    legs, outcomes, cost = solve_cover(items)
    assert cost == pytest.approx(0.98)
    assert {(it.market.market_id, s) for it, s in legs} == {("K-T0.1", "yes"), ("pm≤0.0%", "yes"), ("pm0.1%", "yes")}


def test_bundle_worst_case_is_one_dollar_end_to_end():
    k_ev = Event("kalshi", "KXCPICORE-26OCT", "Core CPI in October",
                 [km("KXCPICORE-26OCT-T0.1", "greater", 0.1, label="Above 0.1%", yes=0.87, no=0.14)])
    p_ev = Event("polymarket", "9", "Core CPI MoM - October 2026",
                 [pm(l, y, n) for l, y, n in [("≤0.0%", .02, .995), ("0.1%", .09, .97), ("0.2%", .48, .54),
                                              ("0.3%", .35, .67), ("0.4%", .09, .93), ("0.5%", .01, .997),
                                              ("0.6%+", .003, .999)]], mutually_exclusive=True)
    ops = numeric_bundles({"kalshi": [k_ev], "polymarket": [p_ev]})
    assert len(ops) == 1 and ops[0].risk_tier == "B"
    e = evaluate(ops[0], 100, now=NOW, use_depth=False)
    assert e.worst_payout == pytest.approx(100)
    assert e.profit == pytest.approx(2.0)
    # every printed value from <=-0.1 to >=0.3 is a scenario, each with exactly one winner here
    assert all(len(w) == 1 for _, w in ops[0].scenarios)


def test_no_bundle_when_fairly_priced():
    items = [Priced(km("K-T0.1", "greater", 0.1, yes=0.90, no=0.12), Interval(2, INF)),
             Priced(pm("≤0.1%", 0.11, 0.9), Interval(-INF, 1))]
    _, _, cost = solve_cover(items)
    assert cost >= 1


def test_overlapping_buckets_rejected():
    from arb.numeric import _validate_buckets
    assert not _validate_buckets([Interval(1, 3), Interval(3, 5)])
    assert _validate_buckets([Interval(-INF, 0), Interval(1, 1), Interval(2, INF)])


def test_choose_prefers_lower_risk_when_expected_is_similar():
    from arb.numeric import choose_cover
    # Same-venue cover (Kalshi ">0.1" YES + Kalshi exact "<=0.1" style legs) vs cross-venue cover
    # with nearly identical cost: the chooser should take the single-venue one.
    k_above = km("K-T0.1", "greater", 0.1, label="Above 0.1%", yes=0.86, no=0.15)
    k_le = km("K-L0.1", "less_or_equal", cap=0.1, label="<=0.1%", yes=0.115, no=0.9)
    items = [Priced(k_above, Interval(2, INF)), Priced(k_le, Interval(-INF, 1))]
    for lab, y, n in [("≤0.0%", .02, .995), ("0.1%", .09, .97), ("0.2%", .48, .54), ("0.3%", .35, .67),
                      ("0.4%", .09, .93), ("0.5%", .01, .997), ("0.6%+", .003, .999)]:
        items.append(Priced(pm(lab, y, n), polymarket_interval(lab, CPI, False)))
    pick, alts, _ = choose_cover(items, CPI)
    assert len(alts) >= 2
    assert {it.market.platform for it, _ in pick.legs} == {"kalshi"}     # 0.975 (2 legs, 1 venue) beats 0.97 (3 legs, 2 venues) risk-adjusted
    assert pick.risk < max(a.risk for a in alts)


def test_implied_distribution_sums_to_one_and_matches_buckets():
    from arb.numeric import implied_distribution
    items = [Priced(pm(l, y, 1 - y + 0.01), polymarket_interval(l, CPI, False))
             for l, y in [("≤0.0%", .05), ("0.1%", .15), ("0.2%", .5), ("0.3%", .25), ("0.4%+", .05)]]
    outcomes = list(range(-1, 6))
    p = implied_distribution(items, outcomes)
    assert abs(p.sum() - 1) < 1e-9 and p[outcomes.index(2)] == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("plat,eid,title,close,want", [
    ("kalshi", "KXUSPPIYOY-26NOV13", "PPI YoY in October", "2026-11-13", ("M", 2026, 10)),   # ticker = release date
    ("polymarket", "1", "PPI YoY - October 2026", "2026-11-14", ("M", 2026, 10)),
    ("kalshi", "KXJOLTSOPEN-26DEC01", "US job openings in October", "2026-12-01", ("M", 2026, 10)),
    ("kalshi", "KXCPI-26DEC", "CPI in December", "2027-01-13", ("M", 2026, 12)),             # year rolls back
    ("polymarket", "4", "Fed Decision in December?", "2026-12-09", ("M", 2026, 12)),
    ("polymarket", "5", "Fed Decision in January?", "2027-01-28", ("M", 2027, 1)),      # "decision" != "dec"
    ("kalshi", "KXFEDDECISION-27MAR", "Fed decision in Mar 2027?", "2027-03-17", ("M", 2027, 3)),
    ("kalshi", "KXCPICORE-26OCT", "Core CPI in October", "2026-11-10", ("M", 2026, 10)),
])
def test_period_uses_reference_month(plat, eid, title, close, want):
    m = pm("x")
    m.close_time = datetime.fromisoformat(close).replace(tzinfo=timezone.utc)
    assert period_of(Event(plat, eid, title, []), m) == want


def test_daily_period_for_gas():
    gas = next(u for u in REGISTRY if u.id == "us_gas_daily")
    m = pm("x")
    m.close_time = datetime(2026, 11, 9, 15, tzinfo=timezone.utc)
    assert period_of(Event("kalshi", "KXAAAGASD-26NOV10", "US gas prices tomorrow", []), m, "D") == ("D", 2026, 11, 10)
    assert period_of(Event("polymarket", "9", "Gas prices on November 9?", []), m, gas.period) == ("D", 2026, 11, 9)
