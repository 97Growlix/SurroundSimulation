from datetime import datetime, timezone

from arb.models import BinaryMarket, Event
from arb.strategies import ladder_pairs


def m(mid, title, label, yes, no, close):
    return BinaryMarket("polymarket", mid, "E", title, label, yes_ask=yes, no_ask=no,
                        close_time=datetime.fromisoformat(close).replace(tzinfo=timezone.utc))


def legs(o):
    return [(l.market.market_id, l.side) for l in o.legs]


def test_threshold_ladder_direction():
    ev = Event("polymarket", "E", "x", [
        m("lo", "Will unemployment reach at least 6.0% in 2026?", "6.0%", 0.10, 0.92, "2027-01-01T04:59:00"),
        m("hi", "Will unemployment reach at least 7.0% in 2026?", "7.0%", 0.12, 0.86, "2027-01-01T04:59:00"),
    ])
    ops = ladder_pairs(ev)
    # hi (stricter) priced above lo: buy YES lo + NO hi = 0.96
    assert [legs(o) for o in ops] == [[("lo", "yes"), ("hi", "no")]]


def test_deadline_ladder_dec31_endtime_rolls_into_next_year():
    # "by December 31" ends 2027-01-01T04:59Z; it must be read as Dec 31 2026,
    # i.e. STRICTER than "by June 30, 2027".
    dec = m("dec", "Iran leadership change by December 31?", "December 31", 0.08, 0.93, "2027-01-01T04:59:00")
    jun = m("jun", "Iran leadership change by June 30, 2027?", "June 30, 2027", 0.15, 0.78, "2027-07-01T03:59:00")
    ops = ladder_pairs(Event("polymarket", "E", "x", [dec, jun]))
    # consistent pricing (dec 0.08 < jun 0.15...) -> correct direction is YES jun + NO dec = 1.08: no arb
    assert ops == []
    # make it inconsistent: earlier deadline priced higher than later one
    dec.no_ask = 0.80          # YES bid ~0.20 on the stricter Dec market
    jun.yes_ask = 0.15
    ops = ladder_pairs(Event("polymarket", "E", "x", [dec, jun]))
    assert [legs(o) for o in ops] == [[("jun", "yes"), ("dec", "no")]]


def test_different_words_not_nested():
    ev = Event("polymarket", "E", "x", [
        m("a", "What will Wingstop say on the call?", "India (2+ times)", 0.36, 0.7, "2026-12-01T00:00:00"),
        m("b", "What will Wingstop say on the call?", "Club Wingstop (3+ times)", 0.9, 0.11, "2026-12-01T00:00:00"),
    ])
    assert ladder_pairs(ev) == []


def test_in_month_buckets_are_not_deadlines():
    ev = Event("polymarket", "E", "x", [
        m("a", "Will X happen in November 2026?", "November", 0.3, 0.6, "2026-12-01T00:00:00"),
        m("b", "Will X happen in December 2026?", "December", 0.5, 0.4, "2027-01-01T00:00:00"),
    ])
    assert ladder_pairs(ev) == []
