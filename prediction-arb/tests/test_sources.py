from arb.matching import _labels_ok, numbers, tokens
from arb.models import BinaryMarket
from arb.sources import kalshi


def test_kalshi_orderbook_bids_become_asks(monkeypatch):
    monkeypatch.setattr(kalshi, "get_json", lambda *a, **k: {"orderbook_fp": {
        "yes_dollars": [["0.06", "100"], ["0.10", "5"]],
        "no_dollars": [["0.85", "50"], ["0.89", "4"]]}})
    ob = kalshi.orderbook("X")
    assert ob["yes"][0] == (0.11, 4.0)     # best YES offer = 1 - best NO bid
    assert ob["no"][0] == (0.90, 5.0)


def test_numbers_must_match():
    assert numbers("cpi above 0.10% in 2026") == {"0.1", "2026"}
    assert numbers("rise more than 0.1%") != numbers("rise more than 0.2%")


def test_labels_must_appear_on_both_sides():
    a = BinaryMarket("predictit", "1", "e", "Who will be next UN SG — Macky Sall", outcome_label="Macky Sall")
    b = BinaryMarket("polymarket", "2", "e", "Will Rafael Grossi be the next UN SG?", outcome_label="Rafael Grossi")
    assert not _labels_ok(a, a.title.lower(), b, b.title.lower())
    assert "the" not in tokens("the next leader")
