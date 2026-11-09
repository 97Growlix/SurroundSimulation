from datetime import datetime, timedelta, timezone

import pytest

from arb import policy
from arb.watch import PaperPortfolio, cadence_seconds

NOW = datetime(2026, 11, 9, 12, tzinfo=timezone.utc)


def test_cadence_tightens_near_release():
    cal = lambda h: [{"release": NOW + timedelta(hours=h)}]  # noqa: E731
    assert cadence_seconds(cal(1), NOW) == 15
    assert cadence_seconds(cal(30), NOW) == 60
    assert cadence_seconds(cal(200), NOW) == 600
    assert cadence_seconds([], NOW) == 600


def test_policy_rules():
    # bundle paying $100 in 100 days, sellable now for $99 -> ~3.7%/yr to keep holding
    h = policy.hold_apy(99, 100, 100)
    assert 0.03 < h < 0.04
    assert policy.should_switch(1.0, h, 0.5) and not policy.should_switch(0.4, h, 0.5)
    assert policy.should_sell_now(100.2, 100) and not policy.should_sell_now(99.9, 100)


def test_paper_portfolio_accounting(tmp_path):
    pf = PaperPortfolio(tmp_path / "p.json", 100)
    pf.state["positions"].append({"status": "open", "cost": 60, "fees": 1, "title": "x"})
    pf.state["cash"] -= 61
    assert pf.equity_at_cost() == pytest.approx(100)
    pf.close(pf.state["positions"][0], 64, "exited")
    assert pf.cash == pytest.approx(103) and pf.state["positions"][0]["pnl"] == pytest.approx(3)
    pf.save()
    assert PaperPortfolio(tmp_path / "p.json", 0).cash == pytest.approx(103)
