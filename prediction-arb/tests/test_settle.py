import json

from arb import settle


def test_settle_marks_payout_and_flags_mismatch(tmp_path, monkeypatch):
    rec = {"id": "x", "title": "cpi", "status": "open", "qty": 100, "cost": 98.0, "fees": 0.5,
           "worst_payout": 100.0,
           "legs": [{"platform": "kalshi", "market_id": "K", "side": "yes"},
                    {"platform": "polymarket", "market_id": "P1", "side": "yes"},
                    {"platform": "polymarket", "market_id": "P2", "side": "yes"}]}
    led = tmp_path / "l.jsonl"
    led.write_text(json.dumps(rec) + "\n")
    vals = {"K": 1.0, "P1": 0.0, "P2": 0.0}
    monkeypatch.setattr(settle, "yes_value", lambda p, m: vals[m])
    r = settle.settle_ledger(led)[0]
    assert r["status"] == "settled" and r["payout"] == 100 and r["realized_profit"] == 1.5
    assert r["matched_expectation"]
    # a "hedge" where every leg lost must be flagged
    rec2 = dict(rec, id="y")
    led.write_text(json.dumps(rec2) + "\n")
    vals["K"] = 0.0
    r = settle.settle_ledger(led)[0]
    assert r["payout"] == 0 and not r["matched_expectation"]


def test_unresolved_stays_open(tmp_path, monkeypatch):
    rec = {"id": "x", "title": "t", "status": "open", "qty": 1, "cost": 0.9, "fees": 0, "worst_payout": 1,
           "legs": [{"platform": "kalshi", "market_id": "K", "side": "yes"}]}
    led = tmp_path / "l.jsonl"
    led.write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr(settle, "yes_value", lambda p, m: None)
    assert settle.settle_ledger(led)[0]["status"] == "open"
