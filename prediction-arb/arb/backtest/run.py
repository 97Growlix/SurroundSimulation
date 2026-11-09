"""Run the backtest scenarios and write data/backtest/results.json + a text summary."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone

from .collect import OUT
from .sim import SimConfig, load, run  # noqa: F401

START = datetime(2026, 5, 8, tzinfo=timezone.utc)
END = datetime(2026, 11, 8, tzinfo=timezone.utc)

V1 = SimConfig(label="v1: cheapest cover, hold to settlement, no hurdle")
NEW = replace(V1, label="v2: 50% hurdle + early exit/switch + risk chooser + 50% max per trade",
              min_apy=0.5, early_exit=True, risk_chooser=True, max_trade_frac=0.5)
SCENARIOS = [
    V1,
    replace(V1, label="v1 + 50% hurdle", min_apy=0.5),
    replace(V1, label="v1 + 50% hurdle + early exit/switch", min_apy=0.5, early_exit=True),
    replace(V1, label="v1 + 50% hurdle + early exit/switch + risk chooser", min_apy=0.5, early_exit=True,
            risk_chooser=True),
    NEW,
    replace(NEW, label="v2 + volume cap 25%", max_participation=0.25),
    replace(NEW, label="v2 + volume cap 10% + conservative", max_participation=0.10, pm_half_spread=0.02,
            depth_kalshi=100, depth_pm=75, latency_hours=2, replenish_hours=24),
    replace(NEW, label="v2 original underlyings only", platforms=("kalshi", "polymarket")),
    replace(NEW, label="v2 + maker bids", maker=True),
    replace(NEW, label="v2 + maker bids + volume cap 25%", maker=True, max_participation=0.25),
    replace(NEW, label="v2 + maker, cautious (2c edge, 6h TTL, 3h blackout) + vol cap 25%", maker=True,
            maker_min_edge=0.02, maker_ttl_hours=6, maker_blackout_hours=3, max_participation=0.25),
] + [replace(NEW, label=f"v2 + volume cap 25%, bankroll ${b:,}", bankroll=float(b), max_participation=0.25)
     for b in (1_000, 10_000, 100_000)]
ORIGINAL = {"us_core_cpi_mom", "us_cpi_mom", "us_core_cpi_yoy", "us_cpi_yoy", "us_core_pce_mom",
            "us_gdp_q_saar", "fed_decision_bps"}

_STATE = {}


def _run_one(cfg):
    if not _STATE:
        _STATE["d"] = load(OUT / "raw.json")
    _, groups, sims = _STATE["d"]
    if cfg.label == "v2 original underlyings only":
        groups = {k: v for k, v in groups.items() if k[0] in ORIGINAL}
    return run(groups, sims, START, END, cfg)


def main():
    raw, groups, sims = load(OUT / "raw.json")
    venues = Counter()
    for (uid, per), items in groups.items():
        venues[(uid, tuple(sorted({i.market.platform for i in items})))] += 1
    print(f"{len(groups)} underlying/period groups; {len(sims)} markets with history")
    for (uid, v), n in sorted(venues.items()):
        print(f"   {uid:18} {'+'.join(v):18} {n} periods")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(_run_one, SCENARIOS))
    for cfg, r in zip(SCENARIOS, results):
        print(f"\n== {cfg.label}: ${cfg.bankroll:,.0f} -> ${r['final']:,.2f} "
              f"({r['return'] * 100:+.2f}% in 6 mo, {r['annualized'] * 100:+.1f}%/yr) | "
              f"{r['executed']} trades, {r['vanished']} vanished before execution, "
              f"{len(r['mismatches'])} worst-case violations")
    (OUT / "results.json").write_text(json.dumps(results, default=str))
    for r in results:
        g = sum(x["guaranteed_min"] - x["cost"] for x in r["trades"] if not x["exited_early"])
        w = sum(x["pnl"] for x in r["trades"]) - g
        ex = sum(1 for x in r["trades"] if x["exited_early"])
        print(f"   {r['config']['label'][:70]:70} early exits {ex:4}  held-to-settle guaranteed ${g:,.2f}  other ${w:,.2f}")
        if r["config"].get("maker"):
            print(f"      maker: {r['maker']}")
    base = results[4]
    print("\nv2 trades ($100):")
    for x in base["trades"]:
        print(f"  {x['opened'][:13]} {x['group'][:34]:34} qty {x['qty']:4} cost ${x['cost']:7.2f} "
              f"-> ${x['payout']:7.2f}  P&L ${x['pnl']:+6.2f}  | {'; '.join(x['legs'])[:110]}")


if __name__ == "__main__":
    main()
