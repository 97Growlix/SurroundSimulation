"""Run the backtest scenarios and write data/backtest/results.json + a text summary."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone

from .collect import OUT
from .sim import SimConfig, load, run

START = datetime(2026, 5, 8, tzinfo=timezone.utc)
END = datetime(2026, 11, 8, tzinfo=timezone.utc)

BASE = SimConfig(label="Base (realistic)")
SCENARIOS = [
    replace(BASE, label="Frictionless (mid prices, infinite depth, no delay)", pm_half_spread=0.0,
            unlimited_depth=True, latency_hours=0, replenish_hours=0, min_edge=0.001),
    BASE,
    replace(BASE, label="Conservative (2c PM spread, half depth, 2h delay, 24h refill)", pm_half_spread=0.02,
            depth_kalshi=100, depth_pm=75, latency_hours=2, replenish_hours=24),
    replace(BASE, label="Kalshi-only (no cross-venue)", platforms=("kalshi",)),
    replace(BASE, label="Base, only lock capital if APY >= 50%", min_apy=0.5),
    replace(BASE, label="Base, only lock capital if APY >= 200%", min_apy=2.0),
] + [replace(BASE, label=f"Base, bankroll ${b:,}", bankroll=float(b)) for b in (1_000, 10_000, 100_000)]


_STATE = {}


def _run_one(cfg):
    if not _STATE:
        _STATE["d"] = load(OUT / "raw.json")
    _, groups, sims = _STATE["d"]
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
    base = results[1]
    print("\nBase-case trades:")
    for x in base["trades"]:
        print(f"  {x['opened'][:13]} {x['group'][:34]:34} qty {x['qty']:4} cost ${x['cost']:7.2f} "
              f"-> ${x['payout']:7.2f}  P&L ${x['pnl']:+6.2f}  | {'; '.join(x['legs'])[:110]}")


if __name__ == "__main__":
    main()
