"""Command line interface.

  python -m arb scan  --min-apy 15 --max-capital 500
  python -m arb paper --min-apy 15            # scan + record dry-run trades
  python -m arb match                          # suggest cross-venue pairs for review
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from .engine import Opportunity
from .execution import DryRunExecutor
from .matching import suggest_pairs
from .scan import DATA, ScanConfig, fetch_all, run_scan


def pct(x: float) -> str:
    return ">9999%" if x > 99.99 else f"{x * 100:.1f}%"


def fmt(o: Opportunity) -> str:
    lines = [f"[{o.risk_tier}] {o.kind:<10} APY {pct(o.apy):>8}  ROI {o.roi * 100:6.2f}%  "
             f"profit ${o.profit:8.2f} on ${o.cost + o.fees:9.2f}  ({o.days:.0f}d)  qty {o.qty:g}",
             f"    {o.title}"]
    for l, f in zip(o.legs, o.fills):
        lines.append(f"      {l.describe():<70} avg {f.avg_price:.4f} fee ${f.fee:.2f}")
    lines.append(f"    worst case: {o.worst_scenario} -> payout ${o.worst_payout:.2f}")
    if o.kind == "cross_pair":
        cap = o.cost + o.fees
        lines.append(f"    if the venues resolve differently against you, you lose ${cap:.2f}; "
                     f"the trade is +EV only if that chance is < {o.profit / (o.profit + cap) * 100:.1f}%")
    for n in o.notes:
        lines.append(f"    note: {n}")
    return "\n".join(lines)


def to_json(o: Opportunity) -> dict:
    return {"kind": o.kind, "risk_tier": o.risk_tier, "title": o.title, "apy": o.apy, "roi": o.roi,
            "profit": o.profit, "apy_capped": min(o.apy, 99.99), "capital": o.cost + o.fees, "days": o.days, "qty": o.qty,
            "worst_scenario": o.worst_scenario,
            "breakeven_mismatch_prob": (o.profit / (o.profit + o.cost + o.fees)) if o.kind == "cross_pair" else None, "notes": o.notes,
            "legs": [{"leg": l.describe(), "avg_price": f.avg_price, "fee": f.fee,
                      "close": l.market.close_time.isoformat() if l.market.close_time else None,
                      "url": l.market.url} for l, f in zip(o.legs, o.fills)]}


def cfg_from(a) -> ScanConfig:
    return ScanConfig(min_apy=a.min_apy / 100, min_profit=a.min_profit, max_capital=a.max_capital,
                      max_days=a.max_days, settle_buffer_days=a.settle_buffer_days,
                      unknown_days=a.unknown_days, platforms=tuple(a.platforms.split(",")),
                      include_unverified=a.include_unverified, assume_exhaustive=a.assume_exhaustive,
                      cache_minutes=a.cache_minutes)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="arb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("settle", help="resolve open paper trades against actual outcomes")
    sp.add_argument("--ledger", default=str(DATA / "paper_ledger.jsonl"))
    for name in ("scan", "paper", "match"):
        p = sub.add_parser(name)
        p.add_argument("--min-apy", type=float, default=10.0, help="minimum annualised return, %%/yr")
        p.add_argument("--min-profit", type=float, default=1.0, help="minimum guaranteed profit, $")
        p.add_argument("--max-capital", type=float, default=1000.0, help="max $ per opportunity")
        p.add_argument("--max-days", type=float, default=400.0)
        p.add_argument("--settle-buffer-days", type=float, default=2.0)
        p.add_argument("--unknown-days", type=float, default=365.0)
        p.add_argument("--platforms", default="kalshi,polymarket,predictit")
        p.add_argument("--include-unverified", action="store_true",
                       help="also show tier-C ideas (auto-matched pairs, assumed exhaustive)")
        p.add_argument("--assume-exhaustive", action="store_true")
        p.add_argument("--cache-minutes", type=float, default=0.0)
        p.add_argument("--paper-tiers", default="A",
                       help="tiers to paper-trade, e.g. 'A' or 'A,B' (B = registry-verified cross-venue bundles)")
    a = ap.parse_args(argv)
    if a.cmd == "settle":
        from .settle import settle_ledger
        recs = settle_ledger(Path(a.ledger))
        done = [r for r in recs if r.get("status") == "settled"]
        print(f"{len(done)} settled / {len(recs) - len(done)} still open")
        for r in done:
            flag = "OK " if r["matched_expectation"] else "!! MISMATCH"
            print(f"  {flag} {r['title'][:60]:60} cost ${r['cost'] + r['fees']:.2f} payout ${r['payout']:.2f} "
                  f"P&L ${r['realized_profit']:+.2f}")
        if done:
            print(f"total realised P&L ${sum(r['realized_profit'] for r in done):+.2f}")
        return
    cfg = cfg_from(a)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if a.cmd == "match":
        events = fetch_all(cfg)
        pairs = suggest_pairs(events)
        out = DATA / "pair_candidates.json"
        out.write_text(json.dumps(pairs, indent=1))
        print(f"{len(pairs)} candidate pairs written to {out}")
        print("Review them; copy verified ones (fix 'invert' if needed) into data/pairs.json")
        for p in pairs[:40]:
            print(f"  {p['score']:5.1f}  {p['a_platform']}:{p['a_text'][:60]!r}  <->  "
                  f"{p['b_platform']}:{p['b_text'][:60]!r}")
        return

    final, screened, _ = run_scan(cfg)
    print("Risk tiers: A = payout guaranteed by one venue's rules; B = cross-venue, same verified "
          "underlying (registry) or human-confirmed pair; C = UNVERIFIED idea, never auto-traded")
    print(f"\n{len(final)} opportunities >= {a.min_apy}%/yr, >= ${a.min_profit} profit "
          f"(after full order-book + fee check; {len(screened)} passed the top-of-book screen)\n")
    for o in final:
        print(fmt(o) + "\n")
    out = DATA / f"scan_{stamp}.json"
    out.write_text(json.dumps([to_json(o) for o in final], indent=1, default=str))
    print(f"saved {out}")
    if a.cmd == "paper":
        ex = DryRunExecutor(DATA / "paper_ledger.jsonl")
        n = 0
        tiers = set(a.paper_tiers.upper().split(","))
        for o in final:
            if o.risk_tier in tiers and not (o.risk_tier == "B" and o.kind != "numeric"):
                ex.execute(o)
                n += 1
        print(f"paper-traded {n} opportunities (tiers {sorted(tiers)}) -> {ex.ledger}")


if __name__ == "__main__":
    main()
