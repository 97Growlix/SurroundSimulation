"""Live in-game cross-venue recorder for two-way game-winner markets (Kalshi vs Polymarket).

A game has two outcomes (A wins / B wins); both venues settle a tie at 50/50 in
the NFL and include overtime elsewhere, so YES-A on one venue + YES-B on the
other pays exactly $1 in every result.  We poll both order books and log the
cheapest such bundle after fees, with available size, so we can measure how
often in-play gaps exist, how big they are and how long they last.

Rule differences that remain (documented, not modelled): Kalshi settles
cancelled/long-postponed games at a "fair price", Polymarket at 50-50.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import fees
from .http import get_json, log
from .sources import kalshi, polymarket
from .sources.kalshi import BASE as K
from .sources.polymarket import GAMMA

LEAGUES = {  # kalshi series -> polymarket tag slug
    "KXNFLGAME": "nfl", "KXNBAGAME": "nba", "KXNHLGAME": "nhl", "KXNCAAFGAME": "cfb",
    "KXNCAAMBGAME": "cbb",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "live"


def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower().replace(".", ""))


def team_match(kalshi_label: str, pm_name: str) -> bool:
    """'Los Angeles C' ~ 'Los Angeles Chargers', 'Tampa Bay' ~ 'Tampa Bay Lightning', 'Kent St' ~ 'Kent State'."""
    kt, pt = _toks(kalshi_label), _toks(pm_name)
    j = 0
    for t in kt:
        while j < len(pt) and not (pt[j] == t or (len(t) <= 2 and pt[j].startswith(t)) or
                                   (t == "st" and pt[j] == "state")):
            j += 1
        if j == len(pt):
            return False
        j += 1
    return bool(kt)


def kalshi_games() -> list[dict]:
    games = []
    for series in LEAGUES:
        d = get_json(f"{K}/events", {"series_ticker": series, "status": "open", "with_nested_markets": "true",
                                     "limit": 200})
        for ev in d.get("events", []):
            mk = [m for m in ev.get("markets", []) if m.get("status") == "active"]
            if len(mk) != 2:       # two-way only (skip 3-way soccer-style)
                continue
            mt = re.search(r"-(\d\d)([A-Z]{3})(\d\d)", ev["event_ticker"])
            if not mt:
                continue
            date = datetime.strptime(f"20{mt.group(1)} {mt.group(2)} {mt.group(3)}", "%Y %b %d").date()
            games.append({"series": series, "event": ev["event_ticker"], "date": date,
                          "teams": [(m["yes_sub_title"], m["ticker"]) for m in mk]})
    return games


def polymarket_games(tag: str) -> list[dict]:
    d = []
    for off in range(0, 2000, 200):
        page = get_json(f"{GAMMA}/events", {"tag_slug": tag, "active": "true", "closed": "false", "limit": 200,
                                            "offset": off})
        if not page:
            break
        d += page
    out = []
    for ev in d:
        ml = [m for m in ev.get("markets", []) if m.get("sportsMarketType") == "moneyline"
              and m.get("acceptingOrders") and not m.get("closed")]
        if len(ml) != 1 or not ev.get("startTime"):
            continue
        m = ml[0]
        outcomes, toks = json.loads(m["outcomes"]), json.loads(m["clobTokenIds"])
        abbr = {(t.get("alias") or t.get("name") or "").lower(): (t.get("abbreviation") or "").lower()
                for t in ev.get("teams", [])}
        slug_abbrs = ev.get("slug", "").split("-")[1:3]
        full = [abbr.get(o.lower()) or (slug_abbrs[i] if i < len(slug_abbrs) else "") for i, o in enumerate(outcomes)]
        start = datetime.fromisoformat(ev["startTime"].replace("Z", "+00:00"))
        out.append({"title": ev["title"], "start": start, "et_date": (start - timedelta(hours=5)).date(),
                    "live": ev.get("live"), "score": ev.get("score"), "period": ev.get("period"),
                    "outcomes": list(zip(outcomes, full, toks)),
                    "fee": fees.polymarket(bool(m.get("feesEnabled")), m.get("feeSchedule"))})
    return out


def match_games() -> list[dict]:
    """Match on league + ET date + team abbreviations.  Kalshi event tickers end in
    AWAYHOME abbreviations (e.g. ...26NOV08TBWPG) and Polymarket slugs list the
    same two teams in the same order (nhl-tb-wpg-2026-11-08).  Abbreviations
    occasionally differ (CGY vs cal), so we require one exact abbreviation match
    plus consistent ordering."""
    pm_by_tag = {tag: polymarket_games(tag) for tag in set(LEAGUES.values())}
    pairs = []
    for g in kalshi_games():
        suffix = re.sub(r"^.*-\d\d[A-Z]{3}\d\d", "", g["event"]).lower()
        k_abbr = {t.rsplit("-", 1)[1].lower(): t for _, t in g["teams"]}
        for p in pm_by_tag[LEAGUES[g["series"]]]:
            if p["et_date"] != g["date"]:
                continue
            (o1, a1, tok1), (o2, a2, tok2) = p["outcomes"]
            if not (suffix.startswith(a1) or suffix.endswith(a2)):
                continue
            exact = [a for a in (a1, a2) if a in k_abbr]
            if not exact:
                continue
            first_k = next(t for ab, t in k_abbr.items() if suffix.startswith(ab))
            second_k = next(t for ab, t in k_abbr.items() if t != first_k)
            pairs.append({"kalshi": g, "pm": p,
                          "map": {first_k: (o1, a1, tok1), second_k: (o2, a2, tok2)}})
            break
    return pairs


def _cost(ladder, qty, fee):
    rem, cost, f = qty, 0.0, 0.0
    for p, s in ladder or []:
        take = min(rem, s)
        cost += take * p
        f += fee.taker_fee(p, take, round_up=False)
        rem -= take
        if rem <= 1e-9:
            return cost + f
    return None


def snapshot(pair: dict, sizes=(1, 25, 100)) -> dict:
    g, p = pair["kalshi"], pair["pm"]
    (t1, o1), (t2, o2) = list(pair["map"].items())
    kb = {t: kalshi.orderbook(t) for t in (t1, t2)}
    pb = polymarket.orderbooks([o1[2], o2[2]])
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "game": p["title"], "kalshi_event": g["event"]}
    best = None
    # bundle: team X on Kalshi (YES X, or NO on the other team - pick cheaper) + other team on Polymarket
    for kx, ko, oy in ((t1, t2, o2), (t2, t1, o1)):
        k_yes, k_no_other = kb[kx]["yes"], kb[ko]["no"]
        pm_lad = pb.get(oy[2])
        for qty in sizes:
            ck = [c for c in (_cost(k_yes, qty, fees.KALSHI), _cost(k_no_other, qty, fees.KALSHI)) if c is not None]
            cp = _cost(pm_lad, qty, p["fee"])
            if not ck or cp is None:
                continue
            unit = (min(ck) + cp) / qty
            rec[f"q{qty}_{kx.rsplit('-', 1)[1]}+pm_{oy[0]}"] = round(unit, 4)
            if qty == 1 and (best is None or unit < best):
                best = unit
    rec["best_unit_cost_q1"] = best
    rec["kalshi_mid"] = {t: (round((kb[t]["yes"][0][0] + (1 - kb[t]["no"][0][0])) / 2, 3)
                             if kb[t]["yes"] and kb[t]["no"] else None) for t in (t1, t2)}
    return rec


def record(minutes: float, interval_s: float = 15.0) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"sports_{datetime.now(timezone.utc):%Y%m%dT%H%M}.jsonl"
    end = time.time() + minutes * 60
    pairs, refreshed = [], 0.0
    with path.open("a") as fh:
        while time.time() < end:
            if time.time() - refreshed > 600:
                pairs = match_games()
                refreshed = time.time()
                now = datetime.now(timezone.utc)
                live = [x for x in pairs if x["pm"]["start"] <= now + timedelta(minutes=10)]
                log(f"{len(pairs)} matched games, {len(live)} live/starting")
            now = datetime.now(timezone.utc)
            for x in pairs:
                if x["pm"]["start"] > now + timedelta(minutes=10):
                    continue
                try:
                    rec = snapshot(x)
                except Exception as ex:   # noqa: BLE001 - keep recording other games
                    log(f"snapshot failed {x['pm']['title']}: {ex}")
                    continue
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            time.sleep(interval_s)
    return path


if __name__ == "__main__":
    import sys
    print(record(float(sys.argv[1]) if len(sys.argv) > 1 else 60))
