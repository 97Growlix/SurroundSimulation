"""Cross-venue market matching.

Automatic matching only *suggests* pairs.  Prediction-market questions that look
identical often differ in resolution source, deadline/time zone, or edge cases
(e.g. "by Dec 31" vs "in 2026"; "announced" vs "confirmed").  A pair is only
treated as tradeable once a human copies it into data/pairs.json.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from .models import BinaryMarket, Event

STOP = set("""a an the of in on at by to for will be is are was before after above below
or and vs v market win wins winner yes no who what which than with from end this that
any more less over under""".split())
NUM = re.compile(r"\d+(?:\.\d+)?")
WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def text_of(m: BinaryMarket) -> str:
    t = m.title
    if m.outcome_label and m.outcome_label.lower() not in t.lower():
        t += " " + m.outcome_label
    return t.lower()


def tokens(t: str) -> set[str]:
    return {w for w in WORD.findall(t) if w not in STOP and len(w) > 1}


def numbers(t: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in NUM.findall(t.replace(",", ""))}


def _labels_ok(a: BinaryMarket, ta: str, b: BinaryMarket, tb: str) -> bool:
    for m, other in ((a, tb), (b, ta)):
        lab = tokens((m.outcome_label or "").lower())
        if lab and not lab <= tokens(other):
            return False
    return True


def suggest_pairs(events_by_platform: dict[str, list[Event]], *, min_score: float = 88,
                  max_close_gap_days: float = 3.0, max_df: int = 300) -> list[dict]:
    mkts = {p: [m for ev in evs for m in ev.markets if m.yes_ask or m.no_ask]
            for p, evs in events_by_platform.items()}
    plats = sorted(mkts, key=lambda p: len(mkts[p]))
    results = []
    for i, pa in enumerate(plats):
        for pb in plats[i + 1:]:
            index: dict[str, list[int]] = defaultdict(list)
            b_list = mkts[pb]
            b_txt = [text_of(m) for m in b_list]
            for j, t in enumerate(b_txt):
                for tok in tokens(t):
                    index[tok].append(j)
            for a in mkts[pa]:
                ta = text_of(a)
                toks = [t for t in tokens(ta) if 0 < len(index.get(t, ())) <= max_df]
                if len(toks) < 2:
                    continue
                counts: dict[int, int] = defaultdict(int)
                for t in toks:
                    for j in index[t]:
                        counts[j] += 1
                na = numbers(ta)
                best = None
                for j, c in counts.items():
                    if c < 2:
                        continue
                    b = b_list[j]
                    if numbers(b_txt[j]) != na:
                        continue
                    if a.close_time and b.close_time and \
                            abs((a.close_time - b.close_time).total_seconds()) > max_close_gap_days * 86400:
                        continue
                    # outcome labels (candidate/team names) must appear on both sides
                    if not _labels_ok(a, ta, b, b_txt[j]):
                        continue
                    s = min(fuzz.token_set_ratio(ta, b_txt[j]), fuzz.token_sort_ratio(ta, b_txt[j]) + 15)
                    if s >= min_score and (best is None or s > best[1]):
                        best = (j, s)
                if best:
                    b = b_list[best[0]]
                    results.append({
                        "a_platform": pa, "a_id": a.market_id, "a_text": ta,
                        "b_platform": pb, "b_id": b.market_id, "b_text": b_txt[best[0]],
                        "score": round(best[1], 1), "invert": False,
                    })
    results.sort(key=lambda r: -r["score"])
    return results


def load_confirmed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())
