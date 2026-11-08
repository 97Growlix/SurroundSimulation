"""Registry of numeric underlyings listed on more than one venue.

Each entry was checked against BOTH venues' resolution rules (source, adjustment,
rounding, which release).  Only add an entry after reading both rule texts - the
whole point is that every market in a group settles off the same printed number.

Deliberately NOT included (checked 2026-11-08):
  * City high/low temperatures - Kalshi NYC uses Central Park (CLINYC), Polymarket
    uses LaGuardia (KLGA METAR). Different stations = different numbers.
  * BTC/ETH price - Kalshi uses CF Benchmarks at a given ET time, Polymarket uses a
    Binance candle at a different time.
  * Fed funds *level* vs Fed *decision* - linking them needs the pre-meeting rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Underlying:
    id: str
    step: float                 # resolution of the printed number (0.1 = one decimal)
    kalshi_series: tuple[str, ...]
    polymarket_title: str       # regex on lower-cased Polymarket event title
    source: str                 # what both venues settle on
    caveats: str = ""

    def matches_polymarket(self, title: str) -> bool:
        return re.search(self.polymarket_title, title.lower()) is not None


REGISTRY: tuple[Underlying, ...] = (
    Underlying("us_core_cpi_mom", 0.1, ("KXCPICORE", "KXECONSTATCPICORE"), r"^core cpi mom\b",
               "BLS CPI-U less food & energy, seasonally adjusted 1-month % change, one decimal",
               "release delay (e.g. shutdown) clauses differ between venues"),
    Underlying("us_cpi_mom", 0.1, ("KXCPI", "KXECONSTATCPI"), r"^cpi mom\b",
               "BLS CPI-U all items, seasonally adjusted 1-month % change, one decimal",
               "release delay clauses differ"),
    Underlying("us_core_cpi_yoy", 0.1, ("KXCPICOREYOY", "KXECONSTATCORECPIYOY"), r"^core cpi yoy\b",
               "BLS CPI-U less food & energy, 12-month % change (not seasonally adjusted), one decimal",
               "release delay clauses differ"),
    Underlying("us_cpi_yoy", 0.1, ("KXCPIYOY", "KXECONSTATCPIYOY"), r"^cpi yoy\b",
               "BLS CPI-U all items, 12-month % change (not seasonally adjusted), one decimal",
               "release delay clauses differ"),
    Underlying("us_core_pce_mom", 0.1, ("KXPCECORE",), r"^core pce mom\b",
               "BEA PCE price index ex food & energy, 1-month % change, one decimal",
               "release delay clauses differ"),
    Underlying("us_gdp_q_saar", 0.1, ("KXGDP",), r"^us gdp growth in q[1-4] 20\d\d",
               "BEA real GDP, seasonally adjusted annualised, Advance Estimate, one decimal",
               "Polymarket brackets are [low, high) - a print exactly on a boundary goes to the higher bracket"),
    Underlying("fed_decision_bps", 25, ("KXFEDDECISION",), r"^fed decision in ",
               "change in the upper bound of the fed funds target range at the FOMC meeting, in bps",
               "non-25bp-multiple moves: Polymarket rounds up to the nearest 25bp"),
)
