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
  * Bank of Japan decisions - BoJ moves in 10-15bp steps; Polymarket rounds those
    up to 25bp but Kalshi's BoJ labels have no "1-25bp" bucket.
  * ECB decisions - Polymarket uses the deposit facility rate, Kalshi the "primary"
    rate; they moved by different amounts in 2024.
  * Payrolls - Polymarket's "jobs added" series is irregular (old, ad-hoc brackets).
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
    period: str = "M"           # grouping granularity: M(onth), Q(uarter), D(ay)
    # extra risk, in $ per $1 bundle, for residual rule differences between venues
    # (used by the risk-aware bundle chooser; 0 = rules identical in practice)
    rule_risk: float = 0.0

    def matches_polymarket(self, title: str) -> bool:
        return re.search(self.polymarket_title, title.lower()) is not None


REGISTRY: tuple[Underlying, ...] = (
    Underlying("us_core_cpi_mom", 0.1, ("KXCPICORE", "KXECONSTATCPICORE"), r"^core cpi (\(ex food (and|&) energy\) )?mom\b",
               "BLS CPI-U less food & energy, seasonally adjusted 1-month % change, one decimal",
               "release delay (e.g. shutdown) clauses differ between venues"),
    Underlying("us_cpi_mom", 0.1, ("KXCPI", "KXECONSTATCPI"), r"^(cpi mom\b|\w+ inflation us - monthly$)",
               "BLS CPI-U all items, seasonally adjusted 1-month % change, one decimal",
               "release delay clauses differ"),
    Underlying("us_core_cpi_yoy", 0.1, ("KXCPICOREYOY", "KXECONSTATCORECPIYOY"), r"^core cpi yoy\b",
               "BLS CPI-U less food & energy, 12-month % change (not seasonally adjusted), one decimal",
               "release delay clauses differ"),
    Underlying("us_cpi_yoy", 0.1, ("KXCPIYOY", "KXECONSTATCPIYOY"), r"^(cpi yoy\b|\w+ inflation us - annual$)",
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
               "non-25bp-multiple moves: Polymarket rounds up to the nearest 25bp", rule_risk=0.002),
    # --- added 2026-11-09 after reading both venues' rules ---
    Underlying("us_unemployment", 0.1, ("KXU3", "KXECONSTATU3"), r"^\w+ unemployment rate$",
               "BLS U-3 unemployment rate, seasonally adjusted, first release, one decimal",
               "release delay clauses differ"),
    Underlying("us_ppi_yoy", 0.1, ("KXUSPPIYOY",), r"^(producer price index \(ppi\) |ppi )yoy\b",
               "BLS PPI final demand, 12-month % change (not seasonally adjusted), one decimal",
               "release delay clauses differ"),
    Underlying("us_jolts_openings_m", 0.001, ("KXJOLTSOPEN",), r"^jolts job openings",
               "BLS JOLTS total nonfarm job openings, seasonally adjusted, in thousands (values in millions)",
               "Polymarket brackets are [low, high) - boundary values go to the higher bracket"),
    Underlying("boe_decision_bps", 25, ("KXCBDECISIONENGLAND",), r"^bank of england decision in ",
               "change in Bank Rate at the scheduled MPC meeting, in bps",
               "moves >25bp that are not multiples of 25 may round differently; cancelled meeting: "
               "Kalshi resolves 'maintain'", rule_risk=0.003),
    Underlying("boc_decision_bps", 25, ("KXCBDECISIONCANADA",), r"^bank of canada decision in ",
               "change in the overnight rate target at the scheduled announcement, in bps",
               "non-25bp moves may round differently", rule_risk=0.003),
    Underlying("rba_decision_bps", 25, ("KXCBDECISIONAUSTRALIA", "KXRBADECISION"),
               r"^reserve bank of australia decision in ",
               "change in the RBA cash rate target at the scheduled meeting, in bps",
               "non-25bp moves may round differently; cancelled meeting: Kalshi resolves 'maintain'",
               rule_risk=0.003),
    Underlying("us_gas_daily", 0.001, ("KXAAAGASD",), r"^gas prices on ",
               "AAA US national average regular gas price for the date (3 decimals)",
               "Polymarket reads AAA at 10:00 AM ET; Kalshi does not state a time of day "
               "(AAA posts once daily, so normally identical)", period="D", rule_risk=0.006),
)
