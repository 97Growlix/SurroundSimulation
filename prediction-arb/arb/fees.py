"""Per-platform fee models.

All fees are expressed in dollars for buying `qty` contracts that pay $1 each.
Sources (verify periodically - platforms change fee schedules):
  * Kalshi taker fee: ceil_to_cent(rate * C * P * (1-P)), rate 0.07 for most
    markets (some index series are 0.035; we use 0.07 as the conservative default).
  * Polymarket: fees only on markets with `feesEnabled`; the market's
    `feeSchedule` gives rate/exponent.  fee = C * p * rate * (p*(1-p))**exponent
    (taker only).  This formula reproduces Polymarket's published peak effective
    rates, but should be re-checked against the docs before trading live.
  * PredictIt: no trade fee, but 10% of *profit* on each winning contract, plus
    a 5% withdrawal fee on cash taken off the site (not modelled per trade).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    kind: str = "none"          # 'kalshi' | 'polymarket' | 'predictit' | 'none'
    rate: float = 0.0
    exponent: float = 1.0
    profit_rate: float = 0.0    # fee charged on winnings (PredictIt)

    def taker_fee(self, price: float, qty: float, *, round_up: bool = True) -> float:
        if qty <= 0:
            return 0.0
        p = price
        if self.kind == "kalshi":
            raw = self.rate * qty * p * (1 - p)
            return math.ceil(raw * 100 - 1e-9) / 100 if round_up else raw
        if self.kind == "polymarket":
            return qty * p * self.rate * (p * (1 - p)) ** self.exponent
        return 0.0

    def profit_fee(self, price: float, qty: float) -> float:
        """Fee taken from the payout of a *winning* position bought at `price`."""
        if self.profit_rate <= 0:
            return 0.0
        return self.profit_rate * (1 - price) * qty


NO_FEES = FeeModel()
KALSHI = FeeModel("kalshi", rate=0.07)
PREDICTIT = FeeModel("predictit", profit_rate=0.10)


def polymarket(fees_enabled: bool, schedule: dict | None) -> FeeModel:
    if not fees_enabled:
        return FeeModel("polymarket", rate=0.0)
    schedule = schedule or {}
    return FeeModel("polymarket", rate=float(schedule.get("rate", 0.0) or 0.0),
                    exponent=float(schedule.get("exponent", 1) or 1))
