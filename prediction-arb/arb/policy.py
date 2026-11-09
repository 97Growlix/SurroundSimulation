"""Capital-management rules shared by the backtest and the live paper trader."""
from __future__ import annotations

import math


def annualize(growth: float, days: float) -> float:
    """(1+r)^(365/days)-1 with days floored at 1 (same-day trades aren't annualised to infinity)."""
    if growth <= 0:
        return -1.0
    try:
        return growth ** (365.0 / max(days, 1.0)) - 1
    except OverflowError:
        return math.inf


def hold_apy(exit_value: float, guaranteed_payout: float, days_left: float) -> float:
    """Annualised return from KEEPING a bundle: you give up exit_value now to get the payout later."""
    if exit_value <= 0:
        return math.inf
    return annualize(guaranteed_payout / exit_value, days_left)


def should_sell_now(exit_value: float, guaranteed_payout: float) -> bool:
    """Selling for at least the guaranteed payout frees capital at no cost."""
    return exit_value >= guaranteed_payout


def should_switch(new_apy: float, held_apy: float, margin: float) -> bool:
    return new_apy > held_apy + margin
