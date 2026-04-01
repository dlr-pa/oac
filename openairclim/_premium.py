"""
Integrates premium functionality.
"""

import logging
from typing import Mapping, Sequence, Callable

OAC_PREMIUM_AVAILABLE: bool = False

# fallback values
pm_factor_low: Callable[[float, float, float, Sequence[float]], float] | None = None
LOW_SOOT_CASES: Mapping[str, Sequence[float]] | None = None

__all__ = [
    "OAC_PREMIUM_AVAILABLE",
    "pm_factor_low",
    "LOW_SOOT_CASES",
]

try:
    from openairclim_premium import (
        pm_factor_low as _pm_factor_low,
        LOW_SOOT_CASES as _LOW_SOOT_CASES,
    )

    pm_factor_low = _pm_factor_low
    LOW_SOOT_CASES = _LOW_SOOT_CASES
    OAC_PREMIUM_AVAILABLE = True
    logging.warning("OpenAirClim premium functionality loaded.")
except ImportError as e:
    pass
