"""
Integrates premium functionality.
"""

import logging

OAC_PREMIUM_AVAILABLE = False

# fallback values
pm_factor_low = None
LOW_SOOT_CASES = None

__all__ = [
    "OAC_PREMIUM_AVAILABLE",
    "pm_factor_low",
    "LOW_SOOT_CASES",
]

try:
    from openairclim_premium import (
        pm_factor_low,
        LOW_SOOT_CASES,
    )
    OAC_PREMIUM_AVAILABLE = True
    logging.warning("OpenAirClim premium functionality loaded.")
except ImportError as e:
    pass
