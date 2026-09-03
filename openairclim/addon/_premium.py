"""
Integrates premium functionality.
"""

import logging
from typing import Mapping, Sequence, Callable

logger = logging.getLogger(__name__)

OAC_PREMIUM_AVAILABLE: bool = False

# fallback values
pm_factor_low: Callable[[float, float, float, Sequence[float]], float] | None = None
LOW_SOOT_CASES: Mapping[str, Sequence[float]] | None = None

try:
    from openairclim_premium import (
        pm_factor_low as _pm_factor_low,
        LOW_SOOT_CASES as _LOW_SOOT_CASES,
    )

    pm_factor_low = _pm_factor_low
    LOW_SOOT_CASES = _LOW_SOOT_CASES
    OAC_PREMIUM_AVAILABLE = True
    logger.warning("OpenAirClim premium functionality loaded.")
except ImportError as e:
    pass
