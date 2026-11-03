"""Integrates premium functionality."""

import logging
from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

OAC_PREMIUM_AVAILABLE: bool = False

# fallback values
pm_factor_low: Callable[[float, float, float, Sequence[float]], float] | None = None
LOW_SOOT_CASES: Mapping[str, Sequence[float]] | None = None

try:
    from openairclim_premium import (
        LOW_SOOT_CASES as _LOW_SOOT_CASES,
    )
    from openairclim_premium import (
        pm_factor_low as _pm_factor_low,
    )

    pm_factor_low = _pm_factor_low
    LOW_SOOT_CASES = _LOW_SOOT_CASES
    OAC_PREMIUM_AVAILABLE = True
except ImportError:
    pass

# fallback values for the coupled Hermite cubic low-soot regime. Imported
# separately so an openairclim_premium release that only has the original
# low-soot definition degrades gracefully
pm_factor_low_hermite: Callable[[float, float], float] | None = None
LOW_SOOT_CASES_HERMITE: Mapping[str, float] | None = None
fsc_factor: Callable[[float], float] | None = None
FSC_BREAKPOINTS_PPM: Sequence[float] | None = None

try:
    from openairclim_premium import (
        FSC_BREAKPOINTS_PPM as _FSC_BREAKPOINTS_PPM,
    )
    from openairclim_premium import (
        LOW_SOOT_CASES_HERMITE as _LOW_SOOT_CASES_HERMITE,
    )
    from openairclim_premium import (
        fsc_factor as _fsc_factor,
    )
    from openairclim_premium import (
        pm_factor_low_hermite as _pm_factor_low_hermite,
    )

    pm_factor_low_hermite = _pm_factor_low_hermite
    LOW_SOOT_CASES_HERMITE = _LOW_SOOT_CASES_HERMITE
    fsc_factor = _fsc_factor
    FSC_BREAKPOINTS_PPM = _FSC_BREAKPOINTS_PPM
except ImportError:
    pass
