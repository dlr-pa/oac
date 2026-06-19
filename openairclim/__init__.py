"""
Initialise OpenAirClim.
"""

from .core import run
from .addon import OAC_PREMIUM_AVAILABLE
from .__about__ import (
    __title__,
    __version__,
    __author__,
    __email__,
    __license__,
    __copyright__,
    __url__,
)

# only run and __version__ are publicly exported
__all__ = ["run", "OAC_PREMIUM_AVAILABLE", "__version__"]
