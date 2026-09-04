"""
Initialise OpenAirClim.
"""

from typing import Any

from .__about__ import (
    __title__,
    __version__,
    __author__,
    __email__,
    __license__,
    __copyright__,
    __url__,
)

__all__ = [
    "run",  # pylint: disable=undefined-all-variable
    "OAC_PREMIUM_AVAILABLE",  # pylint: disable=undefined-all-variable
    "__title__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
    "__url__",
]


def __getattr__(name: str) -> Any:
    """Lazily import `run` and `OAC_PREMIUM_AVAILABLE` on first access.

    Entry points that don't need the core simulation engine (e.g.
    oac-download-data, oac-download-zenodo) import submodules of this
    package too, and Python always initialises a package's __init__.py
    before importing any of its submodules. Without this, every entry
    point would eagerly import `core`/`addon` (and trigger their
    import-time log messages) even when it never uses either.
    """
    if name == "run":
        from .core import run

        return run
    if name == "OAC_PREMIUM_AVAILABLE":
        from .addon import OAC_PREMIUM_AVAILABLE

        return OAC_PREMIUM_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
