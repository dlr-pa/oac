"""Provides utility functions for the OpenAirClim GUI."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


# define unicode superscripts, rather than using math mode
_SUPERSCRIPTS = str.maketrans(
    "0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b"
)

# visual style
COLORS = ["#2271B2", "#3DB7E9", "#F748A5", "#359B73", "#D55E00", "#E69F00", "#F0E442"]
MARKERS = [
    "circle",
    "square",
    "triangle",
    "inverted_triangle",
    "plus",
    "x",
    "hex",
    "diamond",
]


def superscript(n: int) -> str:
    """Convert an integer to Unicode superscript characters.

    Args:
        n (int): Number to convert.

    Returns:
        str: Unicode superscript representation.
    """
    return str(n).translate(_SUPERSCRIPTS)


def load_inventory(working_dir: str, inv_dir: str, inv_file: str) -> "xr.Dataset":
    """Load a single emission inventory, promoting spatial fields to coords.

    Args:
        working_dir (str): Project working directory.
        inv_dir (str): Inventory subdirectory from config.
        inv_file (str): Inventory filename.

    Returns:
        xarray.Dataset: Loaded inventory with plev, lat, lon as coordinates.
    """
    import xarray as xr

    filepath = Path(working_dir) / inv_dir / inv_file
    ds = xr.load_dataset(filepath)

    # make lat, lon and plev into coordinates for easier manipulation
    coord_names = [c for c in ("plev", "lat", "lon") if c in ds.data_vars]
    if coord_names:
        ds = ds.set_coords(coord_names)

    return ds


def get_numeric_vars(ds: "xr.Dataset") -> list:
    """Return names of plottable numeric data variables. Allows for all
    numeric data within the inventories to be visualised, even if it is not
    used by OpenAirClim.

    Args:
        ds (xarray.Dataset): Emission inventory.

    Returns:
        list: Names of numeric data variables, excluding spatial fields and ac.
    """
    skip = {"ac", "plev", "lat", "lon"}
    numeric_kinds = "iuf"  # signed int, unsigned int, float
    return [
        name
        for name, var in ds.data_vars.items()
        if name not in skip and var.dtype.kind in numeric_kinds
    ]


def auto_scale(max_val: float) -> tuple:
    """Determine a scaling factor for clean axis labels.

    Values in the range [0.1, 1000) are left unscaled. Otherwise, the
    appropriate power of 10 is extracted and returned as a label prefix
    using Unicode superscript characters.

    Args:
        max_val (float): Maximum value on the axis.

    Returns:
        tuple: (divisor, label_prefix) where divisor is the power of 10
            to divide data by, and label_prefix is a string like
            ``"10\u2078 "`` or ``""`` if no scaling is needed.
    """
    import numpy as np

    if max_val == 0 or not np.isfinite(max_val):
        return 1.0, ""
    exponent = int(np.floor(np.log10(abs(max_val))))
    if -1 <= exponent <= 2:
        return 1.0, ""
    return 10.0**exponent, f"10{superscript(exponent)} "
