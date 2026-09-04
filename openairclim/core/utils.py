"""
Utility functions used over the entire framework
"""

import re
from pathlib import Path
import numpy as np
import xarray as xr
import pint

UREG: pint.UnitRegistry = pint.UnitRegistry()


def find_basenames(path_lst: list) -> list:
    """Find basenames of a list of paths

    Args:
        path_arr (list): List of paths

    Returns:
        list: List of basenames
    """
    basename_lst = []
    for path in path_lst:
        basename = Path(path).stem
        basename_lst.append(basename)
    return basename_lst


def convert_to_regular(inv: xr.Dataset) -> xr.Dataset:
    """Convert flat / unstructured xarray into xarray
    with regular 3D grid lon/lat/plev

    Args:
        inv (xarray): flat / unstructured xarray

    Returns:
        xarray: regular xarray with dimension lon/lat/plev
    """
    inv_reg = inv.set_coords(["lon", "lat", "plev"])
    inv_reg = inv_reg.set_xindex(["lon", "lat", "plev"])
    inv_reg = inv_reg.unstack("index")
    return inv_reg


def convert_nested_to_series(nested_dict: dict) -> dict:
    """Convert nested dictionary to dictionary of np.arrays / time series

    Args:
        nested_dict (dict): Dictionary of dictionaries, keys are species, years
        {spec: {year: np.array, ...}, ...}

    Returns:
        dict: Dictionary of np.arrays / time series, keys are species
        {spec: np.array, np.array, ...}
    """
    plain_dict = {}
    for key, inner_dict in nested_dict.items():
        plain_dict[key] = np.array(list(inner_dict.values()))
    return plain_dict


def tgco2_to_tgc(co2: float | np.ndarray) -> float | np.ndarray:
    """Converts mass of CO2 in Tg to mass of C in Tg

    Args:
        co2 (float or np.ndarray): Mass of CO2 in Tg

    Returns:
        float or np.ndarray: Mass of C in Tg
    """
    tgc = co2 * 12.0 / 44.0
    return tgc


def kgco2_to_tgc(co2: float) -> float:
    """Converts mass of CO2 in kg to mass of C in Tg

    Args:
        co2 (float): Mass of CO2 in kg

    Returns:
        float: Mass of C in Tg
    """
    tgc = co2 * 12.0 / 44.0 * 1e-9
    return tgc


def to_pint_units(unit_str: str | None) -> str:
    """Rewrite a UDUNITS/CF-style unit string into pint syntax.

    pint doesn't parse UDUNITS' compact compound-unit notation (space
    means multiply, a trailing integer means exponent, e.g. "Tg yr-1" or
    "kg m-2 s-1") on its own, so this rewrites each whitespace-separated
    token's trailing exponent into pint's "**" form and joins tokens with
    "*".

    Args:
        unit_str (str or None): UDUNITS/CF-style unit string, e.g. "kg",
            "Tg yr-1", "1" or "" for dimensionless.

    Returns:
        str: Equivalent unit string in pint syntax.

    Raises:
        ValueError: If unit_str already contains "**" — CF/UDUNITS never
            uses it (exponents are a bare suffix, e.g. "m-2"), so its
            presence means the string is in the wrong convention (e.g.
            already pint syntax) rather than just an unusual CF string.
    """
    unit_str = (unit_str or "").strip()
    if not unit_str:
        return "1"
    if "**" in unit_str:
        raise ValueError(
            f"Unit string {unit_str!r} contains '**', which is not a valid "
            "CF/UDUNITS style. Please check that your emission inventories "
            "are defined following the CF/UDUNITS convention (e.g. 'Tg yr-1')."
        )
    parts = []
    for tok in unit_str.split():
        match = re.match(r"^([^\d\-][^\d]*?)(-?\d+)$", tok)
        parts.append(f"{match.group(1)}**{match.group(2)}" if match else tok)
    return "*".join(parts)


def quantity(value: float | np.ndarray, unit_str: str | None) -> pint.Quantity:
    """Build a pint Quantity from a value and a UDUNITS/CF-style unit string.

    Args:
        value (float or np.ndarray): Numeric value(s).
        unit_str (str or None): UDUNITS/CF-style unit string.

    Returns:
        pint.Quantity: The value tagged with its parsed unit.

    Raises:
        ValueError: If unit_str isn't parseable. pint raises a mix of
            pint.errors.PintError subclasses and bare TypeError (e.g. for
            "incorrect-unit", where the "-" is parsed as a subtraction
            operator between two undefined identifiers).
    """
    try:
        return UREG.Quantity(value, to_pint_units(unit_str))
    except (pint.errors.PintError, TypeError) as exc:
        raise ValueError(str(exc)) from exc


def to_value(qty: pint.Quantity, target_units: str) -> float:
    """Convert a pint Quantity to a plain float in target_units.

    Args:
        qty (pint.Quantity): Quantity to convert.
        target_units (str): Target UDUNITS/CF-style unit string.

    Returns:
        float: qty's magnitude expressed in target_units.

    Raises:
        ValueError: If target_units is incompatible or unparseable.
    """
    try:
        return qty.to(to_pint_units(target_units)).magnitude
    except pint.errors.PintError as exc:
        raise ValueError(str(exc)) from exc


def convert_units(value: float, src_units: str, target_units: str) -> float:
    """Convert a value between two UDUNITS/CF-style unit strings.

    Args:
        value (float): Value to convert.
        src_units (str): Source unit string.
        target_units (str): Target unit string.

    Returns:
        float: Converted value.

    Raises:
        ValueError: If the units aren't parseable or compatible.
    """
    return to_value(quantity(value, src_units), target_units)


def convert_mass_or_annual_rate(
    value: float | np.ndarray, src_units: str, target_units: str
) -> float | np.ndarray:
    """Convert a mass, or a mass accumulated per year, to target_units.

    Some inputs (e.g. a time evolution file's "fuel" variable) declare
    units as a rate (mass per year, e.g. "Tg yr-1") that actually
    represents a total accumulated over exactly one year, not a true
    rate meant to be integrated over an arbitrary duration -- so such a
    rate is cancelled by multiplying by exactly one year (exact, via
    unit algebra), rather than converted as a rate. A plain mass (e.g.
    "kg", "Tg") is converted directly.

    Args:
        value (float or np.ndarray): Value(s) to convert.
        src_units (str): Source unit string -- either a mass or a mass
            accumulated per year.
        target_units (str): Target unit string (a mass, e.g. "kg").

    Returns:
        float or np.ndarray: Converted value(s).

    Raises:
        ValueError: If src_units is neither a mass nor a mass-per-year
            rate, or target_units is incompatible/unparseable.
    """
    qty = quantity(value, src_units)
    if qty.dimensionality != UREG.get_dimensionality("[mass]"):
        qty = qty * quantity(1, "yr")
    return to_value(qty, target_units)
