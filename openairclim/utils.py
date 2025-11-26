"""
Utility functions used over the entire framework
"""

from pathlib import Path
import numpy as np
import cf_units


def find_basenames(path_lst):
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


def convert_to_regular(inv):
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


def convert_nested_to_series(nested_dict):
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


def tgco2_to_tgc(co2):
    """Converts mass of CO2 in Tg to mass of C in Tg

    Args:
        co2 (float): Mass of CO2 in Tg

    Returns:
        float: Mass of C in Tg
    """
    tgc = co2 * 12.0 / 44.0
    return tgc


def kgco2_to_tgc(co2):
    """Converts mass of CO2 in kg to mass of C in Tg

    Args:
        co2 (float): Mass of CO2 in kg

    Returns:
        float: Mass of C in Tg
    """
    tgc = co2 * 12.0 / 44.0 * 1e-9
    return tgc


def tg_to_kg(val):
    """Convert mass in Tg to mass in kg

    Args:
        val (float): Mass in Tg

    Returns:
        float: Mass in kg
    """
    # return 1.0e9 * val
    kilogram = cf_units.Unit("kg")
    teragram = cf_units.Unit("Tg")
    return teragram.convert(val, kilogram)


def kg_to_tg(val):
    """Convert mass in kg to mass in Tg

    Args:
        val (float): Mass in kg

    Returns:
        float: Mass in Tg
    """
    # return 1.0e-9 * val
    kilogram = cf_units.Unit("kg")
    teragram = cf_units.Unit("Tg")
    return kilogram.convert(val, teragram)
