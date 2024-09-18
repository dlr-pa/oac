"""
Constructs concentrations
"""

import numpy as np
import xarray as xr
from openairclim.interpolate_time import interp_linear
from openairclim.utils import kg_to_tg


def get_emissions(inv_dict, species):
    """Get total emissions in Tg for each inventory and given species
    TODO Unit conversions for other units than kg

    Args:
        species (str): String or list of strings, species names
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years
    Raises:
        TypeError: if species argument has wrong type

    Returns:
        dict: Dictionary with arrays of emissions in Tg, keys are spec
    """
    if isinstance(species, list) and all(
        isinstance(ele, str) for ele in species
    ):
        pass
    elif not isinstance(species, list) and isinstance(species, str):
        species = [species]
    else:
        raise TypeError("Species argument is not of type str or list of str")
    emis_dict = {}
    for spec in species:
        emis = calc_inv_sums(spec, inv_dict)
        # Convert kg to Tg
        emis = kg_to_tg(emis)
        emis_dict[spec] = emis
    return emis_dict


def calc_inv_sums(spec, inv_dict):
    """Calculates the emission sums for a given species for a dictionary
    of emission inventories

    Args:
        spec (str): Name of species
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years

    Returns:
        array: Time series array, each value is the sum over one inventory
    """
    inv_years = []
    inv_sums_arr = []
    for year, inv in inv_dict.items():
        check_inv_values(inv, year, spec)
        inv_years.append(year)
        tot = float(inv[spec].sum())
        inv_sums_arr.append(tot)
    inv_years = np.array(inv_years)
    inv_sums = np.array(inv_sums_arr)
    return inv_sums


def check_inv_values(inv, year, spec):
    """
    Checks values in given inventory for a specific species.

    Args:
        inv (xarray.Dataset): Emission inventory dataset for a specific year.
        year (str): Year of the inventory.
        spec (str): Species name.

    Raises:
        ValueError: If there are any negative emissions for the given species in the inventory.
    """
    inv_arr = inv[spec].values
    if np.any(inv_arr < 0.0):
        msg = (
            "Negative emissions detected for inventory year "
            + str(year)
            + " and species "
            + spec
            + ". Only positive emission values are allowed!"
        )
        raise ValueError(msg)


def interp_bg_conc(config, spec):
    """Interpolates background concentrations for given species
    within time_range, for a background file and scenario set in config
    TODO Take into account various conc units in background file

    Args:
        config (dict): Configuration dictionary from config
        spec (str): Species name

    Returns:
        dict: Dictionary with np.ndarray of interpolated concentrations,
            key is species
    """
    inp_file = config["background"][spec]["file"]
    scenario = config["background"][spec]["scenario"]
    conc = xr.load_dataset(inp_file)[scenario]
    conc_dict = {spec: conc}
    years = conc["year"].values
    _, interp_conc = interp_linear(config, years, conc_dict)
    return interp_conc
