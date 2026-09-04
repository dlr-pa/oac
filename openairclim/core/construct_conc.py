"""
Constructs concentrations
"""

from pathlib import Path
import numpy as np
import xarray as xr
from .interpolate_time import interp_linear
from .utils import convert_units


def get_emissions(inv_dict: dict, species: str | list[str]) -> tuple[np.ndarray, dict]:
    """Get total emissions in Tg for each inventory and given species

    Args:
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years
        species (str | list[str]): String or list of strings, species names

    Raises:
        TypeError: if species argument has wrong type

    Returns:
        tuple[np.ndarray, dict]: Inventory years and dictionary with arrays of
            emissions in Tg, keys are spec
    """
    if isinstance(species, list) and all(isinstance(ele, str) for ele in species):
        pass
    elif not isinstance(species, list) and isinstance(species, str):
        species = [species]
    else:
        raise TypeError("Species argument is not of type str or list of str")
    emis_dict = {}
    for spec in species:
        target_units = "km" if spec == "distance" else "Tg"  # distance remains in km
        inv_years, emis = calc_inv_sums(spec, inv_dict, target_units=target_units)
        emis_dict[spec] = emis
    return inv_years, emis_dict


def calc_inv_sums(
    spec: str, inv_dict: dict, target_units: str = "kg"
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the emission sums for a given species for a dictionary
    of emission inventories, converted to target_units using each
    inventory's own declared units.

    Args:
        spec (str): Name of species
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years
        target_units (str): Unit string the returned sums are converted
            to. Defaults to "kg".

    Returns:
        np.ndarray, np.ndarray: Inventory years and inventory sums for given
            species, in target_units
    """
    inv_years_lst = []
    inv_sums_lst = []
    for year, inv in inv_dict.items():
        check_inv_values(inv, year, spec)
        inv_years_lst.append(year)
        tot = float(inv[spec].sum())
        # check_spec_attributies already checks that appropriate units exist
        units = inv[spec].attrs.get("units", target_units)
        inv_sums_lst.append(convert_units(tot, units, target_units))
    inv_years = np.array(inv_years_lst)
    inv_sums = np.array(inv_sums_lst)
    return inv_years, inv_sums


def check_inv_values(inv: xr.Dataset, year: str, spec: str) -> None:
    """
    Checks values in given inventory for a specific species.

    Args:
        inv (xarray.Dataset): Emission inventory dataset for a specific year.
        year (str): Year of the inventory.
        spec (str): Species name.

    Raises:
        ValueError: If there are any negative emissions for the given species
            in the inventory.
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


def interp_bg_conc(config: dict, spec: str) -> dict:
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
    dir_name = config["background"]["dir"]
    inp_file = Path(dir_name) / config["background"][spec]["file"]
    scenario = config["background"][spec]["scenario"]
    conc = xr.load_dataset(inp_file)[scenario]
    conc_dict = {spec: conc}
    years = conc["year"].values
    _, interp_conc = interp_linear(config, years, conc_dict)
    return interp_conc
