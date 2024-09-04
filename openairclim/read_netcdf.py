"""
Methods for reading netCDF input
"""

from pathlib import Path
import logging
import numpy as np
import xarray as xr

# CONSTANTS
INV_SPEC_UNITS = ["kg"]


def open_netcdf(netcdf):
    """Converts netCDF file or list of netCDF files to dictionary of xarray Datasets

    Args:
        netcdf (str or list): (List of) netCDF file names

    Returns:
        dict: Dictionary of xarray Datasets, keys are basenames of input netCDF
    """
    xr_dict = {}
    if isinstance(netcdf, list) and all(
        isinstance(ele, str) for ele in netcdf
    ):
        netcdf_arr = netcdf
    elif not isinstance(netcdf, list) and isinstance(netcdf, str):
        netcdf_arr = [netcdf]
    else:
        raise TypeError("Argument is not of type str or list of str")
    for ele in netcdf_arr:
        netcdf_name = Path(ele).stem
        xr_dict[netcdf_name] = xr.load_dataset(ele)
    return xr_dict


def open_inventories(config):
    """Open inventories from config, check attribute sections
    and time constraints

    Args:
        config (dict): Configuration dictionary from config

    Raises:
        IndexError: if no inv_year is within time_range,
            for evolution_type = "norm" or "scaling" or False
        IndexError: if time_range is not within evolution_time,
            for evolution_type = "norm" or "scaling"
        IndexError: if no inv_year is within evolution_time,
            for evolution_tpye = "norm" or "scaling"
        IndexError: if time_range first and last year are not in inv_years,
            for evolution_type = "scaling"

    Returns:
        dict: Dictionary of xarray Datasets, keys are years of input inventories
    """
    # Get list of file names of inventories
    inv_arr = []
    if "dir" in config["inventories"]:
        inv_dir = config["inventories"]["dir"]
    else:
        inv_dir = ""
    files_arr = config["inventories"]["files"]
    for inv_file in files_arr:
        inv_arr.append(inv_dir + inv_file)
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    # Open inventories as dictionary of xarray Datasets
    inv_inp_dict = open_netcdf(inv_arr)
    # Check attribute sections for all emission species given in config
    check_spec_attributes(config, inv_inp_dict)
    # Get years of inventories, keys of new dictionary are years
    # Check time constraint: At least one inv_year must be within time_range
    # Inventories relevant for simulation: inv_year overlapping with time_range,
    # For all evolution_type: "norm" or "scaling" or False
    inv_dict = {}
    for inv_name, inv in inv_inp_dict.items():
        try:
            year = inv.attrs["Inventory_Year"]
            if time_range[0] <= year <= time_range[-1]:
                inv_dict[year] = inv
            else:
                pass
        except KeyError as exc:
            msg = "No Inventory_Year attribute found in inventory" + inv_name
            raise KeyError(msg) from exc
    # Check if new dictionary of inventories empty and sort
    if inv_dict:
        inv_dict = dict(sorted(inv_dict.items()))
        inv_years = list(inv_dict.keys())
    else:
        raise IndexError("At least one inv_year must be within time_range!")
    # Get evolution_type
    evolution_type = get_evolution_type(config)
    if evolution_type in ("scaling", "norm"):
        time_dir = config["time"]["dir"]
        evolution_name = config["time"]["file"]
        evolution_file = time_dir + evolution_name
        evolution = xr.load_dataset(evolution_file)
        try:
            evolution_time = evolution.time.values
        except AttributeError as exc:
            raise AttributeError(
                "No time coordinate found in evolution file"
            ) from exc
        # Check time constraint: time_range must be within evolution_time
        if (
            time_range[0] >= evolution_time[0]
            and time_range[0] <= evolution_time[-1]
            and time_range[-1] >= evolution_time[0]
            and time_range[-1] <= evolution_time[-1]
        ):
            pass
        else:
            raise IndexError("time_range must be within evolution_time!")
        # Check time constraint: At least one inv_year must be within evolution_time
        overlap = False
        for year in inv_years:
            if evolution_time[0] <= year <= evolution_time[-1]:
                overlap = True
            else:
                pass
        if not overlap:
            raise IndexError(
                "At least one inv_year must be within evolution_time!"
            )
    # For evolution_type = False, check if part of time_range is outside
    # of inventories interval. If so, print warning
    elif evolution_type is False:
        if time_range[0] not in inv_years or time_range[-1] not in inv_years:
            logging.warning(
                "time_range is partly outside interval of inventories, "
                "emissions are assumed to be zero during that time period!"
            )
    else:
        raise ValueError(
            "evolution_type must be either 'scaling', 'norm' or False."
        )
    # evolution_type = "scaling"
    # Check time constraint: time_range first and last year must be inventory years
    if evolution_type == "scaling":
        if time_range[0] not in inv_years or time_range[-1] not in inv_years:
            raise IndexError(
                "time_range first and last year must be inventory years!"
            )
    else:
        pass
    logging.info(
        "Emission inventories openend, attribute sections "
        "and time constraints checked successfully."
    )
    return inv_dict


def get_evolution_type(config):
    """Get evolution type

    Args:
        config (dict): Configuration dictionary from config

    Raises:
        ValueError: if type attribute in evolution file is invalid

    Returns:
        str, bool: evolution_tpye: norm or scaling or False
    """
    evolution_type = False
    if "file" in config["time"]:
        time_dir = config["time"]["dir"]
        file_name = config["time"]["file"]
        file_path = time_dir + file_name
        try:
            evolution = xr.load_dataset(file_path)
            evolution_type = evolution.attrs["Type"]
        except ValueError as exc:
            raise ValueError("No evolution file found") from exc
        except KeyError as exc:
            raise KeyError(
                "No Type attribute found in evolution file"
            ) from exc
        if evolution_type in ("norm", "scaling"):
            pass
        else:
            raise ValueError(
                "Type attribute in evolution file must be either scaling or norm."
            )
    else:
        pass
    return evolution_type


def open_netcdf_from_config(config, section, species, resp_type):
    """Open netcdf files and convert to xarray Datasets for given
    section in config and given species

    Args:
        config (dict): Configuration dictionary from config
        section (str): Section in config
        species (list): List of considered species
        resp_type (str): Response type, e.g. "conc", "rf"

    Returns:
        dict: Dictionary of xarray Datasets, one Dataset for each species,
            keys are species names
    """
    xr_dict = {}
    section_dict = config[section]
    for spec in species:
        inp_file = section_dict[spec][resp_type]["file"]
        xr_dict[spec] = xr.load_dataset(inp_file)
    return xr_dict


def get_results(config: dict) -> dict:
    """Get the simulation results from the output netCDF file.

    Args:
        config (dict): Configuration from config file.

    Returns:
        dict:  dictionaries of numpy arrays containing the simulation results,
            keys are species.
    """
    results_file = config["output"]["dir"] + config["output"]["name"] + ".nc"
    results = xr.load_dataset(results_file)
    emis_dict = {}
    conc_dict = {}
    rf_dict = {}
    dtemp_dict = {}
    for var_name, value_arr in results.items():
        var_name = var_name.split("_")
        result_type = var_name[0]
        spec = var_name[-1]
        if result_type == "emis":
            emis_dict[spec] = value_arr
        elif result_type == "conc":
            conc_dict[spec] = value_arr
        elif result_type == "RF":
            rf_dict[spec] = value_arr
        elif result_type == "dT":
            dtemp_dict[spec] = value_arr
        else:
            pass
    return emis_dict, conc_dict, rf_dict, dtemp_dict


def check_spec_attributes(config, inv_dict):
    """Check emission attributes in inventories for given species in config
    TODO Expand list of possible emission units

    Args:
        config (dict): Configuration dictionary from config
        inv_dict (dict): Dictionary of xarray Datasets,
            keys are years of input inventories

    Raises:
        KeyError: if incorrect units found
    """
    species = config["species"]["inv"]
    for year, inv in inv_dict.items():
        for spec in species:
            try:
                attrs = inv[spec].attrs
            except KeyError as exc:
                msg = (
                    "No attributes found in inventory for year "
                    + str(year)
                    + " and species "
                    + spec
                )
                raise KeyError(msg) from exc
            try:
                units = attrs["units"]
            except KeyError as exc:
                msg = (
                    "No units founds in inventory for year "
                    + str(year)
                    + " and species "
                    + spec
                )
                raise KeyError(msg) from exc
            if units in INV_SPEC_UNITS:
                pass
            else:
                msg = (
                    "Incorrect units found in inventory for year "
                    + str(year)
                    + " and species "
                    + spec
                )
                raise KeyError(msg)