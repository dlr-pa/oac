"""
Methods for reading netCDF input
"""

from pathlib import Path
import logging
import numpy as np
import xarray as xr

# CONSTANTS
INV_SPEC_UNITS = ["kg", "km"]


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


def open_inventories(config, base=False):
    """Open inventories from config, check attribute sections
    and time constraints

    Args:
        config (dict): Configuration dictionary from config
        base (bool): If TRUE, loads base inventory, else input inventory

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

    # initialise array of inventories
    inv_arr = []

    # if base is TRUE, base inventories are loaded
    if base:
        if "dir" in config["inventories"]["base"]:
            inv_dir = config["inventories"]["base"]["dir"]
        else:
            inv_dir = ""
        files_arr = config["inventories"]["base"]["files"]

    # otherwise, load input inventories
    else:
        if "dir" in config["inventories"]:
            inv_dir = config["inventories"]["dir"]
        else:
            inv_dir = ""
        files_arr = config["inventories"]["files"]

    # load files
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
        # Update longitudes to be between 0 and 360 degrees
        if inv.lon.min() < 0.0:
            logging.warning(
                "Longitude values have been automatically updated to be between "
                "0 and 360 degrees to match pre-calculated data."
            )
            inv = inv.assign(lon=inv.lon % 360.0)
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


def split_inventory_by_aircraft(config, inv_dict):
    """Split dictionary of emission inventories by aircraft identifiers defined
    in the config file.

    Args:
        config (dict): Configuration dictionary from config
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.

    Returns:
        dict: Nested dictionary of emission inventories. Keys are aircraft
            identifier, followed by year.
    """

    # check which aircraft are defined in inventories and config
    ac_lst_inv = sorted({
        ac
        for _, inv in inv_dict.items()
        if "ac" in inv
        for ac in np.unique(inv.ac.data)
    })
    ac_lst_config = config["aircraft"]["types"]

    # TEMPORARY
    # since contrail attribution methodologies have not yet been implemented,
    # contrails cannot be calculated for multiple aircraft
    if len(ac_lst_inv) > 1 and "cont" in config["species"]["out"]:
        raise ValueError(
            "In the current version of OpenAirClim, it is not possible to "
            "calculate the contrail climate impact for multiple aircraft "
            "within the same emission inventory."
        )

    # check to ensure all aircraft are defined in config
    if not np.isin(ac_lst_inv, ac_lst_config).all():
        missing = ac_lst_inv[~np.isin(ac_lst_inv, ac_lst_config)]
        raise ValueError(
            "The following aircraft identifiers are present in the emission "
            f"inventories but not defined in config: {missing}."
        )

    # if no "ac" data variable, check whether "DEFAULT" is defined in config
    # only necessary if contrails are to be calculated
    if not ac_lst_inv and "cont" in config["species"]["out"]:
        if "DEFAULT" in ac_lst_config:
            ac_lst = ["DEFAULT"]
            logging.info(
                "No ac data variable found in the emission inventories. "
                "Reverting to 'DEFAULT' aircraft from config file."
            )
        else:
            raise ValueError(
                "No ac data variable found in the emission inventories and "
                "'DEFAULT' aircraft not defined in config. G_250, eff_fac and "
                "PMrel parameters are required for contrail calculations."
            )
    else:
        ac_lst = ac_lst_inv

    # initialise full dictionary
    full_inv_dict = {
        ac: {
            year: {}
            for year in inv_dict.keys()
        }
        for ac in ac_lst + ["TOTAL"]
    }

    # loop through emission inventories
    for year, inv in inv_dict.items():
        # if emission inventory does not contain "ac" data variable
        if "ac" not in inv.data_vars:
            if "DEFAULT" in full_inv_dict:
                full_inv_dict["DEFAULT"].update({year: inv})
            full_inv_dict["TOTAL"].update({year: inv})
        else:
            for ac in ac_lst:
                # if ac in inv, add subset of inventory
                if ac in inv.ac:
                    full_inv_dict[ac].update({
                        year: inv.where(inv.ac == ac, drop=True)
                    })
                # if ac not in inv, add a zero-value inventory
                else:
                    vars_in_inv = set(inv.data_vars)
                    data_vars = {
                        v: (("index",), [0.0])
                        for v in sorted(vars_in_inv - {"plev", "ac"})
                    }
                    data_vars["plev"] = (("index",), [300.0])  # random plev
                    zero_inv = xr.Dataset(
                        data_vars=data_vars,
                        coords={"index": np.array([0], dtype=np.int64)},
                        attrs={"Inventory_Year": year}
                    )
                    full_inv_dict[ac].update({year: zero_inv})

                    # add warning
                    logging.warning(
                        "Created zero-inventory for ac %s in year %s", ac, year
                    )

            # add "TOTAL"
            full_inv_dict["TOTAL"].update({year: inv.copy().drop_vars("ac")})

    return full_inv_dict


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
    dir_name = section_dict["dir"]
    for spec in species:
        inp_file = dir_name + section_dict[spec][resp_type]["file"]
        xr_dict[spec] = xr.load_dataset(inp_file)
    return xr_dict


def get_results(config: dict, ac="TOTAL") -> tuple[dict, dict, dict, dict]:
    """Get the simulation results from the output netCDF file.

    Args:
        config (dict): Configuration from config file.
        ac (str, optional): Aircraft identifier, defaults to TOTAL

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
        # handle multi-aircraft results
        if "ac" in value_arr.dims:
            if ac in value_arr.coords["ac"].values:
                value_arr = value_arr.sel(ac=ac)
            else:
                raise ValueError(
                    f"'ac' coordinate exists in {var_name}, but no '{ac}'"
                    "entry found."
                )
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
