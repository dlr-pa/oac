"""Methods for reading netCDF input."""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from .utils import UREG, convert_mass_or_annual_rate, quantity

logger = logging.getLogger(__name__)

# CONSTANTS
# expected physical dimension of each inventory species' units; species not
# listed here default to mass.
INV_SPEC_DIMENSIONS = {"distance": "[length]"}


def open_netcdf(netcdf: str | Path | Sequence[str | Path]) -> dict:
    """Converts netCDF file or list of netCDF files to dictionary of xarray Datasets.

    Args:
        netcdf (str or list): (List of) netCDF file names

    Returns:
        dict: Dictionary of :class:`xarray.Datasets`, keys are basenames of input netCDF
    """
    xr_dict = {}
    if isinstance(netcdf, list) and all(isinstance(ele, (str, Path)) for ele in netcdf):
        netcdf_arr = netcdf
    elif not isinstance(netcdf, list) and isinstance(netcdf, (str, Path)):
        netcdf_arr = [netcdf]
    else:
        raise TypeError("Argument is not of type str/Path or list of str/Path")
    for ele in netcdf_arr:
        netcdf_name = Path(ele).stem
        xr_dict[netcdf_name] = xr.load_dataset(ele)
    return xr_dict


def _resolve_inventory_paths(config: dict, base: bool) -> list[Path]:
    """Resolves the file paths of the configured (base) emission inventories.

    Args:
        config (dict): Configuration dictionary from config
        base (bool): If True, resolves base inventory paths, else input
            inventory paths.

    Returns:
        list[Path]: One path per referenced inventory file.
    """
    if base:
        inv_dir = config["inventories"]["base"].get("dir", "")
        files_arr = config["inventories"]["base"]["files"]
    else:
        inv_dir = config["inventories"].get("dir", "")
        files_arr = config["inventories"]["files"]
    return [Path(inv_dir) / inv_file for inv_file in files_arr]


def _filter_inventories_to_time_range(
    inv_inp_dict: dict, time_range: np.ndarray
) -> dict:
    """Filters loaded inventories to those overlapping time_range.

    Also normalises longitude values to between 0 and 360 degrees.

    Args:
        inv_inp_dict (dict): Dictionary of xarray Datasets, keys are the
            basenames of the loaded inventory files.
        time_range (numpy.ndarray): Simulation time range from config.

    Raises:
        KeyError: if an inventory has no Inventory_Year attribute.

    Returns:
        dict: Dictionary of xarray Datasets, keys are inventory years within
        time_range.
    """
    inv_dict = {}
    for inv_name, inv in inv_inp_dict.items():
        # Update longitudes to be between 0 and 360 degrees
        if inv.lon.min() < 0.0:
            logger.warning(
                "Longitude values have been automatically updated to be between "
                "0 and 360 degrees to match pre-calculated data."
            )
            inv_norm = inv.assign(lon=inv.lon % 360.0)
        else:
            inv_norm = inv
        try:
            year = inv_norm.attrs["Inventory_Year"]
        except KeyError as exc:
            msg = "No Inventory_Year attribute found in inventory" + inv_name
            raise KeyError(msg) from exc
        if time_range[0] <= year <= time_range[-1]:
            inv_dict[year] = inv_norm
    return inv_dict


def _check_evolution_time_constraints(
    config: dict, inv_years: list[int], time_range: np.ndarray
) -> None:
    """Checks time-range constraints against the configured evolution type.

    Args:
        config (dict): Configuration dictionary from config
        inv_years (list[int]): Sorted years present in the (already
            time_range-filtered) inventory dictionary.
        time_range (numpy.ndarray): Simulation time range from config.

    Raises:
        IndexError: if time_range is not within evolution_time, for
            evolution_type = "norm" or "scaling"
        IndexError: if no inv_year is within evolution_time, for
            evolution_type = "norm" or "scaling"
        IndexError: if time_range first and last year are not in inv_years,
            for evolution_type = "scaling"
        ValueError: if evolution_type is not "norm", "scaling" or False
    """
    evolution_type = get_evolution_type(config)
    if evolution_type in ("scaling", "norm"):
        time_dir = config["time"]["dir"]
        evolution_name = config["time"]["file"]
        evolution_file = Path(time_dir) / evolution_name
        evolution = xr.load_dataset(evolution_file)
        check_evolution_attributes(evolution)
        try:
            evolution_time = evolution.time.values
        except AttributeError as exc:
            raise AttributeError("No time coordinate found in evolution file") from exc
        # Check time constraint: time_range must be within evolution_time
        if not (
            evolution_time[0] <= time_range[0] <= evolution_time[-1]
            and evolution_time[0] <= time_range[-1] <= evolution_time[-1]
        ):
            raise IndexError("time_range must be within evolution_time!")
        # Check time constraint: At least one inv_year must be within evolution_time
        overlap = any(
            evolution_time[0] <= year <= evolution_time[-1] for year in inv_years
        )
        if not overlap:
            raise IndexError("At least one inv_year must be within evolution_time!")
    # For evolution_type = False, check if part of time_range is outside
    # of inventories interval. If so, print warning
    elif evolution_type is False:
        if time_range[0] not in inv_years or time_range[-1] not in inv_years:
            logger.warning(
                "time_range is partly outside interval of inventories, "
                "emissions are assumed to be zero during that time period!"
            )
    else:
        raise ValueError("evolution_type must be either 'scaling', 'norm' or False.")

    # evolution_type = "scaling"
    # Check time constraint: time_range first and last year must be inventory years
    if evolution_type == "scaling" and (
        time_range[0] not in inv_years or time_range[-1] not in inv_years
    ):
        raise IndexError("time_range first and last year must be inventory years!")


def open_inventories(config: dict, base: bool = False) -> dict:
    """Open inventories from config, check attribute sections and time constraints.

    Args:
        config (dict): Configuration dictionary from config
        base (bool): If True, loads base inventory, else input inventory

    Raises:
        IndexError: if no inv_year is within time_range,
            for evolution_type = "norm" or "scaling" or False
        IndexError: if time_range is not within evolution_time,
            for evolution_type = "norm" or "scaling"
        IndexError: if no inv_year is within evolution_time,
            for evolution_type = "norm" or "scaling"
        IndexError: if time_range first and last year are not in inv_years,
            for evolution_type = "scaling"

    Returns:
        dict: Dictionary of xarray Datasets, keys are years of input inventories
    """
    inv_arr = _resolve_inventory_paths(config, base)
    time_config = config["time"]["range"]
    time_range = np.arange(time_config[0], time_config[1], time_config[2], dtype=int)

    # Open inventories as dictionary of xarray Datasets
    inv_inp_dict = open_netcdf(inv_arr)
    # Check attribute sections for all emission species given in config
    check_spec_attributes(config, inv_inp_dict)

    # Get years of inventories overlapping time_range, keys of new dictionary
    # are years. At least one inv_year must be within time_range, for all
    # evolution_type: "norm" or "scaling" or False
    inv_dict = _filter_inventories_to_time_range(inv_inp_dict, time_range)
    if not inv_dict:
        raise IndexError("At least one inv_year must be within time_range!")
    inv_dict = dict(sorted(inv_dict.items()))
    inv_years = list(inv_dict.keys())

    _check_evolution_time_constraints(config, inv_years, time_range)

    logger.info(
        "Emission inventories openend, attribute sections "
        "and time constraints checked successfully."
    )
    return inv_dict


def _resolve_aircraft_list(config: dict, inv_dict: dict) -> list[str]:
    """Determines which aircraft identifiers to split the inventories by.

    Args:
        config (dict): Configuration dictionary from config
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.

    Raises:
        ValueError: If an aircraft identifier present in the inventories is
            not defined in config, or if no "ac" data variable is found and
            contrails are being calculated without a "DEFAULT" aircraft.

    Returns:
        list[str]: Aircraft identifiers to split the inventories by.
    """
    # check which aircraft are defined in inventories and config
    ac_lst_inv = np.array(
        sorted(
            {
                str(ac)
                for inv in inv_dict.values()
                if "ac" in inv
                for ac in np.unique(inv.ac.data)
            }
        ),
        dtype=str,
    )
    ac_lst_config = np.array(config["aircraft"]["types"], dtype=str)

    # check to ensure all aircraft are defined in config
    if not np.isin(ac_lst_inv, ac_lst_config).all():
        missing = ac_lst_inv[~np.isin(ac_lst_inv, ac_lst_config)]
        raise ValueError(
            "The following aircraft identifiers are present in the emission "
            f"inventories but not defined in config: {missing}."
        )

    if ac_lst_inv.size > 0:
        return ac_lst_inv.astype(str).tolist()

    # if no "ac" data variable, check whether "DEFAULT" is defined in config
    # only necessary if contrails are to be calculated
    if "cont" not in config["species"]["out"]:
        return ac_lst_inv.astype(str).tolist()
    if "DEFAULT" in ac_lst_config:
        logger.info(
            "No ac data variable found in the emission inventories. "
            "Reverting to 'DEFAULT' aircraft from config file."
        )
        return ["DEFAULT"]
    raise ValueError(
        "No ac data variable found in the emission inventories and "
        "'DEFAULT' aircraft not defined in config. G_250, b and "
        "PMrel parameters are required for contrail calculations."
    )


def split_inventory_by_aircraft(
    config: dict, inv_dict: dict, base: bool = False
) -> dict:
    """Split dictionary of emission inventories by aircraft identifiers.

    Aircraft identifiers are defined in the config file.

    Args:
        config (dict): Configuration dictionary from config
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
        base (bool, optional): Whether an input or base emission inventory is
            to be split. If True, ``"BASE_"`` is added before the aircraft identifiers.
            Defaults to False.

    Returns:
        dict: Nested dictionary of emission inventories. Keys are aircraft
        identifier, followed by year.
    """
    ac_lst = _resolve_aircraft_list(config, inv_dict)

    def _create_zero_inv(year: int, inv: xr.Dataset) -> xr.Dataset:
        # creates a zero inventory - necessary if values don't exist for a
        # given aircraft identifier in an inventory year
        vars_in_inv = {str(v) for v in inv.data_vars}
        data_vars = {
            v: (("index",), [0.0])
            for v in sorted(vars_in_inv - {"plev", "ac"})
        }
        data_vars["plev"] = (("index",), [300.0])  # random plev
        zero_inv = xr.Dataset(
            data_vars=data_vars,
            coords={"index": np.array([0], dtype=np.int64)},
            attrs={"Inventory_Year": year},
        )
        return zero_inv

    # initialise full dictionary
    full_inv_dict = {
        ac: {year: _create_zero_inv(year, inv) for year, inv in inv_dict.items()}
        for ac in [*ac_lst, "TOTAL"]
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
                    full_inv_dict[ac].update({year: inv.where(inv.ac == ac, drop=True)})

            # add "TOTAL"
            full_inv_dict["TOTAL"].update({year: inv.copy().drop_vars("ac")})

    if base:
        full_inv_dict = {
            f"BASE_{ac}": inner for ac, inner in full_inv_dict.items()
        }

    return full_inv_dict


def get_evolution_type(config: dict) -> str | bool:
    """Get evolution type.

    Args:
        config (dict): Configuration dictionary from config

    Raises:
        ValueError: if type attribute in evolution file is invalid

    Returns:
        str, bool: evolution_type: norm or scaling or False
    """
    evolution_type: str | bool = False
    if "file" in config["time"]:
        time_dir = config["time"]["dir"]
        file_name = config["time"]["file"]
        file_path = Path(time_dir) / file_name
        try:
            evolution = xr.load_dataset(file_path)
            evolution_type = evolution.attrs["Type"]
        except ValueError as exc:
            raise ValueError("No evolution file found") from exc
        except KeyError as exc:
            raise KeyError("No Type attribute found in evolution file") from exc
        if evolution_type not in ("norm", "scaling"):
            raise ValueError(
                "Type attribute in evolution file must be either scaling or norm."
            )
    return evolution_type


def open_netcdf_from_config(
    config: dict, section: str, species: list[str], resp_type: str
) -> dict:
    """Open netcdf files and convert to xarray Datasets.

    For a given section in config and given species.

    Args:
        config (dict): Configuration dictionary from config
        section (str): Section in config
        species (list[str]): List of considered species
        resp_type (str): Response type, e.g. "conc", "rf"

    Returns:
        dict: Dictionary of xarray Datasets, one Dataset for each species,
        keys are species names
    """
    xr_dict = {}
    section_dict = config[section]
    dir_name = section_dict["dir"]
    for spec in species:
        inp_file = Path(dir_name) / section_dict[spec][resp_type]["file"]
        xr_dict[spec] = xr.load_dataset(inp_file)
    return xr_dict


def get_results(config: dict, ac: str = "TOTAL") -> tuple[dict, dict, dict, dict]:
    """Get the simulation results from the output netCDF file.

    Args:
        config (dict): Configuration from config file.
        ac (str, optional): Aircraft identifier, defaults to TOTAL

    Returns:
        dict:  dictionaries of numpy arrays containing the simulation results,
        keys are species.
    """
    results_file = Path(config["output"]["dir"]) / f"{config['output']['name']}.nc"
    results = xr.load_dataset(results_file)
    emis_dict = {}
    conc_dict = {}
    rf_dict = {}
    dtemp_dict = {}
    for raw_var_name, raw_value_arr in results.items():
        var_name = str(raw_var_name)
        # handle multi-aircraft results
        if "ac" in raw_value_arr.dims:
            if ac not in raw_value_arr.coords["ac"].values:
                raise ValueError(
                    f"'ac' coordinate exists in {var_name}, but no '{ac}' "
                    "entry found."
                )
            value_arr = raw_value_arr.sel(ac=ac)
        else:
            value_arr = raw_value_arr
        var_name_parts: list[str] = var_name.split("_")
        result_type = var_name_parts[0]
        spec = var_name_parts[-1]
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


def check_spec_attributes(config: dict, inv_dict: dict) -> None:
    """Check emission attributes in inventories for given species in config.

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
            location = f" in inventory for year {year} and species {spec}"
            try:
                attrs = inv[spec].attrs
            except KeyError as exc:
                raise KeyError(f"No attributes found{location}") from exc
            try:
                units = attrs["units"]
            except KeyError as exc:
                raise KeyError(f"No units found{location}") from exc
            # inputs are assumed to have dimension [mass] except for distance
            # see CONSTANTS at the top of this file
            expected_dim = UREG.get_dimensionality(
                INV_SPEC_DIMENSIONS.get(spec, "[mass]")
            )
            try:
                if quantity(1.0, units).dimensionality != expected_dim:
                    raise ValueError("Units are not dimensionally compatible")
            except ValueError as exc:
                raise KeyError(f"Incorrect units found{location}") from exc


def check_evolution_attributes(evolution: xr.Dataset) -> None:
    """Check the "fuel" variable's units in a time evolution file, if present.

    "fuel" may be declared as a mass (e.g. "Tg") or a mass accumulated
    per year (e.g. "Tg yr-1").

    Args:
        evolution (xarray.Dataset): Loaded time evolution dataset.

    Raises:
        KeyError: if "fuel" is present but its units are missing or not
            mass-compatible.
    """
    if "fuel" not in evolution.data_vars:
        return
    try:
        units = evolution["fuel"].attrs["units"]
    except KeyError as exc:
        raise KeyError(
            "No units found for 'fuel' in time evolution file"
        ) from exc
    try:
        convert_mass_or_annual_rate(1.0, units, "kg")
    except ValueError as exc:
        msg = "Incorrect units found for 'fuel' in time evolution file: " + str(units)
        raise KeyError(msg) from exc
