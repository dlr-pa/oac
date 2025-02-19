"""
Interpolation methods in the time domain
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from openairclim.read_netcdf import get_evolution_type
from openairclim.utils import tg_to_kg

# from scipy.interpolate import interp1d

# CONSTANTS
# Translation table from evolution keys to inventory keys
KEY_TABLE = {
    "fuel": "fuel",
    "EI_CO2": "CO2",
    "EI_H2O": "H2O",
    "dis_per_fuel": "distance",
}


def interp_linear(
    config, years, val_dict, bounds_error=True, fill_value=np.nan
):
    """Interpolate values from discrete years to time_range set in config

    Args:
        config (dict): Configuration dictionary from config
        years (array): Numpy array with discrete years
        val_dict (dict): Dictionary with time series numpy arrays, keys are species
        bounds_error (bool, optional): See documentation of scipy.interpolate.interp1d
        fill_value (float or None, optional):
            See documentation of scipy.interpolate.RegularGridInterpolator

    Raises:
        IndexError: if too few discrete years or inappropriate options set

    Returns:
        array, dict: Time range where to interpolate,
            and interpolated values, keys are the same as in val_dict
    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    interp_dict = {}
    for key, values in val_dict.items():
        # Under certain circumstances, interpolation works also for 1 given inventory
        if (
            len(years) > 1
            or len(years) == 1
            and isinstance(fill_value, (float, int))
        ):
            # interp_func = interp1d(
            #    years,
            #    values,
            #    kind="linear",
            #    bounds_error=bounds_error,
            #    fill_value=fill_value,
            # )
            # interp_dict[key] = interp_func(time_range)
            interp_func = RegularGridInterpolator(
                [years],
                values,
                method="linear",
                bounds_error=bounds_error,
                fill_value=fill_value,
            )
            new_grid = np.ix_(time_range)
            interp_dict[key] = interp_func(new_grid)
        # only one discrete year given,
        # repeat value array times number of years in time_range
        elif len(years) == 1 and fill_value is None:
            # interp_dict[key] = np.ones(len(time_range)) * values
            interp_dict[key] = np.squeeze(np.array([values] * len(time_range)))
        else:
            raise IndexError(
                "Interpolation not possible! Too few discrete values or "
                "inappropriate options set for function interp_linear"
            )
    return time_range, interp_dict


def apply_evolution(config, val_dict, inv_dict):
    """Determine evolution_type, apply normalization, scaling or no evolution

    Args:
        config (dict): Configuration dictionary from config
        val_dict (dict): Dictionary with time series numpy arrays, keys are species
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of input inventories

    Raises:
        ValueError: if no valid evolution_type in evolution file

    Returns:
        array, dict: time_range and normalized/scaled/unmodified dictionary,
            np.ndarray, {spec: np.ndarray}
    """
    evolution_type = get_evolution_type(config)
    if evolution_type == "scaling":
        time_range, out_dict = apply_scaling(config, val_dict, inv_dict)
    elif evolution_type == "norm":
        time_range, out_dict = apply_norm(config, val_dict, inv_dict)
    elif evolution_type is False:
        time_range, out_dict = apply_no_evolution(config, val_dict, inv_dict)
    else:
        raise ValueError(
            "Invalid evolution_type "
            "evolution_type can be either 'scaling', 'norm' or False"
        )
    return time_range, out_dict


def apply_scaling(config, val_dict, inv_dict):
    """Apply scaling on dictionary of time series,
    scaling factors are from evolution file,
    time series and scaling factors are interpolated on time_range
    before multiplication
    TODO: implement scaling for individual species

    Args:
        config (dict): Configuration dictionary from config
        val_dict (dict): Dictionary with time series numpy arrays, keys are species
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of input inventories

    Returns:
        array, dict: time_range and scaled dictionary
    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    # Get inventory years
    inv_years = np.array(list(inv_dict.keys()))
    # Get evolution data
    time_dir = config["time"]["dir"]
    file_name = config["time"]["file"]
    file_path = time_dir + file_name
    evolution = xr.load_dataset(file_path)
    evolution_time = evolution.time.values
    evo_scaling = evolution.scaling.values
    # Interpolate scaling factors on time_range
    _time_range, scaling_dict = interp_linear(
        config, evolution_time, {"scaling": evo_scaling}
    )
    # Interpolate time series data on time_range
    _time_range, interp_dict = interp_linear(config, inv_years, val_dict)
    out_dict = {}
    # for key, values in interp_dict.items():
    #    scaled_dict[key] = np.multiply(scaling_factors_dict["scaling"], values)
    for spec, series_arr in interp_dict.items():
        shape_tp = np.shape(np.transpose(series_arr))
        scaling = np.transpose(np.resize(scaling_dict["scaling"], shape_tp))
        out_dict[spec] = np.multiply(scaling, series_arr)
    return time_range, out_dict


def apply_norm(config, val_dict, inv_dict):
    """Apply normalization on time series,
    get data from evolution file and inventories,
    interpolate time series and evolution data over time_range,
    calculate normalization factors and apply (multiplication)

    Args:
        config (dict): Configuration dictionary from config
        val_dict (dict): Dictionary with time series numpy array, keys are species
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of input inventories

    Returns:
        array, dict: time_range and normalized dictionary
    """
    out_dict = {}
    # Interpolate evolution over time_range
    _time_range, evo_interp_dict = interp_evolution(config)
    # Calculate fuel sums and emission indices from inventories
    # {"fuel": np.ndarray, "EI_CO2": np.ndarray, ..}
    _inv_years, _inv_sum_dict, inv_emi_indices_dict = calc_inv_emi_indices(
        config, inv_dict
    )
    # Interpolate emission indices from inventories over time_range
    # {"fuel": np.ndarray, "EI_CO2": np.ndarray, ..}
    inv_years = np.array(list(inv_dict.keys()))
    # TODO Check bounds, extrapolate
    time_range, inv_interp_dict = interp_linear(
        config,
        inv_years,
        inv_emi_indices_dict,
        bounds_error=False,
        fill_value=None,
    )
    # Calculate normalization factors, over time_range
    norm_dict = calc_norm(evo_interp_dict, inv_interp_dict)
    norm_fuel = norm_dict["fuel"]
    # Interpolate time series data on time_range
    _time_range, interp_dict = interp_linear(
        config,
        inv_years,
        val_dict,
        bounds_error=False,
        fill_value=None,
    )
    # Apply normalization factors to interp_dict
    for spec, series_arr in interp_dict.items():
        shape_tp = np.shape(np.transpose(series_arr))
        if spec in norm_dict:
            norm = np.transpose(np.resize(norm_dict[spec], shape_tp))
            normalized = np.multiply(norm, series_arr)
        else:
            norm = np.transpose(np.resize(norm_fuel, shape_tp))
            normalized = np.multiply(norm, series_arr)
        out_dict[spec] = normalized
    return time_range, out_dict


def apply_no_evolution(config, val_dict, inv_dict):
    """Apply interpolation only on time series data over time_range

    Args:
        config (dict): Configuration dictionary from config
        val_dict (dict): Dictionary with time series numpy arrays, keys are species
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of input inventories

    Returns:
        array, dict: time_range and dictionary of time series data
    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    # Get inventory years
    inv_years = np.array(list(inv_dict.keys()))
    # Interpolate time series data on time_range
    _time_range, interp_dict = interp_linear(
        config, inv_years, val_dict, bounds_error=False, fill_value=0.0
    )
    return time_range, interp_dict


def interp_evolution(config):
    """Interpolate values in evolution file over time_range

    Args:
        config (dict): Configuration dictionary from config

    Returns:
        array, dict: time_range and Dictionary with interpolated values
    """
    time_dir = config["time"]["dir"]
    file_name = config["time"]["file"]
    file_path = time_dir + file_name
    evolution = xr.load_dataset(file_path)
    evo_years = evolution.time.values
    evo_dict = {}
    for key, var in evolution.items():
        arr = var.values
        if key == "fuel":
            if "Tg" in var.attrs["units"]:
                arr = tg_to_kg(arr)
        evo_dict[key] = arr
    time_range, evo_interp_dict = interp_linear(config, evo_years, evo_dict)
    return time_range, evo_interp_dict


def calc_inv_emi_indices(config, inv_dict):
    """Calculate fuel sums, emission sums and emission indices from inventories,
    Sums and emission indices are only calculated from species included in time evolution

    Args:
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of inventories

    Returns:
        np.ndarry, dict, dict: Array of inventory years,
            Dictionary of arrays of summed inventory emissions, keys are species
            Dictionary of arrays of fuel sums and inventory emission indices,
            keys are data variable names of evolution file
    """
    # Translation table from evolution keys to inventory keys
    evo_inv_table = KEY_TABLE
    time_dir = config["time"]["dir"]
    file_name = config["time"]["file"]
    file_path = time_dir + file_name
    evolution = xr.load_dataset(file_path)
    # Get required names of emission indices from evolution file
    evo_key_arr = evolution.keys()
    # Initialize output array and dictionaries with correct keys and empty lists
    inv_years = []
    inv_emi_indices_dict = {}
    inv_sum_dict = {}
    for evo_key in evo_key_arr:
        inv_emi_indices_dict[evo_key] = []
        inv_key = evo_inv_table[evo_key]
        inv_sum_dict[inv_key] = []
    for year, inv in inv_dict.items():
        inv_years.append(year)
        fuel_sum = inv["fuel"].sum().values.item()
        # Fuel sum for normalization of species without emission index
        for evo_key in evo_key_arr:
            inv_key = evo_inv_table[evo_key]
            if inv_key == "fuel":
                inv_sum_dict[inv_key].append(fuel_sum)
                inv_emi_indices_dict[evo_key].append(fuel_sum)
            else:
                spec_sum = inv[inv_key].sum().values.item()
                inv_emi_index = spec_sum / fuel_sum
                inv_sum_dict[inv_key].append(spec_sum)
                inv_emi_indices_dict[evo_key].append(inv_emi_index)
    # Convert lists into numpy arrays
    inv_years = np.array(inv_years)
    for inv_key, sum_arr in inv_sum_dict.items():
        inv_sum_dict[inv_key] = np.array(sum_arr)
    for evo_key, emi_index_arr in inv_emi_indices_dict.items():
        inv_emi_indices_dict[evo_key] = np.array(emi_index_arr)
    return inv_years, inv_sum_dict, inv_emi_indices_dict


def calc_norm(evo_dict, ei_inv_dict):
    """Calculate normalization factors, either by fuel use only,
    or combined with evolution of emission index of species

    Args:
        evo_dict (dict): Evolution
            {"fuel": np.ndarray, "EI_CO2": np.ndarray, ..}
        ei_inv_dict (dict): Emission indices, calculated from inventories,
            {"fuel": np.ndarray, "EI_CO2": np.ndarray, ..}

    Returns:
        dict: Dictionary of normalization factors, keys are "fuel" and species
            {"fuel": np.ndarray (norm_fuel = evo_fuel / inv_fuel),
             "CO2": np.ndarray (norm_fuel * evo_EI / inv_EI), ..}
    """
    norm_dict = {}
    # Translation table from evolution to inventory variable names
    key_table = KEY_TABLE
    evo_fuel = evo_dict["fuel"]
    inv_fuel = ei_inv_dict["fuel"]
    norm_fuel = np.divide(evo_fuel, inv_fuel)
    for key, inv_emi_index in ei_inv_dict.items():
        if key == "fuel":
            norm_arr = norm_fuel
        else:
            evo_emi_index = evo_dict[key]
            norm_spec = np.divide(evo_emi_index, inv_emi_index)
            norm_arr = np.multiply(norm_fuel, norm_spec)
        norm_key = key_table[key]
        norm_dict[norm_key] = norm_arr
    return norm_dict


def filter_evo(
    inv_years: np.ndarray, time_range: np.ndarray, evo_interp_dict: dict
) -> dict:
    """Filters interpolated evolution to values for inventory years only

    Args:
        inv_years (np.ndarray): Array of inventory years
        time_range (np.ndarray): time_range from config
        evo_interp_dict (dict): Evolution, interpolated over time_range

    Returns:
        dict: Evolution, filtered to inventory years
    """
    evo_filtered_dict = {}
    index_arr = []
    i = 0
    for year in time_range:
        if year in inv_years:
            index_arr.append(i)
        i = i + 1
    for key, evo_arr in evo_interp_dict.items():
        evo_inv_arr = evo_arr[index_arr]
        evo_filtered_dict[key] = evo_inv_arr
    return evo_filtered_dict


def multiply_inv(inv_dict: dict, norm_dict: dict) -> dict:
    """Multiply data variables in emission inventories by normalization factors,
    depending on available keys in norm_dict, data variables are multiplied by
    norm_fuel or (norm_fuel * norm_EI), with norm_EI = evo_EI / inv_EI

    Args:
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of inventories
        norm_dict (dict): Dictionary of normalization factors, keys are "fuel" and species
            {"fuel": np.ndarray (norm_fuel = evo_fuel / inv_fuel),
             "CO2": np.ndarray (norm_fuel * evo_EI / inv_EI), ..}

    Returns:
        dict: Dictionary of xarray Datasets (normalized emission inventories),
            keys are years of inventories
    """
    out_inv_dict = {}
    i = 0
    for year, inv in inv_dict.items():
        # Create normalization sub dictionary for current inventory, with scalar values
        norm_sub_dict = {}
        for key, norm_arr in norm_dict.items():
            norm_sub_dict[key] = norm_arr[i]
        # Iterate over data variables in inventory
        for data_key, data_arr in inv.items():
            # Check if data_key is within norm_inv_dict
            if data_key in norm_sub_dict:
                # fuel: multiply with norm_fuel
                if data_key == "fuel":
                    # Multiply inv.fuel by normalization factor norm_fuel
                    data_arr = data_arr * norm_sub_dict["fuel"]
                # species: multiply by (norm_fuel * evo_EI / inv_EI)
                else:
                    data_arr = data_arr * norm_sub_dict[data_key]
            # lon, lat, plev: do NOT multiply
            elif data_key in ["lon", "lat", "plev"]:
                pass
            # species not in norm_inv_dict: multiply with norm_fuel
            else:
                data_arr = data_arr * norm_sub_dict["fuel"]
            inv.update({data_key: data_arr})
        out_inv_dict[year] = inv
        i = i + 1
    return out_inv_dict
