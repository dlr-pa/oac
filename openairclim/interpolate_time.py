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
    "EI_NOx": "NOx",
    "dis_per_fuel": "distance",
}


def interpolate(
    config: dict, years: np.ndarray, val_dict: dict
) -> tuple[np.ndarray, dict]:
    """Interpolate values from discrete years to time_range set in config

    Args:
        config (dict): Configuration dictionary from config
        years (array): Numpy array with discrete years
        val_dict (dict): Dictionary with time series numpy arrays, keys are species

    Returns:
        array, dict: Time range over which interpolation takes place,
            and interpolated values, keys are the same as in val_dict
    """
    # TODO extend this function with several interpolation methods defined in config
    time_range, interp_dict = interp_linear(config, years, val_dict)
    return time_range, interp_dict


def interp_linear(
    config: dict,
    years: np.ndarray,
    val_dict: dict,
    bounds_error=True,
    fill_value=np.nan,
) -> tuple[np.ndarray, dict]:
    """Interpolate linearly values from discrete years to time_range set in config

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
        array, dict: Time range over which interpolation takes place,
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


def adjust_inventories(config: dict, inv_dict: dict) -> dict:
    """Determine evolution_type: norm, scaling or no evolution,
    and adjust inventories depending on type,
    data variables to be adjusted: fuel, species emissions and distance

    Args:
        config (dict): Configuration dictionary from config
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of input inventories

    Raises:
        ValueError: if no valid evolution_type in evolution file

    Returns:
        dict: normalized/scaled/unmodified dictionary of xarray Datasets,
            keys are years of input inventories
    """
    evolution_type = get_evolution_type(config)
    if evolution_type == "norm":
        out_dict = norm_inventories(config, inv_dict)
    elif evolution_type == "scaling":
        out_dict = scale_inventories(config, inv_dict)
    elif evolution_type is False:
        out_dict = inv_dict
    else:
        raise ValueError(
            "Invalid evolution_type "
            "evolution_type can be either 'scaling', 'norm' or False"
        )
    return out_dict


def apply_evolution(
    config: dict,
    val_dict: dict,
    inv_dict: dict,
    inventories_adjusted: bool = False,
) -> tuple[np.ndarray, dict]:
    """Determine evolution_type, apply normalization, scaling or no evolution

    Args:
        config (dict): Configuration dictionary from config
        val_dict (dict): Dictionary with time series numpy arrays, keys are species
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of input inventories
        inventories_adjusted (bool): True if inventories have been adjusted beforehand
            through adjust_inventories(config, inv_dict), defaults to False
    Raises:
        ValueError: if no valid evolution_type in evolution file

    Returns:
        array, dict: time_range and normalized/scaled/unmodified dictionary,
            np.ndarray, {spec: np.ndarray}
    """
    evolution_type = get_evolution_type(config)
    if evolution_type == "scaling":
        time_range, out_dict = apply_scaling(
            config, val_dict, inv_dict, inventories_adjusted
        )
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


def apply_scaling(
    config: dict, val_dict: dict, inv_dict: dict, inventories_adjusted: bool
) -> tuple[np.ndarray, dict]:
    """Apply scaling on dictionary of time series,
    scaling factors are from evolution file,
    time series and scaling factors are interpolated on time_range
    before multiplication

    Args:
        config (dict): Configuration dictionary from config
        val_dict (dict): Dictionary with time series numpy arrays, keys are species
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of input inventories
        inventories_adjusted (bool): True if inventories have been adjusted beforehand
            through adjust_inventories(config, inv_dict), False otherwise

    Returns:
        array, dict: time_range and scaled dictionary
    """
    # Get inventory years
    inv_years = np.array(list(inv_dict.keys()))
    
    # Interpolate scaling factors linearly on time_range
    time_range, evo_interp_dict = interp_evolution(config)
    
    # If inventories have been adjusted beforehand, normalize scaling factors to inv_years
    if inventories_adjusted:
        # Filter evo_interp_dict to inv_years
        evo_filtered_dict = filter_to_inv_years(
            inv_years, time_range, evo_interp_dict
        )
        # Interpolate linearly evo_filtered_dict
        _time_range, evo_filtered_interp_dict = interp_linear(
            config, inv_years, evo_filtered_dict
        )
        # Normalize evolution scaling factors to evo_filtered_interp_dict
        evo_norm_scaling_arr = np.divide(
            evo_interp_dict["scaling"], evo_filtered_interp_dict["scaling"]
        )
        evo_interp_dict = {"scaling": evo_norm_scaling_arr}
    
    # Interpolate time series data linearly on time_range
    _time_range, interp_dict = interp_linear(config, inv_years, val_dict)
    
    # Get species order for proper scaling
    time_dir = config["time"]["dir"]
    file_name = config["time"]["file"]
    file_path = time_dir + file_name
    evolution = xr.load_dataset(file_path)
    if "species" in evolution.coords:
        species_order = evolution.species.values.tolist()
    else:
        # Default order based on DEPA file
        species_order = ["fuel", "CO2", "H2O", "NOx", "distance"]
    
    # Multiply time series data by correct scaling factors
    out_dict = {}
    for spec, series_arr in interp_dict.items():
        # Find the correct scaling factor index for this species
        if spec in species_order:
            species_idx = species_order.index(spec)
            # Extract the scaling factors for this specific species
            scaling_factors = evo_interp_dict["scaling"][:, species_idx]
        else:
            # If species not found, use fuel scaling as default
            scaling_factors = evo_interp_dict["scaling"][:, species_order.index("fuel")]
        
        # Multiply element-wise
        out_dict[spec] = series_arr * scaling_factors
    
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
    # TODO This step might be redundant if normalization is applied beforehand
    # on input inventories. In this case, ei_inv_dict valures are exactly
    # the values in evo_interp_dict for inventory years.
    _inv_years, _inv_sum_dict, ei_inv_dict = calc_inv_quantities(
        config, inv_dict
    )
    # Filter emission indices dictionary to those species specified in time evolution
    ei_inv_dict = filter_dict_to_evo_keys(config, ei_inv_dict)
    # Interpolate emission indices from inventories over time_range
    # {"fuel": np.ndarray, "EI_CO2": np.ndarray, ..}
    inv_years = np.array(list(inv_dict.keys()))
    # TODO Check bounds, extrapolate
    time_range, ei_inv_interp_dict = interp_linear(
        config,
        inv_years,
        ei_inv_dict,
        bounds_error=False,
        fill_value=None,
    )
    # Calculate normalization factors, over time_range
    norm_dict = calc_norm(evo_interp_dict, ei_inv_interp_dict)
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
        # adapted for multiplication of multi-dimensional arrays
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


def calc_inv_quantities(config, inv_dict):
    """Calculate inventory quantities: fuel sums, emission sums and emission indices,
    Sums and emission indices are only calculated from species included in config

    Args:
        config (dict): Configuration dictionary from config
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of inventories

    Returns:
        np.ndarry, dict, dict: Array of inventory years,
            Dictionary of arrays of summed inventory emissions, keys are species
            Dictionary of arrays of fuel sums and inventory emission indices,
            keys are data variable names of evolution file
    """
    #
    # Translation table from evolution keys to inventory keys
    evo_inv_table = KEY_TABLE
    # Invert translation table: inventory keys to evolution keys
    inv_evo_table = {v: k for k, v in evo_inv_table.items()}
    # Initialize output array and dictionaries
    # Inventory years array
    inv_years = []
    # Inventory sums, keys are species, values are arrays over years
    inv_sum_dict = {}
    # Emission indices, keys are species, values are arrays over years
    ei_inv_dict = {}
    # Initialize lists in inv_sum_dict from inventory species array defined in config
    # TODO If inventories have missing species for different years, this is an issue!
    # Check for missing species in read_netcdf.py -> open_inventories()
    spec_lst = ["fuel", *config["species"]["inv"]]
    for spec in spec_lst:
        inv_sum_dict[spec] = []
    # Iterate over inventories
    for year, inv in inv_dict.items():
        inv_years.append(year)
        for spec, data_arr in inv.items():
            # Skip coordinates
            if spec in ["lon", "lat", "plev", "ac"]:
                pass
            # Skip species not defined in config or not fuel
            elif spec not in spec_lst:
                pass
            else:
                spec_sum = data_arr.sum().values.item()
                inv_sum_dict[spec].append(spec_sum)
    # Convert lists into numpy arrays
    inv_years = np.array(inv_years)
    for spec, sum_arr in inv_sum_dict.items():
        inv_sum_dict[spec] = np.array(sum_arr)
    fuel_sum_arr = inv_sum_dict["fuel"]
    # Add array of inventory fuel sums to emission index dictionary
    ei_inv_dict["fuel"] = fuel_sum_arr
    # Calculate emission indices for each species
    for spec, sum_arr in inv_sum_dict.items():
        if spec != "fuel":
            # Calculate emission index for spec (array over inventory years)
            ei_arr = np.divide(sum_arr, fuel_sum_arr)
            # Get right evolution key from translation table
            evo_key = inv_evo_table[spec]
            # Add array of emission indices for spec to emission index dictionary
            ei_inv_dict[evo_key] = ei_arr
    return inv_years, inv_sum_dict, ei_inv_dict


def filter_dict_to_evo_keys(config: dict, inp_dict: dict) -> dict:
    """
    Filter input dictionary to items matching with keys from time evolution file

    This function loads the time evolution file as an xarray dataset, and iterates over its keys.
    An item from the input dictionary is transferred to the output dictionary
    if the corresponding input key matches with a key from the time evolution.

    Args:
        config (dict): Configuration dictionary
        inp_dict (dict): Input dictionary

    Returns:
        dict: Dictionary with filtered items

    Raises:│
        KeyError: If no matches are found between both sets of keys.
    """
    time_dir = config["time"]["dir"]
    file_name = config["time"]["file"]
    file_path = time_dir + file_name
    # Output dictionary
    out_dict = {}
    # Load the time evolution file as an xarray dataset
    evolution = xr.load_dataset(file_path)
    # Iterate over keys in evolution
    for evo_key in evolution.keys():
        # Check if evolution key is present in the input dictionary
        if evo_key in inp_dict.keys():
            # If it is, add its corresponding item to the output dictionary
            out_dict[evo_key] = inp_dict[evo_key]
    # If no matching data variables were found, raise a KeyError
    if not out_dict:
        raise KeyError(
            "No matches found between keys in time evolution file and input dictionary!"
        )
    return out_dict


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


def filter_to_inv_years(
    inv_years, time_range: np.ndarray, interp_dict: dict
) -> dict:
    """Filters dictionary of interpolated arrays to items for inventory years only

    Args:
        inv_years (np.ndarray): Array of inventory years
        time_range (np.ndarray): time_range from config
        interp_dict (dict): Dictionary of arrays, interpolated over time_range,
            keys are e.g. evolution keys or species names

    Returns:
        dict: Dictionary of arrays, filtered to inventory years, same keys as interp_dict
    """
    mask = np.isin(time_range, inv_years)
    filtered_dict = {key: arr[mask] for key, arr in interp_dict.items()}
    return filtered_dict


def norm_inv(inv_dict: dict, norm_dict: dict) -> dict:
    """Multiply data variables in emission inventories by normalization factors,
    depending on available keys in norm_dict, data variables are multiplied by
    norm_fuel or (norm_fuel * norm_EI), with norm_EI = evo_EI / inv_EI

    Args:
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of inventories
        norm_dict (dict): Dictionary of normalization factors, keys are "fuel" and species

    Returns:
        dict: Dictionary of xarray Datasets (normalized emission inventories),
            keys are years of inventories
    """
    # Initialize output inventory dictionary
    out_inv_dict = {}
    # Array index corresponding to inventory years
    i = 0
    for year, inv in inv_dict.items():
        # Get global inventory attributes
        inv_attrs = inv.attrs
        # Initialize output inventory
        out_inv = xr.Dataset()
        # Create normalization sub dictionary for current inventory, with scalar values
        norm_sub_dict = {}
        for key, norm_arr in norm_dict.items():
            norm_sub_dict[key] = norm_arr[i]
        # Iterate over data variables in inventory
        for data_key, data_arr in inv.items():
            # Get attributes of data variable
            data_attrs = data_arr.attrs
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
            elif data_key in ["lon", "lat", "plev", "ac"]:
                pass
            # species not in norm_inv_dict: multiply with norm_fuel
            else:
                data_arr = data_arr * norm_sub_dict["fuel"]
            # Add data variable to output inventory
            data_arr.attrs = data_attrs
            out_inv = out_inv.merge({data_key: data_arr})
        # Set global inventory attributes
        out_inv.attrs = inv_attrs
        out_inv_dict[year] = out_inv
        i = i + 1
    return out_inv_dict


def norm_inventories(config: dict, inv_dict: dict) -> dict:
    """Applies normalization to a dictionary of emission inventories.
    This function first interpolates evolution data variables to time_range from config.
    It then gets inventory years, calculates inventory sums and emission indices
    (dictionaries with spec keys and arrays over inventory years).
    The emission indices dictionary is filtered to those species specified in time evolution.
    The arrays in evolution data are filtered to inventory years only.
    Next, the multipliers used for normalization are calculated.
    Finally, the normalization is performed by multiplying the inventory data variables
    with the normalization factors.

    Args:
        config (dict): Configuration dictionary
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of inventories

    Returns:
        dict: Dictionary of normalized emission inventories, keys are years of inventories
    """
    # Interpolate evolution data variables to time_range from config
    time_range, evo_interp_dict = interp_evolution(config)
    # Get inventory years, calculate inventory sums and emission indices
    # (dictionaries with spec keys and arrays over inventory years)
    inv_years, _inv_sum_dict, ei_inv_dict = calc_inv_quantities(
        config, inv_dict
    )
    # Filter emission indices dictionary to those species specified in time evolution
    ei_inv_dict = filter_dict_to_evo_keys(config, ei_inv_dict)
    # Filter arrays in evolution data to inventory years only
    evo_filtered_dict = filter_to_inv_years(
        inv_years, time_range, evo_interp_dict
    )
    # Calclulate multipliers used for normalization (dictionary with keys "fuel" and species)
    norm_dict = calc_norm(evo_filtered_dict, ei_inv_dict)
    # Perform actual normalization: Multiply inventory data variables by normalization factors
    out_inv_dict = norm_inv(inv_dict, norm_dict)
    return out_inv_dict


def scale_inv(inv_dict: dict, scale_dict: dict) -> dict:
    """Multiply data variables in emission inventories by scaling factors

    Args:
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of inventories
        scale_dict (dict): Dictionary of scaling factors {"scaling": np.ndarray}
            where scaling array has shape (n_years, n_species)

    Returns:
        dict: Dictionary of xarray Datasets (scaled emission inventories),
            keys are years of inventories
    """
    # Get array with scaling multipliers - shape (n_years, n_species)
    scale_arr = scale_dict["scaling"]
    
    # Get the species order from the scaling data (if available)
    # This should match the order in the scaling file
    if "species" in scale_dict:
        species_order = scale_dict["species"]
    else:
        # Default species order based on DEPA file label order
        species_order = ["fuel", "CO2", "H2O", "NOx", "distance"]
    
    # Initialize output inventory dictionary
    out_inv_dict = {}
    
    # Array index corresponding to inventory years
    i = 0
    for year, inv in inv_dict.items():
        # Get global inventory attributes
        inv_attrs = inv.attrs
        # Initialize output inventory
        out_inv = xr.Dataset()        
        # Iterate over data variables in inventory
        for data_key, data_arr in inv.items():
            # Get attributes of data variable
            data_attrs = data_arr.attrs            
            # lon, lat, plev: do NOT multiply
            if data_key in ["lon", "lat", "plev", "ac"]:
                pass
            # multiply fuel, species emissions, and distance by scaling multiplier
            else:
                # Find the correct scaling factor for this data variable
                if data_key in species_order:
                    species_idx = species_order.index(data_key)
                    scaling_value = scale_arr[i, species_idx]
                else:
                    # If species not found, use fuel scaling as default
                    scaling_value = scale_arr[i, species_order.index("fuel")]
                data_arr = data_arr * scaling_value
            
            # Add data variable to output inventory
            data_arr.attrs = data_attrs
            out_inv = out_inv.merge({data_key: data_arr})
        
        # Set global inventory attributes
        out_inv.attrs = inv_attrs
        out_inv_dict[year] = out_inv
        i = i + 1
    
    return out_inv_dict


def scale_inventories(config: dict, inv_dict: dict) -> dict:
    """Applies scaling to a dictionary of emission inventories.
    This function first interpolates evolution data variables to time_range from config.
    It then gets inventory years. The array in evolution data is filtered to inventory years only.
    Finally, the scaling is performed by multiplying the inventory data variables
    with the scaling factors in the filtered evolution data.

    Args:
        config (dict): Configuration dictionary
        inv_dict (dict): Dictionary of xarray Datasets, keys are years of inventories

    Returns:
        dict: Dictionary of scaled emission inventories, keys are years of inventories
    """
    # Interpolate evolution data variables to time_range from config
    time_range, evo_interp_dict = interp_evolution(config)
    # Get inventory years
    inv_years = []
    for year in inv_dict.keys():
        inv_years.append(year)
    # Filter scaling array in evolution data to inventory years only
    evo_filtered_dict = filter_to_inv_years(
        inv_years, time_range, evo_interp_dict
    )
    # Perform actual scaling: Multiply inventory data variables by scaling factors
    out_inv_dict = scale_inv(inv_dict, evo_filtered_dict)
    return out_inv_dict
