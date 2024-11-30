"""
Writes output: results netCDF file and diagnositics files
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import joblib


# CONSTANTS
RESULT_TYPE_DICT = {
    "emis": {
        "long_name": "Emission",
        "units": {"CO2": "Tg", "H2O": "Tg", "NOx": "Tg", "distance": "km"},
    },
    "conc": {
        "long_name": "Concentration",
        "units": {"CO2": "ppmv", "H2O": "?", "O3": "?", "CH4": "ppbv"},
    },
    "RF": {
        "long_name": "Radiative Forcing",
        "units": {
            "CO2": "W/m²",
            "H2O": "W/m²",
            "O3": "W/m²",
            "CH4": "W/m²",
            "PMO": "W/m²",
            "cont": "W/m²",
        },
    },
    "dT": {
        "long_name": "Temperature change",
        "units": {
            "CO2": "K",
            "H2O": "K",
            "O3": "K",
            "CH4": "K",
            "PMO": "K",
            "cont": "K",
        },
    },
    "ATR": {"long_name": "Average Temperature Response", "units": "K"},
    "AGWP": {
        "long_name": "Absolute Global Warming Potential",
        "units": "W m-2 year",
    },
    "AGTP": {
        "long_name": "Absolute Global Temperature Change Potential",
        "units": "K",
    },
}
# dtype of output data variables in xarray Datasets and netCDF
OUT_DTYPE = "float32"
# Cache settings
CHECKSUM_PATH = "../cache/weights/"
# CHECKSUM_FILENAME = "checksum_neighbours.csv"
CHECKSUM_FILENAME = "checksum_weights.csv"


def write_to_netcdf(config, val_arr_dict, result_type, mode="w"):
    """Convert dictionary with time series numpy arrays into xarray Dataset
    and write to netCDF file

    Args:
        config (dict): Configuration from config file
        val_arr_dict (dict): Dictionary of time series numpy arrays,
            keys are species names
        result_type (str): Type of computed result
        mode (str): Can be "w" for write or "a" for append

    Returns:
        xarray: xarray Dateset of results time series
    """
    # TODO "distance" is not really an emission, so being saved as "distance emission" doesn't really make sense

    output_dir = config["output"]["dir"]
    output_name = config["output"]["name"]
    output_filename = output_dir + output_name + ".nc"
    time_config = config["time"]["range"]
    time_arr = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    var_arr = []
    # data_vars=dict(time=(["time"], time_arr, {"long_name": "time", "units": "years"}))
    xr_time = xr.Dataset(
        data_vars={
            "time": (
                ["time"],
                time_arr,
                {"long_name": "time", "units": "years"},
            )
        }
    )
    var_arr.append(xr_time)
    descr = RESULT_TYPE_DICT[result_type]
    for spec, val_arr in val_arr_dict.items():
        var = xr.Dataset(
            data_vars={
                (result_type + "_" + spec): (
                    ["time"],
                    val_arr,
                    {
                        "long_name": spec + " " + descr["long_name"],
                        "units": descr["units"][spec],
                    },
                )
            },
            coords={"time": time_arr},
        )
        var_arr.append(var)
    output = xr.merge(var_arr)
    output.attrs = {"Title": output_name}
    output = output.astype(OUT_DTYPE)
    output.to_netcdf(output_filename, mode=mode)
    return output


def write_climate_metrics(
    config: dict, metrics_dict: dict, mode: str = "w"
) -> xr.Dataset:
    """
    Writes climate metrics to netCDF file.

    Args:
        config (dict): Configuration from config file
        metrics_dict (dict): Dictionary of climate metrics, keys are metric types
        mode (str, optional): Can be "w" for write or "a" for append. Defaults to "w".

    Returns:
        xr.Dataset: xarray Dataset of climate metrics
    """
    output_dir = config["output"]["dir"]
    output_name = config["output"]["name"]
    output_filename = output_dir + output_name + "_metrics.nc"
    metrics_type_arr = config["metrics"]["types"]
    t_zero_arr = config["metrics"]["t_0"]
    horizon_arr = config["metrics"]["H"]
    # Construct xarray Datasets
    var_arr = []
    # get species names from keys of first dictionary in metrics_dict
    # --> xarray dimension
    first_dict = next(iter(metrics_dict.values()))
    species_arr = np.array(list(first_dict.keys()))
    xr_species = xr.Dataset(
        data_vars={
            "species": (
                ["species"],
                species_arr,
                {"long_name": "species"},
            )
        }
    )
    var_arr.append(xr_species)
    # Iterate over all combinations of climate metrics
    i = 0
    for metrics_type in metrics_type_arr:
        for t_zero in t_zero_arr:
            t_zero_str = format(t_zero, ".0f")
            for horizon in horizon_arr:
                horizon_str = format(horizon, ".0f")
                # get climate metrics values --> xarray Variable
                #
                # element i in outer dictionary --> inner dict = {'spec': val, ..}
                inner_dict = list(metrics_dict.values())[i]
                val_arr = np.array(list(inner_dict.values()))
                descr = RESULT_TYPE_DICT[metrics_type]
                var = xr.Dataset(
                    data_vars={
                        (
                            metrics_type + "_" + horizon_str + "_" + t_zero_str
                        ): (
                            ["species"],
                            val_arr,
                            {
                                "long_name": descr["long_name"],
                                "units": descr["units"],
                                "t_0": t_zero_str,
                                "H": horizon_str,
                            },
                        )
                    },
                    coords={"species": species_arr},
                )
                var_arr.append(var)
                i = i + 1
    output = xr.merge(var_arr)
    output.attrs = {"Title": (output_name + " climate metrics")}
    output = output.astype(OUT_DTYPE)
    output.to_netcdf(output_filename, mode=mode)
    return output


def query_checksum_table(spec, resp, inv):
    """Look up in checksum table, if for the particular spec/resp/inv combination
    pre-calculated data exists

    Args:
        spec (str): Name of the species
        resp (xarray): Response xarray Dataset
        inv (xarray): Emission inventory xarray Dataset

    Returns:
        xarray/None, int: xarray Dataset with weight parameters,
            Number of rows in checksum table
    """
    checksum_path = CHECKSUM_PATH
    checksum_file = checksum_path + CHECKSUM_FILENAME
    # Open checksum_file, if file or/and parent folder(s) not existing, create those
    try:
        checksum_df = pd.read_csv(checksum_file)
    except IOError:
        msg = (
            "No checksum file "
            + checksum_file
            + " available, will create a new checksum file."
        )
        print(msg)
        checksum_df = pd.DataFrame(
            columns=["spec", "resp", "inv", "cache_file"]
        )
        try:
            os.makedirs(checksum_path)
        except FileExistsError:
            pass
        checksum_df.to_csv(checksum_file, index=False)
    # Query if argument checksum combination is in checksum_file
    resp_hash = joblib.hash(resp, hash_name="md5")
    inv_hash = joblib.hash(inv, hash_name="md5")
    found = False
    for _index, row in checksum_df.iterrows():
        if [row["spec"], row["resp"], row["inv"]] == [
            spec,
            resp_hash,
            inv_hash,
        ]:
            cache_file = row["cache_file"]
            found = True
            break
    if found:
        weights = xr.load_dataset(cache_file)
    else:
        weights = None
    # Return existing xarray Dataset or None and the number of rows in checksum_file
    return weights, checksum_df.shape[0]


def update_checksum_table(spec, resp, inv, cache_file):
    """Add a row to the existing checksum table with hashes of resp and inv,
        and path to cache_file

    Args:
        spec (str): Name of the species
        resp (xarray): Response xarray
        inv (xarray): Emission inventory
        cache_file (str): Path to cache file

    Returns:
        pd.DataFrame: Representation of the current checksum table
    """
    checksum_path = CHECKSUM_PATH
    checksum_file = checksum_path + CHECKSUM_FILENAME
    checksum_df = pd.read_csv(checksum_file)
    resp_hash = joblib.hash(resp, hash_name="md5")
    inv_hash = joblib.hash(inv, hash_name="md5")
    checksum_df.loc[len(checksum_df)] = [spec, resp_hash, inv_hash, cache_file]
    checksum_df.to_csv(checksum_file, index=False)
    return checksum_df


def write_concentrations(config, resp_dict, conc_dict):
    """Output of concentration changes,
    Convert dictionary of time series numpy arrays into dictionary of xarray Datasets,
    write to netCDF files, one per species

    Args:
        config (dict): Configuration from config file
        resp_dict (dict): Dictionary of response xarray Datasets, keys are species
        conc_dict (dict): Dictionary of time series numpy arrays (time, lat, plev),
            keys are species

    Returns:
        dict: Dictionary of concentration changes (time, lat, plev), keys are species
    """
    output_dir = config["output"]["dir"]
    output_name = config["output"]["name"]
    time_config = config["time"]["range"]
    time_arr = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    output_dict = {}
    for spec, resp in resp_dict.items():
        output_path = output_dir + "conc_" + spec + ".nc"
        conc_arr = conc_dict[spec]
        output = xr.Dataset(
            data_vars={
                "time": (
                    ["time"],
                    time_arr,
                    {"long_name": "time", "units": "years"},
                )
            },
            attrs={
                "Title": (output_name + " - concentration change"),
                "Species": spec,
            },
        )
        output["lat"] = resp.lat
        output["plev"] = resp.plev
        var = xr.DataArray(
            data=conc_arr,
            dims=["time", "lat", "plev"],
            attrs={
                "long_name": ("conc. change " + spec),
                "units": "mol/mol",
            },
        )
        var_name = "conc_" + spec
        output[var_name] = var
        output_dict[spec] = output
        output.to_netcdf(output_path, mode="w")
    return output_dict
