"""
Reads a config file, assigns values to variables and creates an output directory
"""

# TODO Add check function for valid inv_species / out_species combinations

import os
import shutil
import tomllib
import logging
from collections.abc import Iterable
import numpy as np

# CONSTANTS
# List types of all mandatory config parameters
CONFIG_TYPES = {
    "species": {"inv": Iterable, "out": Iterable},
    "inventories": {"dir": str, "files": Iterable},
    "output": {
        "full_run": bool,
        "dir": str,
        "name": str,
        "overwrite": bool,
        "concentrations": bool,
    },
    "time": {"range": Iterable},
    "background": {"CO2": {"file": str, "scenario": str}},
    "responses": {"CO2": {"response_grid": str}},
    "temperature": {"method": str, "CO2": {"lambda": float}},
    "metrics": {"types": Iterable, "t_0": Iterable, "H": Iterable},
}


def get_config(file_name):
    """load_config, check_config and create_output_dir

    Args:
        file_name (str): Name of config file

    Raises:
        ValueError: if configuration not valid

    Returns:
        dict: Dictionary containing key-value pairs
    """
    config = load_config(file_name)
    if check_config(config):
        create_output_dir(config)
    else:
        raise ValueError("Configuration is not valid.")
    return config


def load_config(file_name):
    """Loads config file in toml format.

    Args:
        file_name (str): Name of config file

    Returns:
        dict: Dictionary of key-value pairs
    """
    try:
        with open(file_name, "rb") as config_file:
            config = tomllib.load(config_file)
        return config
    except FileNotFoundError as exc:
        raise FileNotFoundError("No Config file found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise tomllib.TOMLDecodeError(
            "Config file is not a valid TOML document."
        ) from exc


def check_config(config):
    """Checks if configuration is complete and correct

    Args:
        config (dict): Dictionary of key-value pairs

    Raises:
        KeyError: if no response file defined

    Returns:
        bool: True if configuration correct, False otherwise
    """
    flag = check_config_types(config, CONFIG_TYPES)
    if flag:
        # Check response section
        _species_0d, species_2d, species_cont = classify_species(config)
        response_files = []
        for spec in species_2d:
            resp_flag = False
            resp_dir = config["responses"]["dir"]
            # At least one resp_type must be defined in config
            for resp_type in ["conc", "rf", "tau", "resp"]:
                try:
                    filename = (
                        resp_dir + config["responses"][spec][resp_type]["file"]
                    )
                    response_files.append(filename)
                    resp_flag = True
                except KeyError:
                    pass
            if not resp_flag:
                flag = False
                raise KeyError("No response file defined for", spec)
        # Check if files exist
        # TODO check evolution file
        emi_inv_files = []
        if "dir" in config["inventories"]:
            inv_dir = config["inventories"]["dir"]
        else:
            inv_dir = ""
        files_arr = config["inventories"]["files"]
        for inv_file in files_arr:
            emi_inv_files.append(inv_dir + inv_file)
        # Inventories and response files
        all_files = emi_inv_files + response_files
        for filename in all_files:
            if not os.path.exists(filename):
                msg = "File " + filename + " does not exist."
                logging.error(msg)
                flag = False
        # Climate metrics time settings
        if not check_metrics_time(config):
            flag = False
    if flag:
        logging.info("Configuration file checked.")
    else:
        logging.error("Configuration is not valid.")
    return flag


def check_config_types(config, types):
    """Checks config against table of types

    Args:
        config (dict): Dictionary of key-value pairs
        types (dict): Table of valid types for config entries

    Returns:
        bool: True if configuration types correct, False otherwise
    """
    flag = True
    # Default data types of configuration values
    for key, value in types.items():
        # For nested dict, call this function again
        if isinstance(value, dict):
            sub_config = config.get(key)
            sub_types = value
            if not check_config_types(sub_config, sub_types):
                flag = False
                break
        else:
            config_value = config.get(key)
            # Checks if required configuration variables are set
            if config_value is None:
                msg = key + " is not defined in configuration file."
                logging.error(msg)
                flag = False
                break
            # Checks if data types are as expected
            if not isinstance(config_value, types.get(key)):
                msg = key + " has wrong data type."
                logging.error(msg)
                flag = False
                break
    return flag


def create_output_dir(config):
    """Check for existing output directory, results file,
    overwrite and full_run settings. Create new output directory if needed.

    Args:
        config (dict): Dictionary of key-value pairs

    Raises:
        OSError: if no output directory is created or
            results file not existing with full_run = false
    """
    dir_path = config["output"]["dir"]
    output_name = config["output"]["name"]
    overwrite = config["output"]["overwrite"]
    full_run = config["output"]["full_run"]
    results_file = dir_path + output_name + ".nc"
    metrics_file = dir_path + output_name + "_metrics.nc"
    if not full_run and os.path.exists(results_file):
        msg = (
            "Compute climate metrics only, using results file " + results_file
        )
        logging.info(msg)
        if os.path.exists(metrics_file):
            msg = "Overwrite existing metrics file " + metrics_file
            logging.info(msg)
    elif not full_run and not os.path.exists(results_file):
        raise OSError(
            "Results file "
            + results_file
            + " does not exist."
            + " Repeat simulation with full_run = true"
        )
    elif overwrite and not os.path.isdir(dir_path):
        msg = "Create new output directory " + dir_path
        logging.info(msg)
        os.makedirs(dir_path)
    elif overwrite and os.path.isdir(dir_path):
        msg = "Overwrite existing output directory " + dir_path
        logging.info(msg)
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        raise OSError(
            "No output directory is created. Set output overwrite = true for "
            "overwriting existing directory or define a different directory path."
        )


def classify_species(config):
    """Classifies species into applied response modelling methods

    Args:
        config (dict): Configuration, dictionary of key-value pairs

    Raises:
        KeyError: if no valid response_grid in config
        KeyError: if no response defined for a spec

    Returns:
        list: Lists of strings (species)
    """
    species = config["species"]["out"]
    responses = config["responses"]
    species_0d = []
    species_2d = []
    species_cont = []
    for spec in species:
        exists = False
        for key, item in responses.items():
            if key == spec:
                exists = True
                if item["response_grid"] == "0D":
                    species_0d.append(spec)
                elif item["response_grid"] == "2D":
                    species_2d.append(spec)
                elif item["response_grid"] == "cont":
                    species_cont.append(spec)
                else:
                    raise KeyError(
                        "No valid response_grid in config for", spec
                    )
            else:
                pass
        if exists is False:
            raise KeyError("Responses not defined in config for", spec)
    return species_0d, species_2d, species_cont


def classify_response_types(config, species_arr):
    """
    Classifies species into categories based on their response types defined in the config

    Args:
        config (dict): Configuration, dictionary of key-value pairs
        species_arr (list): A list of strings representing the species

    Returns:
        tuple: A tuple of lists. list (species_rf) contains species with response type 'rf',
            i.e. a response file must be given comprising the response surface
            from emissions to RF,
            list (species_tau) contains species with response type 'tau',
            i.e. a response file must be given comprising the response surface
            from emissions to inverse species lifetime.

    Raises:
        KeyError: If no valid response type is defined in the configuration for a species.
    """
    species_rf = []
    species_tau = []
    for spec in species_arr:
        if "tau" in config["responses"][spec]:
            if spec != "CH4":
                raise KeyError(f'Response type "tau" not supported for {spec}')
            species_tau.append(spec)
        elif (
            "rf" in config["responses"][spec]
            and "file" in config["responses"][spec]["rf"]
        ):
            species_rf.append(spec)
        else:
            raise KeyError(
                "No valid response type defined in config for", spec
            )
    return species_rf, species_tau


def check_metrics_time(config: dict) -> bool:
    """
    Checks if metrics time settings are within the defined time range.

    Args:
        config (dict): Configuration from config file

    Returns:
        bool: True if metrics time settings are within the defined time range,
            False otherwise.

    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    delta_t = time_config[2]
    if delta_t != 1.0:
        msg = (
            "Time step in time range is NOT 1.0 years which could "
            "produce wrong metrics values."
        )
        logging.warning(msg)
    t_zero_arr = config["metrics"]["t_0"]
    horizon_arr = config["metrics"]["H"]
    #  Iterate through all metrics time ranges
    flag = True
    for t_zero, horizon in zip(t_zero_arr, horizon_arr):
        time_metrics = np.arange(t_zero, (t_zero + horizon), delta_t)
        for year_metrics in time_metrics:
            if year_metrics not in time_range:
                flag = False
        if not flag:
            msg = (
                "Metrics time settings with "
                + "t_0 = "
                + str(t_zero)
                + " and "
                + "H = "
                + str(horizon)
                + " are outside defined time range."
            )
            logging.error(msg)
        # Check if last year of time_metrics previous to last year in time range
        if time_metrics[-1] < time_range[-1]:
            msg = (
                "Last year in metrics time with "
                + "t_0 = "
                + str(t_zero)
                + " and "
                + "H = "
                + str(horizon)
                + " is earlier than last year in time range."
            )
            logging.warning(msg)
    return flag
