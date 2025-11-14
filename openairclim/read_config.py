"""
Reads a config file, assigns values to variables and creates an output directory
"""

# TODO Add check function for valid inv_species / out_species combinations

import os
import shutil
import tomllib
import logging
from pathlib import Path
from typing import Any
from collections.abc import Iterable
from deepmerge import Merger
import numpy as np
import pandas as pd
from openairclim.calc_cont import calc_sac_slope

# CONSTANTS
# Template of config dictionary with types of MANDATORY input settings
CONFIG_TEMPLATE = {
    "species": {"inv": Iterable, "out": Iterable},
    "inventories": {"dir": str, "files": Iterable, "rel_to_base": bool},
    "output": {
        "full_run": bool,
        "dir": str,
        "name": str,
        "overwrite": bool,
        "concentrations": bool,
    },
    "time": {"range": Iterable},
    "background": {"CO2": {"file": str, "scenario": str}},
    "responses": {"CO2": {"response_grid": str, "rf": {"method": str}},
                  "cont": {"method": str}},
    "temperature": {"method": str, "CO2": {"lambda": float}},
    "metrics": {"types": Iterable, "t_0": Iterable, "H": Iterable},
    "aircraft": {"types": Iterable},
}

# Default config settings to be added if not specified by user in config file,
# default settings are ONLY added if corresponding type defined in CONFIG_TEMPLATE
DEFAULT_CONFIG = {"responses":
    {"CO2": {"rf": {"method": "Etminan_2016"}},
     "cont": {"method": "Megill_2025"}},
}

# Species for which responses are calculated subsequently,
# i.e. dependent on computed response of other species
SPECIES_SUB_ARR = ["PMO"]


def get_config(file_name):
    """load_config, check_config and create_output_dir

    Args:
        file_name (str): Name of config file

    Returns:
        dict: Configuration dictionary
    """
    config = load_config(file_name)
    config = check_config(
        config, config_template=CONFIG_TEMPLATE, default_config=DEFAULT_CONFIG
    )
    create_output_dir(config)
    return config


def load_config(file_name):
    """Loads config file in toml format.

    Args:
        file_name (str): Name of config file

    Returns:
        dict: Configuration dictionary
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


def load_ac_data(config):
    """Load aircraft identifier parameters from a separate csv file.

    Args:
        config (dict): Configuration dictionary

    Raises:
        FileNotFoundError: File does not exist
        KeyError: If a required column or value does not exist
        ValueError: If a duplicate identifier is found

    Returns:
        dict: Configuration dictionary modified in-place
    """

    # check file is not defined, then return
    ac_file = config["aircraft"].get("file")
    if ac_file is None or (isinstance(ac_file, str) and not ac_file.strip()):
        return config

    # check whether file exists
    file_path = Path(config["aircraft"]["dir"]) / config["aircraft"]["file"]
    if not file_path.exists():
        logging.error("File %s does not exist.", file_path)
        raise FileNotFoundError(f"File {file_path} does not exist.")

    # helper function that checks whether a column is present
    def _check_column_present(df, col):
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not present in aircraft identifier "
                "input file."
            )

    # load file, check whether columns "ac"
    df = pd.read_csv(file_path)
    _check_column_present(df, "ac")

    # check for NaNs or duplicates
    if df["ac"].isna().any():
        raise ValueError("NaN value found for aircraft identifier.")
    if df["ac"].duplicated().any():
        raise ValueError(
            "Duplicate values found in column 'ac': "
            f"{df[df['ac'].duplicated()]['ac'].unique()}"
        )

    # update aircraft types
    config["aircraft"]["types"].extend(df["ac"].tolist())
    config["aircraft"]["types"] = list(dict.fromkeys(config["aircraft"]["types"]))

    # if contrails aren't calculated, we don't need to add the
    # contrail-specific variables
    if "cont" not in config["species"]["out"]:
        return config

    # add contrail-specific variables
    # check "eff_fac" column is present and all values are valid
    _check_column_present(df, "eff_fac")
    if df["eff_fac"].isna().any():
        raise ValueError("Invalid 'eff_fac' value found.")

    # check whether G_250 and PMrel columns exist and if not initialise them
    req_cols = ["G_250", "PMrel"]
    for col in req_cols:
        if col not in df.columns:
            df[col] = np.nan

    # calculate missing G_250 values
    df_no_g = df[df["G_250"].isna()]
    if not df_no_g.empty:
        _check_column_present(df, "SAC_eq")
        for idx, row in df_no_g.iterrows():
            cols = ["SAC_eq", "Q_h", "eta", "eta_elec", "EIH2O", "R"]
            args = [row.get(c) for c in cols]
            try:
                g_250 = calc_sac_slope(250e2, *args)
                df.at[idx, "G_250"] = g_250
            # log which aircraft failed
            except ValueError:
                logging.error(
                    "Error in calculation of G_250 of aircraft identifer %s",
                    row["ac"]
                )
                raise

    # calculate missing PMrel values
    df_no_pmrel = df[df["PMrel"].isna()]
    if not df_no_pmrel.empty:
        _check_column_present(df, "PM")
        for idx, row in df_no_pmrel.iterrows():
            if pd.isna(row.get("PM")):
                msg = f"Missing 'PM' value for aircraft identifier {row['ac']}"
                logging.error(msg)
                raise KeyError(msg)
            df.at[idx, "PMrel"] = row.get("PM") / 1.5e15

    # add values to config (no overwriting)
    cols_to_add = ["G_250", "PMrel", "eff_fac"]
    for _, row in df.iterrows():
        ac = row["ac"]
        new_data = {col: round(row[col], 3) for col in cols_to_add}
        if ac not in config["aircraft"]:
            config["aircraft"][ac] = new_data
        else:
            for k, v in new_data.items():
                if k in config["aircraft"][ac]:
                    logging.warning(
                        "Aircraft '%s': value for '%s' already exists in config "
                        "(existing=%r, new=%r) — keeping existing value.",
                        ac, k, config['aircraft'][ac][k], v
                    )
                config["aircraft"][ac].setdefault(k, v)

    return config


def check_config(config, config_template, default_config):
    """Checks if configuration is complete and correct

    Args:
        config (dict): Configuration dictionary

    Raises:
        KeyError: if no response file defined

    Returns:
        dict: Configuration dictionary
    """
    # config = check_config(config, config_template, default_config)
    config = check_against_template(config, config_template, default_config)
    flag = True
    # Check response section
    _species_0d, species_2d, _species_cont, _species_sub = classify_species(
        config
    )
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

    # check base inventories (if rel_to_base is TRUE)
    emi_inv_files = []
    if "rel_to_base" in config["inventories"]:
        if config["inventories"]["rel_to_base"]:
            if "dir" in config["inventories"]["base"]:
                inv_dir = config["inventories"]["base"]["dir"]
            else:
                inv_dir = ""
            files_arr = config["inventories"]["base"]["files"]
            for inv_file in files_arr:
                emi_inv_files.append(inv_dir + inv_file)
    else:
        msg = "Parameter `rel_to_base` not defined."
        logging.error(msg)
        flag = False

    # check aircraft identifiers if contrails are to be calculated
    config = load_ac_data(config)
    ac_lst = config["aircraft"]["types"]
    if "TOTAL" in ac_lst:
        raise ValueError(
            "Aircraft identifier 'TOTAL' is reserved and cannot be defined "
            "in the config file."
        )
    if "cont" in config["species"]["out"]:
        req_cont_vars = ["G_250", "eff_fac", "PMrel"]
        for ac in ac_lst:
            if ac not in config["aircraft"]:
                msg = f"Contrail variables missing for aircraft {ac}."
                logging.error(msg)
                flag = False
                raise ValueError(msg)  # contrail module will fail
            for req_cont_var in req_cont_vars:
                if req_cont_var not in config["aircraft"][ac]:
                    msg = f"Variable {req_cont_var} missing for aircraft {ac}."
                    logging.error(msg)
                    flag = False
                    raise ValueError(msg)  # contrail module will fail

    # check inventories
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
    return config


def get_keys_values(v, key_arr, val_arr, prefix=""):
    """Gets list of (sub) keys and list of values for (nested) dictionary.
    Nested hierarchy is converted to a flattened structure.

    Args:
        v (dict): (Nested) dictionary
        key_arr (list): List of strings, each string comprises all sub keys
            associated to one value, sub keys are separated by blanks.
        val_arr (list): List of values (any type)
        prefix (str, optional): Defaults to ''.
    """
    if isinstance(v, dict):
        for k, v2 in v.items():
            # Append key to string chain
            p2 = f"{prefix}{k} "
            # Recursion if value v is dictionary
            get_keys_values(v2, key_arr, val_arr, p2)
    else:
        # print(prefix, v)
        key_arr.append(prefix)
        val_arr.append(v)


def check_against_template(config, config_template, default_config):
    """Checks config dictionary against template:
    check if config is complete,
    add default settings if required,
    check if values have correct data types.

    Args:
        config (dict): Configuration dictionary
        config_template (dict): Configuration template dictionary
        default_config (dict): Default configuration dictionary

    Raises:
        TypeError: if value in config has not expected data type

    Returns:
        dict: Configuration dictionary, possibly with added default settings
    """
    # Initialize key, value lists
    config_key_arr = []
    config_val_arr = []
    template_key_arr = []
    template_val_arr = []
    # Assign key, value lists with get_keys_values()
    get_keys_values(config, config_key_arr, config_val_arr)
    get_keys_values(config_template, template_key_arr, template_val_arr)
    # Template iterator index
    i = 0
    for key_str in template_key_arr:
        template_type = template_val_arr[i]
        # Check if all required settings defined in template are in config
        if key_str in config_key_arr:
            # Config iterator index
            config_index = config_key_arr.index(key_str)
            # Get value from config for corresponding key_str
            config_val = config_val_arr[config_index]
            # Check if config value has correct date type
            if not isinstance(config_val, template_type):
                msg = key_str + " has incorrect data type in config file"
                raise TypeError(msg)
        # If required setting not in config, try to add from default config
        else:
            msg = "Get default value for: " + key_str
            logging.info(msg)
            config = add_default_config(config, key_str, default_config)
        i = i + 1
    return config


def add_default_config(
    config: dict, key_str: str, default_config: dict
) -> dict:
    """Adds default settings to config if not defined by user,
    but defined in default_config

    Args:
        config (dict): Configuration dictionary
        key_str (str): String of sub keys associated to one value,
            sub keys are separated by blanks.
        default_config (dict): Default configuration dictionary

    Raises:
        KeyError: if required setting from key_str not included in default_config

    Returns:
        dict: Configuration dictionary, with added default setting
    """
    # Initialize key, value lists
    default_key_arr: list[str] = []
    default_val_arr: list[Any] = []
    # Assign key, value lists with get_keys_values()
    get_keys_values(default_config, default_key_arr, default_val_arr)
    # Check if configuration in default_config
    if key_str in default_key_arr:
        # default config iterator index
        default_index = default_key_arr.index(key_str)
        # Get value from default config for corresponding key_str
        default_val = default_val_arr[default_index]
        # Convert string chain into list of sub keys
        sub_key_arr = key_str.split()
        # Iterate (nested) dictionary sub keys from inside out
        added_dict = default_val
        for key in reversed(sub_key_arr):
            added_dict = {key: added_dict}
        # Merge added_dict with config
        my_merger = Merger([(dict, ["merge"])], ["override"], ["override"])
        config = my_merger.merge(config, added_dict)
    else:
        msg = "No valid configuration found for: " + key_str
        raise KeyError(msg)
    return config


def check_config_types(config, types):
    """Checks config against table of types
    TODO legacy code, remove this function?

    Args:
        config (dict): Configuration dictionary
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
        config (dict): Configuration dictionary

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
        config (dict): Configuration dictionary

    Raises:
        KeyError: if no valid response_grid in config
        KeyError: if no response defined for a spec

    Returns:
        tuple: tuple of lists of strings (species names)
    """
    species = config["species"]["out"]
    responses = config["responses"]
    species_0d = []
    species_2d = []
    species_cont = []
    species_sub = []
    for spec in species:
        # Classify species_sub, no response_grid required
        if spec in SPECIES_SUB_ARR:
            species_sub.append(spec)
            exists = True
        else:
            # Initialize exists flag
            exists = False
        # Check if response_grid is defined for spec and classify
        for key, item in responses.items():
            # Check if spec has config settings in response section
            # If True, classify spec according to response_grid
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
    return species_0d, species_2d, species_cont, species_sub


def classify_response_types(config, species_arr):
    """
    Classifies species into categories based on their response types defined in the config

    Args:
        config (dict): Configuration dictionary
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
        config (dict): Configuration dictionary

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
    # Iterate through all metrics time ranges
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
