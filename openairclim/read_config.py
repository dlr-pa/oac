"""
Reads a config file, assigns values to variables and creates an output directory
"""

# TODO Add check function for valid inv_species / out_species combinations

import os
import shutil
import tomllib
import logging
from copy import deepcopy
from pathlib import Path
from collections.abc import Iterable
import numpy as np

_SENTINEL = object()

# CONSTANTS
# Template of config dictionary with types of MANDATORY input settings
CONFIG_TEMPLATE = {
    "species": {"inv": Iterable, "out": Iterable},
    "inventories": {"dir": str, "files": Iterable, "rel_to_base": bool},
    "output": {
        "run_oac": bool,
        "run_metrics": bool,
        "run_plots": bool,
        "dir": str,
        "name": str,
        "overwrite": bool,
        "concentrations": bool,
    },
    "time": {"range": Iterable},
    "background": {
        "dir": str,
        "CO2": {"file": str, "scenario": str},
        "CH4": {"file": str, "scenario": str},
        "N2O": {"file": str, "scenario": str},
    },
    "responses": {"dir": str},
    "temperature": {"method": str, "CO2": {"lambda": float}},
    "metrics": {"types": Iterable, "t_0": Iterable, "H": Iterable},
    "aircraft": {"types": Iterable},
}

# Default config settings to be added if not specified by user in config file
DEFAULT_CONFIG = {
    "responses": {
        "CO2": {
            "response_grid": "0D",
            "conc": {"method": "Sausen&Schumann"},
            "rf": {"method": "Etminan_2016", "attr": "proportional"},
        },
        "H2O": {"response_grid": "2D"},
        "O3": {"response_grid": "2D"},
        "CH4": {
            "response_grid": "2D",
            "rf": {"method": "Etminan_2016", "attr": "proportional"}
        },
        "cont": {"response_grid": "cont", "method": "Megill_2025"},
    },
    "temperature": {"method": "Boucher&Reddy"},
}

# Species for which responses are calculated subsequently,
# i.e. dependent on computed response of other species
SPECIES_SUB_ARR = ["PMO"]

# Alias map that maintains backwards compatibility when config parameters change
ALIAS_MAP = {
    "output.full_run": "output.run_oac",
}


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


def _apply_aliases(config: dict) -> dict:
    """Map deprecated variables to their new counterparts to maintain backwards
    compatibility.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Configuration dictionary, modified in place.
    """

    # loop over aliases
    for old, new in ALIAS_MAP.items():
        cur = config
        parts = old.split(".")
        for p in parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                break  # old path missing
            cur = cur[p]
        else:
            old_key = parts[-1]
            if old_key in cur:  # old value is present
                cur_new = config
                new_parts = new.split(".")
                for p in new_parts[:-1]:
                    cur_new = cur_new.setdefault(p, {})  # create new path
                new_key = new_parts[-1]

                if new_key not in cur_new:
                    cur_new[new_key] = cur.pop(old_key)  # old -> new value
                    logging.warning(
                        "Config key '%s' is deprecated; migrated to '%s'. "
                        "Please update your config file.",
                        old, new
                    )
                else:
                    logging.warning(
                        "Both deprecated key '%s' and new key '%s' exist; "
                        "keeping the new key. Please update your config file.",
                        old, new
                    )
    return config


def _gather_response_files(config: dict) -> list[Path]:
    """Collect required response files for all 2D species.

    Args:
        config (dict): Configuration dictionary

    Raises:
        KeyError: If no response file is found.

    Returns:
        list[Path]: List of paths to all response files.
    """
    _, species_2d, _, _ = classify_species(config)
    resp_dir = Path(config["responses"]["dir"])
    response_files: list[Path] = []

    # for 2D species, find response files
    for spec in species_2d:
        spec_cfg = config["responses"].get(spec, {})
        found_any = False
        for resp_type in ("conc", "rf", "tau", "resp"):
            try:
                filename = spec_cfg[resp_type]["file"]
            except (KeyError, TypeError):
                continue
            response_files.append(resp_dir / filename)
            found_any = True

        # if none are found, raise KeyError
        if not found_any:
            raise KeyError(f"No response file defined for {spec}")

    return response_files


def _gather_inventory_files(config: dict) -> list[Path]:
    """Collect all inventory files, including base inventories if rel_to_base
    is True.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list[Path]: List of paths to all inventory files.
    """

    inv = config["inventories"]
    files: list[Path] = []

    # get emission inventory paths
    inv_dir = Path(inv.get("dir", ""))
    for f in inv["files"]:
        files.append(inv_dir / f)

    # get base emission inventory paths
    if inv.get("rel_to_base"):
        base = inv.get("base", {})
        base_dir = Path(base.get("dir", ""))
        for f in base.get("files", []):
            files.append(base_dir / f)

    return files


def _aircraft_identifier_validation(config: dict) -> None:
    """Check aircraft identifiers and required contrail variables.

    Args:
        config (dict): Configuration dictionary

    Raises:
        ValueError: If a reserved aircraft identifier is used.
        ValueError: If contrail variables are missing for an aircraft identifier.
    """

    # ensure no reserved aircraft identifiers are present
    ac_types = list(config["aircraft"]["types"])
    reserved_acs = ("TOTAL")
    for reserved in reserved_acs:
        if reserved in ac_types:
            raise ValueError(
                f"Aircraft identifier {reserved} is reserved and cannot be"
                "defined in the config file."
            )

    # for the contrail module, test whether required parameters are present
    if "cont" in config["species"]["out"]:
        required = ("G_250", "eff_fac", "PMrel")
        for ac in ac_types:
            ac_cfg = config["aircraft"].get(ac)
            if not isinstance(ac_cfg, dict):
                msg = f"Contrail variables missing for aircraft {ac}."
                logging.error(msg)
                raise ValueError(msg)
            for key in required:
                if key not in ac_cfg:
                    msg = f"Variable {key} missing for aircraft {ac}."
                    logging.error(msg)
                    raise ValueError(msg)


def _assert_files_exist(paths: list[Path]) -> None:
    """Ensure that no files in the input list of paths are missing.

    Args:
        paths (list[Path]): List of paths to check.

    Raises:
        FileNotFoundError: If files are missing.
    """
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        for m in missing:
            logging.error("File %s does not exist.", m)
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )


def _validate_against_template(cfg: dict, tmpl: dict, path=""):
    """Recursively ensure every key in template (tmpl) exists in config (cfg)
    and has the right type. For dict-valued template entries, recurse into
    their children. For leaf template entries, the tempalte value is a type
    (e.g. str, Iterable, bool).count(value)

    Args:
        cfg (dict): Configuration dictionary
        tmpl (dict): Configuration template dictioanry
        path (str, optional): Path within recursive dict. Defaults to "".
    """

    # check that config is a dictionary
    if not isinstance(cfg, dict):
        raise TypeError(f"{path or '<root>'} must be a dict.")

    # recursively loop through keys and values
    for k, v in tmpl.items():
        here = f"{path}.{k}" if path else k

        # if v is a dictionary, then it is a (sub)section of the config
        if isinstance(v, dict):
            if k not in cfg:
                raise KeyError(f"Missing required section: {here}")
            if not isinstance(cfg[k], dict):
                raise TypeError(f"{here} must be a dict.")
            # recurse into (sub)section
            _validate_against_template(cfg[k], v, here)

        # otherwise, v is a value, so check its type
        else:
            val = cfg.get(k, _SENTINEL)
            if val is _SENTINEL:
                raise KeyError(f"Missing required setting: {here}")
            if not isinstance(val, v):
                raise TypeError(
                    f"{here} has incorrect type: {type(val).__name__}"
                    f"(expected {v.__name__})"
                )


def _merge_defaults_inplace(cfg: dict, defaults: dict):
    """Recursively add defaults into cfg (config) without overwriting existing
    user values. If a key is missing, copy the default into cfg. If a key
    exists, leave it as-is (even if the type differs).

    Args:
        cfg (dict): Configuration dictionary
        defaults (dict): Configuration dictionary with default values
    """

    for k, dv in defaults.items():
        # if k does not exist in cfg, copy defaults into cfg
        if k not in cfg:
            cfg[k] = deepcopy(dv)

        # if k does exist and is a value, do not overwrite
        # if k exists and is a dict, recurse
        else:
            cv = cfg[k]
            if isinstance(cv, dict) and isinstance(dv, dict):
                _merge_defaults_inplace(cv, dv)


def check_against_template(config, config_template, default_config):
    """Checks config dictionary against template:
    - check if config is complete,
    - add default settings if required,
    - check if values have correct data types.

    Args:
        config (dict): Configuration dictionary
        config_template (dict): Configuration template dictionary
        default_config (dict): Default configuration dictionary

    Returns:
        dict: Configuration dictionary, possibly with added default settings
    """

    # validate required keys and types from the template
    _validate_against_template(config, config_template)

    # add defaults non-destructively
    _merge_defaults_inplace(config, default_config)

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

    # apply aliases for backwards compatibility of config files
    config = _apply_aliases(config)

    # validate and fill defaults (no overwriting)
    config = check_against_template(config, config_template, default_config)

    # check aircraft identifiers and contrail variables
    _aircraft_identifier_validation(config)

    # collect files and ensure that they exist
    response_files = _gather_response_files(config)
    inventory_files = _gather_inventory_files(config)
    _assert_files_exist(response_files + inventory_files)

    # metrics time settings
    if config["output"]["run_metrics"]:
        _check_metrics(config)

    logging.info("Configuration file checked.")
    return config


def create_output_dir(config):
    """Check for existing output directory, results file,
    overwrite and run_oac settings. Create new output directory if needed.

    Args:
        config (dict): Configuration dictionary

    Raises:
        OSError: if no output directory is created or
            results file not existing with run_oac = false
    """
    dir_path = config["output"]["dir"]
    output_name = config["output"]["name"]
    overwrite = config["output"]["overwrite"]
    run_oac = config["output"]["run_oac"]
    results_file = dir_path + output_name + ".nc"
    metrics_file = dir_path + output_name + "_metrics.nc"
    if not run_oac and os.path.exists(results_file):
        msg = (
            "Compute climate metrics only, using results file " + results_file
        )
        logging.info(msg)
        if os.path.exists(metrics_file):
            msg = "Overwrite existing metrics file " + metrics_file
            logging.info(msg)
    elif not run_oac and not os.path.exists(results_file):
        raise OSError(
            "Results file "
            + results_file
            + " does not exist."
            + " Repeat simulation with run_oac = true"
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


def _check_metrics(config: dict) -> None:
    """
    Checks if metrics are properly defined.

    Args:
        config (dict): Configuration dictionary
    """
    # metric types, H and t_0 must not be empty
    req_keys = ("types", "H", "t_0")
    arrs = {}
    for key in req_keys:
        val = config["metrics"].get(key)
        if not isinstance(val, Iterable):
            raise ValueError(f"config['metrics']['{val}'] must be an Iterable.")
        val_lst = list(val)
        if not val_lst:
            raise ValueError(f"config['metrics']['{val}'] must not be empty.")
        arrs[key] = val_lst

    # get time information
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

    # Iterate through all metrics time ranges
    for t_zero, horizon in zip(arrs["t_0"], arrs["H"]):
        time_metrics = np.arange(t_zero, (t_zero + horizon), delta_t)
        for year_metrics in time_metrics:
            if year_metrics not in time_range:
                msg = (
                    f"Metrics time settings with t_0 = {t_zero} and H = "
                    f"{horizon} are outside of defined time range"
                )
                logging.error(msg)
                raise ValueError(msg)

        # Check if last year of time_metrics previous to last year in time range
        if time_metrics[-1] < time_range[-1]:
            logging.warning(
                "Last year in metrics time with t_0 = %s and H = %s is earlier "
                "than last year in time range.", t_zero, horizon
            )
