"""
Reads a config file, assigns values to variables and creates an output directory
"""

# TODO Add check function for valid inv_species / out_species combinations

import os
import shutil
import tomllib
import logging
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable
import numpy as np
import pandas as pd
from pydantic import TypeAdapter, ValidationError
from .config_model import validate_config, AircraftCsvRow, AIRCRAFT_DERIVATION_MAP

# CONSTANTS
# Species for which responses are calculated subsequently,
# i.e. dependent on computed response of other species
SPECIES_SUB_ARR = ["PMO", "SWV"]


def get_config(file_name):
    """load_config, check_config and create_output_dir

    Args:
        file_name (str): Name of config file

    Returns:
        dict: Configuration dictionary
    """
    config = load_config(file_name)
    config = check_config(config)
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


def _format_ac_csv_errors(exc, df):
    """Format a bulk AircraftCsvRow ValidationError, one line per affected
    aircraft.

    Args:
        exc (pydantic.ValidationError): Raised validating all csv rows at
            once (see load_ac_data).
        df (pandas.DataFrame): The source csv data, for "ac" lookups by
            row index.

    Returns:
        str: Multi-line message, one line per aircraft with a field error.
    """
    by_row = defaultdict(list)
    for err in exc.errors():
        idx, *field_path = err["loc"]
        field = ".".join(str(p) for p in field_path)
        by_row[idx].append(f"{field}: {err['msg']}" if field else err["msg"])
    lines = (
        f"  - '{df.iloc[idx]['ac']}': {'; '.join(msgs)}"
        for idx, msgs in sorted(by_row.items())
    )
    return "Invalid aircraft data:\n" + "\n".join(lines)


def load_ac_data(config: dict) -> dict:
    """Load and validate aircraft identifier parameters from a separate csv
    file. Parameters defined within the config file are checked by
    `config_model.validate_config`.

    Args:
        config (dict): Configuration dictionary

    Raises:
        FileNotFoundError: File does not exist
        KeyError: If the "ac" column does not exist
        ValueError: If a duplicate identifier is found within the csv
            file; if an aircraft's data is invalid or has G_250/PMrel
            that can't be derived from sub-values (see AircraftCsvRow in
            config_model.py); or if an aircraft is defined both inline
            in the config file and in the csv file

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

    # load file, check "ac" column is present
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    if "ac" not in df.columns:
        raise KeyError("Required column 'ac' not present in aircraft data file.")

    # check for duplicates
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

    # an aircraft defined both inline in the config file and in the csv
    # file is ambiguous — this is treated as a conflict the user must resolve
    conflicts = sorted({
        ac for ac in df["ac"]
        if isinstance(config["aircraft"].get(ac), dict)
    })
    if conflicts:
        raise ValueError(
            "Aircraft identifier(s) defined both inline in the config file "
            f"([aircraft.<id>]) and in the aircraft csv file: {conflicts}. "
            "Each aircraft must be defined in exactly one place — remove "
            "the [aircraft.<id>] section for these identifier(s) from the "
            "config file, or remove their row(s) from the csv file."
        )

    # validate all rows of the csv file
    try:
        entries = TypeAdapter(list[AircraftCsvRow]).validate_python(
            df.to_dict("records")
        )
    except ValidationError as exc:
        msg = _format_ac_csv_errors(exc, df)
        logging.error(msg)
        raise ValueError(msg) from exc

    # add csv values to config
    for entry in entries:
        config["aircraft"][entry.ac] = entry.model_dump(
            include={"b", "PMrel", "G_250"}, exclude_none=True
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
    reserved_acs = ["TOTAL"]
    for reserved in reserved_acs:
        if reserved in ac_types:
            raise ValueError(
                f"Aircraft identifier {reserved} is reserved and cannot be"
                "defined in the config file."
            )
    if any(ac.startswith("BASE_") for ac in ac_types):
        raise ValueError(
            "Aircraft identifiers beginning with 'BASE_' are reserved and "
            "cannot be defined in the config file."
        )

    # for the contrail module, test whether required parameters are present
    if "cont" in config["species"]["out"]:
        req_cont_vars = ["G_250", "b", "PMrel"]
        for ac in ac_types:
            ac_cfg = config["aircraft"].get(ac)
            if not isinstance(ac_cfg, dict):
                msg = f"Contrail variables missing for aircraft {ac}."
                logging.error(msg)
                raise ValueError(msg)
            for key in req_cont_vars:
                if key in ac_cfg:
                    continue
                msg = f"Variable {key} missing for aircraft {ac}."
                sub_cols = AIRCRAFT_DERIVATION_MAP.get(key)
                if sub_cols:
                    msg += f" Define it directly, or via its sub-values ({', '.join(sub_cols)})."
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
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def check_config(config):
    """Checks if configuration is complete and correct

    Args:
        config (dict): Configuration dictionary

    Raises:
        KeyError: if no response file defined

    Returns:
        dict: Configuration dictionary
    """

    # validate structure/types, migrate deprecated keys, fill defaults
    # inline [aircraft.<id>] entries are validated/derived here too
    config = validate_config(config)

    # load aircraft data csv and check all aircraft identifiers and required
    # contrail variables
    config = load_ac_data(config)
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
    dir_path = Path(config["output"]["dir"])
    output_name = config["output"]["name"]
    overwrite = config["output"]["overwrite"]
    run_oac = config["output"]["run_oac"]
    results_file = dir_path / f"{output_name}.nc"
    metrics_file = dir_path / f"{output_name}_metrics.nc"
    if not run_oac and os.path.exists(results_file):
        msg = f"Compute climate metrics only, using results file {results_file}"
        logging.info(msg)
        if os.path.exists(metrics_file):
            msg = f"Overwrite existing metrics file {metrics_file}"
            logging.info(msg)
    elif not run_oac and not os.path.exists(results_file):
        raise OSError(
            f"Results file {results_file} does not exist."
            " Repeat simulation with run_oac = true"
        )
    elif overwrite and not os.path.isdir(dir_path):
        msg = f"Create new output directory {dir_path}"
        logging.info(msg)
        os.makedirs(dir_path)
    elif overwrite and os.path.isdir(dir_path):
        msg = f"Overwrite existing output directory {dir_path}"
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
                    raise KeyError("No valid response_grid in config for", spec)
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
            raise KeyError("No valid response type defined in config for", spec)
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
    time_range = np.arange(time_config[0], time_config[1], time_config[2], dtype=int)
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
                "than last year in time range.",
                t_zero,
                horizon,
            )
