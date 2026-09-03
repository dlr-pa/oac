"""
Reads a config file, checks that it is complete and correct, and creates the
output directory.

Configuration checking runs in two layers, split across two modules:

- **Structural validation** (:mod:`openairclim.core.config_model`) — types,
  required/optional keys, valid option strings, and defaults, enforced by the
  pydantic :class:`~openairclim.core.config_model.Config` schema. Pure
  in-memory validation: nothing here touches the filesystem, and it doesn't
  need aircraft/inventory/response files to actually exist. Entered via
  :func:`~openairclim.core.config_model.validate_config`.
- **Everything that needs the filesystem or cross-references other data**
  (this module) — merging in an optional aircraft csv file, and checking
  that referenced files actually exist.

:func:`check_config` runs both layers in order:

1. :func:`~openairclim.core.config_model.validate_config` — validate
   structure/types, migrate deprecated keys (see
   :data:`~openairclim.core.config_model.ALIAS_MAP`), and fill in defaults.
   Inline ``[aircraft.<id>]`` entries are validated here too, deriving
   ``G_250``/``PMrel`` from sub-values where possible (see
   :class:`~openairclim.core.config_model.AircraftEntry`); if
   ``output.run_metrics`` is set, ``metrics.types``/``t_0``/``H`` are checked
   for completeness and consistency with ``time.range``.
2. :func:`_resolve_repository_dirs` - if ``background.dir``/``responses.dir``
   were left unset, fill them in with OpenAirClim's shared repository data
   cache directory (:func:`openairclim.repository.get_cache_dir`). Only
   resolves the *path*; doesn't check whether it actually contains the
   required files or download anything. Downloads only ever happen via an
   explicit ``oac-download-data`` call to prevent surprise network calls.
3. :func:`load_ac_data` - if ``aircraft.file`` is set, load that csv,
   validate each row (:class:`~openairclim.core.config_model.AircraftCsvRow`),
   and merge it into ``config["aircraft"]`` — raising if an aircraft
   identifier is defined both inline and in the csv.
4. :func:`_check_reserved_aircraft_ids` - reject aircraft identifiers that
   collide with core's own internal bookkeeping (``"TOTAL"``, ``"BASE_*"``).
5. :func:`_check_required_contrail_vars` - if ``"cont"`` is an output
   species, every aircraft identifier must end up with complete
   ``G_250``/``b``/``PMrel`` data, from either source.
6. :func:`_check_required_files` - every response, background, emission
   inventory and base inventory file the (now fully resolved) config
   references must actually exist on disk. If a missing file lives under
   the resolved repository data cache, the error points at
   ``oac-download-data``.

:func:`create_output_dir` is a separate step, not part of :func:`check_config`
— it's only run once a config has passed all of the above (see
:func:`get_config`), since it can create or wipe the output directory.
"""

import os
import shutil
import tomllib
import logging
from collections import defaultdict
from pathlib import Path
import pandas as pd
from pydantic import TypeAdapter, ValidationError
from .. import repository
from .config_model import validate_config, AircraftCsvRow, AIRCRAFT_DERIVATION_MAP

logger = logging.getLogger(__name__)

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
        logger.error("File %s does not exist.", file_path)
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
    conflicts = sorted(
        {ac for ac in df["ac"] if isinstance(config["aircraft"].get(ac), dict)}
    )
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
        logger.error(msg)
        raise ValueError(msg) from exc

    # add csv values to config
    for entry in entries:
        config["aircraft"][entry.ac] = entry.model_dump(
            include={"b", "PMrel", "G_250"}, exclude_none=True
        )

    return config


def _resolve_repository_dirs(config: dict) -> None:
    """Fill in background.dir/responses.dir from OpenAirClim's shared
    repository-data cache if left unset in the config (config_model.py's
    ``Path("")`` default, which pathlib normalises to ``Path(".")``).
    Mutates config in place.

    Does not download anything, and does not check whether the resolved
    directory actually contains the required files.

    Args:
        config (dict): Configuration dictionary, modified in-place.
    """
    cache_dir = None
    for section in ("background", "responses"):
        if Path(config[section].get("dir", "")) == Path("."):
            if cache_dir is None:
                cache_dir = str(repository.get_cache_dir())
            config[section]["dir"] = cache_dir


def _check_reserved_aircraft_ids(config: dict) -> None:
    """Ensure no reserved aircraft identifiers are used. "TOTAL" and
    "BASE_*" are reserved for core's own internal bookkeeping (see
    calc_cont.calc_contrails and read_netcdf.split_inventory_by_aircraft).

    Args:
        config (dict): Configuration dictionary

    Raises:
        ValueError: If a reserved aircraft identifier is used.
    """
    ac_types = config["aircraft"]["types"]
    if "TOTAL" in ac_types:
        raise ValueError(
            "Aircraft identifier TOTAL is reserved and cannot be defined "
            "in the config file."
        )
    if any(ac.startswith("BASE_") for ac in ac_types):
        raise ValueError(
            "Aircraft identifiers beginning with 'BASE_' are reserved and "
            "cannot be defined in the config file."
        )


def _check_required_contrail_vars(config: dict) -> None:
    """If contrails are being calculated, ensure every aircraft identifier
    has complete G_250/b/PMrel data. Both sources (inline config or csv)
    have already been merged into config["aircraft"][<ac_id>] by this point.

    Args:
        config (dict): Configuration dictionary

    Raises:
        ValueError: If contrail variables are missing for an aircraft
            identifier.
    """
    if "cont" not in config["species"]["out"]:
        return

    req_cont_vars = ["G_250", "b", "PMrel"]
    for ac in config["aircraft"]["types"]:
        ac_cfg = config["aircraft"].get(ac)
        if not isinstance(ac_cfg, dict):
            msg = f"Contrail variables missing for aircraft {ac}."
            logger.error(msg)
            raise ValueError(msg)
        for key in req_cont_vars:
            if key in ac_cfg:
                continue
            msg = f"Variable {key} missing for aircraft {ac}."
            sub_cols = AIRCRAFT_DERIVATION_MAP.get(key)
            if sub_cols:
                msg += f" Define it directly, or via its sub-values ({', '.join(sub_cols)})."
            logger.error(msg)
            raise ValueError(msg)


def _background_file_paths(config: dict) -> list:
    """Paths to the configured background concentration files.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list[Path]: One path per background species (CO2, CH4, N2O).
    """
    bg = config["background"]
    bg_dir = Path(bg["dir"])
    return [bg_dir / bg[species]["file"] for species in ("CO2", "CH4", "N2O")]


def _inventory_file_paths(config: dict) -> list:
    """Paths to the configured emission inventory files, including base
    inventories if rel_to_base is set.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list[Path]: One path per referenced inventory file.
    """
    inv = config["inventories"]
    inv_dir = Path(inv.get("dir", ""))
    paths = [inv_dir / f for f in inv["files"]]
    if inv.get("rel_to_base"):
        base = inv.get("base", {})
        base_dir = Path(base.get("dir", ""))
        paths.extend(base_dir / f for f in base.get("files", []))
    return paths


def _missing_files_message(missing: list) -> str:
    """Build the FileNotFoundError message for a list of missing paths.

    Args:
        missing (list[str]): Missing file paths, as strings.

    Returns:
        str: The error message, with a pointer to `oac-download-data`
            appended if any missing path falls under the shared
            repository-data cache directory.
    """
    msg = "Missing required files:\n" + "\n".join(missing)
    cache_dir = str(repository.get_cache_dir())
    if any(m.startswith(cache_dir) for m in missing):
        msg += (
            "\n\nSome of these files are expected in OpenAirClim's shared "
            "repository data cache. Run `oac-download-data` to fetch them, "
            "or set background.dir / responses.dir explicitly in your "
            "config to point at your own data."
        )
    return msg


def _check_required_files(config: dict) -> None:
    """Ensure every response, background, emission inventory and
    base inventory file the config references actually exists.

    Args:
        config (dict): Configuration dictionary

    Raises:
        KeyError: If no response file is defined for a 2D species.
        FileNotFoundError: If any referenced file is missing.
    """
    _, species_2d, _, _ = classify_species(config)
    resp_dir = Path(config["responses"]["dir"])
    paths: list[Path] = []

    # response files, for 2D species
    for spec in species_2d:
        spec_cfg = config["responses"].get(spec, {})
        found_any = False
        for resp_type in ("conc", "rf", "tau", "resp"):
            try:
                filename = spec_cfg[resp_type]["file"]
            except (KeyError, TypeError):
                continue
            paths.append(resp_dir / filename)
            found_any = True
        if not found_any:
            raise KeyError(f"No response file defined for {spec}")

    # background concentration files
    paths.extend(_background_file_paths(config))

    # emission inventory files, including base inventories if rel_to_base
    paths.extend(_inventory_file_paths(config))

    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        for m in missing:
            logger.error("File %s does not exist.", m)
        raise FileNotFoundError(_missing_files_message(missing))


def check_config(config):
    """Checks if configuration is complete and correct.

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Configuration dictionary
    """

    # validate structure/types, migrate deprecated keys, fill defaults;
    # inline [aircraft.<id>] entries validated/derived here too; metrics
    # (if enabled) checked for consistency with the simulation time range
    config = validate_config(config)

    # fill in background.dir/responses.dir from the shared repository-data
    # cache if left unset (does not download anything)
    _resolve_repository_dirs(config)

    # load aircraft data csv and check all aircraft identifiers and
    # required contrail variables
    config = load_ac_data(config)
    _check_reserved_aircraft_ids(config)
    _check_required_contrail_vars(config)

    # ensure all referenced files exist
    _check_required_files(config)

    logger.info("Configuration file checked.")
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
        logger.info(msg)
        if os.path.exists(metrics_file):
            msg = f"Overwrite existing metrics file {metrics_file}"
            logger.info(msg)
    elif not run_oac and not os.path.exists(results_file):
        raise OSError(
            f"Results file {results_file} does not exist."
            " Repeat simulation with run_oac = true"
        )
    elif overwrite and not os.path.isdir(dir_path):
        msg = f"Create new output directory {dir_path}"
        logger.info(msg)
        os.makedirs(dir_path)
    elif overwrite and os.path.isdir(dir_path):
        msg = f"Overwrite existing output directory {dir_path}"
        logger.info(msg)
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        raise OSError(
            "No output directory is created. Set output overwrite = true for "
            "overwriting existing directory or define a different directory path."
        )


def classify_species(config):
    """Classifies output species by response modelling method.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: tuple of lists of strings (species names)
    """
    species_0d = []
    species_2d = []
    species_cont = []
    species_sub = []
    for spec in config["species"]["out"]:
        if spec in SPECIES_SUB_ARR:
            species_sub.append(spec)
            continue
        grid = config["responses"][spec]["response_grid"]
        if grid == "0D":
            species_0d.append(spec)
        elif grid == "2D":
            species_2d.append(spec)
        elif grid == "cont":
            species_cont.append(spec)
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
