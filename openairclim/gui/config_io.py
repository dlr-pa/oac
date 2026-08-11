"""Configuration loading, validation, and saving logic.

Split into two validation stages so the sidebar can build an editable
form as soon as a config is structurally sound, without forcing every
referenced file to already exist:

- :func:`parse_and_check_structure` — parse TOML, apply aliases, check
  required keys/types, fill in defaults. A failure here means the
  config can't be safely edited (keys may be missing), so the caller
  should not build a form from it.
- :func:`check_full_config` — run the core's own full config check
  (`core.read_config.check_config`): structure, aircraft/contrail setup,
  and that every referenced inventory/response file actually exists.
  This function is run explicitly when a config file is loaded or when
  the user clicks the "validate" button.
"""

import os
from copy import deepcopy
from pathlib import Path


def blank_config():
    """Return a configuration skeleton seeded with `DEFAULT_CONFIG`.

    Satisfies the structural requirements of `CONFIG_TEMPLATE` so that an
    unfinished configuration can still be validated (and will report
    sensible errors, e.g. missing files) rather than crashing on missing
    keys. Where the core's own `DEFAULT_CONFIG` provides a default
    (e.g. `responses.CO2.rf.method`), it's merged in here too, so a
    brand-new config starts with the same defaults that a loaded file would
    get via check_against_template. A handful of fields that have no
    universal default in `DEFAULT_CONFIG` (efficacies, parametric ATR20 ratios)
    are seeded with the values from the example config instead.

    Returns:
        dict: Blank configuration dictionary.
    """
    from ..core.read_config import DEFAULT_CONFIG, _merge_defaults_inplace

    config = {
        "species": {"inv": [], "out": [], "nox": "NO"},
        "inventories": {
            "dir": "",
            "files": [],
            "rel_to_base": False,
            "base": {"dir": "", "files": []},
        },
        "output": {
            "run_oac": True,
            "run_metrics": True,
            "run_plots": True,
            "dir": "",
            "name": "new_config",
            "overwrite": True,
            "concentrations": False,
        },
        "time": {"dir": "", "range": [2020, 2021, 1]},
        "background": {
            "dir": "",
            "CO2": {"file": "", "scenario": ""},
            "CH4": {"file": "", "scenario": ""},
            "N2O": {"file": "", "scenario": ""},
        },
        "responses": {"dir": ""},
        "temperature": {
            "method": "Boucher&Reddy",
            "CO2": {"lambda": 0.73},
            "H2O": {"efficacy": 1.14},
            "O3": {"efficacy": 1.37},
            "PMO": {"efficacy": 1.37},
            "CH4": {"efficacy": 1.14},
            "cont": {"efficacy": 0.59},
            "SWV": {"efficacy": 1.0},
        },
        "metrics": {"types": [], "t_0": [], "H": []},
        "aircraft": {"types": ["DEFAULT"]},
        "parametric": {
            "enabled": False,
            "CO2": 1.0019972,
            "H2O": 0.25401992,
            "O3": 0.7016167,
            "CH4": 1.246515,
            "cont": 0.22705537,
        },
    }

    # Pulls in e.g. responses.{CO2,H2O,O3,CH4,cont} skeletons — only
    # adds keys that are missing, never overwrites what's set above.
    _merge_defaults_inplace(config, DEFAULT_CONFIG)

    return config


def parse_and_check_structure(working_dir, config_path):
    """Load a TOML config file and validate its structure (keys/types).

    Does not check that referenced files exist. This is done by
    :func:`check_files_exist`.

    Args:
        working_dir (str): Project working directory.
        config_path (str): Path to the config file, absolute or relative
            to working_dir.

    Returns:
        tuple: (config dict or None, list of error message strings).
    """
    from ..core.read_config import (
        CONFIG_TEMPLATE,
        DEFAULT_CONFIG,
        _apply_aliases,
        check_against_template,
        load_config,
    )

    config_p = Path(config_path)
    if not config_p.is_absolute():
        config_p = Path(working_dir) / config_p

    try:
        config = load_config(str(config_p))
    except FileNotFoundError:
        return None, [f"Config file not found: `{config_p}`"]
    except Exception as e:
        return None, [f"Failed to parse TOML: {e}"]

    try:
        config = _apply_aliases(config)
        config = check_against_template(config, CONFIG_TEMPLATE, DEFAULT_CONFIG)
    except (KeyError, TypeError) as e:
        return None, [f"Structural validation error: {e}"]

    return config, []


def check_full_config(working_dir, config):
    """Run the core's own full configuration check on a local config file.

    Reuses ``core.read_config.check_config`` as-is: applies aliases,
    validates against `CONFIG_TEMPLATE` (filling in defaults from
    `DEFAULT_CONFIG`), validates aircraft identifiers and contrail setup,
    checks that every referenced inventory/response file exists, and
    checks the metrics time settings if enabled.

    Operates on a deep copy, since `check_config` mutates/returns its
    input (applies aliases and merges in defaults in place). The live
    config being edited in the GUI shouldn't change as a side effect of
    validating it.

    Args:
        working_dir (str): Project working directory — paths inside the
            config are resolved relative to this, so the check runs
            with the cwd temporarily switched there.
        config (dict): Structurally-seeded configuration dictionary
            (e.g. state.edited_config).

    Raises:
        Exception: Whatever `check_config` raises for an invalid config
            (KeyError, TypeError, ValueError, FileNotFoundError, ...).
    """
    from ..core.read_config import CONFIG_TEMPLATE, DEFAULT_CONFIG, check_config

    old_cwd = os.getcwd()
    try:
        os.chdir(working_dir)
        check_config(deepcopy(config), CONFIG_TEMPLATE, DEFAULT_CONFIG)
    finally:
        os.chdir(old_cwd)


def run_config(working_dir, config_path):
    """Run OpenAirClim using a saved config file.

    All paths inside a saved config (inventory dir, response dir, etc.)
    are relative to `working_dir` — exactly like check_files_exist, this
    temporarily changes into `working_dir` so they resolve correctly,
    then restores the original directory regardless of outcome.

    Args:
        working_dir (str): Project working directory.
        config_path (str): Path to a saved config TOML file.
    """
    from ..core import run as oac_run

    old_cwd = os.getcwd()
    try:
        os.chdir(working_dir)
        oac_run(config_path)
    finally:
        os.chdir(old_cwd)


# ======================================================================
# Directory path helpers (absolute during editing, relative when saved)
# ======================================================================


def resolve_dir(working_dir, dir_str):
    """Resolve a (possibly relative) directory string against `working_dir`.

    Args:
        working_dir (str): Project working directory.
        dir_str (str): Directory path, absolute or relative.

    Returns:
        Path: Resolved absolute path, with any "." / ".." segments
            collapsed — doesn't require the path to exist.
    """
    p = Path(dir_str)
    if not p.is_absolute():
        p = Path(working_dir) / p
    return p.resolve()


def to_relative(working_dir, absolute_path):
    """Convert an absolute path to one relative to `working_dir`, if possible.

    Falls back to the absolute path unchanged if it lies outside
    `working_dir` (e.g. on a different drive on Windows).

    Since OpenAirClim's core code currently builds file paths by
    plain string concatenation — ``dir + filename`` — rather than via
    pathlib. That means every directory value in the config MUST end
    with a trailing slash, or the joined path is missing a separator
    (e.g. ``"input" + "x.nc"`` -> ``"inputx.nc"``). This is enforced
    here, the single place all directory values pass through before
    being written to TOML.

    Args:
        working_dir (str): Project working directory.
        absolute_path (str): Absolute path to convert.

    Returns:
        str: Relative path (forward-slash separated, trailing slash)
            if possible, otherwise the absolute path (also with a
            trailing slash).
    """
    try:
        rel = os.path.relpath(absolute_path, working_dir)
    except ValueError:
        result = absolute_path
    else:
        result = absolute_path if rel.startswith("..") else Path(rel).as_posix()

    if not result.endswith("/"):
        result += "/"
    return result


def list_nc_files(directory_path):
    """List NetCDF filenames in a directory.

    Args:
        directory_path (Path): Directory to scan.

    Returns:
        list: Sorted list of ``.nc`` filenames found, or an empty list
            if the directory does not exist.
    """
    if not directory_path.is_dir():
        return []
    return sorted(f.name for f in directory_path.glob("*.nc"))


def list_nc_data_vars(filepath):
    """List data variable names in a NetCDF file.

    Used to discover the available scenario names inside a background
    concentration file (e.g. "SSP2-4.5"), which are stored as data
    variables rather than declared anywhere in the config.

    Args:
        filepath (Path or str): Path to the NetCDF file.

    Returns:
        list: Sorted list of data variable names. Empty list if the
            file doesn't exist or can't be opened.
    """
    import xarray as xr

    try:
        with xr.open_dataset(filepath) as ds:
            return sorted(ds.data_vars)
    except (FileNotFoundError, OSError, ValueError):
        return []


# ======================================================================
# TOML writer
# ======================================================================


def _format_toml_value(value):
    """Format a Python value as a TOML literal.

    Args:
        value: Value to format (bool, int, float, str, list, or tuple).

    Returns:
        str: TOML-formatted literal.

    Raises:
        TypeError: If the value type is not supported.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_toml_value(v) for v in value)
        return f"[{inner}]"
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


def _flatten_dict(d, parent_key=""):
    """Flatten a nested dict into dotted-key / value pairs.

    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): Dotted key prefix (used during recursion).

    Returns:
        list: List of (dotted_key, value) tuples for all leaf values.
    """
    items = []
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, full_key))
        else:
            items.append((full_key, v))
    return items


def write_toml(config, filepath):
    """Write a configuration dictionary to a TOML file.

    Each top-level key becomes a ``[section]`` header; nested dicts
    within a section are flattened to dotted keys (e.g.
    ``CO2.file = "..."``), matching OpenAirClim's existing config style.

    Args:
        config (dict): Configuration dictionary to write.
        filepath (str or Path): Destination file path.
    """
    lines = []
    for section, content in config.items():
        lines.append(f"[{section}]")
        if isinstance(content, dict):
            for key, value in _flatten_dict(content):
                lines.append(f"{key} = {_format_toml_value(value)}")
        else:
            lines.append(f"{section} = {_format_toml_value(content)}")
        lines.append("")
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")


def prepare_for_save(config, working_dir):
    """Return a copy of config with directory paths made relative.

    During editing, directory fields are kept as absolute paths so that
    they remain correct regardless of when the working directory
    happens to be set. This converts them back to working-dir-relative
    form, once, right before writing a portable TOML file.

    Args:
        config (dict): Configuration dictionary (absolute dir paths).
        working_dir (str): Project working directory.

    Returns:
        dict: Deep copy of config with directory fields made relative
            to working_dir where possible.
    """
    prepared = deepcopy(config)
    dir_paths = [
        ("inventories", "dir"),
        ("output", "dir"),
        ("background", "dir"),
        ("responses", "dir"),
        ("time", "dir"),
    ]
    for section, key in dir_paths:
        val = prepared.get(section, {}).get(key)
        if val:
            prepared[section][key] = to_relative(working_dir, val)

    base_dir = prepared.get("inventories", {}).get("base", {}).get("dir")
    if base_dir:
        prepared["inventories"]["base"]["dir"] = to_relative(working_dir, base_dir)

    return prepared
