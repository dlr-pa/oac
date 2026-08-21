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


def _stringify_paths(obj):
    """Recursively convert `Path` values back to plain strings.

    `core.config_model.Config` types dir fields as `Path` so core can
    join them without requiring a trailing slash — but `Path("")`
    normalizes to `Path(".")`, which would otherwise show up as a
    resolved "." folder in the GUI (and get saved to TOML) for fields
    the user hasn't actually filled in yet. `edited_config` is meant to
    hold plain str/bool/list/dict values throughout, matching what the
    FilePicker/TextInput widgets read and write.

    Args:
        obj: A (possibly nested) config value — dict, list, Path, or
            already-plain value.

    Returns:
        The same structure, with every Path replaced by a string
        ("" for Path(".")/Path(""), str(path) otherwise).
    """
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    if isinstance(obj, Path):
        return "" if obj in (Path("."), Path("")) else str(obj)
    return obj


def blank_config():
    """Return a configuration skeleton satisfying the required fields of
    `core.config_model.Config`, with everything the model can default
    itself (responses.*, temperature.*, metrics, parametric.*,
    inventories.base/rel_to_base, ...) filled in by `validate_config`.

    Only holds fields Config has no default for.

    Returns:
        dict: Blank configuration dictionary with stringified paths.
    """
    from ..core.config_model import validate_config

    config = {
        "species": {"inv": [], "out": []},
        "inventories": {"dir": "", "files": [], "rel_to_base": False},
        "output": {
            "dir": "",
            "name": "new_config",
        },
        # we add a placeholder valid range here, so that validate_config()
        # doesn't reject the skeleton. It is swapped back for the "not set yet"
        # sentinel later once validation has completed.
        "time": {"range": [0, 1, 1]},
        "background": {
            "dir": "",
            "CO2": {"file": "", "scenario": ""},
            "CH4": {"file": "", "scenario": ""},
            "N2O": {"file": "", "scenario": ""},
        },
        "responses": {"dir": ""},
        "aircraft": {"types": ["DEFAULT"]},
    }

    validated = _stringify_paths(validate_config(config))
    validated["time"]["range"] = [0, 0, 1]
    return validated


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
    from pydantic import ValidationError

    from ..core.config_model import validate_config
    from ..core.read_config import load_config

    config_p = Path(config_path)
    if not config_p.is_absolute():
        config_p = Path(working_dir) / config_p

    try:
        config = load_config(str(config_p))
    except FileNotFoundError:
        return None, [f"Config file not found: `{config_p}`"]
    except Exception as e:  # pylint: disable=broad-exception-caught
        return None, [f"Failed to parse TOML: {e}"]

    try:
        config = validate_config(config)
    except ValidationError as e:
        return None, [f"Structural validation error: {e}"]

    return _stringify_paths(config), []


def parse_toml_text(text):
    """Parse and structurally validate a TOML config given as text.

    Mirrors :func:`parse_and_check_structure`, for config content that
    hasn't been saved to a file yet — e.g. hand-edited on the GUI's
    "Config (Expert)" text tab.

    Args:
        text (str): TOML config content.

    Returns:
        tuple: (config dict or None, list of error message strings).
    """
    import tomllib

    from pydantic import ValidationError

    from ..core.config_model import validate_config

    try:
        config = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        return None, [f"Failed to parse TOML: {e}"]

    try:
        config = validate_config(config)
    except ValidationError as e:
        return None, [f"Structural validation error: {e}"]

    return _stringify_paths(config), []


def run_full_validation(state):
    """Run the full validation pipeline against ``state.edited_config``.

    Args:
        state (AppState): Shared application state.

    Returns:
        tuple: (bool valid, str markdown status message).
    """
    from .tabs import config as config_tab

    if not state.working_dir:
        return False, "⚠️ Select a working directory first."
    if not state.edited_config:
        return False, "⚠️ No configuration to validate yet."
    if state.aircraft_csv_dirty:
        return False, (
            "⚠️ You have unsaved edits to the aircraft CSV — save it "
            "first (validation reads the file on disk)."
        )

    problems = config_tab.check_required_fields(state.edited_config)
    if problems:
        lines = ["⚠️ Fields missing or invalid\n"]
        lines += [f"- {status} **{title}**" for title, status in problems]
        return False, "\n".join(lines)

    try:
        check_full_config(state.working_dir, state.edited_config)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"❌ Configuration invalid\n\n{e}"

    return True, "✅ Configuration valid"


def check_full_config(working_dir, config):
    """Run the core's own full configuration check on a local config file.

    Reuses ``core.read_config.check_config`` as-is. Operates on a deep copy,
    since `check_config` mutates/returns its input (migrates deprecated keys
    and merges in defaults in place). The live config being edited in the GUI
    shouldn't change as a side effect of validating it.

    Args:
        working_dir (str): Project working directory — paths inside the
            config are resolved relative to this, so the check runs
            with the cwd temporarily switched there.
        config (dict): Structurally-seeded configuration dictionary
            (e.g. state.edited_config).

    Raises:
        Exception: Whatever `check_config` raises for an invalid config
            (pydantic.ValidationError, ValueError, KeyError,
            FileNotFoundError, ...).
    """
    from ..core.read_config import check_config

    old_cwd = os.getcwd()
    try:
        os.chdir(working_dir)
        check_config(deepcopy(config))
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


# ---------------------------------------------------------------------
# Directory path helpers (absolute during editing, relative when saved)
# ---------------------------------------------------------------------


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
    `working_dir`.

    Args:
        working_dir (str): Project working directory.
        absolute_path (str): Absolute path to convert.

    Returns:
        str: Relative path (forward-slash separated) if possible,
            otherwise the absolute path unchanged.
    """
    try:
        rel = os.path.relpath(absolute_path, working_dir)
    except ValueError:
        return absolute_path
    return absolute_path if rel.startswith("..") else Path(rel).as_posix()


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


# ---------------------------------------------------------------------
# TOML writer
# ---------------------------------------------------------------------


def _format_toml_value(value):
    """Format a Python value as a TOML literal.

    Args:
        value: Value to format (bool, int, float, str, Path, list, or tuple).

    Returns:
        str: TOML-formatted literal.

    Raises:
        TypeError: If the value type is not supported.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Path):
        value = str(value)
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


def to_toml_string(config):
    """Format a configuration dictionary as TOML text.

    Each top-level key becomes a `[section]` header; nested dicts
    within a section are flattened to dotted keys (e.g.
    `CO2.file = "..."`), matching OpenAirClim's existing config style.

    Args:
        config (dict): Configuration dictionary to format.

    Returns:
        str: TOML-formatted text.
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
    return "\n".join(lines)


def write_toml(config, filepath):
    """Write a configuration dictionary to a TOML file.

    Args:
        config (dict): Configuration dictionary to write.
        filepath (str or Path): Destination file path.
    """
    Path(filepath).write_text(to_toml_string(config), encoding="utf-8")


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
