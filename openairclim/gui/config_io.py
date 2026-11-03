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


# ======================================================================
# Per-card required-field status
#
# Each card on the Config tab gets a small "status" check that returns
# None (nothing missing) or "⚠️" (required data missing/incomplete) —
# shown as an icon appended to the card's title, and used below by
# run_full_validation to short-circuit before reaching check_full_config
# (which assumes a structurally-complete config and would otherwise raise
# on things like blank_config()'s "not set yet" time sentinel).
#
# Most cards just need a flat list of dotted-key paths that must be
# non-empty (REQUIRED_FIELDS_*) - add or remove a path there to change
# what a card requires. A few cards have conditional or multi-level
# logic (e.g. Background/Responses warn on any unset dropdown, Emission
# inventories' base folder is only required when "Relative to base" is on,
# Simulation period warns on end <= start rather than a blank check) and
# get their own _check_* function instead, following the same pattern.
# ======================================================================

REQUIRED_FIELDS_SPECIES = ["species.inv", "species.out"]
REQUIRED_FIELDS_TIME_EVOLUTION: list[str] = []
REQUIRED_FIELDS_TEMPERATURE: list[str] = []
REQUIRED_FIELDS_PARAMETRIC: list[str] = []
REQUIRED_FIELDS_OUTPUT = ["output.dir", "output.name"]


def _is_blank(value):
    """Return True if a config field counts as "not filled in".

    Args:
        value: Field value read from the edited config dict.

    Returns:
        bool: True for None, "", [], or {}.
    """
    return value in (None, "", [], {})


def _get_path(edited, path):
    """Look up a dotted key path in the config dict, e.g. "species.inv".

    Args:
        edited (dict): Working configuration dict.
        path (str): Dotted key path.

    Returns:
        The value at that path, or None if any part of the path is
        missing.
    """
    value = edited
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _required_fields_status(edited, paths):
    """Return "⚠️" if any dotted-path field in `paths` is blank.

    Args:
        edited (dict): Working configuration dict.
        paths (list): Dotted key paths that must be non-empty.

    Returns:
        str or None: "⚠️" if something's missing, else None.
    """
    for path in paths:
        if _is_blank(_get_path(edited, path)):
            return "⚠️"
    return None


def _check_species(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_SPECIES)


def _check_time(edited):
    """Warn if the simulation period is unset or empty (end <= start) —
    same condition the Config tab's time section checks live, so the
    card ⚠️ and the inline widget warning always agree."""
    start, end, _ = edited["time"]["range"]
    return "⚠️" if end <= start else None


def _check_time_evolution(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_TIME_EVOLUTION)


def _check_temperature(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_TEMPERATURE)


def _check_parametric(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_PARAMETRIC)


def _check_output(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_OUTPUT)


def _has_missing_files(dir_str, files):
    """Return True if any of `files` doesn't exist in `dir_str`.

    `dir_str` is expected to already be an absolute path — that's how
    it's stored once a folder is picked in the Config tab (see
    `tabs.config._build_dir_files_widgets`) — so this is a plain
    existence check, no resolution against working_dir needed.

    Args:
        dir_str (str): Absolute folder path, or "" if unset.
        files (list): Filenames expected to live in that folder.

    Returns:
        bool: True if dir_str is set and at least one file is missing.
    """
    if not dir_str:
        return False
    base = Path(dir_str)
    return any(not (base / f).exists() for f in files)


def _check_inventories(edited):
    """Folder + files are always required; base folder + files only
    when "Relative to base" is enabled. Also warns if any selected
    file (main, or base when relevant) can't be found on disk — the
    same "not found here" files flagged in the multiselect itself."""
    inv = edited["inventories"]
    paths = ["inventories.dir", "inventories.files"]
    if inv.get("rel_to_base"):
        paths = paths + ["inventories.base.dir", "inventories.base.files"]

    status = _required_fields_status(edited, paths)
    if status:
        return status

    if _has_missing_files(inv.get("dir", ""), inv.get("files", [])):
        return "⚠️"
    if inv.get("rel_to_base"):
        base = inv.get("base", {})
        if _has_missing_files(base.get("dir", ""), base.get("files", [])):
            return "⚠️"
    return None


def default_repository_dir() -> Path:
    """OpenAirClim's shared repository data cache directory. Used as the
    implicit default for the background and responses section directories when
    left blank. This is also what `core.read_config._resolve_repository_dirs`
    does.

    Returns:
        Path: The resolved shared cache directory (see
            `openairclim.repository.get_cache_dir`).
    """
    from .. import repository

    return repository.get_cache_dir()


def _check_background(edited):
    """Warn if any species' file or scenario is unset, or if the
    resolved folder (explicit, or OpenAirClim's shared repository data
    cache when left blank) is missing any of the referenced files."""
    bg = edited["background"]
    for species in ("CO2", "CH4", "N2O"):
        sub = bg.get(species, {})
        if _is_blank(sub.get("file")) or _is_blank(sub.get("scenario")):
            return "⚠️"

    dir_str = bg.get("dir") or str(default_repository_dir())
    files = [bg.get(species, {}).get("file") for species in ("CO2", "CH4", "N2O")]
    if _has_missing_files(dir_str, files):
        return "⚠️"
    return None


def _check_responses(edited):
    """Warn if any response file is unset, or if the resolved folder
    (explicit, or OpenAirClim's shared repository data cache when left
    blank) is missing any of the referenced files."""
    resp = edited["responses"]
    file_fields = [
        resp.get("H2O", {}).get("rf", {}).get("file"),
        resp.get("O3", {}).get("rf", {}).get("file"),
        resp.get("CH4", {}).get("tau", {}).get("file"),
        resp.get("cont", {}).get("resp", {}).get("file"),
    ]
    if any(_is_blank(v) for v in file_fields):
        return "⚠️"

    dir_str = resp.get("dir") or str(default_repository_dir())
    if _has_missing_files(dir_str, file_fields):
        return "⚠️"
    return None


def _check_metrics(edited):
    """Warn if the metrics fields are missing/incomplete.

    OpenAirClim needs at least one value in each of types/H/t_0 to run
    metrics. Warns if:
    - "Calculate climate metrics" (Output) is on and any of the three
      is unset, or
    - only one or two of the three are filled in, regardless of
      whether metrics calculation is enabled (partially-filled card).
    """
    metrics = edited["metrics"]
    filled = sum(
        0 if _is_blank(metrics.get(key)) else 1 for key in ("types", "H", "t_0")
    )
    run_metrics = bool(edited.get("output", {}).get("run_metrics"))

    if run_metrics and filled < 3:
        return "⚠️"
    if 0 < filled < 3:
        return "⚠️"
    return None


# Card title -> check function, in card order
CARD_CHECKS = {
    "Species": _check_species,
    "Simulation period": _check_time,
    "Time evolution (optional)": _check_time_evolution,
    "Emission inventories": _check_inventories,
    "Temperature": _check_temperature,
    "Metrics": _check_metrics,
    "Parametric": _check_parametric,
    "Output": _check_output,
    "Background": _check_background,
    "Responses": _check_responses,
}


def check_required_fields(edited_config):
    """Run every card's required-field check against a config dict.

    Lets other modules (the sidebar's Validate button, run_full_validation
    below) run the exact same checks shown as ⚠️ icons on the Config tab's
    card titles, without needing the cards to actually be rendered.

    Args:
        edited_config (dict): Working configuration dict.

    Returns:
        list: (card_title, status_icon) tuples for cards with a
            non-None status, in card order. Empty if nothing's missing.
    """
    problems = []
    for title, check_fn in CARD_CHECKS.items():
        status = check_fn(edited_config)
        if status:
            problems.append((title, status))
    return problems


# exact text of run_full_validation's success message
VALID_CONFIG_MESSAGE = "✅ Configuration valid"


def run_full_validation(state):
    """Run the full validation pipeline against ``state.edited_config``.

    Args:
        state (AppState): Shared application state.

    Returns:
        tuple: (bool valid, str markdown status message).
    """
    if not state.working_dir:
        return False, "⚠️ Select a working directory first."
    if not state.edited_config:
        return False, "⚠️ No configuration to validate yet."
    if state.aircraft_csv_dirty:
        return False, (
            "⚠️ You have unsaved edits to the aircraft CSV — save it "
            "first (validation reads the file on disk)."
        )

    problems = check_required_fields(state.edited_config)
    if problems:
        lines = ["⚠️ Fields missing or invalid\n"]
        lines += [f"- {status} **{title}**" for title, status in problems]
        return False, "\n".join(lines)

    try:
        check_full_config(state.working_dir, state.edited_config)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"❌ Configuration invalid\n\n{e}"

    return True, VALID_CONFIG_MESSAGE


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
