"""Helpers for creating and inspecting OpenAirClim config files."""

import os
from copy import deepcopy
from pathlib import Path

import xarray as xr


def resolve_dir(working_dir: str, dir_str: str) -> Path:
    """Resolve a (possibly relative) directory string against ``working_dir``.

    Args:
        working_dir (str): Base directory that relative paths resolve against.
        dir_str (str): Directory path, absolute or relative.

    Returns:
        Path: Resolved absolute path, with any "." / ".." segments collapsed.
        Doesn't require the path to exist.
    """
    p = Path(dir_str)
    if not p.is_absolute():
        p = Path(working_dir) / p
    return p.resolve()


def to_relative(working_dir: str, absolute_path: str) -> str:
    """Convert an absolute path to one relative to ``working_dir``.

    Uses ".." segments where necessary, so that directories outside
    ``working_dir`` still resolve correctly on another machine or OS, as long
    as their position relative to ``working_dir`` is preserved. Falls back to
    the absolute path unchanged only when no relative path can be computed
    (e.g. paths on different drives on Windows).

    Args:
        working_dir (str): Base directory to make the path relative to.
        absolute_path (str): Absolute path to convert.

    Returns:
        str: Relative path (forward-slash separated) if possible, otherwise the
        absolute path unchanged.
    """
    try:
        rel = os.path.relpath(absolute_path, working_dir)
    except ValueError:
        return Path(absolute_path).as_posix()
    return Path(rel).as_posix()


def list_nc_files(directory_path: Path) -> list[str]:
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


def list_nc_data_vars(filepath: Path | str) -> list[str]:
    """List data variable names in a NetCDF file.

    Used e.g. to discover the available scenario names inside a background
    concentration file (e.g. "SSP2-4.5"), which are stored as data variables.

    Args:
        filepath (Path or str): Path to the NetCDF file.

    Returns:
        list: Sorted list of data variable names. Empty list if the
        file doesn't exist or can't be opened.
    """
    try:
        with xr.open_dataset(filepath) as ds:
            return sorted(str(var) for var in ds.data_vars)
    except (FileNotFoundError, OSError, ValueError):
        return []


# -- TOML writer --------------------------------------------------------------


def _format_toml_value(value: object) -> str:
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
        value = value.as_posix()
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_toml_value(v) for v in value)
        return f"[{inner}]"
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


def _flatten_dict(d: dict, parent_key: str = "") -> list[tuple[str, object]]:
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


def to_toml_string(config: dict) -> str:
    """Format a configuration dictionary as TOML text.

    Each top-level key becomes a ``[section]`` header; nested dicts
    within a section are flattened to dotted keys (e.g.
    ``CO2.file = "..."``), matching OpenAirClim's existing config style.

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


def write_toml(config: dict, filepath: Path | str) -> None:
    """Write a configuration dictionary to a TOML file.

    Args:
        config (dict): Configuration dictionary to write.
        filepath (str or Path): Destination file path.
    """
    Path(filepath).write_text(to_toml_string(config), encoding="utf-8")


def prepare_for_save(config: dict, working_dir: str) -> dict:
    """Return a copy of config with directory paths made relative.

    Args:
        config (dict): Configuration dictionary (absolute dir paths).
        working_dir (str): Base directory the written TOML's paths should
            be relative to.

    Returns:
        dict: Deep copy of config with directory fields made relative
            to ``working_dir`` where possible.
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
