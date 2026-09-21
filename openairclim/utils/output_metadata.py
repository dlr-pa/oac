"""Functions relating to the OpenAirClim-specific output metadata."""

import json
from pathlib import Path
from typing import Any

import xarray as xr
from PIL import Image

from .config_files import prepare_for_save, write_toml

# keys written to output netCDF global attrs by `core.write_output.gen_sim_metadata`
_METADATA_KEYS = (
    "created",
    "user",
    "platform",
    "python_version",
    "oac_version",
    "oac_git_commit",
    "config_hash",
)

# file suffixes get_metadata scans for: netCDF results, and the PNG figures
# `core.write_output.fig_metadata` tags.
_METADATA_SUFFIXES = (".nc", ".png")


def _read_nc_metadata(path: Path) -> dict[str, Any] | None:
    """Read one netCDF file's OpenAirClim run metadata, if present.

    Args:
        path (Path): Path to a netCDF file.

    Returns:
        dict or None: Metadata dict (with ``config_json``, if present, parsed
        into a ``"config"`` entry), or None if the file can't be opened or
        carries no ``config_hash`` attribute (not an OpenAirClim output, or
        predates metadata support).
    """
    try:
        with xr.open_dataset(path) as ds:
            attrs = dict(ds.attrs)
    except (OSError, ValueError):
        return None

    if "config_hash" not in attrs:
        return None

    metadata = {key: attrs[key] for key in _METADATA_KEYS if key in attrs}
    if "config_json" in attrs:
        metadata["config"] = json.loads(attrs["config_json"])
    return metadata


def _read_png_metadata(path: Path) -> dict[str, Any] | None:
    """Read one PNG figure's OpenAirClim metadata, if present.

    Args:
        path (Path): Path to a PNG file.

    Returns:
        dict or None: A subset of ``config_hash``/``oac_version``/``created``
        (whichever text chunks are present), or None if the file can't be
        opened or carries none of them.
    """
    try:
        with Image.open(path) as im:
            info = im.info
    except (OSError, ValueError):
        return None

    metadata = {key: info[key] for key in _METADATA_KEYS if key in info}
    return metadata or None


# currently, only png and nc files are written by OpenAirClim
_READERS = {
    ".nc": _read_nc_metadata,
    ".png": _read_png_metadata,
}


def read_metadata(path: str | Path) -> dict[str, Any] | None:
    """Read OpenAirClim metadata from a single output or figure file.

    Args:
        path (str or Path): Path to a ``.nc`` or ``.png`` file.

    Returns:
        dict or None: Metadata found in the file, or None if its suffix
        isn't recognised, it can't be opened, or it carries no OpenAirClim
        metadata (not an OpenAirClim output, or predating metadata support).
    """
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        return None
    return reader(path)


def get_metadata(
    path: str | Path, full: bool = False, recursive: bool = False
) -> dict[str, dict[str, Any]] | dict[str, str]:
    """Return OpenAirClim metadata for a file or folder of output/figure files.

    Useful for identifying which simulation an output netCDF file or figure
    belongs to, e.g. when you have a ``config_hash`` but can't remember which
    run it came from.

    Args:
        path (str or Path): A single file, or a directory to scan for
            ``.nc`` and ``.png`` files.
        full (bool): Return each file's full metadata dict. Otherwise, return
            just the ``config_hash`` of files that carry one. Defaults to
            False.
        recursive (bool): Scan subdirectories of ``path`` too. Defaults to
            False.

    Returns:
        dict: If ``full``, maps filename -> metadata for every file found that
        carries OpenAirClim metadata. If not ``full``, maps filename ->
        ``config_hash``.
    """
    p = Path(path)
    if p.is_file():
        candidates = [p]
        keyed_by_relpath = False
    else:
        glob = p.rglob if recursive else p.glob
        candidates = sorted(
            f for suffix in _METADATA_SUFFIXES for f in glob(f"*{suffix}")
        )
        keyed_by_relpath = recursive

    results: dict[str, Any] = {}
    for candidate in candidates:
        metadata = read_metadata(candidate)
        if metadata is None:
            continue
        key = str(candidate.relative_to(p)) if keyed_by_relpath else candidate.name
        if full:
            results[key] = metadata
        elif "config_hash" in metadata:
            results[key] = metadata["config_hash"]
    return results


def _format_results(results: dict[str, dict[str, Any]] | dict[str, str]) -> str:
    """Format :func:`get_metadata` results as a human-readable table.

    Groups filenames by their ``config_hash``, so every file belonging to the
    same run is listed together under it.

    Args:
        results: The mapping returned by :func:`get_metadata`, either
            filename -> config_hash (``full=False``) or filename ->
            metadata dict (``full=True``).

    Returns:
        str: One block per ``config_hash`` (sorted), each listing its files
        and, if the results carried full metadata, that file's other
        metadata fields indented beneath it.
    """
    groups: dict[str, dict[str, dict[str, Any] | None]] = {}
    for filename, value in results.items():
        if isinstance(value, dict):
            file_hash = value.get("config_hash", "(no config_hash)")
            metadata = {k: v for k, v in value.items() if k != "config_hash"}
        else:
            file_hash = value
            metadata = None
        groups.setdefault(file_hash, {})[filename] = metadata

    lines = []
    for file_hash in sorted(groups):
        lines.append(f"{file_hash}:")
        for filename in sorted(groups[file_hash]):
            lines.append(f"  {filename}")
            for key, val in (groups[file_hash][filename] or {}).items():
                shown = json.dumps(val) if isinstance(val, dict | list) else val
                lines.append(f"    {key}: {shown}")
    return "\n".join(lines)


def main_get_metadata():
    """Parse command-line arguments and print/save OpenAirClim metadata."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Get OpenAirClim metadata for an OpenAirClim output file or folder."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a single file or a directory to scan.",
    )
    parser.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="Show full metadata for each file, instead of just its config_hash.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan subdirectories of path too.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Optional file to write the results to, instead of printing them.",
    )
    args = parser.parse_args()
    results = get_metadata(args.path, full=args.full, recursive=args.recursive)
    text = _format_results(results)
    if args.output_file:
        Path(args.output_file).write_text(text, encoding="utf-8")
        print(f"Wrote {len(results)} entries to {args.output_file}")
    else:
        print(text)


def write_config_from_netcdf(
    nc_path: str | Path,
    toml_path: str | Path,
    relative_to: str | Path | None = None,
) -> dict:
    """Recover the config embedded in an OpenAirClim output file as TOML.

    Useful for replicating a previous simulation if you have the results netCDF
    file but not the config TOML file. Or, for viewing the config of a previous
    simulation in a human-readable way.

    Args:
        nc_path (str or Path): A netCDF output file written by OpenAirClim,
            whose global attrs include a ``config_json`` dump written by
            :func:`~openairclim.core.write_output.gen_sim_metadata`.
        toml_path (str or Path): Destination path for the recovered config.
        relative_to (str or Path, optional): If given, the config's directory
            fields (inventories/background/responses/output/time dir, and
            inventories.base.dir) are rewritten relative to this directory,
            rather than being left as absolute files. See also
            :func:`~openairclim.utils.config_files.prepare_for_save`.

    Returns:
        dict: The configuration written to ``toml_path``.

    Raises:
        KeyError: If ``nc_path`` carries no embedded config metadata.
    """
    metadata = _read_nc_metadata(Path(nc_path))
    if metadata is None or "config" not in metadata:
        raise KeyError(
            f"{nc_path} has no embedded config metadata. It may predate "
            "OpenAirClim's metadata support."
        )
    config = metadata["config"]
    if relative_to is not None:
        config = prepare_for_save(config, str(relative_to))
    write_toml(config, toml_path)
    return config


def main_write_config_from_netcdf():
    """Parse command-line arguments and recover a config from an output file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Recover a config embedded in an OpenAirClim output file."
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to netCDF file to read"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Path to output TOML file",
    )
    parser.add_argument(
        "-r",
        "--relative-to",
        default=None,
        type=str,
        help="Directory for config paths to be relative to (optional)",
    )
    args = parser.parse_args()
    write_config_from_netcdf(args.input_file, args.output_file, args.relative_to)
