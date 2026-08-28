"""
Resolves, downloads, and caches OpenAirClim's repository data (background
concentration scenarios and response-surface lookup tables).

That data is published independently of this package, in
https://github.com/dlr-pa/oac-data, with its own Zenodo-backed
versioning (see REPOSITORY_DATA_RECORD_DOI).
"""

import argparse
import logging
import os
from pathlib import Path

import platformdirs

from .utils.download_zenodo import (
    download_file,
    fetch_record_json,
    fetch_record_versions,
    verify_checksum,
)

#: DOI of any already-published record (any version) in the dlr-pa/oac-data
#: Zenodo deposition - NOT the "concept DOI" Zenodo shows in its "Cite all
#: versions" box, which isn't itself a queryable record via the API. Any real
#: version's DOI works as an anchor and returns every sibling version via
#: fetch_record_versions, so this never needs updating as new versions are
#: published - it is NOT tied to DEFAULT_REPOSITORY_DATA_VERSION below.
#: Synced by dlr-pa/oac-data's own "Sync to Zenodo" GitHub Actions workflow,
#: not Zenodo's built-in GitHub integration, because that only archives
#: whole-repo zips, which doesn't match the per-file record this module expects.
REPOSITORY_DATA_RECORD_DOI = "10.5281/zenodo.22146823"

#: Data repository release that this installed version of openairclim expects
#: by default. Deliberately independent of openairclim's own version number.
DEFAULT_REPOSITORY_DATA_VERSION = "0.1.0"

#: Env var to override the shared cache location entirely.
ENV_CACHE_DIR = "OPENAIRCLIM_DATA_DIR"

#: Filenames the repository dataset must contain.
REQUIRED_FILES = [
    "co2_bg.nc",
    "ch4_bg.nc",
    "n2o_bg.nc",
    "ch4_for_swv_calc.nc",
    "resp_RF.nc",
    "resp_RF_O3.nc",
    "resp_ch4.nc",
    "resp_cont.nc",
    "resp_cont_lf.nc",
]


def get_cache_dir(data_version: str | None = None) -> Path:
    """Resolve the shared cache directory for repository data.

    Does not create the directory or check its contents.

    Args:
        data_version (str, optional): Repository data version to namespace the
            cache directory by. Defaults to DEFAULT_REPOSITORY_DATA_VERSION.

    Returns:
        Path: The resolved cache directory. If the ENV_CACHE_DIR environment
            variable is set, it is returned as-is. Otherwise, a per-OS user
            data directory, namespaced by `data_version`.
    """
    env_override = os.environ.get(ENV_CACHE_DIR)
    if env_override:
        return Path(env_override)
    version = data_version or DEFAULT_REPOSITORY_DATA_VERSION
    return Path(
        platformdirs.user_data_dir(
            "openairclim", appauthor=False, version=version
        )
    )


def resolve_record_id(data_version: str | None = None) -> str:
    """Resolve the Zenodo record ID for a repository data version.

    Args:
        data_version (str, optional): Data repository release to look for (e.g.
            "0.1.0"). Defaults to DEFAULT_REPOSITORY_DATA_VERSION.

    Returns:
        str: The matching Zenodo record ID.

    Raises:
        ValueError: If no record with a matching ``metadata.version`` is
            found among REPOSITORY_DATA_RECORD_DOI's sibling versions (no
            fallback to "latest".)
    """
    version = data_version or DEFAULT_REPOSITORY_DATA_VERSION
    versions = fetch_record_versions(REPOSITORY_DATA_RECORD_DOI)
    for record in versions:
        if record.get("metadata", {}).get("version") == version:
            return str(record["id"])
    available = sorted(
        {v for r in versions if (v := r.get("metadata", {}).get("version"))}
    )
    raise ValueError(
        f"No repository data release tagged version {version!r} found "
        f"(available: {available}). Pass an explicit record/DOI to "
        "override, or update DEFAULT_REPOSITORY_DATA_VERSION."
    )


def check_data(cache_dir: str | Path) -> list[str]:
    """Return the names of any REQUIRED_FILES missing from cache_dir.

    Args:
        cache_dir (str or Path): Directory to check.

    Returns:
        list[str]: Filenames from REQUIRED_FILES not found in cache_dir.
            Empty if everything is present.
    """
    cache_dir = Path(cache_dir)
    return [f for f in REQUIRED_FILES if not (cache_dir / f).is_file()]


def is_data_present(
    cache_dir: str | Path,
    record: dict | None = None,
    verify_checksums: bool = False
) -> bool:
    """Check whether cache_dir already holds a complete, valid data set.

    Args:
        cache_dir (str or Path): Directory to check.
        record (dict, optional): A fetched Zenodo record's metadata, used
            for checksum verification. Required if verify_checksums=True.
        verify_checksums (bool, optional): If True, also verify each
            required file's checksum against `record`. Defaults to False
            (existence-only check).

    Returns:
        bool: True if every file in REQUIRED_FILES is present (and, if
            requested, checksum-valid).
    """
    cache_dir = Path(cache_dir)
    if check_data(cache_dir):  # if a file is not found in cache, return False
        return False

    # verify checksums
    if not verify_checksums:
        return True
    checksums = {
        f["key"]: f.get("checksum", "")
        for f in (record or {}).get("files", [])
    }
    return all(
        verify_checksum(cache_dir / f, checksums.get(f, ""))
        for f in REQUIRED_FILES
    )


def download_data(
    record_or_doi: str | None = None,
    output_dir=None,
    data_version: str | None = None,
    force: bool = False,
) -> Path:
    """Download OpenAirClim's repository data from Zenodo into a local cache.

    Args:
        record_or_doi (str, optional): Zenodo record ID or DOI to fetch,
            overriding the version-matched default.
        output_dir (str or Path, optional): Directory to download into.
            Defaults to get_cache_dir(data_version).
        data_version (str, optional): Data-repo release to fetch if
            record_or_doi isn't given. Defaults to
            DEFAULT_REPOSITORY_DATA_VERSION. Ignored if record_or_doi is given.
        force (bool, optional): Re-download and overwrite even if a file
            already exists and passes checksum verification. Defaults to
            False.

    Returns:
        Path: The directory the files were downloaded into.

    Raises:
        RuntimeError: If a downloaded file's checksum doesn't match the
            Zenodo record's metadata.
    """
    record_id = record_or_doi or resolve_record_id(data_version)
    record = fetch_record_json(record_id)

    target_dir = (
        Path(output_dir) if output_dir is not None
        else get_cache_dir(data_version)
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_entry in record["files"]:
        filename = file_entry["key"]
        checksum = file_entry.get("checksum", "")
        dest = target_dir / filename

        if not force and verify_checksum(dest, checksum):
            logging.info("%s already present and valid, skipping.", filename)
            continue

        logging.info("Downloading %s...", filename)
        download_file(file_entry["links"]["self"], dest)

        if checksum and not verify_checksum(dest, checksum):
            raise RuntimeError(
                f"Checksum mismatch for {filename} after download; the file "
                "may be corrupt, or the Zenodo record may have changed. "
                "Try again, or pass force=True."
            )

    return target_dir


def main():
    """Parse command-line arguments and download OpenAirClim's repository data."""
    parser = argparse.ArgumentParser(
        prog="oac-download-data",
        description="Download OpenAirClim's repository data (background "
        "concentrations and response surfaces) into the shared cache "
        "OpenAirClim uses by default.",
    )
    parser.add_argument(
        "-r",
        "--record",
        type=str,
        default=None,
        help="Zenodo record ID or DOI to fetch, overriding the "
        "version-matched default.",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default=None,
        help="Repository data version to fetch (default: "
        f"{DEFAULT_REPOSITORY_DATA_VERSION}). Ignored if --record is given.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to download into (default: the shared per-version "
        "cache directory; see openairclim.repository.get_cache_dir).",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Re-download and overwrite even if files already exist and "
        "pass checksum verification.",
    )
    args = parser.parse_args()

    # TODO openairclim.addon._premium has already configured the root logger
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    path = download_data(
        record_or_doi=args.record,
        output_dir=args.output_dir,
        data_version=args.version,
        force=args.force,
    )
    print(f"Repository data downloaded to {path}")


if __name__ == "__main__":
    main()
