"""Download files from a Zenodo record using the requests library."""

import argparse
import fnmatch
import hashlib
import re
from pathlib import Path

import requests

#: Timeout (seconds) applied to every Zenodo API/download request.
REQUEST_TIMEOUT = 30

#: Chunk size (bytes) used when streaming file downloads.
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _extract_record_id(record_or_doi: str) -> str:
    """Extract the numeric Zenodo record ID from a record ID, DOI or URL.

    Args:
        record_or_doi (str): Zenodo record ID, or a DOI/URL containing one
            (e.g. "https://doi.org/10.5281/zenodo.11442322")

    Returns:
        str: The numeric record ID.

    Raises:
        ValueError: if no record ID can be found in `record_or_doi`
    """
    match = re.search(r"(\d+)\D*$", str(record_or_doi))
    if not match:
        raise ValueError(f"Could not find a Zenodo record ID in {record_or_doi!r}")
    return match.group(1)


def fetch_json(url: str) -> dict:
    """Fetch and parse JSON from a URL.

    Args:
        url (str): The URL to fetch.

    Returns:
        dict: The parsed JSON response.

    Raises:
        requests.HTTPError: if the request does not succeed.
    """
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_record_json(record_or_doi: str) -> dict:
    """Fetch a Zenodo record's metadata.

    Args:
        record_or_doi (str): Zenodo record ID, or a DOI/URL containing one.

    Returns:
        dict: The record's metadata, as returned by the Zenodo API.
    """
    record_id = _extract_record_id(record_or_doi)
    return fetch_json(f"https://zenodo.org/api/records/{record_id}")


def fetch_record_versions(record_or_doi: str) -> list[dict]:
    """Fetch every published version of a Zenodo concept record.

    Args:
        record_or_doi (str): Concept record ID, or a DOI/URL containing one.

    Returns:
        list[dict]: The version records (each a full Zenodo record dict).
    """
    record_id = _extract_record_id(record_or_doi)
    data = fetch_json(f"https://zenodo.org/api/records/{record_id}/versions")
    return data.get("hits", {}).get("hits", [])


def verify_checksum(path: str | Path, expected: str) -> bool:
    """Verify a local file's checksum against a Zenodo-format checksum string.

    Args:
        path (str or Path): Path to the local file.
        expected (str): Checksum in Zenodo's "<algorithm>:<hexdigest>"
            format, e.g. "md5:1234..." - see https://developers.zenodo.org/

    Returns:
        bool: True if the file exists and its checksum matches, False
            otherwise.
    """
    path = Path(path)
    if not path.is_file() or not expected:
        return False
    algorithm, _, digest = expected.partition(":")
    hasher = hashlib.new(algorithm or "md5")
    with open(path, "rb") as opened_file:
        for chunk in iter(lambda: opened_file.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest


def download_file(url: str, dest: str | Path) -> None:
    """Stream a single file from url to dest.

    Args:
        url (str): The URL to download.
        dest (str or Path): Local path to write the file to.

    Raises:
        requests.HTTPError: if the request does not succeed.
    """
    response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    with open(dest, "wb") as opened_file:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            opened_file.write(chunk)


def download(
        record_or_doi: str,
        output_dir: str | Path,
        file_glob: str = "*"
    ) -> None:
    """Download files from a Zenodo record matching file_glob into output_dir

    Args:
        record_or_doi (str): Zenodo record ID, or a DOI/URL containing one
            (e.g. "https://doi.org/10.5281/zenodo.11442322")
        output_dir (str or Path): Directory to download files into,
            created if it doesn't already exist
        file_glob (str): Glob pattern (as understood by fnmatch) used to
            filter which files in the record are downloaded. Defaults to
            "*", i.e. every file in the record.

    Raises:
        ValueError: if no record ID can be found in record_or_doi
    """
    record = fetch_record_json(record_or_doi)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_entry in record["files"]:
        filename = file_entry["key"]
        if not fnmatch.fnmatch(filename, file_glob):
            continue
        download_file(file_entry["links"]["self"], output_dir / filename)


def main():
    """Parse command-line arguments and download the matching files"""
    parser = argparse.ArgumentParser(
        description="Download files from a Zenodo record."
    )
    parser.add_argument(
        "record_or_doi", type=str, help="Zenodo record ID or DOI"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=".", help="Output directory"
    )
    parser.add_argument(
        "-g",
        "--glob",
        type=str,
        default="*",
        help="Glob pattern to filter which files are downloaded",
    )
    args = parser.parse_args()
    download(args.record_or_doi, args.output_dir, args.glob)


if __name__ == "__main__":
    main()
