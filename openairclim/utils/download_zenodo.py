"""Download files from a Zenodo record using the requests library."""

import time
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


def fetch_json(
    url: str,
    max_attempts: int = 3,
    backoff_seconds: float = 5.0,
) -> dict:
    """Fetch and parse JSON from a URL, retrying on transient errors.

    Args:
        url (str): The URL to fetch.
        max_attempts (int): Number of attempts before giving up. Must be
            at least 1.
        backoff_seconds (float): Base delay between retries; doubles each
            attempt (5s, 10s, 20s, ...).

    Returns:
        dict: The parsed JSON response.

    Raises:
        requests.HTTPError: if the request does not succeed.
        ValueError: if max_attempts is less than 1.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == max_attempts:
                raise
            wait = backoff_seconds * (2 ** (attempt - 1))
            print(f"Zenodo API request failed ({exc}); retrying in {wait:.0f}s "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(wait)

    raise AssertionError("unreachable")


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


def download_file(
    url: str,
    dest: str | Path,
    max_attempts: int = 3,
    backoff_seconds: float = 5.0,
) -> None:
    """Stream a single file from url to dest, retrying on transient errors.

    Args:
        url (str): The URL to download.
        dest (str or Path): Local path to write the file to.
        max_attempts (int): Number of attempts before giving up.
        backoff_seconds (float): Base delay between retries; doubles each
            attempt (5s, 10s, 20s, ...).

    Raises:
        requests.HTTPError: if the request does not succeed.
        requests.exceptions.RequestException: if all attempts fail.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            with open(dest, "wb") as opened_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    opened_file.write(chunk)
            return
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == max_attempts:
                raise
            wait = backoff_seconds * (2 ** (attempt - 1))
            print(f"Download failed ({exc}); retrying in {wait:.0f}s "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(wait)


def download(
    record_or_doi: str,
    output_dir: str | Path,
    file_glob: str = "*",
    force: bool = False,
    max_attempts: int = 3,
    backoff_seconds: float = 5.0,
) -> None:
    """Download files from a Zenodo record matching file_glob into output_dir,
    skipping any file that already exists and passes checksum verification.

    Raises:
        ValueError: if no record ID can be found in record_or_doi
        RuntimeError: if a downloaded file's checksum doesn't match the
            Zenodo record's metadata
    """
    record = fetch_record_json(record_or_doi)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_entry in record["files"]:
        filename = file_entry["key"]
        if not fnmatch.fnmatch(filename, file_glob):
            continue
        dest = output_dir / filename
        checksum = file_entry.get("checksum", "")

        if not force and verify_checksum(dest, checksum):
            print(f"{filename} already present and valid, skipping.")
            continue

        download_file(file_entry["links"]["self"], dest, max_attempts, backoff_seconds)

        if checksum and not verify_checksum(dest, checksum):
            raise RuntimeError(
                f"Checksum mismatch for {filename} after download; the file "
                "may be corrupt, or the Zenodo record may have changed. "
                "Try again, or pass force=True."
            )


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
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Number of attempts before giving up",
    )
    parser.add_argument(
        "--backoff-seconds", type=float, default=5.0,
        help="Base delay between retries",
    )
    args = parser.parse_args()
    download(
        args.record_or_doi,
        args.output_dir,
        args.glob,
        args.max_attempts,
        args.backoff_seconds
    )


if __name__ == "__main__":
    main()
