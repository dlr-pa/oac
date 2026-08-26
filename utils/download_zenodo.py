"""Download files from a Zenodo record using standard Python library."""

import argparse
import fnmatch
import json
import re
import urllib.request
from pathlib import Path


def download(record_or_doi, output_dir, file_glob="*"):
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
    match = re.search(r"(\d+)\D*$", str(record_or_doi))
    if not match:
        raise ValueError(f"Could not find a Zenodo record ID in {record_or_doi!r}")
    record_id = match.group(1)

    with urllib.request.urlopen(
        f"https://zenodo.org/api/records/{record_id}"
    ) as response:
        record = json.load(response)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_entry in record["files"]:
        filename = file_entry["key"]
        if not fnmatch.fnmatch(filename, file_glob):
            continue
        urllib.request.urlretrieve(
            file_entry["links"]["self"], output_dir / filename
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
    args = parser.parse_args()
    download(args.record_or_doi, args.output_dir, args.glob)


if __name__ == "__main__":
    main()
