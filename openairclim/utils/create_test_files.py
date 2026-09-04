"""Create files for testing purposes"""

try:
    from .create_test_data import create_test_inv, create_test_rf_resp
except ImportError:
    from create_test_data import (  # type: ignore[import-not-found,no-redef]
        create_test_inv,
        create_test_rf_resp,
    )

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# CONSTANTS
INV_NAME = "test_inv.nc"
RESP_NAME = "test_resp.nc"
TOML_NAME = "test.toml"
TOML_INVALID_NAME = "test_invalid.toml"


def create_test_directories(path_arr: list) -> None:
    """
    Create new test directories if they do not exist.

    Args:
        path_arr (list): A list of paths to be created.

    Returns:
        None

    Raises:
        OSError: If the creation of a directory fails.
    """
    for path in path_arr:
        if not os.path.isdir(path):
            msg = f"Create new test directory {path}"
            print(msg)
            os.makedirs(path)


def create_test_config_files(
    repo_path: str, valid_name: str, invalid_name: str
) -> None:
    """
    Create two configuration files for testing.

    Args:
        repo_path (str): The path to the repository.
        valid_name (str): The name of the valid configuration file.
        invalid_name (str): The name of the invalid configuration file.

    Returns:
        None

    Raises:
        OSError: If the creation of a file fails.
    """
    file_path = os.path.join(repo_path, valid_name)
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    with open(file_path, mode="w", encoding="utf-8") as valid_file:
        valid_file.write('# Key-Value pair\
            \nkey = "value"')
    file_path = os.path.join(repo_path, invalid_name)
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    with open(file_path, mode="w", encoding="utf-8") as invalid_file:
        invalid_file.write('# Invalid Toml syntax\
            \nkey ! "value"')


def create_test_inv_nc(repo_path: str, inv_name: str) -> None:
    """
    Create an emission inventory netCDF file for testing.

    Args:
        repo_path (str): The path to the repository.
        inv_name (str): The name of the emission inventory file.

    Returns:
        None

    Raises:
        OSError: If the creation of a file fails.
    """
    file_path = os.path.join(repo_path, inv_name)
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    inv = create_test_inv()
    inv.to_netcdf(file_path)


def create_test_resp_nc(repo_path: str, resp_name: str) -> None:
    """
    Create a response netCDF file for testing.

    Args:
        repo_path (str): The path to the repository.
        resp_name (str): The name of the response file.

    Returns:
        None

    Raises:
        OSError: If the creation of a file fails.
    """
    file_path = os.path.join(repo_path, resp_name)
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    resp = create_test_rf_resp()
    resp.to_netcdf(file_path)


def main() -> None:
    """Parse command-line arguments and create test fixture files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create files needed to run the pytest suite (dev-only "
        "fixture generator).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the test fixture files into, "
        "e.g. tests/core/repository/ (run from the repo root).",
    )
    args = parser.parse_args()

    create_test_directories([args.output_dir])
    create_test_config_files(args.output_dir, TOML_NAME, TOML_INVALID_NAME)
    create_test_inv_nc(args.output_dir, INV_NAME)
    create_test_resp_nc(args.output_dir, RESP_NAME)


if __name__ == "__main__":
    main()
