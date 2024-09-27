"""Create files for testing purposes"""

import os
import create_test_data as ctd


# CONSTANTS
REPO_PATH = "../tests/repository/"
INV_NAME = "test_inv.nc"
RESP_NAME = "test_resp.nc"
TOML_NAME = "test.toml"
TOML_INVALID_NAME = "test_invalid.toml"


def create_test_directories(path_arr: list):
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


def create_test_config_files(repo_path, valid_name, invalid_name):
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
    file_path = repo_path + valid_name
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    with open(file_path, mode="w", encoding="utf-8") as valid_file:
        valid_file.write(
            '# Key-Value pair\
            \nkey = "value"'
        )
    file_path = repo_path + invalid_name
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    with open(file_path, mode="w", encoding="utf-8") as invalid_file:
        invalid_file.write(
            '# Invalid Toml syntax\
            \nkey ! "value"'
        )


def create_test_inv_nc(repo_path, inv_name):
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
    file_path = repo_path + inv_name
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    inv = ctd.create_test_inv()
    inv.to_netcdf(file_path)


def create_test_resp_nc(repo_path, resp_name):
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
    file_path = repo_path + resp_name
    if os.path.isfile(file_path):
        msg = "Overwrite existing file " + file_path
        print(msg)
    resp = ctd.create_test_rf_resp()
    resp.to_netcdf(file_path)


if __name__ == "__main__":
    create_test_directories([REPO_PATH])
    create_test_config_files(REPO_PATH, TOML_NAME, TOML_INVALID_NAME)
    create_test_inv_nc(REPO_PATH, INV_NAME)
    create_test_resp_nc(REPO_PATH, RESP_NAME)
