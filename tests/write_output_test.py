"""
Provides tests for module write_output
"""

import os
import shutil
import numpy as np
import xarray as xr
import pytest
import openairclim as oac

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# CONSTANTS
REPO_PATH = "repository/"
CACHE_PATH = "repository/cache/weights/"
INV_NAME = "test_inv.nc"
RESP_NAME = "test_resp.nc"
CACHE_NAME = "000.nc"


# TODO Instead of creating and removing directories, use patch or monkeypatch
#      fixtures for the simulation of os functionalities (test doubles)
@pytest.fixture(name="make_remove_dir", scope="class")
def fixture_make_remove_dir(request):
    """Arrange and Cleanup fixture, create an output directory for testing
        and remove it afterwards, setup and the directory name can be reused
        in several test functions of the same class.

    Args:
        request (_pytest.fixtures.FixtureRequest): pytest request parameter
            for injecting objects into test functions
    """
    dir_path = "results/"
    request.cls.dir_path = dir_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    yield
    shutil.rmtree(dir_path)


@pytest.fixture(name="setup_arguments", scope="class")
def fixture_setup_arguments():
    """Setup config and val_arr_dict arguments for write_to_netcdf

    Returns:
        dict, dict: configuration dictionary and
            dictionary of time series numpy array
    """
    config = {
        "output": {"dir": "results/", "name": "example", "overwrite": True},
        "time": {"range": [2020, 2022, 1]},
    }
    val_arr_dict = {"CO2": np.array([1.0, 2.0])}
    return config, val_arr_dict


@pytest.mark.usefixtures("make_remove_dir", "setup_arguments")
class TestWriteToNetcdf:
    """Tests function write_to_netcdf(config, val_arr_dict, result_type, mode)"""

    def test_correct_input(self, setup_arguments):
        """Correct input returns xarray Dataset time series"""
        config, val_arr_dict = setup_arguments
        output = oac.write_to_netcdf(config, val_arr_dict, "emis")
        assert isinstance(output, xr.Dataset)

    def test_incorrect_result_type(self, setup_arguments):
        """Incorrect result_type returns KeyError"""
        config, val_arr_dict = setup_arguments
        with pytest.raises(KeyError):
            oac.write_to_netcdf(
                config, val_arr_dict, "not-existing-result_type"
            )
