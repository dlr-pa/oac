"""
Provides tests for module read_netcdf
"""

import os
import xarray as xr
import pytest
import openairclim as oac

# from unittest.mock import patch

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# CONSTANTS
REPO_PATH = "repository/"
INV_NAME = "test_inv.nc"
BG_NAME = "co2_bg.nc"


@pytest.fixture(name="open_nc", scope="class")
def fixture_open_nc():
    """Open netCDF file for multiple tests

    Returns:
        dict: Dictionary of xarrays
    """
    xr_dict = oac.open_netcdf((REPO_PATH + BG_NAME))
    return xr_dict


@pytest.mark.usefixtures("open_nc")
class TestOpenNetcdf:
    """Tests function open_netcdf(netcdf)"""

    def test_type(self, open_nc):
        """Open netcdf file and test if output is of type dictionary"""
        xr_dict = open_nc
        assert isinstance(xr_dict, dict)

    def test_key(self, open_nc):
        """Open netcdf file and test if keys of dictionary are input file basenames"""
        xr_dict = open_nc
        assert "co2_bg" in xr_dict

    def test_xarray(self, open_nc):
        """Open netcdf file and test if dictionary values are of type xarray.Dataset"""
        xr_dict = open_nc
        val = xr_dict["co2_bg"]
        assert isinstance(val, xr.Dataset)


@pytest.fixture(name="setup_arguments", scope="class")
def fixture_setup_arguments():
    """Setup config and inv_dict arguments for check_spec_attributes

    Returns:
        dict: Configuration dictionary from config
        dict: Dictionary of inventory xarrays, keys are years of input inventories
    """
    config = {"species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]}}
    file_path = REPO_PATH + INV_NAME
    inv = xr.load_dataset(file_path)
    key = inv.attrs["Inventory_Year"]
    inv_dict = {key: inv}
    return config, inv_dict


@pytest.mark.usefixtures("setup_arguments")
class TestCheckSpecAttributes:
    """Tests function check_spec_attributes(config, inv_dict)"""

    def test_correct_input(self, setup_arguments):
        "Correct input returns no Error"
        config, inv_dict = setup_arguments
        oac.check_spec_attributes(config, inv_dict)

    def test_no_attributes(self, setup_arguments):
        """Missing attributes in inventory for species raises KeyError"""
        config, inv_dict = setup_arguments
        inv_dict[2020]["CO2"].attrs = {}
        with pytest.raises(KeyError):
            oac.check_spec_attributes(config, inv_dict)

    def test_no_units(self, setup_arguments):
        """Missing units in inventory for species raises KeyError"""
        config, inv_dict = setup_arguments
        inv_dict[2020]["CO2"].attrs = {"long_name": "CO2"}
        with pytest.raises(KeyError):
            oac.check_spec_attributes(config, inv_dict)

    def test_incorrect_units(self, setup_arguments):
        """Incorrect units in inventory for species raises KeyError"""
        config, inv_dict = setup_arguments
        inv_dict[2020]["CO2"].attrs["units"] = "incorrect-unit"
        with pytest.raises(KeyError):
            oac.check_spec_attributes(config, inv_dict)
