"""
Provides tests for module construct_conc
"""

import os
import numpy as np
import xarray as xr
import pytest
import openairclim as oac

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# CONSTANTS
REPO_PATH = "repository/"
INV_NAME = "test_inv.nc"


@pytest.fixture(name="load_inv", scope="class")
def fixture_load_inv():
    """Load example emission inventory and reuse xarray in multiple tests

    Returns:
        dict: Dictionary of xarray, key is inventory years
    """
    file_path = REPO_PATH + INV_NAME
    inv = xr.load_dataset(file_path)
    inv_dict = {2020: inv}
    return inv_dict


@pytest.mark.usefixtures("load_inv")
class TestCalcInvSums:
    """Tests function calc_inv_sums(spec, inv_dict)"""

    def test_correct_input(self, load_inv):
        """Correct species name and inventory inputs returns array of sums"""
        inv_dict = load_inv
        _inv_years, inv_sums = oac.calc_inv_sums("CO2", inv_dict)
        assert isinstance(inv_sums, np.ndarray)

    def test_incorrect_input(self, load_inv):
        """Incorrect species name returns KeyError"""
        inv_dict = load_inv
        with pytest.raises(KeyError):
            oac.calc_inv_sums("not-existing-species", inv_dict)


@pytest.mark.usefixtures("load_inv")
class TestCheckInvValues:
    """Tests function check_inv_values(inv, year, spec)"""

    def test_negative_emissions(self, load_inv):
        """Load dictionary of emission inventory with positive emissions"""
        inv_dict = load_inv
        year = 2020
        spec = "CO2"
        inv = inv_dict[year]
        inv_arr = inv[spec].values
        # Convert first element of CO2 inventory array into negative emission
        inv_arr[0] = -inv_arr[0]
        inv[spec].values = inv_arr
        with pytest.raises(ValueError):
            oac.check_inv_values(inv, year, spec)
