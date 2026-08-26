"""
Provides tests for module construct_conc
"""

import os
import numpy as np
import xarray as xr
import pytest
from openairclim.core import construct_conc

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
        _inv_years, inv_sums = construct_conc.calc_inv_sums("CO2", inv_dict)
        assert isinstance(inv_sums, np.ndarray)

    def test_incorrect_input(self, load_inv):
        """Incorrect species name returns KeyError"""
        inv_dict = load_inv
        with pytest.raises(KeyError):
            construct_conc.calc_inv_sums("not-existing-species", inv_dict)

    def test_target_units_conversion(self, load_inv):
        """Sums are converted to a non-default target_units."""
        inv_dict = load_inv
        _inv_years, inv_sums_kg = construct_conc.calc_inv_sums("CO2", inv_dict)
        _inv_years, inv_sums_tg = construct_conc.calc_inv_sums(
            "CO2", inv_dict, target_units="Tg"
        )
        assert inv_sums_tg[0] == pytest.approx(inv_sums_kg[0] * 1.0e-9)

    def test_declared_units_are_read_per_year(self, load_inv):
        """Sums are converted from each inventory's own declared units,
        not assumed to already be in target_units."""
        inv_dict = load_inv
        year = next(iter(inv_dict))
        original_units = inv_dict[year]["CO2"].attrs["units"]
        try:
            inv_dict[year]["CO2"].attrs["units"] = "g"
            _inv_years, inv_sums_from_g = construct_conc.calc_inv_sums(
                "CO2", inv_dict, target_units="kg"
            )
            inv_dict[year]["CO2"].attrs["units"] = "kg"
            _inv_years, inv_sums_from_kg = construct_conc.calc_inv_sums(
                "CO2", inv_dict, target_units="kg"
            )
        finally:
            inv_dict[year]["CO2"].attrs["units"] = original_units
        assert inv_sums_from_g[0] == pytest.approx(inv_sums_from_kg[0] * 1.0e-3)


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
            construct_conc.check_inv_values(inv, year, spec)
