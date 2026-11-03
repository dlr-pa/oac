"""Provides tests for module interpolate_space."""

import os

import numpy as np
import pytest
import xarray as xr

from openairclim.core import interpolate_space as intsp

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# CONSTANTS
REPO_PATH = "repository/"
INV_NAME = "test_inv.nc"
RESP_NAME = "test_resp.nc"


@pytest.fixture(name="setup_arguments", scope="class")
def fixture_setup_arguments() -> tuple[str, xr.Dataset, xr.Dataset]:
    """Setup arguments for calc_weights(spec, resp, inv).

    Returns:
        str, xarray.Dataset, xarray.Dataset: species name, response, emission
            inventory
    """
    spec = "H2O"
    resp = xr.load_dataset(REPO_PATH + RESP_NAME)
    inv = xr.load_dataset(REPO_PATH + INV_NAME)
    return spec, resp, inv


@pytest.mark.usefixtures("setup_arguments")
class TestCalcWeights:
    """Tests function calc_weights(spec, resp, inv)."""

    def test_correct_input(self, setup_arguments):
        """Valid input returns :class:`xarray.Dataset` with non-empty weights."""
        spec, resp, inv = setup_arguments
        output = intsp.calc_weights(spec, resp, inv)
        assert isinstance(output, xr.Dataset)
        assert output["weights"].values.size


class TestUniqueLatPlevVals:
    """Tests function unique_lat_plev_vals(inv) and its use in calc_weights."""

    def test_reconstructs_inventory_locations(self, setup_arguments):
        """Scattering unique locations back reproduces lat and plev."""
        _, _, inv = setup_arguments
        locations, inverse = intsp.unique_lat_plev_vals(inv)
        np.testing.assert_array_equal(locations[inverse, 0], inv.lat.values)
        np.testing.assert_array_equal(locations[inverse, 1], inv.plev.values)

    def test_duplicates_collapsed(self):
        """Repeated (lat, plev) pairs are returned once."""
        inv = xr.Dataset(
            {
                "lat": ("index", [10.0, 10.0, -5.0, 10.0]),
                "plev": ("index", [250.0, 250.0, 300.0, 300.0]),
            }
        )
        locations, inverse = intsp.unique_lat_plev_vals(inv)
        assert locations.shape == (3, 2)
        assert inverse.shape == (4,)
        assert inverse[0] == inverse[1]

    def test_weights_identical(self, setup_arguments):
        """calc_weights gives identical weights with and without unique_locs."""
        spec, resp, inv = setup_arguments
        direct = intsp.calc_weights(spec, resp, inv)
        unique = intsp.calc_weights(spec, resp, inv, intsp.unique_lat_plev_vals(inv))
        xr.testing.assert_identical(direct, unique)

    def test_nan_locations_preserved(self):
        """NaN locations are kept as their own category, not coded as -1."""
        inv = xr.Dataset(
            {
                "lat": ("index", [10.0, np.nan, 10.0]),
                "plev": ("index", [250.0, 250.0, 250.0]),
            }
        )
        locations, inverse = intsp.unique_lat_plev_vals(inv)
        np.testing.assert_array_equal(locations[inverse, 0], inv.lat.values)
