"""Provides tests for module interpolate_space."""

import os

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
