"""
Provides tests for module calc_response
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


@pytest.fixture(name="setup_arguments", scope="class")
def fixture_setup_arguments():
    """Setup arguments for calc_resp

    Returns:
        str, xr.Dataset, xr.Dataset: species name, emission inventory, weights
    """
    spec = "H2O"
    file_path = REPO_PATH + INV_NAME
    inv = xr.load_dataset(file_path)
    weights = xr.Dataset(
        data_vars={
            "lat": inv.lat,
            "plev": inv.plev,
            "weights": (
                ["index"],
                np.ones(len(inv.lat.values)),
                {
                    "long_name": "weights",
                },
            ),
        }
    )
    return spec, inv, weights


@pytest.mark.usefixtures("setup_arguments")
class TestCalcResp:
    """Tests function calc_resp(spec, inv, weights)"""

    def test_correct_input(self, setup_arguments):
        """Valid input returns float value"""
        spec, inv, weights = setup_arguments
        output = oac.calc_resp(spec, inv, weights)
        # Check the result
        assert isinstance(output, float)
