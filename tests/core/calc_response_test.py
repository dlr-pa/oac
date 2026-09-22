"""Provides tests for module calc_response."""

import os

import numpy as np
import pytest
import xarray as xr

from openairclim.core import calc_response
from openairclim.utils.create_test_data import create_test_inv


class TestCalcResp:
    """Tests function calc_resp(spec, inv, weights)."""

    @pytest.fixture(scope="class")
    def inv(self):
        """Fixture to create an example inventory."""
        return create_test_inv(year=2020)

    @pytest.fixture(scope="class")
    def weights(self, inv):
        """Fixture to create weights Dataset."""
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
        return weights

    def test_correct_input(self, inv, weights):
        """Valid input returns float value."""
        spec = "H2O"
        output = calc_response.calc_resp(spec, inv, weights)
        # Check the result
        assert isinstance(output, float)

    def test_invalid_species_raises_key_error(self, inv, weights):
        """A species without a response mapping raises KeyError."""
        with pytest.raises(KeyError):
            calc_response.calc_resp("NOT_A_SPECIES", inv, weights)
