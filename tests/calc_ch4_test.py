"""
Provides tests for module calc_ch4
"""

import numpy as np
import xarray as xr
import pytest
import openairclim as oac


class TestCalcCh4Rf:
    """Tests function calc_ch4_rf(config, conc_dict, conc_ch4_bg_dict, conc_no2_bg_dict)"""

    def test_invalid_method(self):
        """Invalid method returns ValueError"""
        config = {"responses": {"CH4": {"rf": {"method": "invalid_method"}}}}
        conc_dict = {"CH4": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError):
            oac.calc_ch4_rf(conc_dict, config)

    def test_empty_conc_dict(self):
        """Empty concentration dictionary returns KeyError"""
        config = {"responses": {"CO2": {"rf": {"method": "Etminan_2016"}}}}
        conc_dict = {}
        with pytest.raises(KeyError):
            oac.calc_ch4_rf(conc_dict, config)


@pytest.fixture(name="create_rf_dict", scope="class")
def fixture_load_inv():
    """Create example dictionary with computed RF values

    Returns:
        dict: Dictionary of xarray DataArray, key are species
    """
    rf_dict = {
        "CH4": xr.DataArray(
            data=np.array([1.0, 1.0, 1.0]),
            coords={"time": np.array([2020, 2030, 2040])},
            dims=["time"],
            name="RF_CH4",
        )
    }
    return rf_dict


class TestCalcPmoRF:
    """Tests function calc_pmo_rf(rf_dict)"""

    def test_valid_input(self):
        """Valid input (dictionary of xr.DataArray) returns expected dictionary"""
        out_dict = {"RF_CH4": np.array([1.0, 1.0, 1.0])}
        expected_dict = {"PMO": np.array([0.29, 0.29, 0.29])}
        np.testing.assert_array_almost_equal(
            expected_dict["PMO"], oac.calc_pmo_rf(out_dict)["PMO"]
        )

    def test_missing_ch4(self):
        """out_dict without CH4 returns KeyError"""
        out_dict = {}
        with pytest.raises(KeyError):
            oac.calc_pmo_rf(out_dict)
