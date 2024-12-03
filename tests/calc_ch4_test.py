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
        conc_ch4_bg_dict = {"CH4": np.array([1750.0, 1800.0, 1850.0])}
        conc_n2o_bg_dict = {"N2O": np.array([300.0, 325.0, 350.0])}
        with pytest.raises(ValueError):
            oac.calc_ch4_rf(
                config, conc_dict, conc_ch4_bg_dict, conc_n2o_bg_dict
            )

    def test_empty_conc_dict(self):
        """Empty concentration dictionary returns KeyError"""
        config = {"responses": {"CO2": {"rf": {"method": "Etminan_2016"}}}}
        conc_dict = {}
        conc_ch4_bg_dict = {"CH4": np.array([1750.0, 1800.0, 1850.0])}
        conc_n2o_bg_dict = {"N2O": np.array([300.0, 325.0, 350.0])}
        with pytest.raises(KeyError):
            oac.calc_ch4_rf(
                config, conc_dict, conc_ch4_bg_dict, conc_n2o_bg_dict
            )


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


@pytest.mark.usefixtures("create_rf_dict")
class TestCalcPmoRF:
    """Tests function calc_pmo_rf(rf_dict)"""

    def test_valid_input(self, create_rf_dict):
        """Valid input (dictionary of xr.DataArray) returns expected dictionary"""
        rf_dict = create_rf_dict
        expected_dict = {"PMO": np.array([0.29, 0.29, 0.29])}
        np.testing.assert_array_almost_equal(
            expected_dict["PMO"], oac.calc_pmo_rf(rf_dict)["PMO"]
        )

    def test_missing_ch4(self):
        """rf_dict without CH4 returns KeyError"""
        rf_dict = {}
        with pytest.raises(KeyError):
            oac.calc_pmo_rf(rf_dict)
