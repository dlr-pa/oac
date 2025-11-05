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
            oac.calc_ch4_rf(config, conc_dict, conc_ch4_bg_dict, conc_n2o_bg_dict)

    def test_empty_conc_dict(self):
        """Empty concentration dictionary returns KeyError"""
        config = {"responses": {"CO2": {"rf": {"method": "Etminan_2016"}}}}
        conc_dict = {}
        conc_ch4_bg_dict = {"CH4": np.array([1750.0, 1800.0, 1850.0])}
        conc_n2o_bg_dict = {"N2O": np.array([300.0, 325.0, 350.0])}
        with pytest.raises(KeyError):
            oac.calc_ch4_rf(config, conc_dict, conc_ch4_bg_dict, conc_n2o_bg_dict)


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


import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from ambiance import Atmosphere

# replace with actual module name


class TestCalcSWV:
    @pytest.mark.parametrize(
        "delta_ch4, expected_mass",
        [
            ([0, 0, 0, 0], [0, 0, 0, 0]),
            ([10, 20, 30, 40, 50, 60, 70, 90], [0, 4, 24, 50, 94, 138, 182, 226]),
        ],
    )
    @patch("openairclim.calc_ch4.get_alpha_AOA")
    @patch("openairclim.calc_ch4.get_volume_matrix")
    @patch("openairclim.calc_ch4.Atmosphere")
    def test_calc_swv_mass_conc_basic(
        self,
        mock_atmosphere,
        mock_get_volume,
        mock_get_alpha_aoa,
        delta_ch4,
        expected_mass,
    ):
        # --- Mock get_volume_matrix ---
        mock_get_volume.return_value = np.ones((2, 2))  # simple 2x2 grid of 1.0

        # --- Mock Atmosphere ---
        mock_atm_instance = MagicMock()
        mock_atm_instance.density = np.ones(2) * 1.0  # constant density
        mock_atm_instance.number_density = np.ones(2) * 1e25  # arbitrary number density
        mock_atmosphere.return_value = mock_atm_instance

        # --- Mock get_alpha_AOA ---
        alpha = pd.DataFrame([[0.9, 0.8], [0.2, 0.3]])  # fractional release factor
        AoA = pd.DataFrame([[4, 2], [1, 3]])  # years as lags
        mock_get_alpha_aoa.return_value = alpha, AoA

        # --- Input ---
        # delta_ch4 = [10, 20, 30]

        # --- Run ---
        delta_mass_swv, delta_conc_swv, _ = oac.calc_swv_mass_conc(
            delta_ch4, display_distribution=False
        )

        # --- Assertions ---
        assert isinstance(delta_mass_swv, np.ndarray)
        assert isinstance(delta_conc_swv, np.ndarray)
        assert delta_mass_swv.shape == (len(delta_ch4),)
        assert delta_conc_swv.shape == (len(delta_ch4),)

        # No NaNs or infs in output
        assert np.all(np.isfinite(delta_mass_swv))
        assert np.all(np.isfinite(delta_conc_swv))

        # Since everything mocked is constant, outputs should be > 0

        M_h2o = 18.01528 * 10**-3  # kg/mol
        M_air = 28.97 * 10**-3  # kg/mol
        for i in range(len(delta_ch4)):
            print(i, delta_mass_swv[i] * 1e18, expected_mass[i] * M_h2o / M_air)
            assert delta_mass_swv[i] * 1e18 == pytest.approx(
                expected_mass[i] * M_h2o / M_air, rel=1e-8
            )
