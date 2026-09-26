"""Provides tests for module calc_ch4."""

import numpy as np
import pytest
import xarray as xr

from openairclim.core import calc_ch4


class TestCalcCh4Concentration:
    """Tests function calc_ch4_concentration(config, tau_dict)."""

    @pytest.fixture
    def tau_dict(self, valid_config):
        """Fixture to create tau_dict."""
        time_config = valid_config["time"]["range"]
        num = len(np.arange(time_config[0], time_config[1], time_config[2], dtype=int))
        tau_arr = np.ones(num) * (-0.001)
        return {"CH4": tau_arr}

    @pytest.fixture
    def fake_bg(self, monkeypatch):
        """Fixture to mock background concentration."""

        def fake_interp_bg_conc(valid_config, spec):
            time_config = valid_config["time"]["range"]
            num = len(
                np.arange(time_config[0], time_config[1], time_config[2], dtype=int)
            )
            assert spec == "CH4"
            return {"CH4": np.linspace(1800, 1899, num=num)}

        monkeypatch.setattr(calc_ch4, "interp_bg_conc", fake_interp_bg_conc)
        return fake_interp_bg_conc

    def test_invalid_tau_approach(self, valid_config, tau_dict, fake_bg):
        """Invalid approach returns ValueError."""
        config_invalid_tau_approach = valid_config
        config_invalid_tau_approach["responses"]["CH4"]["tau"]["approach"] = "invalid"
        with pytest.raises(ValueError):
            calc_ch4.calc_ch4_concentration(
                config=config_invalid_tau_approach, tau_dict=tau_dict
            )

    def test_tau_arr_different_size(self, valid_config, tau_dict, fake_bg):
        """Different size of tau_arr than time_range returns ValueError."""
        tau_arr = tau_dict["CH4"]
        with pytest.raises(ValueError):
            calc_ch4.calc_ch4_concentration(
                config=valid_config, tau_dict={"CH4": tau_arr[1:]}
            )

class TestCalcCh4Rf:
    """Tests function calc_ch4_rf(conc_dict, config)."""

    def test_invalid_method(self):
        """Invalid method returns ValueError."""
        config = {"responses": {"CH4": {"rf": {"method": "invalid_method"}}}}
        conc_dict = {"CH4": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError):
            calc_ch4.calc_ch4_rf(conc_dict, config)

    def test_empty_conc_dict(self):
        """Empty concentration dictionary returns KeyError."""
        config = {"responses": {"CO2": {"rf": {"method": "Etminan_2016"}}}}
        conc_dict = {}
        with pytest.raises(KeyError):
            calc_ch4.calc_ch4_rf(conc_dict, config)


@pytest.fixture(name="create_rf_dict", scope="class")
def fixture_load_inv():
    """Create example dictionary with computed RF values.

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
    """Tests function calc_pmo_rf(rf_dict)."""

    def test_valid_input(self):
        """Valid input (dict of :class:`xarray.DataArray`) returns expected dict."""
        out_dict = {"RF_CH4": np.array([1.0, 1.0, 1.0])}
        expected_dict = {"PMO": np.array([0.29, 0.29, 0.29])}
        np.testing.assert_array_almost_equal(
            expected_dict["PMO"], calc_ch4.calc_pmo_rf(out_dict)["PMO"]
        )

    def test_missing_ch4(self):
        """out_dict without CH4 returns KeyError."""
        out_dict = {}
        with pytest.raises(KeyError):
            calc_ch4.calc_pmo_rf(out_dict)
