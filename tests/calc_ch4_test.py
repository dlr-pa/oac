"""
Provides tests for module calc_ch4
"""

import numpy as np
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

    def test_rf_etminan_2016_calculation(self, mocker):
        """Test the Etminan 2016 RF formula with controlled values"""
        config = {"responses": {"CH4": {"rf": {"method": "Etminan_2016"}}}}
        conc_dict = {"CH4": np.array([10.0])}  # delta CH4

        # Mock N2O background to a constant
        mock_n2o = {"N2O": np.array([330.0])}
        mocker.patch("openairclim.calc_ch4.interp_bg_conc", return_value=mock_n2o)

        result = oac.calc_ch4_rf(conc_dict, config)
        assert "CH4" in result
        assert len(result["CH4"]) == 1
        # Verify it's a reasonable number (not NaN or Inf)
        assert np.isfinite(result["CH4"][0])


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


class TestCalcCh4Concentration:
    """Tests function calc_ch4_concentration(config, tau_dict)"""

    @pytest.fixture
    def base_config(self):
        """Base configuration for CH4 concentration tests"""
        return {
            "time": {"range": [2020, 2030, 1]},
            "responses": {
                "CH4": {"tau": {"method": "tagging"}, "rf": {"method": "Etminan_2016"}}
            },
            "background": {
                "dir": "dummy/",
                "CH4": {"file": "ch4.nc", "scenario": "ssp126"},
            },
        }

    @pytest.fixture
    def base_tau_dict(self):
        """Base tau dictionary for CH4 concentration tests"""
        # Length should match time_range (2020 to 2029 = 10 elements)
        return {"CH4": np.linspace(0.1, 0.2, 10)}

    def test_tagging_method(self, mocker, base_config, base_tau_dict):
        """Verify correctness of tagging method"""
        # Mock interp_bg_conc to avoid file I/O
        # It should return a dict with "CH4" as a numpy array of length 10
        mock_bg = {"CH4": np.linspace(1800, 1900, 10)}
        mocker.patch("openairclim.calc_ch4.interp_bg_conc", return_value=mock_bg)

        base_config["responses"]["CH4"]["tau"]["method"] = "tagging"

        result = oac.calc_ch4_concentration(base_config, base_tau_dict)

        assert "CH4" in result
        assert isinstance(result["CH4"], np.ndarray)
        assert len(result["CH4"]) == 10
        # Result should not be all zeros since background and tau are non-zero
        assert not np.all(result["CH4"] == 0)

    def test_perturbation_method(self, mocker, base_config, base_tau_dict):
        """Verify correctness of perturbation method"""
        mock_bg = {"CH4": np.linspace(1800, 1900, 10)}
        mocker.patch("openairclim.calc_ch4.interp_bg_conc", return_value=mock_bg)

        base_config["responses"]["CH4"]["tau"]["method"] = "perturbation"

        result = oac.calc_ch4_concentration(base_config, base_tau_dict)

        assert "CH4" in result
        assert isinstance(result["CH4"], np.ndarray)
        assert len(result["CH4"]) == 10
        assert not np.all(result["CH4"] == 0)

    def test_invalid_method(self, mocker, base_config, base_tau_dict):
        """Invalid method in config raises ValueError"""
        mock_bg = {"CH4": np.linspace(1800, 1900, 10)}
        mocker.patch("openairclim.calc_ch4.interp_bg_conc", return_value=mock_bg)

        base_config["responses"]["CH4"]["tau"]["method"] = "invalid_method"

        with pytest.raises(
            ValueError, match="CH4.tau.method in config file is invalid."
        ):
            oac.calc_ch4_concentration(base_config, base_tau_dict)

    def test_tau_array_length_mismatch(self, mocker, base_config):
        """Mismatch between tau_dict length and time_range should
        raise ValueError (from interp1d)"""
        mock_bg = {"CH4": np.linspace(1800, 1900, 10)}
        mocker.patch("openairclim.calc_ch4.interp_bg_conc", return_value=mock_bg)

        # tau_dict length is 5 instead of 10
        tau_dict = {"CH4": np.linspace(0.1, 0.2, 5)}

        with pytest.raises(ValueError):
            oac.calc_ch4_concentration(base_config, tau_dict)
