"""
Provides tests for module parametric
"""

import numpy as np
import pytest
import openairclim as oac


class TestAdaptCo2Emissions:
    """Tests function adapt_co2_emission(config, emis_interp_dict)"""

    @pytest.fixture(autouse=True)
    def patch_get_factor(self, monkeypatch):
        """
        Replace the private helper `_get_factor` with a deterministic stub
        that simply returns the value from config or 1 if absent.
        """

        def fake_get_factor(config, spec):
            return config.get("parametric").get(spec, 1)

        monkeypatch.setattr("openairclim.parametric._get_factor", fake_get_factor)

    def test_factor_application(self):
        """Tests correct multiplication by given factor."""
        emis_interp_dict = {"CO2": np.array([1.0, 2.0, 3.0])}
        config = {"parametric": {"CO2": 2.0}}
        result_dict = oac.adapt_co2_emission(config, emis_interp_dict)
        expected_dict = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result_dict["CO2"], expected_dict)

    def test_co2_not_in_emis_interp_dict(self):
        """Missing CO2 key in emis_interp_dict raises KeyError."""
        emis_interp_dict = {"H2O": np.array([1.0])}
        config = {"parametric": {"CO2": 2.0}}
        with pytest.raises(KeyError):
            oac.adapt_co2_emission(config, emis_interp_dict)


class TestAdaptRf:
    """Tests function adapt_rf(config, rf_interp_dict, spec_lst)"""

    @pytest.fixture(autouse=True)
    def patch_get_factor(self, monkeypatch):
        """
        Replace the private helper `_get_factor` with a deterministic stub
        that simply returns the value from config or 1 if absent.
        """

        def fake_get_factor(config, spec):
            return config.get("parametric").get(spec, 1)

        monkeypatch.setattr("openairclim.parametric._get_factor", fake_get_factor)

    def test_factor_application(self):
        """Tests correct multiplication by given factor."""
        rf_interp_dict = {"H2O": np.array([1.0, 2.0, 3.0])}
        config = {"parametric": {"H2O": 2.0}}
        result_dict = oac.adapt_rf(config, rf_interp_dict, ["H2O"])
        expected_dict = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result_dict["H2O"], expected_dict)

    def test_spec_not_in_rf_interp_dict(self):
        """Missing species key in rf_interp_dict raises KeyError."""
        rf_interp_dict = {"H2O": np.array([1.0])}
        config = {"parametric": {"CH4": 2.0}}
        with pytest.raises(KeyError):
            oac.adapt_rf(config, rf_interp_dict, ["CH4"])
