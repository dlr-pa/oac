"""
Provides tests for module calc_co2
"""

import numpy as np
import pytest
import openairclim as oac


class TestCalcCo2Concentration:
    """Tests function calc_co2_concentration(config, emis_dict)"""

    def test_invalid_method(self):
        """Invalid method returns ValueError"""
        config = {"responses": {"CO2": {"conc": {"method": "InvalidMethod"}}}}
        emis_dict = {
            "CO2": np.array(
                [1000.0, 2000.0, 3000.0]
            )  # Example emissions in Tg
        }
        with pytest.raises(ValueError):
            oac.calc_co2_concentration(config, emis_dict)


class TestCalcCo2Ss:
    """Tests function calc_co2_ss(config, emis_dict)"""

    def test_zero_emissions(self):
        """Zero CO2 emissions return zero concentration changes"""
        config = {"time": {"range": [2000, 2010, 1]}}
        emis_dict = {"CO2": np.zeros(10)}
        result = oac.calc_co2_ss(config, emis_dict)
        expected = {"CO2": np.zeros(10)}
        np.testing.assert_array_equal(result["CO2"], expected["CO2"])


class TestCalcCo2Rf:
    """Tests function calc_co2_rf(config, conc_dict, conc_co2_bg)"""

    def test_invalid_method(self):
        """Invalid method returns ValueError"""
        config = {"responses": {"CO2": {"rf": {"method": "invalid_method"}}}}
        conc_dict = {"CO2": np.array([1.0, 2.0, 3.0])}
        conc_co2_bg = np.array([410.0, 420.0, 430.0])
        with pytest.raises(ValueError):
            oac.calc_co2_rf(config, conc_dict, conc_co2_bg)

    def test_empty_conc_dict(self):
        """Empty concentration dictionary returns KeyError"""
        config = {"responses": {"CO2": {"rf": {"method": "IPCC_2001_1"}}}}
        conc_dict = {}
        conc_co2_bg = np.array([410.0, 420.0, 430.0])
        with pytest.raises(KeyError):
            oac.calc_co2_rf(config, conc_dict, conc_co2_bg)
