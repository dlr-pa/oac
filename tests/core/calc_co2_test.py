"""Provides tests for module calc_co2."""

import numpy as np
import pytest

from openairclim.core import calc_co2


class TestCalcCo2Concentration:
    """Tests function calc_co2_concentration(config, emis_dict)."""

    def test_invalid_method(self):
        """Invalid method returns ValueError."""
        config = {"responses": {"CO2": {"conc": {"method": "InvalidMethod"}}}}
        emis_dict = {
            "CO2": np.array([1000.0, 2000.0, 3000.0])  # Example emissions in Tg
        }
        with pytest.raises(ValueError):
            calc_co2.calc_co2_concentration(config, emis_dict)


class TestCalcCo2Ss:
    """Tests function calc_co2_ss(config, emis_dict)."""

    def test_zero_emissions(self):
        """Zero CO2 emissions return zero concentration changes."""
        config = {"time": {"range": [2000, 2010, 1]}}
        emis_dict = {"CO2": np.zeros(10)}
        result = calc_co2.calc_co2_ss(config, emis_dict)
        expected = {"CO2": np.zeros(10)}
        np.testing.assert_array_equal(result["CO2"], expected["CO2"])


class TestCalcCo2Rf:
    """Tests function calc_co2_rf(conc_dict, config)."""

    def test_invalid_method(self):
        """Invalid method returns ValueError."""
        config = {"responses": {"CO2": {"rf": {"method": "invalid_method"}}}}
        conc_dict = {"CO2": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError):
            calc_co2.calc_co2_rf(conc_dict, config)

    def test_empty_conc_dict(self):
        """Empty concentration dictionary returns KeyError."""
        config = {"responses": {"CO2": {"rf": {"method": "IPCC_2001_1"}}}}
        conc_dict = {}
        with pytest.raises(KeyError):
            calc_co2.calc_co2_rf(conc_dict, config)


class TestCalcCo2SsConvolution:
    """Compares calc_co2_ss against a direct double-loop convolution."""

    def test_matches_loop(self):
        """Vectorised convolution equals the explicit nested-loop result."""
        config = {"time": {"range": [2000, 2100, 2]}}
        years = np.arange(2000, 2100, 2)
        emis = np.random.default_rng(1).random(len(years))
        emis[::3] = 0.0
        emis_c = calc_co2.tgco2_to_tgc(emis)
        expected = np.zeros(len(years))
        for i, year in enumerate(years):
            for j, year_dash in enumerate(years[: i + 1]):
                expected[i] += calc_co2.greens_c(year - year_dash) * emis_c[j] * 2
        expected /= 1000.0
        result = calc_co2.calc_co2_ss(config, {"CO2": emis})["CO2"]
        np.testing.assert_allclose(result, expected, rtol=1e-12)
