"""Provides tests for module calc_dt."""

import numpy as np
import pytest

from openairclim.core import calc_dt


class TestCalcDtempBr2008Co2:
    """Tests function calc_dtemp_br2008(config, spec, rf_arr)."""

    def test_zero_rf(self):
        """RF array with zeros results in temperature arrays with zeros."""
        config = {
            "time": {"range": [2000, 2100, 1]},
            "temperature": {"CO2": {"lambda": 1.0}},
        }
        spec = "CO2"
        rf_arr = np.zeros(100)
        expected_result = np.zeros(100)
        np.testing.assert_array_equal(
            calc_dt.calc_dtemp_br2008(config, spec, rf_arr), expected_result
        )


class TestCalcDtempBr2008Convolution:
    """Compares calc_dtemp_br2008 against a direct double-loop convolution."""

    @pytest.mark.parametrize("step", [1, 5])
    def test_matches_loop(self, step):
        """Vectorised convolution equals the explicit nested-loop result."""
        config = {
            "time": {"range": [2000, 2100, step]},
            "temperature": {"CO2": {"lambda": 0.8}, "CH4": {"efficacy": 1.3}},
        }
        years = np.arange(2000, 2100, step)
        rf_arr = np.random.default_rng(0).random(len(years))
        lambda_spec = 1.3 * 0.8
        expected = np.zeros(len(years))
        for i, year in enumerate(years):
            for j, year_dash in enumerate(years[: i + 1]):
                expected[i] += (
                    (lambda_spec / 1.06)
                    * rf_arr[j]
                    * calc_dt.calc_delta_temp_br2008(
                        year - year_dash, calc_dt.C_ARR, calc_dt.D_ARR
                    )
                    * step
                )
        np.testing.assert_allclose(
            calc_dt.calc_dtemp_br2008(config, "CH4", rf_arr), expected, rtol=1e-12
        )
