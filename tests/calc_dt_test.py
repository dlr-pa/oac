"""
Provides tests for module calc_dt
"""

import numpy as np
import openairclim as oac


class TestCalcDtempBr2008Co2:
    """Tests function calc_dtemp_br2008_co2(config, rf_arr)"""

    def test_zero_rf(self):
        """RF array with zeros results in temperature arrays with zeros"""
        config = {
            "time": {"range": [2000, 2100, 1]},
            "temperature": {"CO2": {"lambda": 1.0}},
        }
        spec = "CO2"
        rf_arr = np.zeros(100)
        expected_result = np.zeros(100)
        np.testing.assert_array_equal(
            oac.calc_dtemp_br2008(config, spec, rf_arr), expected_result
        )
