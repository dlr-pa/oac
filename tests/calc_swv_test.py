"""
Provides tests for module calc_ch4
"""

import numpy as np
import xarray as xr
import pytest
import openairclim as oac

# TODO this is just ch4 test copied for reference, first cleanup the code to properly test it


class TestCalcSwvRf:
    # TODO """Tests function calc_swv_rf(total_swv_dict)"""
    def test_calc_swv_rf(self):
        """
        Tests if calc_swv_rf is working properly when inputting correct values.:
        """
        total_swv_mass = {"SWV": np.array([-10, 100, 160])}
        rf_swv_dict = oac.calc_swv_rf(total_swv_mass)
        assert np.allclose(
            rf_swv_dict["SWV"],
            np.array([-0.00390254, 0.03782624, 0.05252204]),
            rtol=1e-08,
            atol=1e-11,
        )
        with pytest.raises(ValueError):
            rf_swv_dict = oac.calc_swv_rf({"SWV": np.array([170])})
        with pytest.raises(ValueError):
            rf_swv_dict = oac.calc_swv_rf({"SWV": np.array([1.5])})

    def test_invalid_entry(self):
        """
        Invalid input type returns TypeError
        """
        with pytest.raises(TypeError):
            total_swv_mass = [10, 100]
            rf_swv_dict = oac.calc_swv_rf(total_swv_mass)


# SOne test that raises an error when there is no methane concentration available/in OAC
