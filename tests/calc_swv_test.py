"""
Provides tests for module calc_ch4
"""

import numpy as np
import xarray as xr
import pytest
import openairclim as oac

#TODO this is just ch4 test copied for reference, first cleanup the code to properly test it

class TestCalcSwvRf:
    #TODO """Tests function calc_ch4_rf(config, conc_dict, conc_ch4_bg_dict, conc_no2_bg_dict)"""
    def test_calc_swv_rf(self):
        """
        Tests if calc_swv_rf is working properly when inputting correct values.:
        TODO what to do with 0 as i suppose the values there are invalid, raise a warning?
        """
        total_SWV_mass = {'SWV':np.array([-10,100,160])}
        rf_swv_dict = oac.calc_swv_rf(total_SWV_mass)
        assert np.allclose(rf_swv_dict['SWV'],np.array([-0.00390254, 0.03782624, 0.05252204]),rtol=1e-08,atol=1e-11)
        with pytest.raises(ValueError):
            rf_swv_dict = oac.calc_swv_rf({'SWV':np.array([170])})
        # with pytest.raises(ValueError):
        #     rf_swv_dict = oac.calc_swv_rf({'SWV':np.array([0])}) # TODO SHOULD THIS BE?


    def test_invalid_entry(self):
        """
        Invalid input type returns TypeError
        """
        with pytest.raises(TypeError):
            total_SWV_mass = [10,100]
            rf_swv_dict = oac.calc_swv_rf(total_SWV_mass)

# SOne test that raises an error when there is no methane concentration available/in OAC


