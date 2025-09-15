"""
Provides tests for module calc_ch4
"""

import numpy as np
import xarray as xr
import pytest
import openairclim as oac

#TODO this is just ch4 test copied for reference, first cleanup the code to properly test it

class TestCalcSWVRf:
    #TODO """Tests function calc_ch4_rf(config, conc_dict, conc_ch4_bg_dict, conc_no2_bg_dict)"""
    def test_calc_swv_rf(self):
        total_SWV_mass = {'SWV':np.array([-10,100])}
        rf_swv_dict = oac.calc_swv_rf(total_SWV_mass)
        assert np.allclose(rf_swv_dict['SWV'],np.array([-0.00390254, 0.03782624]),rtol=1e-08,atol=1e-11)
        assert type(rf_swv_dict) == dict
        with pytest.raises(ValueError):
            rf_swv_dict = oac.calc_swv_rf({'SWV':np.array([0,170])})

    def test_invalid_entry(self):
        with pytest.raises(TypeError):
            total_SWV_mass = [10,100]
            rf_swv_dict = oac.calc_swv_rf(total_SWV_mass)

# SOne test that raises an error when there is no methane concentration available/in OAC






    def test_invalid_method(self):
        """Invalid method returns ValueError"""
        config = {"responses": {"CH4": {"rf": {"method": "invalid_method"}}}}
        conc_dict = {"CH4": np.array([1.0, 2.0, 3.0])}
        conc_ch4_bg_dict = {"CH4": np.array([1750.0, 1800.0, 1850.0])}
        conc_n2o_bg_dict = {"N2O": np.array([300.0, 325.0, 350.0])}
        with pytest.raises(ValueError):
            oac.calc_ch4_rf(
                config, conc_dict, conc_ch4_bg_dict, conc_n2o_bg_dict
            )

    def test_empty_conc_dict(self):
        """Empty concentration dictionary returns KeyError"""
        config = {"responses": {"CO2": {"rf": {"method": "Etminan_2016"}}}}
        conc_dict = {}
        conc_ch4_bg_dict = {"CH4": np.array([1750.0, 1800.0, 1850.0])}
        conc_n2o_bg_dict = {"N2O": np.array([300.0, 325.0, 350.0])}
        with pytest.raises(KeyError):
            oac.calc_ch4_rf(
                config, conc_dict, conc_ch4_bg_dict, conc_n2o_bg_dict
            )