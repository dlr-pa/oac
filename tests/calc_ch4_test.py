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
