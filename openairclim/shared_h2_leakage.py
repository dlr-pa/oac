import openairclim as oac
from openairclim.shared_uncertainties import get_uncertainty

"""
Contains the H2 Leakage RF for O3, CH4 and SWV. Only to be used in case of H2 fuel.
ERF quantified by Sand et al. (2023). ERF/RF factors from Lee et al. (2021)

Implementation J.S. Dunselman (2026) MSc Thesis.
"""

# Sand et al. (2023)
O3_ERF_H2 = 0.40e-3 #W/m2/Tg/y
CH4_ERF_H2 = 0.46e-3 #W/m2/Tg/y
SWV_ERF_H2 = 0.19e-3 #W/m2/Tg/y

# Lee et al. (2021)
O3_ERF_RF = 1.37
CH4_ERF_RF = 1.18
SWV_ERF_RF = 1.18

def calc_h2_leakage_rf(H2_consumption_dict, file_name):
    """Calculate and STORE the H2 leakage RF arrays as module attributes."""
    config = oac.get_config(file_name)
    H2_Leakage_Modeling = config["hydrogen_leakage"]["Leakage_modeling"]
    H2_leakage_factor = config["hydrogen_leakage"]["Leakage_rate"]
    global O3_H2_RF_array, CH4_H2_RF_array, SWV_H2_RF_array
    binary = 1
    if H2_Leakage_Modeling == False:
        binary = 0
    fuel = H2_consumption_dict["fuel"]

    O3_H2_RF_array = fuel * (H2_leakage_factor * 
                             O3_ERF_H2 * get_uncertainty("O3 H2 Leakage ERF")
                             / O3_ERF_RF) * binary
    CH4_H2_RF_array = fuel * (H2_leakage_factor * 
                              CH4_ERF_H2 * get_uncertainty("CH4 H2 Leakage ERF")
                              / CH4_ERF_RF) * binary
    SWV_H2_RF_array = fuel * (H2_leakage_factor * 
                              SWV_ERF_H2 * get_uncertainty("SWV H2 Leakage ERF")
                              / SWV_ERF_RF) * binary

    return O3_H2_RF_array, CH4_H2_RF_array, SWV_H2_RF_array