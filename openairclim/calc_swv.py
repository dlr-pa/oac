"""Calculates the impact of SWV"""

import numpy as np


def calc_swv_rf(total_swv_mass: dict):  # mass in Tg
    """
    Function to calculate the RF due to a certain SWV perturbation mass
    Args:
        total_swv_mass (dict): A dict with the key "SWV" with an array with the SWV mass in Tg for corresponding year
    Raises:
        TypeError: if total_SWV_mass is not a dict
        ValueError: if the total mass is out of range of the plot of Pletzer (2024)

    Returns:
        rf_swv_dict (dict): A dict that contains the forcing due to SWV at that time
    """
    # based on the formula of Pletzer 2024
    if not isinstance(total_swv_mass, dict):
        raise TypeError("total SWV mass must be a float or integer")

    rf_swv_list = []
    a = -0.00088
    b = 0.47373
    c = -0.74676
    for value in total_swv_mass["SWV"]:
        negative = False
        if value < 0:
            negative = True
            value = abs(value)
        if value > 160 or value < -1.58:
            raise ValueError("Total SWV mass out of range of Pletzer plot")
        rf_value = (a * value**2 + b * value + c) / 1000  # to make it W/m2 from mW/m2
        if negative == True:
            rf_value = rf_value * -1
        rf_swv_list.append(rf_value)
    rf_swv_array = np.array(rf_swv_list)
    rf_swv_dict = {"SWV": rf_swv_array}
    return rf_swv_dict
