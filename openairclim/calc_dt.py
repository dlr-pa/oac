"""
Calculates temperature changes for each species and scenario
"""

import logging
import numpy as np

# CONSTANTS
#
# from Boucher & Reddy (2008)
# https://doi.org/10.1016/j.enpol.2007.08.039
C_ARR = [0.631, 0.429]  # in K / (W m-2)
D_ARR = [8.4, 409.5]  # in years


def calc_dtemp(config, spec, rf_arr):
    """
    Calculates the temperature changes for a single species

    Args:
        config (dict): Configuration dictionary from config
        spec (str): species
        rf_arr (np.ndarray): array of radiative forcing values
            for time range as defined in config

    Returns:
        np.ndarray: array of temperature values for time range as defined in config
    """
    if config["temperature"]["method"] == "Boucher&Reddy":
        dtemp_arr = calc_dtemp_br2008(config, spec, rf_arr)
    else:
        # TODO Move this check to module read_config
        msg = "Method for temperature change calculation is not valid."
        logging.warning(msg)
    return dtemp_arr


def calc_dtemp_br2008(
    config: dict, spec: str, rf_arr: np.ndarray
) -> np.ndarray:
    """
    Calculates temperature changes after Boucher and Reddy (2008)
    https://doi.org/10.1016/j.enpol.2007.08.039


    Args:
        config (dict): configuration dictionary from config
        spec (str): species
        rf_arr (np.ndarray): array of radiative forcing values

    Returns:
        np.ndarray: array of temperature values
    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    delta_t = time_config[2]
    lambda_co2 = config["temperature"]["CO2"]["lambda"]
    if spec == "CO2":
        efficacy = 1.0
    else:
        efficacy = config["temperature"][spec]["efficacy"]
    lambda_spec = efficacy * lambda_co2
    dtemp_arr = np.zeros(len(time_range))
    i = 0
    for year in time_range:
        j = 0
        dtemp = 0
        for year_dash in time_range[: (i + 1)]:
            dtemp = (
                dtemp
                + (lambda_spec / lambda_co2)
                * rf_arr[j]
                * calc_delta_temp_br2008((year - year_dash), C_ARR, D_ARR)
                * delta_t
            )
            j = j + 1
        dtemp_arr[i] = dtemp
        i = i + 1
    return dtemp_arr


def calc_delta_temp_br2008(t: float, c_arr, d_arr):
    """
    Impulse response function according to Boucher and Reddy (2008), Appendix A

    Args:
        t (float): time
        c_arr (list): parameter array of impulse response function,
            Table A1: ci in (K / (W m-2))
        d_arr (list): parameter array of impulse response function,
            Table A1: di in (years)

    Returns:
        float: temperature change according to the Boucher and Reddy (2008) model
    """
    delta_temp = 0.0
    for c, d in zip(c_arr, d_arr):
        delta_temp = delta_temp + (c / d) * np.exp(-t / d)
    return delta_temp
