"""
Calculates CH4 response
"""

import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from openairclim.construct_conc import interp_bg_conc
from openairclim.calc_co2 import N2O_0

# CONSTANTS
TAU_GLOBAL = 8.0
CH4_0 = 731.41  # pre-industrial CH4 concentration [ppb] used as reference


def calc_ch4_concentration(config: dict, tau_inverse_dict: dict) -> dict:
    """
    Calculates the methane (CH4) concentration over time based on methane background and
    inverse methane lifetime of idealized emission boxes.

    Args:
        config (dict): Configuration dictionary from config
        tau_inverse_dict (dict): Dictionary of an np.ndarray of inverse lifetime for methane.
    Returns:
        dict: A dictionary containing the calculated methane concentration for each time step.
            The dictionary has a single key "CH4" with corresponding values as a numpy array.
    """
    time_config = config["time"]["range"]
    time_range = np.arange(time_config[0], time_config[1], time_config[2], dtype=int)
    ch4_bg_dict = interp_bg_conc(config, "CH4")
    ch4_bg_arr = ch4_bg_dict["CH4"]
    ch4_bg = interp1d(x=time_range, y=ch4_bg_arr)
    tau_inverse_arr = tau_inverse_dict["CH4"]
    tau_inverse = interp1d(x=time_range, y=tau_inverse_arr)
    solution = solve_ivp(
        func_tagging,
        [time_range[0], time_range[-1]],
        [0],
        t_eval=time_range,
        dense_output=True,
        args=(ch4_bg, TAU_GLOBAL, tau_inverse),
    )
    conc_ch4_dict = {"CH4": solution.sol(time_range)[0]}
    return conc_ch4_dict


def func_tagging(t, y, ch4_bg, tau_global, tau_inverse):
    """Differential equation, contribution (tagging) method, for evaluating CH4 concentratrion
    after equation 4.49 in Rieger, V.S., A new method to assess the climate effect of mitigation
    strategies for road traffic, Delft University of Technology, PhD, 2018,
    https://doi.org/10.4233/uuid:cc96a7c7-1ec7-449a-84b0-2f9a342a5be5

    Args:
        t (float): time
        y (float): CH4 concentration, tagged, required solution of differential equation
        ch4_bg (float): CH4 background concentration
        tau_global (float): global CH4 lifetime
        tau_inverse (float): inverse CH4 lifetime, tagged

    Returns:
        float: d/dt CH4 (CH4 concentration, tagged)
    """
    return (-0.5) * (tau_inverse(t) * ch4_bg(t) + (1.0 / tau_global) * y)


def calc_ch4_rf(conc_dict: dict, config: dict) -> dict:
    """Calculates the Radiative Forcing values for emitted CH4 concentrations.

    Args:
        config (dict): Configuration dictionary from config
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
    Raises:
        ValueError: if CH4.rf.method not valid

    Returns:
        dict: Dictionary with np.ndarray of CH4 Radiative Forcing values
            between the starting and ending years, key is species CH4
    """
    method = config["responses"]["CH4"]["rf"]["method"]
    if method == "Etminan_2016":
        conc_n2o_bg_dict = interp_bg_conc(config, "N2O")
        rf_dict = calc_ch4_rf_etminan_2016(conc_dict, conc_n2o_bg_dict)
        return rf_dict

    # unknown or invalid CH4 RF method
    raise ValueError("CH4.rf.method in config file is invalid.")


def calc_ch4_rf_etminan_2016(conc_dict: dict, conc_n2o_bg_dict: dict) -> dict:
    """Calculates the Radiative Forcing values for emitted CH4 concentrations after
    Etminan, Maryam, et al. Radiative forcing of carbon dioxide, methane, and nitrous oxide:
    A significant revision of the methane radiative forcing.
    Geophysical Research Letters 43.24 (2016): 12-614.
    https://doi.org/10.1002/2016GL071930

    Args:
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
        conc_ch4_bg_dict (dict): Dictionary of np.ndarray of background CH4 concentrations
            between the starting and ending years, key is species
        conc_n2o_bg_dict (dict): Dictionary of np.ndarray of background N2O concentrations
            between the starting and ending years, key is species

    Returns:
        dict: Dictionary with np.ndarray of CH4 Radiative Forcing values
            between the starting and ending years, key is species CH4
    """
    # concentrations
    d_ch4_conc = conc_dict["CH4"]  # ΔCH4 concentration (compared to background)
    ch4_conc = d_ch4_conc + CH4_0  # CH4 concentration on top of background
    n2o_conc = conc_n2o_bg_dict["N2O"]
    ch4_conc_mean = 0.5 * (ch4_conc + CH4_0)
    n2o_conc_mean = 0.5 * (n2o_conc + N2O_0)

    # check validity range: 340-3500 ppb for CH4 from Etminan et al. (2016)
    if np.any((ch4_conc < 340.0) | (3500.0 < ch4_conc)):
        logging.warning(
            "CH4 concentration is outside of the validity range 340 - 3500 ppb"
            "given by Etminan et al. (2016)."
        )

    # coefficients
    a3 = -1.3e-6  # W/m²/ppb
    b3 = -8.2e-6  # W/m²/ppb

    # calculate RF
    x = a3 * ch4_conc_mean + b3 * n2o_conc_mean + 0.043
    y = np.sqrt(ch4_conc) - np.sqrt(CH4_0)
    rf_ch4_arr = x * y
    return {"CH4": rf_ch4_arr}


def calc_ch4_drf_dconc(conc_dict: dict, config: dict) -> dict:
    """
    Calculates the derivative of the radiative forcing values for emitted CH4
    concentrations with respect to CH4 concentration. This is used for the
    differential and marginal RF attribution methods. The CH4 method is taken
    from the config file

    Args:
        conc_dict (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species
        config (dict): Configuration dictionary from config

    Returns:
        dict: Dictionary with np.ndarray of CH4 radiative forcing derivative
            values between the starting and ending years, key is species CH4
    """
    method = config["responses"]["CH4"]["rf"]["method"]
    if method == "Etminan_2016":
        conc_n2o_bg_dict = interp_bg_conc(config, "N2O")
        drf_dconc_dict = calc_ch4_drf_dconc_etminan_2016(conc_dict, conc_n2o_bg_dict)
        return drf_dconc_dict

    raise ValueError(
        "CH4.rf.method does not have a valid concentration derivative method"
        "and thus cannot be used with the selected attribution method."
    )


def calc_ch4_drf_dconc_etminan_2016(conc_dict: dict, conc_n2o_bg_dict: dict) -> dict:
    """Calculates the derivative of the radiative forcing values for emitted CH4
    concentrations with respect to CH4 concentration after Etminan, M., Myhre,
    G., Highwood, E. J., & Shine, K. P. (2016). Radiative forcing of carbon
    dioxide, methane, and nitrous oxide: A significant revision of the methane
    radiative forcing. Geophysical Research Letters, 43(24), 12-614.
    https://doi.org/10.1002/2016GL071930

    Args:
        conc_dict (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species
        conc_n2o_bg_dict (dict): Dictionary of np.ndarray of background N2O
            concentrations between the starting and ending years, key is species

    Returns:
        dict: Dictionary with np.ndarray of dRF(CH4)/dconc values
            between the starting and ending years, key is species CH4
    """
    # concentrations
    d_ch4_conc = conc_dict["CH4"]  # ΔCH4 concentration (compared to background)
    ch4_conc = d_ch4_conc + CH4_0  # CH4 concentration on top of background
    n2o_conc = conc_n2o_bg_dict["N2O"]
    ch4_conc_mean = 0.5 * (ch4_conc + CH4_0)
    n2o_conc_mean = 0.5 * (n2o_conc + N2O_0)

    # coefficients
    a3 = -1.3e-6  # W/m²/ppb
    b3 = -8.2e-6  # W/m²/ppb

    # calculate derivative of RF w.r.t. concentration using product rule
    x = a3 * ch4_conc_mean + b3 * n2o_conc_mean + 0.043
    x_prime = a3 / 2.0
    y = np.sqrt(ch4_conc) - np.sqrt(CH4_0)
    y_prime = 1.0 / (2.0 * np.sqrt(ch4_conc))
    drf_dconc_dict = x * y_prime + x_prime * y
    return {"CH4": drf_dconc_dict}


def calc_pmo_rf(out_dict):
    """
    Calculates PMO RF

    Args:
        out_dict (dict): Dictionary with computed responses, keys are e.g.
            'RF_CH4'

    Returns:
        dict: Dictionary of np.ndarray of computed RF, key is PMO

    Raises:
        KeyError: If computed CH4 RF is not available.
    """
    if "RF_CH4" in out_dict:
        rf_ch4_arr = out_dict["RF_CH4"]
        rf_pmo_arr = 0.29 * rf_ch4_arr
    else:
        msg = "PMO RF requires computed CH4 RF which is not available!"
        raise KeyError(msg)
    return {"PMO": rf_pmo_arr}
