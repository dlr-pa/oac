"""Calculates CH4 response."""

import logging
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .calc_co2 import N2O_0
from .construct_conc import interp_bg_conc

logger = logging.getLogger(__name__)

# CONSTANTS
TAU_GLOB = 8.0  # CH4 lifetime
TAU_PERT = 12.0  # Perturbation lifetime
CH4_0 = 731.41  # pre-industrial CH4 concentration [ppb] used as reference
CH4_VALIDITY_MIN = 340.0  # lower bound of Etminan et al. (2016) validity range [ppb]
CH4_VALIDITY_MAX = 3500.0  # upper bound of Etminan et al. (2016) validity range [ppb]

# Define the shared signature of ODE functions explicitly
OdeFunc = Callable[
    [float, np.ndarray, Callable[[float], float], Callable[[float], float]],
    np.ndarray,
]


def calc_ch4_concentration(config: dict, tau_dict: dict) -> dict:
    """Calculates the methane (CH4) concentration over time.

    This method uses methane background and methane lifetimes of idealized
    emission boxes. Either the tagging or the perturbation method is used.

    Args:
        config (dict): Configuration dictionary from config
        tau_dict (dict): Dictionary of a np.ndarray, either inverse lifetimes (1 / tau)
            or relative changes in lifetime (delta), key is "CH4"
            NOTE: array must be of same size as time_range
    Returns:
        dict: A dictionary containing the calculated methane concentration
        for each time step. The dictionary has a single key "CH4" with
        corresponding values as a numpy array.
    """
    method = config["responses"]["CH4"]["tau"]["method"]
    time_config = config["time"]["range"]
    time_range = np.arange(time_config[0], time_config[1], time_config[2], dtype=int)
    ch4_bg_dict = interp_bg_conc(config, "CH4")
    ch4_bg_arr = ch4_bg_dict["CH4"]
    # Define function for evaluation at continous times,
    # Use boundaries of array as fill_value, i.e. outside time_range
    ch4_bg_func = interp1d(
        x=time_range,
        y=ch4_bg_arr,
        bounds_error=False,
        fill_value=(ch4_bg_arr[0], ch4_bg_arr[-1]),
    )
    # tau_arr comprising either inverse lifetimes (1 / tau) for tagging
    # or relative changes in lifetime (delta) for perturbation method
    tau_arr = tau_dict["CH4"]
    # Define function for evaluation at continous times
    tau_func = interp1d(
        x=time_range,
        y=tau_arr,
        # Constant values between integer years
        kind="zero",
        bounds_error=False,
        fill_value=(tau_arr[0], tau_arr[-1]),
    )
    # Explicit annotation of ode
    ode: OdeFunc
    if method == "tagging":
        # Ordinary Differential Equation
        ode = ode_ch4_tagging
        # Initial condition
        # From ODE defined in ch4_tagging, d/dt y = A*y + B
        # tau_arr: inverse lifetimes (1 / tau)
        y0 = -2.0 * (TAU_GLOB * tau_arr[0]) * ch4_bg_arr[0]
    elif method == "perturbation":
        # Ordinary Differential Equation
        ode = ode_ch4_perturbation
        # Initial condition
        # From ODE defined in ch4_perturbation, d/dt y = A*y + B
        # tau_arr: relative changes in lifetime (delta)
        y0 = 2.0 * tau_arr[0] * ch4_bg_arr[0]
    else:
        raise ValueError("CH4.tau.method in config file is invalid.")
    solution = solve_ivp(
        ode,
        [time_range[0], time_range[-1]],
        [y0],
        method="RK45",
        t_eval=time_range,
        dense_output=False,
        args=(ch4_bg_func, tau_func),
    )
    conc_ch4_dict = {"CH4": solution.y[0]}
    return conc_ch4_dict


def ode_ch4_tagging(
    t: float,
    y: np.ndarray,
    ch4_bg: Callable[[float], float],
    tau_inverse: Callable[[float], float],
) -> np.ndarray:
    """Differential equation, contribution (tagging) method.

    This differential equation determines CH4 concentration, after equation 4.49
    in Rieger, V.S., A new method to assess the climate effect of mitigation
    strategies for road traffic, Delft University of Technology, PhD, 2018,
    https://doi.org/10.4233/uuid:cc96a7c7-1ec7-449a-84b0-2f9a342a5be5

    Args:
        t (float): time
        y (np.ndarray): CH4 concentration, tagged, required solution
        ch4_bg (Callable[[float], float]): CH4 background concentration
        tau_inverse (Callable[[float], float]): inverse CH4 lifetime, tagged

    Returns:
        np.ndarray: d/dt CH4 (CH4 concentration, tagged)
    """
    return (-0.5) * (tau_inverse(t) * ch4_bg(t) + (1.0 / TAU_GLOB) * y)


def ode_ch4_perturbation(
    t: float,
    y: np.ndarray,
    ch4_bg: Callable[[float], float],
    delta: Callable[[float], float],
) -> np.ndarray:
    """Differential equation for evaluating CH4 concentration changes.

    Perturbation method after equation (3) in Grewe & Stenke (2008),
    https://doi.org/10.5194/acp-8-4621-2008

    Args:
        t (float): time
        y (np.ndarray): CH4 concentration changes, required solution
        ch4_bg (Callable[[float], float]): CH4 background concentration
        delta (Callable[[float], float]): relative change in lifetime

    Returns:
        np.ndarray: d/dt CH4 (CH4 concentration change, perturbation)
    """
    return (delta(t) / (1 + delta(t))) * (1 / TAU_PERT) * ch4_bg(t) - (
        1 / (1 + delta(t))
    ) * (1 / TAU_PERT) * y


def calc_ch4_rf(conc_dict: dict, config: dict) -> dict:
    """Calculates the Radiative Forcing values for emitted CH4 concentrations.

    Args:
        config (dict): Configuration dictionary from config
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
    Raises:
        ValueError: if CH4.rf.method not valid

    Returns:
        dict: Dictionary with :class:`numpy.ndarray` of CH4 Radiative Forcing values
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
    """Calculates the CH4 Radiative Forcing values for emitted concentrations.

    Reference: Etminan, Maryam, et al. Radiative forcing of carbon dioxide,
    methane, and nitrous oxide: A significant revision of the methane
    radiative forcing. Geophysical Research Letters 43.24 (2016): 12-614.
    https://doi.org/10.1002/2016GL071930.

    Args:
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
        conc_ch4_bg_dict (dict): Dictionary of :class:`numpy.ndarray` of background CH4
            concentrations between the starting and ending years, key is
            species
        conc_n2o_bg_dict (dict): Dictionary of :class:`numpy.ndarray` of background N2O
            concentrations between the starting and ending years, key is
            species

    Returns:
        dict: Dictionary with :class:`numpy.ndarray` of CH4 Radiative Forcing values
        between the starting and ending years, key is species CH4
    """
    # concentrations
    d_ch4_conc = conc_dict["CH4"]  # ΔCH4 concentration (compared to background)
    ch4_conc = d_ch4_conc + CH4_0  # CH4 concentration on top of background
    n2o_conc = conc_n2o_bg_dict["N2O"]
    ch4_conc_mean = 0.5 * (ch4_conc + CH4_0)
    n2o_conc_mean = 0.5 * (n2o_conc + N2O_0)

    # check validity range: 340-3500 ppb for CH4 from Etminan et al. (2016)
    if np.any((ch4_conc < CH4_VALIDITY_MIN) | (ch4_conc > CH4_VALIDITY_MAX)):
        logger.warning(
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
    """Calculates the derivative of CH4 radiative forcing w.r.t. concentration.

    This is used for the differential and marginal RF attribution methods.
    The CH4 method is taken from the config file.

    Args:
        conc_dict (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species
        config (dict): Configuration dictionary from config

    Returns:
        dict: Dictionary with :class:`numpy.ndarray` of CH4 radiative forcing derivative
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
    """Calculates the derivative of CH4 radiative forcing w.r.t. concentration.

    Reference: Etminan, M., Myhre, G., Highwood, E. J., & Shine, K. P. (2016).
    Radiative forcing of carbon dioxide, methane, and nitrous oxide: A
    significant revision of the methane radiative forcing. Geophysical
    Research Letters, 43(24), 12-614.
    https://doi.org/10.1002/2016GL071930.

    Args:
        conc_dict (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species
        conc_n2o_bg_dict (dict): Dictionary of :class:`numpy.ndarray` of background N2O
            concentrations between the starting and ending years, key is species

    Returns:
        dict: Dictionary with :class:`numpy.ndarray` of dRF(CH4)/dconc values
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


def calc_pmo_rf(out_dict: dict) -> dict:
    """Calculates PMO RF.

    Args:
        out_dict (dict): Dictionary with computed responses, keys are e.g.
            'RF_CH4'

    Returns:
        dict: Dictionary of :class:`numpy.ndarray` of computed RF, key is PMO

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
