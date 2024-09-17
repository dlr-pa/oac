"""
Calculates CH4 response
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from openairclim.construct_conc import interp_bg_conc

# CONSTANTS
TAU_GLOBAL = 8.0


def calc_ch4_concentration(config: dict, tau_inverse_dict: dict) -> dict:
    """
    Calculates the methane (CH4) concentration over time based on methane background and
    inverse methane lifetime of idealized emission boxes.
    Args:
        config (dict): Configuration dictionary from config
        tau_inverse_arr (numpy.ndarray): An array of inverse lifetime for methane.
    Returns:
        dict: A dictionary containing the calculated methane concentration for each time step.
            The dictionary has a single key "CH4" with corresponding values as a numpy array.
    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    ch4_bg_arr = interp_bg_conc(config, "CH4")
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


def calc_ch4_rf(config: dict, conc_arr):
    # TODO
    return 1.0
