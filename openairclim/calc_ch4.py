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


def calc_ch4_rf(
    config: dict,
    conc_dict: dict,
    conc_ch4_bg_dict: dict,
    conc_n2o_bg_dict: dict,
) -> dict:
    """Calculates the Radiative Forcing values for emitted CH4 concentrations

    Args:
        config (dict): Configuration dictionary from config
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
        conc_ch4_bg_dict (dict): Dictionary of np.ndarray of background CH4 concentrations
            between the starting and ending years, key is species
        conc_n2o_bg_dict (dict): Dictionary of np.ndarray of background N2O concentrations
            between the starting and ending years, key is species
    Raises:
        ValueError: if CH4.rf.method not valid

    Returns:
        dict: Dictionary with np.ndarray of CH4 Radiative Forcing values
            between the starting and ending years, key is species CH4
    """
    method = config["responses"]["CH4"]["rf"]["method"]
    if method == "Etminan_2016":
        rf_dict = calc_ch4_rf_etminan_2016(
            conc_dict, conc_ch4_bg_dict, conc_n2o_bg_dict
        )
        return rf_dict
    else:
        raise ValueError("CH4.rf.method in config file is invalid.")


def calc_ch4_rf_etminan_2016(
    conc_dict: dict, conc_ch4_bg_dict: dict, conc_n2o_bg_dict: dict
) -> dict:
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
    conc_ch4_arr = conc_dict["CH4"]
    conc_ch4_bg_arr = conc_ch4_bg_dict["CH4"]
    conc_n2o_bg_arr = conc_n2o_bg_dict["N2O"]
    a3 = -1.3e-6  # W/m²/ppb
    b3 = -8.2e-6
    m_0_arr = conc_ch4_bg_arr
    m_arr = conc_ch4_bg_arr + conc_ch4_arr
    m_mean_arr = conc_ch4_bg_arr + 0.5 * conc_ch4_arr
    n_mean_arr = conc_n2o_bg_arr
    rf_ch4_arr = (a3 * m_mean_arr + b3 * n_mean_arr + 0.043) * (
        np.sqrt(m_arr) - np.sqrt(m_0_arr)
    )
    return {"CH4": rf_ch4_arr}


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


from ATZE.Myhre2 import *


def get_alpha_AOA():
    df = construct_myhre_2a_df(cp_lat=87, cp_a=60)
    df["value"] = df["value"] * 1046.6
    df["value"] = 1.8 - df["value"]

    delta_h = 100.0  # height increment in meters
    delta_deg = 1.0  # latitude increment
    heights = np.arange(0, 60000 + delta_h, delta_h)  # 0 to 60 km
    latitudes = np.arange(-85, 86, delta_deg)  # -85° to 85°
    grid = get_griddata(df, heights, latitudes, plot_data=True)
    ### This grid will resemble the HAlOE CH4 concentrtion (ppmv)data of myhre fig 1, it is 'gebeunt' but approximately correct

    ch4_e = 1.8  # ppmv
    alpha = (ch4_e - grid) / ch4_e

    aoa = 0.3 + 15.2 * alpha - 21.2 * alpha**2 + 10.4 * alpha**3

    AoA = pd.DataFrame(aoa.round(0))
    return alpha, AoA


# %%
def calc_swv_mass(delta_ch4, alpha, AoA):
    delta_swv = np.ones(len(delta_ch4))

    volume = get_volume_matrix(heights, latitudes, delta_h, delta_deg)
    density = Atmosphere(heights).density
    mass_mat = volume * density[:, np.newaxis]

    # loop
    for t in range(len(delta_ch4)):
        multiplier_map_old = {
            1: delta_ch4[t - 1] if t - 1 >= 0 else 0.0,
            2: delta_ch4[t - 2] if t - 2 >= 0 else 0.0,
            3: delta_ch4[t - 3] if t - 3 >= 0 else 0.0,
            4: delta_ch4[t - 4] if t - 4 >= 0 else 0.0,
            5: delta_ch4[t - 5] if t - 5 >= 0 else 0.0,
        }
        # print(multiplier_map_old)
        df_multipliers_old = AoA.replace(multiplier_map_old)
        swv = 2 * alpha * df_multipliers_old

        SWV_mass_mat = swv * 10**-9 * M_h2o / molar_mass_air * mass_mat
        swv_mass = np.nansum(SWV_mass_mat) / 1e9  # Tg
        print(swv_mass)
        delta_swv[t] = swv_mass
    return delta_swv
