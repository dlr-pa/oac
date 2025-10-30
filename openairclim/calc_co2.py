"""Calculates CO2 response."""

import logging
import numpy as np
from openairclim.construct_conc import calc_inv_sums
from openairclim.construct_conc import interp_bg_conc
from openairclim.utils import tgco2_to_tgc
from openairclim.utils import kg_to_tg

# CONSTANTS
#
# alpha_j (list): [ppbv/Tg(C)] alpha_j coefficients of the impulse response function G_C
# for CO2 (e.g. from Table 1 of Sausen & Schumann (2000))
# with j being the eigenmodes of the impulse response
# https://doi.org/10.1023/A:1005579306109
ALPHA_ARR = [0.067, 0.1135, 0.152, 0.0970, 0.041]
# m_j (list): [1/yr] inverse of tau_j coefficients of the impulse response function G_C
# for CO2 (e.g. from Table I of Sausen & Schumann (2000))
M_ARR = [0.0, 1.0 / 313.8, 1.0 / 79.8, 1.0 / 18.8, 1.0 / 1.7]

# pre-industrial CO2 and N2O concentration used as reference
# these values are from the SSP scenarios
C_0 = 284.32  # [ppm]
N_0 = 273.87  # [ppb]


def get_co2_emissions(inv_dict):
    """Get total CO2 emissions in Tg for each inventory.

    Args:
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years

    Returns:
        dict: Dictionary with array of CO2 emissions in Tg
    """
    # Sum up CO2 emissions in inventories
    _inv_years, emis_co2 = calc_inv_sums("CO2", inv_dict)
    # Convert kg to Tg
    emis_co2_dict = {"CO2": kg_to_tg(emis_co2)}
    return emis_co2_dict


def greens_c(time):
    """Green's function / Impulse response for CO2 concentration
    after (5) in Sausen & Schumann 2000

    Args:
        time (float): Time

    Returns:
        float: Impulse response for a certain point in time
    """
    return np.sum(ALPHA_ARR * np.exp(np.multiply(M_ARR, -time)))


def calc_co2_concentration(config: dict, emis_dict: dict) -> dict:
    """Calculates the CO2 concentration values in ppmv for emitted CO2 in Tg,
    get method from config and execute corresponding subroutine

    Args:
        config (dict): Configuration dictionary from config
        emis_dict (dict): Dictionary with arrays of emissions
            for time range as defined in config, keys are species

    Returns:
        dict: Dictionary with array of CO2 concentration in ppmv
            for time range as defined in config, key is species CO2
    """
    method = config["responses"]["CO2"]["conc"]["method"]
    if method == "Sausen&Schumann":
        conc_co2_dict = calc_co2_ss(config, emis_dict)
        return conc_co2_dict
    else:
        raise ValueError("CO2.conc.method in config file is invalid.")


def calc_co2_ss(config, emis_dict):
    """Calculates the CO2 concentration values in ppmv for emitted CO2 in Tg
    after Sausen&Schumann, 2000, formulas (4) and (5)

    Args:
        config (dict): Configuration dictionary from config
        emis_dict (dict): Dictionary with arrays of emissions
            for time range as defined in config, keys are species

    Returns:
        dict: Dictionary with array of CO2 concentration in ppmv
            for time range as defined in config, key is species CO2
    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    delta_t = time_config[2]
    # Convert Tg CO2 to Tg C
    emis_co2_arr = tgco2_to_tgc(emis_dict["CO2"])
    conc_co2_arr = np.zeros(len(time_range))
    i = 0
    for year in time_range:
        j = 0
        conc_co2 = 0
        for year_dash in time_range[: (i + 1)]:
            if emis_co2_arr[j] != 0.0:  # optimize code
                # (4) in Sausen & Schumann, 2000
                conc_co2 = (
                    conc_co2
                    + greens_c(year - year_dash) * emis_co2_arr[j] * delta_t
                )
            j = j + 1
        conc_co2_arr[i] = conc_co2 / 1000.0  # convert ppbv -> ppmv
        i = i + 1
    return {"CO2": conc_co2_arr}


def calc_co2_rf(conc_dict, config):
    """
    Calculates the radiative forcing values for emitted CO2 concentrations. The
    CO2 method is taken from the config file.

    Args:
        conc_co2 (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species
        config (dict): Configuration dictionary from config

    Raises:
        ValueError: if CO2.rf.method not valid

    Returns:
        dict: Dictionary with np.ndarray of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    method = config["responses"]["CO2"]["rf"]["method"]
    if method == "IPCC_2001_1":
        rf_dict = calc_co2_rf_ipcc_2001_1(conc_dict)
        return rf_dict
    if method == "IPCC_2001_2":
        rf_dict = calc_co2_rf_ipcc_2001_2(conc_dict)
        return rf_dict
    if method == "IPCC_2001_3":
        rf_dict = calc_co2_rf_ipcc_2001_3(conc_dict)
        return rf_dict
    if method == "Etminan_2016":
        conc_n2o_bg_dict = interp_bg_conc(config, "N2O")
        rf_dict = calc_co2_rf_etminan_2016(
            conc_dict, conc_n2o_bg_dict
        )
        return rf_dict

    # unknown CO2 RF method
    raise ValueError("CO2.rf.method in config file is invalid.")


def calc_co2_rf_ipcc_2001_1(conc_dict):
    """Calculates the radiative forcing values for emitted CO2 concentrations,
    after IPCC 2001, Table 6.2, first row

    Args:
        conc_co2 (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species

    Returns:
        dict: Dictionary with array of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    conc_co2_arr = conc_dict["CO2"]
    rf_co2_arr = 5.35 * np.log(1 + conc_co2_arr / C_0)
    return {"CO2": rf_co2_arr}


def calc_co2_rf_ipcc_2001_2(conc_dict):
    """Calculates the radiative forcing values for emitted CO2 concentrations,
    after IPCC 2001, Table 6.2, second row

    Args:
        conc_co2 (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species

    Returns:
        dict: Dictionary with array of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    conc_co2_arr = conc_dict["CO2"]
    rf_co2_arr = 4.841 * np.log(1 + conc_co2_arr / C_0) + 0.0906 * (
        np.sqrt(conc_co2_arr + C_0) - np.sqrt(C_0)
    )
    return {"CO2": rf_co2_arr}


def calc_co2_rf_ipcc_2001_3(conc_dict):
    """Calculates the radiative forcing values for emitted CO2 concentrations,
    after IPCC 2001, Table 6.2, third row

    Args:
        conc_co2 (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species

    Returns:
        dict: Dictionary with array of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """

    def g(conc):
        return np.log(1.0 + 1.2 * conc + 0.005 * conc**2 + 1.4e-6 * conc**3)

    conc_co2_arr = conc_dict["CO2"]
    rf_co2_arr = 3.35 * (g(conc_co2_arr + C_0) - g(C_0))
    return {"CO2": rf_co2_arr}


def calc_co2_rf_etminan_2016(
    conc_dict: dict, conc_n2o_bg_dict: dict
) -> dict:
    """Calculates the radiative forcing values for emitted CO2 concentrations after
    Etminan, M., Myhre, G., Highwood, E. J., & Shine, K. P. (2016). Radiative forcing
    of carbon dioxide, methane, and nitrous oxide: A significant revision of the
    methane radiative forcing. Geophysical Research Letters, 43(24), 12-614.
    https://doi.org/10.1002/2016GL071930

    Args:
        conc_dict (dict): Dictionary with array of concentrations (not including
            background) between the starting and ending years, keys is species
        conc_n2o_bg_dict (dict): Dictionary of np.ndarray of background N2O concentrations
            between the starting and ending years, key is species

    Returns:
        dict: Dictionary with np.ndarray of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    # concentrations
    dc = conc_dict["CO2"]  # ΔCO2 concentration (compared to background)
    c = dc + C_0  # CO2 concentration on top of background
    n = conc_n2o_bg_dict["N2O"]
    n_mean = 0.5 * (n + N_0)

    # check validity range: 180-2000 ppm for CO2 from Etminan et al. (2016)
    if np.any((c < 180.0) | (2000.0 < c)):
        logging.warning(
            "CO2 concentration is outside of the validity range 180 - 2000 ppm"
            "given by Etminan et al. (2016)."
        )

    # coefficients
    a1 = -2.4e-7  # W/m²/ppm
    b1 = 7.2e-4   # W/m²/ppm
    c1 = -2.1e-4  # W/m²/ppb

    # calculate RF
    x = a1 * dc ** 2.0 + b1 * np.abs(dc) + c1 * n_mean + 5.36
    y = np.log(c / C_0)
    rf_co2_arr = x * y
    return {"CO2": rf_co2_arr}


def calc_co2_drf_dconc(
    conc_dict: dict, config: dict
) -> dict:
    """
    Calculates the derivative of the radiative forcing values for emitted CO2
    concentrations with respect to CO2 concentration. This is used for the
    differential and marginal RF attribution methods. The CO2 method is taken
    from the config file.

    Args:
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, key is species
        config (dict): Configuration dictionary from config

    Raises:
        ValueError: if CO2.rf.method not valid or the derivative undefined

    Returns:
        dict: Dictionary with np.ndarray of CO2 radiative forcing derivative
            values between the starting and ending years, key is species CO2
    """
    method = config["responses"]["CO2"]["rf"]["method"]
    if method == "Etminan_2016":
        conc_n2o_bg_dict = interp_bg_conc(config, "N2O")
        drf_dconc_dict = calc_co2_drf_dconc_etminan_2016(
            conc_dict, conc_n2o_bg_dict
        )
        return drf_dconc_dict

    raise ValueError(
        "CO2.rf.method does not have a valid concentration derivative method" 
        "and thus cannot be used with the selected attribution method."
    )


def calc_co2_drf_dconc_etminan_2016(
    conc_dict: dict, conc_n2o_bg_dict: dict
) -> dict:
    """Calculates the derivative of the radiative forcing values for emitted CO2
    concentrations with respect to CO2 concentration after Etminan, M., Myhre,
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
        dict: Dictionary with np.ndarray of dRF(CO2)/dconc values
            between the starting and ending years, key is species CO2
    """
    # concentrations
    dc = conc_dict["CO2"]  # ΔCO2 concentration (compared to background)
    c = dc + C_0  # CO2 concentration on top of background
    n = conc_n2o_bg_dict["N2O"]
    n_mean = 0.5 * (n + N_0)

    # coefficients
    a1 = -2.4e-7  # W/m²/ppm
    b1 = 7.2e-4   # W/m²/ppm
    c1 = -2.1e-4  # W/m²/ppb

    # calculate derivative of RF w.r.t. concentration using product rule
    x = a1 * dc ** 2.0 + b1 * np.abs(dc) + c1 * n_mean + 5.36
    x_prime = 2.0 * a1 * dc + b1 * np.sign(dc)
    y = np.log(c / C_0)
    y_prime = 1.0 / c
    drf_dconc_dict = x * y_prime + x_prime * y
    return {"CO2": drf_dconc_dict}
