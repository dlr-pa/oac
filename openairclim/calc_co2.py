"""Calculates CO2 response."""

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


def get_co2_emissions(inv_dict):
    """Get total CO2 emissions in Tg for each inventory.

    Args:
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years

    Returns:
        dict: Dictionary with array of CO2 emissions in Tg
    """
    # Sum up CO2 emissions in inventories
    emis_co2 = calc_inv_sums("CO2", inv_dict)
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


def calc_co2_rf(config, conc_dict, conc_co2_bg_dict):
    """
    Calculates the radiative forcing values for emitted CO2 concentrations,
    after IPCC 2001, Ramaswamy, V. et al. in "Climate Change 2001:
    The Scientific Basis. Contribution of Working Group I to the Third Assessment
    Report of the Intergovernmental Panel of Climate Change"Table 6.2,
    get method from config and execute corresponding subroutine

    Args:
        config (dict): Configuration dictionary from config
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, key is species
        conc_co2_bg_dict (dict): Dictionary of np.ndarray of background CO2 concentrations
            between the starting and ending years, key is species
    Raises:
        ValueError: if CO2.rf.method not valid

    Returns:
        dict: Dictionary with np.ndarray of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    method = config["responses"]["CO2"]["rf"]["method"]
    if method == "IPCC_2001_1":
        rf_dict = calc_co2_rf_ipcc_2001_1(conc_dict, conc_co2_bg_dict)
        return rf_dict
    elif method == "IPCC_2001_2":
        rf_dict = calc_co2_rf_ipcc_2001_2(conc_dict, conc_co2_bg_dict)
        return rf_dict
    elif method == "IPCC_2001_3":
        rf_dict = calc_co2_rf_ipcc_2001_3(conc_dict, conc_co2_bg_dict)
        return rf_dict
    elif method == "Etminan_2016":
        conc_n2o_bg_dict = interp_bg_conc(config, "N2O")
        rf_dict = calc_co2_rf_etminan_2016(
            conc_dict, conc_co2_bg_dict, conc_n2o_bg_dict
        )
        return rf_dict
    else:
        raise ValueError("CO2.rf.method in config file is invalid.")


def calc_co2_rf_ipcc_2001_1(conc_dict, conc_co2_bg_dict):
    """Calculates the radiative forcing values for emitted CO2 concentrations,
    after IPCC 2001, Table 6.2, first row

    Args:
        conc_co2 (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
        conc_co2_bg_dict (dict): Dictionary of np.ndarray of background CO2 concentrations
            between the starting and ending years, key is species

    Returns:
        dict: Dictionary with array of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    conc_co2_arr = conc_dict["CO2"]
    conc_co2_bg = conc_co2_bg_dict["CO2"]
    rf_co2_arr = 5.35 * np.log(1 + conc_co2_arr / conc_co2_bg)
    return {"CO2": rf_co2_arr}


def calc_co2_rf_ipcc_2001_2(conc_dict, conc_co2_bg_dict):
    """Calculates the radiative forcing values for emitted CO2 concentrations,
    after IPCC 2001, Table 6.2, second row

    Args:
        conc_co2 (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
        conc_co2_bg_dict (dict): Dictionary of np.ndarray of background CO2 concentrations
            between the starting and ending years, key is species

    Returns:
        dict: Dictionary with array of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    conc_co2_arr = conc_dict["CO2"]
    conc_co2_bg = conc_co2_bg_dict["CO2"]
    rf_co2_arr = 4.841 * np.log(1 + conc_co2_arr / conc_co2_bg) + 0.0906 * (
        np.sqrt(conc_co2_arr + conc_co2_bg) - np.sqrt(conc_co2_bg)
    )
    return {"CO2": rf_co2_arr}


def calc_co2_rf_ipcc_2001_3(conc_dict, conc_co2_bg_dict):
    """Calculates the radiative forcing values for emitted CO2 concentrations,
    after IPCC 2001, Table 6.2, third row

    Args:
        conc_co2 (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
        conc_co2_bg_dict (dict): Dictionary of np.ndarray of background CO2 concentrations
            between the starting and ending years, key is species

    Returns:
        dict: Dictionary with array of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """

    def g(conc):
        return np.log(1.0 + 1.2 * conc + 0.005 * conc**2 + 1.4e-6 * conc**3)

    conc_co2_arr = conc_dict["CO2"]
    conc_co2_bg = conc_co2_bg_dict["CO2"]
    rf_co2_arr = 3.35 * (g(conc_co2_arr + conc_co2_bg) - g(conc_co2_bg))
    return {"CO2": rf_co2_arr}


def calc_co2_rf_etminan_2016(
    conc_dict: dict, conc_co2_bg_dict: dict, conc_n2o_bg_dict: dict
) -> dict:
    """Calculates the radiative forcing values for emitted CO2 concentrations after
    Etminan, M., Myhre, G., Highwood, E. J., & Shine, K. P. (2016). Radiative forcing
    of carbon dioxide, methane, and nitrous oxide: A significant revision of the
    methane radiative forcing. Geophysical Research Letters, 43(24), 12-614.
    https://doi.org/10.1002/2016GL071930

    Args:
        conc_dict (dict): Dictionary with array of concentrations
            between the starting and ending years, keys is species
        conc_co2_bg_dict (dict): Dictionary of np.ndarray of background CO2 concentrations
            between the starting and ending years, key is species
        conc_n2o_bg_dict (dict): Dictionary of np.ndarray of background N2O concentrations
            between the starting and ending years, key is species

    Returns:
        dict: Dictionary with np.ndarray of CO2 radiative forcing values
            between the starting and ending years, key is species CO2
    """
    # TODO Check this method! Check units: ppmv vs. ppm ?
    conc_co2_arr = conc_dict["CO2"]
    conc_co2_bg_arr = conc_co2_bg_dict["CO2"]
    conc_n2o_bg_arr = conc_n2o_bg_dict["N2O"]
    a1 = -2.4e-7  # W/m²/ppm
    b1 = 7.2e-4
    c1 = -2.1e-4
    c_0_arr = conc_co2_bg_arr
    c_arr = conc_co2_bg_arr + conc_co2_arr
    n_mean_arr = conc_n2o_bg_arr
    rf_co2_arr = (
        a1 * (c_arr - c_0_arr) ** 2
        + b1 * abs(c_arr - c_0_arr)
        + c1 * n_mean_arr
        + 5.36
    ) * np.log(c_arr / c_0_arr)
    return {"CO2": rf_co2_arr}
