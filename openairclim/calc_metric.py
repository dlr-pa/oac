"""
Calculates climate metric for each species and scenario
"""

import numpy as np
from openairclim.read_netcdf import get_results


def calc_climate_metrics(config: dict) -> dict:
    """Get all combinations of required climate metrics

    Args:
        config (dict): Configuration from config file

    Returns:
        dict: Dictionary of dictionaries containing climate metrics values,
            keys are unique climate metrics identifiers for each combination
            and species
    """
    metrics_type_arr = config["metrics"]["types"]
    t_zero_arr = config["metrics"]["t_0"]
    horizon_arr = config["metrics"]["H"]
    _emis_dict, _conc_dict, rf_dict, dtemp_dict = get_results(config)
    out_dict = {}
    for metrics_type in metrics_type_arr:
        for t_zero in t_zero_arr:
            for horizon in horizon_arr:
                key = (
                    metrics_type
                    + "_"
                    + format(horizon, ".0f")
                    + "_"
                    + format(t_zero, ".0f")
                )
                if metrics_type == "ATR":
                    metrics_dict = calc_atr(
                        config, t_zero, horizon, dtemp_dict
                    )
                elif metrics_type == "AGWP":
                    metrics_dict = calc_agwp(config, t_zero, horizon, rf_dict)
                elif metrics_type == "AGTP":
                    metrics_dict = calc_agtp(
                        config, t_zero, horizon, dtemp_dict
                    )
                else:
                    pass
                out_dict[key] = metrics_dict
    return out_dict


def calc_atr(
    config: dict, t_zero: float, horizon: float, dtemp_dict: dict
) -> dict:
    """
    Calculates Average Temperature Response (ATR) climate metrics
    for each species and the total

    Args:
        config (dict): Configuration from config file
        t_zero (float): start year for metrics calculation
        horizon (float): time horizon in years
        dtemp_dict (dict): Dictionary containing temperature changes for each species

    Returns:
        dict: Dictionary containing ATR values, keys are species and total
    """
    time_config = config["time"]["range"]
    delta_t = time_config[2]
    dtemp_metrics_dict = get_metrics_dict(config, t_zero, horizon, dtemp_dict)
    # Calcultate ATR for temperature array
    #
    # Dallara, E. S., Kroo, I. M., & Waitz, I. A. (2011).
    # Metric for comparing lifetime average climate impact of aircraft.
    # AIAA journal, 49(8), 1600-1613. http://dx.doi.org/10.2514/1.J050763
    atr_dict = {}
    for spec, dtemp_arr in dtemp_metrics_dict.items():
        atr = 0
        for dtemp in dtemp_arr:
            atr = atr + (dtemp / horizon) * delta_t
        atr_dict[spec] = atr
    #  Calcultate total ATR (sum of all species)
    atr_dict["total"] = sum(atr_dict.values())
    return atr_dict


def calc_agwp(
    config: dict, t_zero: float, horizon: float, rf_dict: dict
) -> dict:
    """
    Calculates the Absolute Global Warming Potential (AGWP) climate metrics
    for each species and the total

    Args:
        config (dict): Configuration from the configuration file.
        t_zero (float): The start year for the metrics calculation.
        horizon (float): The time horizon in years.
        rf_dict (dict): A dictionary containing the RF values for
            each species.

    Returns:
        dict: A dictionary containing the AGWP values for each species and
            the total.
    """
    # Rodhe, H. (1990). A comparison of the contribution of various gases
    # to the greenhouse effect. Science, 248(4960), 1217-1219.
    # http://dx.doi.org/10.1126/science.248.4960.1217
    time_config = config["time"]["range"]
    delta_t = time_config[2]
    rf_metrics_dict = get_metrics_dict(config, t_zero, horizon, rf_dict)
    agwp_dict = {}
    for spec, rf_arr in rf_metrics_dict.items():
        agwp = 0
        for rf in rf_arr:
            agwp = agwp + rf * delta_t
        agwp_dict[spec] = agwp
    # Calculate total AGWP (sum of all species)
    agwp_dict["total"] = sum(agwp_dict.values())
    return agwp_dict


def calc_agtp(
    config: dict, t_zero: float, horizon: float, dtemp_dict: dict
) -> dict:
    """
    Calculates the Absolute Global Temperature Change Potential (AGTP)
    climate metrics for each species and the total

    Args:
        config (dict): Configuration from the configuration file.
        t_zero (float): The start year for the metrics calculation.
        horizon (float): The time horizon in years.
        dtemp_dict (dict): A dictionary containing the temperature changes for
            each species.

    Returns:
        dict: A dictionary containing the AGTP values for each species and
            the total.
    """
    # Shine, K. P., Fuglestvedt, J. S., Hailemariam, K., & Stuber, N. (2005).
    # Alternatives to the global warming potential for comparing climate impacts
    # of emissions of greenhouse gases. Climatic change, 68(3), 281-302.
    # https://doi.org/10.1007/s10584-005-1146-9
    dtemp_metrics_dict = get_metrics_dict(config, t_zero, horizon, dtemp_dict)
    agtp_dict = {}
    for spec, dtemp_arr in dtemp_metrics_dict.items():
        agtp = dtemp_arr[-1]
        agtp_dict[spec] = agtp
    # Calculate total AGTP (sum of all species)
    agtp_dict["total"] = sum(agtp_dict.values())
    return agtp_dict


def get_metrics_dict(
    config: dict, t_zero: float, horizon: float, resp_dict: dict
) -> dict:
    """
    Get subset of timeseries dictionary: only for years in time_metrics

    Args:
        config (dict): Configuration from config file
        t_zero (float): start year for metrics calculation
        horizon (float): time horizon in years
        resp_dict (dict): Dictionary containing response (RF or dtemp)
            values for each species

    Returns:
        dict: Dictionary containig metrics values only for years in time_metrics,
            keys are species (and total)
    """
    time_config = config["time"]["range"]
    time_range = np.arange(
        time_config[0], time_config[1], time_config[2], dtype=int
    )
    delta_t = time_config[2]
    #  Metrics time range
    time_metrics = np.arange(t_zero, (t_zero + horizon), delta_t)
    # Get values in resp_dict for years in time_metrics
    i = 0
    index_arr = []
    for year_config in time_range:
        if year_config in time_metrics:
            index_arr.append(i)
        else:
            pass
        i = i + 1
    resp_metrics_dict = {}
    for spec, resp_arr in resp_dict.items():
        resp_metrics_arr = np.zeros(len(time_metrics))
        i = 0
        for index in index_arr:
            resp_metrics_arr[i] = resp_arr[index]
            i = i + 1
        resp_metrics_dict[spec] = resp_metrics_arr
    return resp_metrics_dict
