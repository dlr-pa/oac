"""Calculates temperature changes for each species and scenario."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# CONSTANTS
#
# from Boucher & Reddy (2008)
# https://doi.org/10.1016/j.enpol.2007.08.039
C_ARR = [0.631, 0.429]  # in K / (W m-2)
D_ARR = [8.4, 409.5]  # in years


def calc_dtemp(config: dict, spec: str, rf_dict: dict) -> dict:
    """Calculates the temperature changes for a single species.

    Args:
        config (dict): Configuration dictionary from config
        spec (str): species
        rf_dict (dict): Dictionary of :class:`numpy.ndarray` of radiative
            forcing values for time range as defined in config

    Returns:
        dict: Dictionary of :class:`numpy.ndarray` of temperature values for
        time range as defined in config

    Raises:
        KeyError: If temperature method is unknown.
    """
    rf_arr = rf_dict[spec]
    if config["temperature"]["method"] == "Boucher&Reddy":
        dtemp_arr = calc_dtemp_br2008(config, spec, rf_arr)
        return {spec: dtemp_arr}
    else:
        msg = "Method for temperature change calculation is not valid."
        logger.warning(msg)
        raise KeyError(msg)


def calc_dtemp_br2008(config: dict, spec: str, rf_arr: np.ndarray) -> np.ndarray:
    """Calculates temperature changes after Boucher and Reddy (2008).

    https://doi.org/10.1016/j.enpol.2007.08.039

    Args:
        config (dict): configuration dictionary from config
        spec (str): species
        rf_arr (numpy.ndarray): array of radiative forcing values

    Returns:
        numpy.ndarray: array of temperature values
    """
    time_config = config["time"]["range"]
    time_range = np.arange(time_config[0], time_config[1], time_config[2], dtype=int)
    delta_t = time_config[2]

    # calculate lambda for the species
    lambda_co2_input = config["temperature"]["CO2"]["lambda"]
    lambda_co2_br2008 = 1.06  # default value (sum of c_i)
    if spec == "CO2":
        efficacy = 1.0
    else:
        efficacy = config["temperature"][spec]["efficacy"]
    lambda_spec = efficacy * lambda_co2_input

    # Discrete convolution of the forcing with the impulse response. The kernel
    # only depends on the time lag, so it is evaluated once for all lags.
    lags = time_range - time_range[0]
    kernel = np.asarray(calc_delta_temp_br2008(lags, C_ARR, D_ARR))
    dtemp_arr = (
        (lambda_spec / lambda_co2_br2008)
        * np.convolve(np.asarray(rf_arr, dtype=float), kernel)[: len(time_range)]
        * delta_t
    )
    return dtemp_arr


def calc_delta_temp_br2008(
    t: float | np.ndarray, c_arr: list[float], d_arr: list[float]
) -> float | np.ndarray:
    """Impulse response function according to Boucher and Reddy (2008), Appendix A.

    Args:
        t (float | numpy.ndarray): time, scalar or array
        c_arr (list[float]): parameter array of impulse response function,
            Table A1: ci in (K / (W m-2))
        d_arr (list[float]): parameter array of impulse response function,
            Table A1: di in (years)

    Returns:
        float | numpy.ndarray: temperature change according to the Boucher and
        Reddy (2008) model, same shape as ``t``
    """
    c = np.asarray(c_arr, dtype=float)
    d = np.asarray(d_arr, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    return np.sum((c / d) * np.exp(-t_arr[..., np.newaxis] / d), axis=-1)
