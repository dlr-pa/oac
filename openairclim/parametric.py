"""
Parametric scenario: Adapt emissions of CO2 and RF of other species.

Post-processing approach after:

Ahmad Saleh Walie, Mitigation of aviation's climate impact:
a scenario-based parametric study in OpenAirClim
AE4997: Educational Research Project, TU Delft, 2025

Refactoring and integration of code by Stefan Völk.
"""

import logging


# Default values for parametric factors from
# Castino (2024): https://doi.org/10.5194/gmd-17-4031-2024
RATIO_DIC_D = {
    "CO2": 1.0812,
    "O3": 0.9404,
    "CH4": 1.2017,
    "H2O": 0.9339,
    "cont": 0.7133,
    "PMO": 1,
}


def _get_factor(config, spec):
    factor = config.get("parametric", {}).get(spec)
    if factor is None or float(factor) < 0:
        logging.info(
            "Invalid or missing %s parametric factor. Using default value.", spec
        )
        factor = RATIO_DIC_D[spec]
    else:
        factor = float(factor)
    return factor


def adapt_co2_emission(config: dict, emis_interp_dict: dict) -> dict:
    """Adapt CO2 emission array by applying multiplication factor from the config

    Args:
        config (dict): Configuration dictionary from config file
        emis_interp_dict (dict): Dictionary of emission time series arrays,
            interpolated over time_range, keys are species names

    Raises:
        KeyError: if missing CO2 in dictionary of emission arrays

    Returns:
        dict: Updated dictionary of emission arrays with adapted CO2 emissions
    """
    if "CO2" in emis_interp_dict:
        emis_interp_dict["CO2"] = emis_interp_dict["CO2"] * _get_factor(config, "CO2")
    else:
        raise KeyError("Parametric scenario is enabled, but no CO2 emissions found.")
    return emis_interp_dict


def adapt_rf(config: dict, rf_interp_dict: dict, spec_lst: list) -> dict:
    """Adapt Radiative Forcing arrays by applying multiplication factor from the config

    Args:
        config (dict): Configuration dictionary from config file
        rf_interp_dict (dict): Dictionary of RF time series arrays,
            interpolated over time_range, keys are species names
        spec_lst (list): list of strings, species names

    Returns:
        dict: Updated dictionary of RF arrays
            with adapted values for species in spec_lst
    """
    for spec in spec_lst:
        rf_interp_dict[spec] = rf_interp_dict[spec] * _get_factor(config, spec)
    return rf_interp_dict
