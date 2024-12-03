"""
Calculates responses for each species and scenario
"""

import logging
import numpy as np
from openairclim.interpolate_space import calc_weights
from openairclim.read_netcdf import get_results
from openairclim.calc_ch4 import calc_pmo_rf


# CONSTANTS
#
# Conversion table: out_species (response species) to inv_species (inventory species)
OUT_INV_DICT = {"CO2": "CO2", "H2O": "H2O", "O3": "NOx", "CH4": "NOx"}
#
# CORRECTION (normalization) factors
#
# Correction H2O emission --> H2O concentration
# Correction factor from AirClim
# TODO Check correction factor
# no correction seconds in year? units mol/mol or ppbv?
# CORR_CONC_H2O = 1.0 / 125.0e-15
# assuming ppbv as units for response surfaces:
CORR_CONC_H2O = 1.0e-9 / 125.0e-15
#
# Correction factor for NO2 inventory emissions (instead NO)
CORR_NO2 = 30.0 / 46.0
#
# Correction NOx emission --> O3 concentration
# EMAC input setting: emission strength for box regions was
# eps = 6.877E-16 kg(NO)/kg(air)/s
# This translates to an emission strength for one year:
# eps * (365 * 24 * 3600)
#
# Correction factor for O3 concentration, tagging
# TODO Check if air mass normalization properly implemented --> calc_weights()
CORR_CONC_O3 = 1.0 / (6.877e-16 * 365 * 24 * 3600)
#
# Correction factor for RF H2O, AirClim (perturbation)
#
# Scaling of water vapour radiative forcing by 1.5 according to findings from
# De Forster, P. M., Ponater, M., & Zhong, W. Y. (2001). Testing broadband radiation schemes
# for their ability to calculate the radiative forcing and temperature response to
# stratospheric water vapour and ozone changes. Meteorologische Zeitschrift, 10(5), 387-393.
# see also: Fichter, C. (2009). Climate impact of air traffic emissions in dependency of the
# emission location and altitude. DLR. PhD thesis, Chapter 6.2
#
# CORR_RF_H2O = 1.5 / (31536000.0 * 125.0e-15)
CORR_RF_H2O = 380517.5038
#
# Correction factor for RF O3, tagging
CORR_RF_O3 = CORR_CONC_O3
#
# Correction factor for RF O3, AirClim (perturbation)
# CORR_RF_O3 = 1.0 / (31536000.0 * 0.45e-15)
# CORR_RF_O3 = 70466204.41
#
# Correction factor for tau CH4, tagging
CORR_TAU_CH4 = CORR_CONC_O3


def calc_resp(spec: str, inv, weights) -> np.ndarray:
    """
    Calculate response from response surfaces, emission inventories
    and pre-computed weighting parameters.

    Args:
        spec (str): Name of response species
        inv (xarray.Dataset): Emission inventory data
        weights (xarray.Dataset): Dataset with weighting parameters
    Raises:
        KeyError: if species not valid

    Returns:
        np.ndarray: Response array
    """
    inv_spec = OUT_INV_DICT[spec]
    inv_arr = inv[inv_spec].values
    weights_arr = weights["weights"].values
    if spec in ["H2O", "O3", "CH4"]:
        pass
    else:
        raise KeyError("calculating response: species not valid")
    # Elememt-wise multiplication of inventory emissions and weights
    out_arr = (np.multiply(inv_arr.T, weights_arr.T)).T
    # Sum over index axis (all steps in emission inventory)
    out_arr = np.sum(out_arr, axis=0)
    return out_arr


def calc_resp_all(config, resp_dict, inv_dict):
    """Loop calc_response function over elements in response dictionary

    Args:
        config (dict): Configuration dictionary from config
        resp_dict (dict): Dictionary of response xarray Datasets, keys are species
        inv_dict (dict): Dictionary of inventory xarray Datasets, keys are years

    Returns:
        dict: Dictionary of dictionary of numpy arrays of computed responses,
            keys are species and inventory years
    """
    # "NO" or "NO2" in emission inventory
    nox = config["species"]["nox"]
    if nox == "NO":
        corr_nox = 1.0
    elif nox == "NO2":
        corr_nox = CORR_NO2
    # default correction factor
    corr = 1.0
    out_dict = {}
    for spec, resp in resp_dict.items():
        # resp_type (str): "conc" or "rf"
        resp_type = resp.attrs["resp_type"]
        if resp_type in "conc":
            if spec == "H2O":
                corr = CORR_CONC_H2O
            elif spec == "O3":
                corr = CORR_CONC_O3 * corr_nox
        elif resp_type == "rf":
            if spec == "H2O":
                corr = CORR_RF_H2O
            elif spec == "O3":
                # Warning message if tagging response surface is used
                if CORR_RF_O3 == CORR_CONC_O3:
                    logging.warning("O3 response surface is not validated!")
                corr = CORR_RF_O3 * corr_nox
        elif resp_type == "tau":
            if spec == "CH4":
                corr = CORR_TAU_CH4 * corr_nox
        else:
            raise ValueError("resp_type not valid")
        out_inv_dict = {}
        for inv in inv_dict.values():
            year = inv.attrs["Inventory_Year"]
            weights = calc_weights(spec, resp, inv)
            # weights = find_weights(spec, resp, inv)
            out_arr = corr * calc_resp(spec, inv, weights)
            # conc = np.sum(conc_arr)
            out_inv_dict[year] = out_arr
        out_dict[spec] = out_inv_dict
    return out_dict


def calc_resp_sub(config, species_sub):
    """
    Calculates responses for specified sub-species.
    The calculation of sub-species responses depends on the results
    of main species which must be calculated and written to output beforehand.

    Args:
        config (dict): Configuration dictionary
        species_sub (list[str]): List of sub-species names, such as 'PMO'

    Returns:
        dict: Dictionary with computed responses, keys are sub-species

    Raises:
        KeyError: If no method defined for the sub-species
    """
    # Get results computed for other species
    _emis_dict, _conc_dict, rf_dict, _dtemp_dict = get_results(config)
    rf_sub_dict = {}
    for spec in species_sub:
        if spec == "PMO":
            rf_pmo_dict = calc_pmo_rf(rf_dict)
            rf_sub_dict = rf_sub_dict | rf_pmo_dict
            logging.warning("PMO response not validated!")
        else:
            msg = "No method defined for sub species " + spec
            raise KeyError(msg)
    return rf_sub_dict
