"""
Calculates responses for each species and scenario
"""

import numpy as np
from openairclim.interpolate_space import calc_weights


# CONSTANTS
#
# Conversion table: out_species (response species) to inv_species (inventory species)
OUT_INV_DICT = {"CO2": "CO2", "H2O": "H2O", "O3": "NOx"}
#
# Correction / Normalization factors
#
# Correction H2O emission --> H2O concentration
# Correction factor from AirClim
# TODO Check correction factor
# no correction seconds in year? units mol/mol or ppbv?
# CORR_CONC_H2O = 1.0 / 125.0e-15
# assuming ppbv as units for response surfaces:
CORR_CONC_H2O = 1.0e-9 / 125.0e-15
#
#
# Correction NOx emission --> O3 concentration
# EMAC input setting: emission strength for box regions was
# eps = 6.877E-16 kg(NO)/kg(air)/s
# This translates to an emission strength for one year:
# eps * (365 * 24 * 3600)
#
# TODO Check if air mass normalization properly implemented --> calc_weights()
#
CORR_CONC_O3 = 1.0 / (6.877e-16 * 365 * 24 * 3600)
#
# CORR_RF_H2O from AirClim, normalization of response
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
# TODO Update correction factor
CORR_RF_O3 = 1.0


def calc_resp(spec: str, inv, weights) -> np.ndarray:
    """
    Calculate response from response surfaces, emission inventories
    and pre-computed weighting parameters.

    Args:
        spec (str): Name of response species
        inv (xarray.Dataset): Emission inventory data
        weights (xarray.Dataset): Dataset with weighting parameters

    Returns:
        np.ndarray: Response array
    """
    inv_spec = OUT_INV_DICT[spec]
    inv_arr = inv[inv_spec].values
    weights_arr = weights["weights"].values
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
                if nox == "NO":
                    corr = CORR_CONC_O3
                elif nox == "NO2":
                    corr = CORR_CONC_O3 * (30.0 / 46.0)
        elif resp_type == "rf":
            if spec == "H2O":
                corr = CORR_RF_H2O
            elif spec == "O3":
                if nox == "NO":
                    corr = CORR_RF_O3
                elif nox == "NO2":
                    corr = CORR_RF_O3 * (30.0 / 46.0)
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
