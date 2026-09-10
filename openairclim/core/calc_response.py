"""Calculates responses for each species and scenario."""

import logging

import numpy as np
import xarray as xr

from .calc_ch4 import calc_pmo_rf
from .calc_swv import calc_swv_mass_conc, calc_swv_rf
from .config_model import OUT_TO_INV_REQUIRED
from .interpolate_space import calc_weights

logger = logging.getLogger(__name__)


# CONSTANTS
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
# De Forster, P. M., Ponater, M., & Zhong, W. Y. (2001). Testing broadband
# radiation schemes for their ability to calculate the radiative forcing and
# temperature response to stratospheric water vapour and ozone changes.
# Meteorologische Zeitschrift, 10(5), 387-393.
# see also: Fichter, C. (2009). Climate impact of air traffic emissions in
# dependency of the emission location and altitude. DLR. PhD thesis, Chapter 6.2
#
# CORR_RF_H2O = 1.5 / (31536000.0 * 125.0e-15)
CORR_RF_H2O = 380517.5038
#
# Correction factor for RF O3, tagging
CORR_RF_O3 = CORR_CONC_O3
# Warning message if tagging response surface is used
if CORR_RF_O3 == CORR_CONC_O3:
    logger.warning("O3 response surface is not validated!")
#
# Correction factor for RF O3, AirClim (perturbation)
# CORR_RF_O3 = 1.0 / (31536000.0 * 0.45e-15)
# CORR_RF_O3 = 70466204.41
#
# Correction factor for tau CH4, tagging
CORR_TAU_CH4 = CORR_CONC_O3

# Correction factors that do not scale with the NOx assumption
_CORR_FACTORS_FIXED = {
    ("conc", "H2O"): CORR_CONC_H2O,
    ("rf", "H2O"): CORR_RF_H2O,
}
# Correction factors that scale with the NOx assumption (corr_nox)
_CORR_FACTORS_NOX_SCALED = {
    ("conc", "O3"): CORR_CONC_O3,
    ("rf", "O3"): CORR_RF_O3,
    ("tau", "CH4"): CORR_TAU_CH4,
}


def calc_resp(spec: str, inv: xr.Dataset, weights: xr.Dataset) -> np.ndarray:
    """Calculate response from response surfaces and emission inventories.

    Uses pre-computed weighting parameters.

    Args:
        spec (str): Name of response species
        inv (xarray.Dataset): Emission inventory data
        weights (xarray.Dataset): Dataset with weighting parameters
    Raises:
        KeyError: if species not valid

    Returns:
        numpy.ndarray: Response array
    """
    inv_spec = OUT_TO_INV_REQUIRED[spec]
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


def _calc_corr_factor(resp_type: str, spec: str, corr_nox: float) -> float:
    """Determines the response correction (normalisation) factor.

    Args:
        resp_type (str): Response type, one of "conc", "rf", "tau"
        spec (str): Name of response species
        corr_nox (float): NOx correction factor (1.0 or :data:`CORR_NO2`)

    Returns:
        float: Correction factor for the given response type and species.
        Defaults to 1.0 if no correction is required.

    Raises:
        ValueError: If resp_type is not valid
    """
    if resp_type not in ("conc", "rf", "tau"):
        raise ValueError("resp_type not valid")
    key = (resp_type, spec)
    if key in _CORR_FACTORS_NOX_SCALED:
        return _CORR_FACTORS_NOX_SCALED[key] * corr_nox
    return _CORR_FACTORS_FIXED.get(key, 1.0)


def calc_resp_all(config: dict, resp_dict: dict, inv_dict: dict) -> dict:
    """Loop calc_response function over elements in response dictionary.

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
    else:
        raise KeyError("Invalid NOx assumption in config['species']['nox'].")
    out_dict = {}
    for spec, resp in resp_dict.items():
        # resp_type (str): "conc" or "rf"
        resp_type = resp.attrs["resp_type"]
        corr = _calc_corr_factor(resp_type, spec, corr_nox)
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


def calc_resp_sub(
    species_sub: list[str], config: dict, output_dict: dict, ac: str
) -> tuple[dict, dict]:
    """Calculates responses for specified sub-species.

    The calculation of sub-species responses depends on the results
    of main species which must be calculated and written to output beforehand.

    Args:
        species_sub (list[str]): List of sub-species names, such as 'PMO'
        config (dict): Configuration dictionary from config
        output_dict (dict): Dictionary with computed responses for main
            species, keyed by aircraft identifier
        ac (str): Aircraft identifier

    Returns:
        tuple[dict, dict]: ``rf_sub_dict``, dictionary with computed RF
        responses, and ``conc_sub_dict``, dictionary with computed
        concentration responses; keys are sub-species

    Raises:
        KeyError: If no method defined for the sub-species
    """
    # Get results computed for other species
    rf_sub_dict: dict = {}
    conc_sub_dict: dict = {}
    for spec in species_sub:
        if spec == "PMO":
            rf_pmo_dict = calc_pmo_rf(output_dict[ac])
            rf_sub_dict = rf_sub_dict | rf_pmo_dict
            logger.warning("PMO response not validated!")
        elif spec == "SWV":
            if "conc_CH4" in output_dict[ac]:
                mass_swv_dict = {}
                conc_swv_dict = {}
                mass_swv_dict["SWV"], conc_swv_dict["SWV"], _ = calc_swv_mass_conc(
                    output_dict[ac]["conc_CH4"], config,
                )

                rf_swv_dict = calc_swv_rf(mass_swv_dict)
                rf_sub_dict = rf_sub_dict | rf_swv_dict
                conc_sub_dict = conc_sub_dict | conc_swv_dict
            else:
                raise KeyError("SWV RF response requires a CH4 concentration")
        else:
            msg = "No method defined for sub species " + spec
            raise KeyError(msg)
    return rf_sub_dict, conc_sub_dict
