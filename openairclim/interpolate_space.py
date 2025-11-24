"""
Interpolation and Regridding methods in the space domain
"""

from scipy.interpolate import interpn
import numpy as np
import xarray as xr
from openairclim.utils import calc_theta
from openairclim.write_output import query_checksum_table
from openairclim.write_output import update_checksum_table


# CONSTANTS
CHECKSUM_PATH = "../cache/weights/"


def calc_weights(spec, resp, inv, interp_theta=False):
    """
    Calculate the weighting factors for a given response and inventory.

    Args:
        spec (str): Name of the species for which the weights are being calculated.
        resp (xr.Dataset): Response dataset
        inv (xr.Dataset): Emission inventory dataset
        interp_theta (bool, optional): Interpolate over potential temperature "theta"
            along altitude dimension. If False interpolate over pressure. Defaults to True

    Returns:
        xarray.Dataset: Dataset with weighting parameters

    """
    # Get the grid points and values from the response dataset
    if interp_theta:
        resp_alt_arr = calc_theta(resp.emi_plev.values)
        inv_alt_arr = calc_theta(inv.plev.values)
    else:
        resp_alt_arr = resp.emi_plev.values
        inv_alt_arr = inv.plev.values
    grid_points = (resp.emi_lat.values, resp_alt_arr)
    # Transposition necessary since numpy broadcasting
    # matches dimensions from right (last dimension)
    grid_values = (np.divide(resp[spec].values.T, resp.emi_air_mass.values.T)).T
    # Get the locations from the inventory dataset
    locations = np.column_stack((inv.lat.values, inv_alt_arr))
    # Use the scipy.interpolate.interpn function to interpolate the response
    # data to the inventory locations
    weights_arr = interpn(
        grid_points,
        grid_values,
        locations,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    # Create the dimensions for the weights dataset
    weights_dims = ["index"]
    for dim_name in resp[spec].dims:
        if dim_name not in ["emi_lat", "emi_plev"]:
            weights_dims.append(dim_name)
    weights_dims = tuple(weights_dims)
    # Create weights attributes and get weights units
    weights_attrs, weights_units = _create_weights_attrs(spec, resp, inv)
    # Create the weights dataset
    weights_ds = xr.Dataset(
        data_vars={
            "lat": inv.lat,
            "plev": inv.plev,
            "weights": (
                weights_dims,
                weights_arr,
                {
                    "long_name": "weights",
                    "units": weights_units,
                },
            ),
        },
        attrs=weights_attrs,
    )
    return weights_ds


def _create_weights_attrs(spec: str, resp: xr.Dataset, inv: xr.Dataset):
    weights_attrs = {
        "Title": "Weighting factors",
        "Species": spec,
    }
    for inv_attrs_key, inv_attrs_value in inv.attrs.items():
        if inv_attrs_key == "Inventory_Year":
            weights_attrs[inv_attrs_key] = inv_attrs_value
    # Initialize weights_units to a default value
    weights_units = "undefined"
    # Add the response type and inventory year to the attributes
    for resp_attrs_key, resp_attrs_value in resp.attrs.items():
        if resp_attrs_key == "resp_type":
            weights_attrs[resp_attrs_key] = resp_attrs_value
            if resp_attrs_value == "rf":
                weights_units = "W/m²/kg"
            elif resp_attrs_value == "conc":
                weights_units = "mol/mol/kg"
            elif resp_attrs_value == "tau":
                weights_units = "1/yr/kg"
            else:
                weights_units = "undefined"
    return weights_attrs, weights_units


def find_weights(spec, resp, inv):
    # TODO Debug find_weights --> cache files are newly created although already present
    """Find weighting parameters on response grid for entire emission inventory,
    response grid is 2d with dimensions lat and plev
    First, query checksum table to get pre-calculated weights from cache
    If weights not pre-calculated for resp / inv combination, execute calc_weights

    Args:
        spec (str): Name of the species
        resp (xr.Dataset): Response Dataset with lat and plev dimensions
        inv (xr.Dataset): Emission inventory Dataset

    Returns:
        xr.Dataset: Dataset with weighting parameters
    """
    checksum_path = CHECKSUM_PATH
    weights, index = query_checksum_table(spec, resp, inv)
    if weights is None:
        cache_file = checksum_path + f"{index:03}" + ".nc"
        weights = calc_weights(spec, resp, inv)
        weights.to_netcdf(cache_file)
        update_checksum_table(spec, resp, inv, cache_file)
    return weights
