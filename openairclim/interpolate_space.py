"""
Interpolation and Regridding methods in the space domain
"""
# TODO Check if one of these python packages are more suitable/flexible
# for example geocat, see https://geocat-comp.readthedocs.io/en/stable/
# for pressure level interpolations geocat.comp.interpolation.interp_hybrid_to_pressure
# maybe this is a more general function: geocat.comp.interpolation.interp_multidim
# or that one: https://unidata.github.io/MetPy/latest/api/generated/metpy.interpolate.interpolate_to_points.html

from scipy.interpolate import interpn
import numpy as np
import xarray as xr
from openairclim.write_output import query_checksum_table
from openairclim.write_output import update_checksum_table


# CONSTANTS
CHECKSUM_PATH = "../cache/weights/"


def calc_weights(spec, resp, inv):
    """
    Calculate the weighting factors for a given response and inventory.

    Args:
        spec (str): Name of the species for which the weights are being calculated.
        resp (xarray.Dataset): Response dataset
        inv (xarray.Dataset): Emission inventory dataset

    Returns:
        xarray.Dataset: Dataset with weighting parameters

    """
    # Get the grid points and values from the response dataset
    grid_points = (resp.emi_lat.values, resp.emi_plev.values)
    # Transposition necessary since numpy broadcasting
    # matches dimensions from right (last dimension)
    grid_values = (
        np.divide(resp[spec].values.T, resp.emi_air_mass.values.T)
    ).T
    # Get the locations from the inventory dataset
    locations = np.column_stack((inv.lat.values, inv.plev.values))
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
    # Create the dimensions and attributes for the weights dataset
    weights_dims = ["index"]
    for dim_name in resp[spec].dims:
        if dim_name not in ["emi_lat", "emi_plev"]:
            weights_dims.append(dim_name)
    weights_dims = tuple(weights_dims)
    weights_attrs = {
        "Title": "Weighting factors",
        "Species": spec,
    }
    # Add the response type and inventory year to the attributes
    for resp_attrs_key, resp_attrs_value in resp.attrs.items():
        if resp_attrs_key == "resp_type":
            weights_attrs[resp_attrs_key] = resp_attrs_value
            if resp_attrs_value == "rf":
                weights_units = "W/m²/kg"
            elif resp_attrs_value == "conc":
                weights_units = "mol/mol/kg"
            else:
                weights_units = "undefined"
    for inv_attrs_key, inv_attrs_value in inv.attrs.items():
        if inv_attrs_key == "Inventory_Year":
            weights_attrs[inv_attrs_key] = inv_attrs_value
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


def find_weights(spec, resp, inv):
    # TODO Debug find_weights --> cache files are newly created although already present
    """Find weighting parameters on response grid for entire emission inventory,
    response grid is 2d with dimensions lat and plev
    First, query checksum table to get pre-calculated weights from cache
    If weights not pre-calculated for resp / inv combination, execute calc_weights

    Args:
        spec (str): Name of the species
        resp (xarray): Response Dataset with lat and plev dimensions
        inv (xarray): Emission inventory Dataset

    Returns:
        xarray: Dataset with weighting parameters
    """
    checksum_path = CHECKSUM_PATH
    weights, index = query_checksum_table(spec, resp, inv)
    if weights is None:
        cache_file = checksum_path + f"{index:03}" + ".nc"
        weights = calc_weights(spec, resp, inv)
        weights.to_netcdf(cache_file)
        update_checksum_table(spec, resp, inv, cache_file)
    return weights
