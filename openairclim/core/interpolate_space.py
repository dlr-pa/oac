"""Interpolation and Regridding methods in the space domain."""

# TODO Check if one of these python packages are more suitable/flexible
# for example geocat, see https://geocat-comp.readthedocs.io/en/stable/
# for pressure level interpolations geocat.comp.interpolation.interp_hybrid_to_pressure
# maybe this is a more general function: geocat.comp.interpolation.interp_multidim
# or that one: https://unidata.github.io/MetPy/latest/api/generated/metpy.interpolate.interpolate_to_points.html

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interpn

from .write_output import query_checksum_table, update_checksum_table

# CONSTANTS
CHECKSUM_PATH = "../cache/weights/"


def unique_lat_plev_vals(inv: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Find the unique (lat, plev) locations of an emission inventory.

    Although OpenAirClim's inventories are flat, they were often built from
    grids and hence contain far fewer unique (lat, plev) pairs than emission
    points. Interpolating over the unique pairs only and pushing the results
    back is significantly cheaper than interpolatinng over every point.

    Args:
        inv (xarray.Dataset): Emission inventory dataset

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``locations``, array of shape
        ``(n_unique, 2)`` with columns lat and plev, and ``inverse``, integer
        array of shape ``(n_points,)`` such that
        ``locations[inverse]`` reproduces the inventory's (lat, plev) pairs.
    """
    # use a hash-based factorisation, which is O(n)
    lat_codes, lat_uniq = pd.factorize(inv.lat.values, use_na_sentinel=False)
    plev_codes, plev_uniq = pd.factorize(inv.plev.values, use_na_sentinel=False)
    key = lat_codes.astype(np.int64) * len(plev_uniq) + plev_codes
    inverse, key_uniq = pd.factorize(key)
    locations = np.column_stack(
        (lat_uniq[key_uniq // len(plev_uniq)], plev_uniq[key_uniq % len(plev_uniq)])
    )
    return locations, inverse.reshape(-1)


def calc_weights(
    spec: str,
    resp: xr.Dataset,
    inv: xr.Dataset,
    unique_locs: tuple[np.ndarray, np.ndarray] | None = None,
) -> xr.Dataset:
    """Calculate the weighting factors for a given response and inventory.

    Args:
        spec (str): Name of the species for which the weights are being
            calculated.
        resp (xarray.Dataset): Response dataset
        inv (xarray.Dataset): Emission inventory dataset
        unique_locs (tuple[numpy.ndarray, numpy.ndarray] | None): Output of
            :func:`~openairclim.core.interpolate_space.unique_lat_plev_vals`
            for ``inv``. If given, the response is interpolated on the unique
            locations only. If ``None``, every inventory point is interpolated
            directly.

    Returns:
        xarray.Dataset: Dataset with weighting parameters
    """
    # Get the grid points and values from the response dataset
    grid_points = (resp.emi_lat.values, resp.emi_plev.values)
    # Transposition necessary since numpy broadcasting
    # matches dimensions from right (last dimension)
    grid_values = (np.divide(resp[spec].values.T, resp.emi_norm.values.T)).T
    # Get the locations from the inventory dataset
    if unique_locs is None:
        locations = np.column_stack((inv.lat.values, inv.plev.values))
        inverse = None
    else:
        locations, inverse = unique_locs
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
    if inverse is not None:
        weights_arr = weights_arr[inverse]
    # Create the dimensions and attributes for the weights dataset
    weights_dims_lst = ["index"]
    weights_dims_lst.extend(
        str(dim_name)
        for dim_name in resp[spec].dims
        if dim_name not in ["emi_lat", "emi_plev"]
    )
    weights_dims = tuple(weights_dims_lst)
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
            elif resp_attrs_value == "tau":
                weights_units = "1/yr/kg"
            else:
                weights_units = "undefined"
    if "Inventory_Year" in inv.attrs:
        weights_attrs["Inventory_Year"] = inv.attrs["Inventory_Year"]
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


def find_weights(spec: str, resp: xr.Dataset, inv: xr.Dataset) -> xr.Dataset:
    # TODO Debug find_weights --> cache files are newly created although already present
    """Find weighting parameters on response grid for entire emission inventory.

    Response grid is 2d with dimensions lat and plev. First, query checksum
    table to get pre-calculated weights from cache. If weights are not
    pre-calculated for the resp / inv combination, execute calc_weights.

    Args:
        spec (str): Name of the species
        resp (xarray.Dataset): Response Dataset with lat and plev dimensions
        inv (xarray.Dataset): Emission inventory Dataset

    Returns:
        xarray.Dataset: Dataset with weighting parameters
    """
    checksum_path = CHECKSUM_PATH
    weights, index = query_checksum_table(spec, resp, inv)
    if weights is None:
        cache_file = checksum_path + f"{index:03}" + ".nc"
        weights = calc_weights(spec, resp, inv)
        weights.to_netcdf(cache_file)
        update_checksum_table(spec, resp, inv, cache_file)
    return weights
