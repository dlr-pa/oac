"""
Creates data objects for testing
"""

import numpy as np
import xarray as xr
from utils import create_artificial_inventories as cai


def create_test_conc_resp():
    """
    Creates an example response dataset for testing purposes
    with resp_type = "conc"

    Returns:
        xr.Dataset: A minimal response dataset with random data.
    """
    lat_arr = np.arange(-75.0, 80.0, 15.0).astype("float32")
    dim_lat = len(lat_arr)
    plev_arr = np.arange(100.0, 1000.0, 100.0).astype("float32")
    dim_plev = len(plev_arr)
    emi_lat_arr = np.array([10.0, 40.0], dtype="float32")
    emi_plev_arr = np.array([250.0, 500.0], dtype="float32")
    emi_loc_arr = np.array([["p10_250", "p10_500"], ["p40_250", "p40_500"]])
    p10_250_arr = np.random.randn(dim_lat, dim_plev).astype("float32")
    p40_250_arr = np.random.randn(dim_lat, dim_plev).astype("float32")
    p10_500_arr = np.random.randn(dim_lat, dim_plev).astype("float32")
    p40_500_arr = np.random.randn(dim_lat, dim_plev).astype("float32")
    resp = xr.Dataset(
        data_vars={
            "emi_loc": (["emi_lat", "emi_plev"], emi_loc_arr),
            "p10_250": (["lat", "plev"], p10_250_arr),
            "p40_250": (["lat", "plev"], p40_250_arr),
            "p10_500": (["lat", "plev"], p10_500_arr),
            "p40_500": (["lat", "plev"], p40_500_arr),
        },
        coords={
            "lat": lat_arr,
            "plev": plev_arr,
            "emi_lat": emi_lat_arr,
            "emi_plev": emi_plev_arr,
        },
        attrs={"resp_type": "conc"},
    )
    return resp


def create_test_rf_resp():
    """
    Creates an example response dataset for testing purposes
    with resp_type = "rf"

    Returns:
        xr.Dataset: A minimal response dataset with random data.
    """
    emi_lat_arr = np.array([10.0, 40.0], dtype="float32")
    emi_plev_arr = np.array([250.0, 500.0], dtype="float32")
    emi_loc_arr = np.array([["p10_250", "p10_500"], ["p40_250", "p40_500"]])
    h2o_arr = np.random.rand(len(emi_lat_arr), len(emi_plev_arr)).astype(
        "float32"
    )
    emi_air_mass_arr = np.ones_like(h2o_arr)
    resp = xr.Dataset(
        data_vars={
            "emi_air_mass": (["emi_lat", "emi_plev"], emi_air_mass_arr),
            "emi_loc": (["emi_lat", "emi_plev"], emi_loc_arr),
            "H2O": (["emi_lat", "emi_plev"], h2o_arr),
        },
        coords={
            "emi_lat": emi_lat_arr,
            "emi_plev": emi_plev_arr,
        },
        attrs={"resp_type": "rf"},
    )
    return resp


def create_test_inv(year=2020, size=3):
    """
    Creates an example inventory dataset for testing purposes.

    Args:
        year (int): inventory year
        size (int): The number of samples to generate.

    Returns:
        xr.Dataset: An xarray dataset with random inventory data.

    """
    inv = cai.ArtificialInventory(year, size=size).create()
    return inv
