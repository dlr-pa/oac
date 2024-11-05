"""
Creates data objects for testing
"""

import numpy as np
import xarray as xr
import utils.create_artificial_inventories as cai


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


def create_test_resp_cont(n_lat=48, n_lon=96, n_plev=39, seed=None):
    """Creates example precalculated contrail input data for testing purposes.

    Args:
        n_lat (int, optional): Number of latitude values. Defaults to 48.
        n_lon (int, optional): Number of longitude values. Defaults to 96.
        n_plev (int, optional): Number of pressure level values. Defaults to 39.
        seed (int, optional): Random seed.

    Returns:
        xr.Dataset: Example precalculated contrail input data.
    """

    # set random seed
    np.random.seed(seed)

    # Create the coordinates
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    lat = np.linspace(90, -90, n_lat + 2)[1:-1]  # do not include 90 or -90
    plev = np.linspace(1014, 10, n_plev)

    # Create the data variables with random values between 0 and 1
    iss = np.random.rand(n_lat, n_lon)
    sac_con = np.random.rand(n_lat, n_lon, n_plev)
    sac_lh2 = np.random.rand(n_lat, n_lon, n_plev)

    # Combine into an xarray Dataset
    ds_cont = xr.Dataset(
        {
            "ISS": (["lat", "lon"], iss),
            "SAC_CON": (["lat", "lon", "plev"], sac_con),
            "SAC_LH2": (["lat", "lon", "plev"], sac_lh2)
        },
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
            "plev": ("plev", plev)
        }
    )

    return ds_cont
