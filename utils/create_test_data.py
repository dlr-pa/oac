"""
Creates data objects for testing
"""

import sys
import os
import numpy as np
import xarray as xr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.create_artificial_inventories import ArtificialInventory


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


def create_test_inv(year=2020, size=3, ac_lst=None):
    """
    Creates an example inventory dataset for testing purposes.

    Args:
        year (int): inventory year
        size (int): The number of samples to generate.
        ac_lst (list, optional): List of aircraft identifiers (strings).

    Returns:
        xr.Dataset: An xarray dataset with random inventory data.

    """
    inv = ArtificialInventory(year, size=size, ac_lst=ac_lst).create()
    return inv


def create_test_resp_cont(method="Megill_2025", n_lat=48, n_lon=96, n_plev=39, seed=None):
    """Creates example precalculated contrail input data for testing purposes.

    Args:
        method (str, optional): Contrail calculation method.
            Options: 'Megill_2025' (default) or 'AirClim'.
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
    plev = np.sort(np.append(np.linspace(1014, 10, n_plev-1), [250]))[::-1]

    # depending on method, create test resp_cont
    # Create the data variables with random values between 0 and 1
    assert method in ["Megill_2025", "AirClim"], "Unknown contrail calculation " \
        "method. Should be one of 'Megill_2025' or 'AirClim'."

    iss = np.random.rand(n_lat, n_lon)  # independent of method

    if method == "AirClim":
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

    else:  # Megill_2025 method
        # define aircraft
        ac = [f"oac{x}" for x in range(5)]
        n_ac = len(ac)

        # create dataset
        ds_cont = xr.Dataset(
            {
                "ISS": (["lat", "lon"], iss),
            },
            coords={
                "lon": ("lon", lon),
                "lat": ("lat", lat),
                "plev": ("plev", plev),
                "AC": ("AC", ac)
            }
        )
        ds_cont.AC.attrs = {"units": "None"}

        # populate dataset
        ds_cont["ppcf"] = (
            ("AC", "plev", "lat", "lon"),
            np.random.rand(n_ac, n_plev, n_lat, n_lon)
        )
        ds_cont["g_250"] = (("AC"), np.random.rand(n_ac))
        fit_vars = ["l_1", "k_1", "x0_1", "d_1", "l_2", "k_2", "x0_2"]
        for fit_var in fit_vars:
            ds_cont[fit_var] = (("plev"), np.random.rand(n_plev))

    # add units
    ds_cont.lat.attrs = {"units": "degrees_north"}
    ds_cont.lon.attrs = {"units": "degrees_east"}
    ds_cont.plev.attrs = {"units": "hPa"}

    return ds_cont
