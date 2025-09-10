"""Create netCDF files controlling time evolution: time scaling and time normalization"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# GENERAL CONSTANTS
OUT_PATH = "../example/input/"

# SCALING CONSTANTS
SCALING_TIME = np.arange(1990, 2200, 1)
SCALING_ARR = np.sin(SCALING_TIME * 0.2) * 0.6 + 1.0
SCALING_ARR = SCALING_ARR.astype("float32")

# NORMALIZATION CONSTANTS
NORM_TIME = np.array(
    [
        2020,
        2025,
        2030,
        2035,
        2040,
        2045,
        2050,
        2055,
        2060,
        2065,
        2070,
        2075,
        2080,
        2085,
        2090,
        2095,
        2100,
        2105,
        2110,
        2115,
        2120,
    ]
)
# Reference for fuel consumption until year 2050:
# Energy Insights’ Global Energy Perspective, Reference Case A3 October 2020; IATA; ICAO
# (fuel consumption values beyond 2050 are customized)
FUEL_ARR = np.array(
    [
        215,
        364,
        407,
        446,
        479,
        503,
        520,
        536,
        552,
        568,
        585,
        603,
        621,
        639,
        658,
        678,
        699,
        720,
        741,
        763,
        786,
    ]
).astype("float32")
EI_CO2_ARR = 3.115 * np.ones(len(NORM_TIME), dtype="float32")
EI_H2O_ARR = 1.25 * np.ones(len(NORM_TIME), dtype="float32")
DIS_PER_FUEL_ARR = 0.3 * np.ones(len(NORM_TIME), dtype="float32")

# TIME SCALING


def plot_time_scaling(scaling_time: np.ndarray, scaling_arr: np.ndarray):
    """
    Plots the time scaling factors.

    Args:
        scaling_time (np.ndarray): The time values for the scaling factors.
        scaling_arr (np.ndarray): The scaling factors to plot.

    Returns:
        None

    """
    _fig, ax = plt.subplots()
    ax.plot(scaling_time, scaling_arr)
    ax.set_xlabel("year")
    ax.set_ylabel("scaling factor")
    plt.show()


def create_time_scaling_xr(
    scaling_time: np.ndarray, scaling_arr: np.ndarray
) -> xr.Dataset:
    """
    Create an xarray dataset containing time scaling factors.

    Args:
        scaling_time (np.ndarray): The time values for the scaling factors.
        scaling_arr (np.ndarray): The scaling factors to plot.

    Returns:
        xr.Dataset: The xarray dataset containing the time scaling factors.

    """
    evolution = xr.Dataset(
        data_vars=dict(scaling=(["time"], scaling_arr)),
        coords=dict(time=scaling_time),
    )
    evolution.time.attrs = {"units": "years"}
    evolution.scaling.attrs = {"species": "all"}
    evolution.attrs = dict(
        Title="Time scaling example",
        Convention="CF-XXX",
        Type="scaling",
        Author="Stefan Völk",
        Contact="stefan.voelk@dlr.de",
    )
    return evolution


# TIME NORMALIZATION


def create_time_normalization_xr(
    time_arr: np.ndarray,
    fuel_arr: np.ndarray,
    ei_co2_arr: np.ndarray,
    ei_h2o_arr: np.ndarray,
    dis_per_fuel_arr: np.ndarray,
) -> xr.Dataset:
    """Create an xarray dataset containing normalization factors

    Args:
        time_arr (np.ndarray): Time values (years)
        fuel_arr (np.ndarray): Fuel consumption
        ei_co2_arr (np.ndarray): Emission indices for CO2
        ei_h2o_arr (np.ndarray): Emission indices for H2O
        dis_per_fuel_arr (np.ndarray): Distance per fuel

    Returns:
        xr.Dataset: The xarray dataset containing the normalization factors
    """
    evolution = xr.Dataset(
        data_vars=dict(
            fuel=(["time"], fuel_arr),
            EI_CO2=(["time"], ei_co2_arr),
            EI_H2O=(["time"], ei_h2o_arr),
            dis_per_fuel=(["time"], dis_per_fuel_arr),
        ),
        coords=dict(time=time_arr),
    )
    evolution.time.attrs = {"units": "years"}
    evolution.fuel.attrs = {
        "long_name": "fuel consumption",
        "units": "Tg yr-1",
    }
    evolution.EI_CO2.attrs = {"long_name": "CO2 emission index", "units": ""}
    evolution.EI_H2O.attrs = {"long_name": "H2O emission index", "units": ""}
    evolution.dis_per_fuel.attrs = {
        "long_name": "distance per fuel",
        "units": "km kg-1",
    }
    evolution.attrs = dict(
        Title="Time normalization example",
        Convention="CF-XXX",
        Type="norm",
        Author="Stefan Völk",
        Contact="stefan.voelk@dlr.de",
    )
    return evolution


def plot_time_norm(evolution):
    """Plot normalized values

    Args:
        evolution (xr.Dataset): The xarray Dataset containing the normalization factors.

    Returns:
        None
    """
    co2_emi_arr = np.multiply(evolution.fuel.values, evolution.EI_CO2.values)

    _fig, axs = plt.subplots(nrows=2)
    axs[0].grid(True)
    axs[1].grid(True)
    evolution.fuel.plot.line("-o", ax=axs[0])
    axs[1].plot(evolution.time.values, co2_emi_arr, "-o")
    axs[1].set_xlabel("time [years]")
    axs[1].set_ylabel("CO2 emissions [Tg]")
    plt.show()


# WRITE OUTPUT netCDF
def convert_xr_to_nc(ds: xr.Dataset, file_name: str, out_path: str = OUT_PATH):
    """
    Convert a xarray dataset to a netCDF file and write to out_path.
    Create out_path if not existing.

    Args:
        ds (xr.Dataset): The xarray dataset to write to netCDF.
        file_name (str): The name of the output file, including the extension.
        out_path (str, optional): The path to the output directory.
            Defaults to OUT_PATH.

    Returns:
        None
    """
    os.makedirs(out_path, exist_ok=True)
    out_file = out_path + file_name + ".nc"  # "time_[scaling|norm]_example.nc"
    ds.to_netcdf(out_file)


if __name__ == "__main__":
    scaling_ds = create_time_scaling_xr(SCALING_TIME, SCALING_ARR)
    convert_xr_to_nc(scaling_ds, "time_scaling_example")
    plot_time_scaling(SCALING_TIME, SCALING_ARR)
    norm_ds = create_time_normalization_xr(
        NORM_TIME, FUEL_ARR, EI_CO2_ARR, EI_H2O_ARR, DIS_PER_FUEL_ARR
    )
    convert_xr_to_nc(norm_ds, "time_norm_examplexxxxxxxxxxx")
    plot_time_norm(norm_ds)
