"""Calculates the impact of SWV"""

__author__ = "Atze Harmsen"
__email__ = "atzeharmsen@gmail.com"
__license__ = "Apache License 2.0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ambiance import Atmosphere
import xarray as xr

M_H2O = 18.01528 * 10**-3  # kg/mol
M_AIR = 28.97 * 10**-3  # kg/mol
PATH_CH4_FOR_SWV_CALC = r"../repository/ch4_for_swv_calc.nc"


def calc_swv_rf(total_swv_mass: dict):  # mass in Tg
    """
    Function to calculate the RF due to a certain SWV perturbation mass.
    Based on Pletzer (2024) The climate impact
    of hypersonic transport.

    Args:
        total_swv_mass (dict): A dict with as key "SWV" with an array containing
        the SWV mass in Tg for corresponding year.
    Raises:
        TypeError: if total_SWV_mass is not a dict
        ValueError: if the total mass is out of range of the plot of Pletzer (2024)

    Returns:
        rf_swv_dict (dict): A dict that contains the forcing due to SWV at that time
    """

    # based on the formula of Pletzer 2024
    if not isinstance(total_swv_mass, dict):
        raise TypeError("total SWV mass must be a float or integer")

    rf_swv_list = []
    # constants from Pletzer
    a = -0.00088
    b = 0.47373
    c = -0.74676
    for value in total_swv_mass["SWV"]:
        negative = False
        if value < 0:
            negative = True
            value = abs(value)
        if value > 160:
            raise ValueError("Total SWV mass out of range of Pletzer plot")
        if value < 1.6:
            # Make sure that values smaller than 1.6 Tg cause 0 impact
            # instead of impact with the wrong sign
            rf_value = 0
        else:
            rf_value = (
                a * value**2 + b * value + c
            ) / 1000  # to make it W/m2 from mW/m2
        if negative is True:
            rf_value = rf_value * -1
        rf_swv_list.append(rf_value)
    rf_swv_array = np.array(rf_swv_list)
    rf_swv_dict = {"SWV": rf_swv_array}
    return rf_swv_dict


def construct_myhre_1m_df():
    """
    A function to reproduce the HALOE data stated in figure 1 from Myhre et al., (2007)
    Radiative forcing due to stratospheric water vapour from CH4 oxidation.
    This function produces the HALOE zonal mean vertical profile of CH4
    over the period October 1991 throughout 1999.
    The function reads the data file called 'ch4_for_swv_calc.nc' in which the data is stored.

    Returns:
        df (DataFrame): a DataFrame that contains all data that is required
        by the get_griddata function to provide a proper grid

    """

    ds = xr.open_dataset(PATH_CH4_FOR_SWV_CALC)

    df = pd.DataFrame(
        {
            "latitude": ds["latitude"].values,
            "altitude": ds["altitude"].values,
            "value": ds["value"].values,
            "source": ds["source"].values.astype(str),
        }
    )
    return df


def get_volume_matrix(heights, latitudes, delta_h, delta_deg):
    """
    A function to get the volume of every box of air in an altitude-latitude graph
    The heights and latitudes arrays should have a spacing equivalent to the corresponding delta
    Args:
        heights: a np.array of heights in meters
        latitudes: a np.array of latitudes in degrees
        delta_h: the step between every height in meters
        delta_deg: the step between every latitude in degrees

    Returns (np.array): A matrix of volumes. Rows correspond
                        to altitude levels, columns to latitudes

    """
    earth_radius = 6371000.0  # Earth radius in meters
    delta_phi = np.deg2rad(delta_deg)

    # Volume of 1deg latitude x 100 m height strip integrated over all longitude
    volumes = np.zeros((len(heights), len(latitudes)))

    for i, h in enumerate(heights):
        for j, lat in enumerate(latitudes):
            volumes[i, j] = (
                2
                * np.pi
                * (earth_radius + h) ** 2
                * np.cos(np.deg2rad(lat))
                * delta_phi
                * delta_h
            )
    return volumes


def get_griddata(df, heights, latitudes, plot_data=False):
    """
    Function to transform the data to an evenly spaced and linearly interpolated grid.
    Args:
        df (DataFrame): a dataframe containing altitudes and latitudes and corresponding values
        heights (np.array): a np.array of heights in meters
        latitudes (np.array): a np.array of latitudes in degrees
        plot_data (bool): whether to plot the data or not:

    Returns:
        grid (ndarray): A grid with x axis latitudes and y axis heights and for all gridpoints an
        interpolated value of df
    """
    # Extract columns
    x = df["latitude"].values
    y = df["altitude"].values / 1000  # due to griddata
    z = df["value"].astype(float).values

    # Create grid
    xi = latitudes
    yi = heights / 1000  # due to gridddata
    x_grid, y_grid = np.meshgrid(xi, yi)

    # Interpolate values onto grid
    grid = griddata((x, y), z, (x_grid, y_grid), method="linear")

    # Make a plot if plot_data is true
    if plot_data:
        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(x_grid, y_grid, grid, shading="auto", cmap="viridis")
        plt.colorbar(heatmap, label="Value")

        plt.xlabel("Latitude (deg)")
        plt.ylabel("Altitude (km)")
        plt.title("SWV ppmv cause by a change of 1 ppbv CH4")
        plt.tight_layout()
        plt.show()
    return grid


def get_alpha_aoa(heights, latitudes, plot_data=False):
    """
    Function to construct the fractional release factor for CH4 (alpha)
    and the age-of air rounded to whole years.

    Args:
        heights (np.array): a np.array of heights in meters
        latitudes (np.array): a np.array of latitudes in degrees
        plot_data (bool): whether to plot the data or not

    Returns:
        alpha (np.ndarray): A matrix of fractional release factors
                            for different altitude and latitude levels.
        rounded_aoa (np.ndarray): A matrix of the rounded age of air
                                  for different altitude and latitude levels.
    """
    tp_value = (
        1.772  # tropospheric methane concentration averaged over the period 1991-1999
    )
    df = construct_myhre_1m_df()
    grid = get_griddata(df, heights, latitudes, plot_data=False)

    ch4_e = tp_value  # ppmv
    alpha = (ch4_e - grid) / ch4_e
    if (alpha > 1.0).any().any():
        raise ValueError("alpha contains a value higher than 1.")
    if (alpha < -0.01).any().any():
        raise ValueError("alpha contains a negative value.")
    aoa = 0.3 + 15.2 * alpha - 21.2 * alpha**2 + 10.4 * alpha**3

    rounded_aoa = pd.DataFrame(aoa.round(0))

    if plot_data is True:
        plot_alpha_aoa(latitudes, heights, alpha, rounded_aoa)
    return alpha, rounded_aoa


def calc_swv_mass_conc(delta_ch4, display_distribution=False):
    """
        Calculates the SWV concentration and mass based on the oxidation of CH4.
        It is based on the tropospheric CH4 change,
        the fractional release factor, and the Age-of-Air.
        Based on the papers of A.J. Harmsen (2026) The Climate Impact of
        Stratospheric Water Vapour Caused by Aviation Emissions
        https://repository.tudelft.nl/record/uuid:98c4bda7-a17d-47a4-9b24-48b7b46e4bb6
    }

        Args:
            delta_ch4 (list): List of yearly changes in CH4 concentration due to an emission.
            display_distribution (bool): Whether to plot the distribution of swv or not.

        Returns:
            delta_mass_swv (list): A list of the total change in SWV mass in Tg due to CH4 oxidation
                                   for each year corresponding to delta_ch4.
            delta_conc_swv (list): A list with the average stratospheric concentration
                                   change of SWV in ppbv due to CH4 oxidation for each
                                   year corresponding to delta_ch4.
            final_swv_distribution (DataFrame): A DataFrame of the final distribution of
                                                SWV concentration change in ppbv
    """
    # initialize
    delta_mass_swv = np.ones(len(delta_ch4))
    delta_conc_swv = np.ones(len(delta_ch4))

    # define constants
    delta_h = 100.0  # height increment in meters
    delta_deg = 1.0  # latitude increment
    heights = np.arange(0, 60000 + delta_h, delta_h)  # 0 to 60 km

    latitudes = np.arange(-85, 85, delta_deg)

    volume = get_volume_matrix(heights, latitudes, delta_h, delta_deg)
    density = Atmosphere(heights).density
    mass_mat = volume * density[:, np.newaxis]  # kg
    alpha, rounded_aoa = get_alpha_aoa(heights, latitudes, plot_data=False)

    if (rounded_aoa >= 6.0).any().any():
        # 6 is not allowed due to the timelag map is defined till 5
        raise ValueError("rounded_aoa contains a value of 6 or higher.")
    if (rounded_aoa < 0.0).any().any():
        raise ValueError("rounded_aoa contains a negative value.")
    for t in range(len(delta_ch4)):
        # get swv distribution
        timelag_map = {
            1: delta_ch4[t - 1] if t - 1 >= 0 else 0.0,
            2: delta_ch4[t - 2] if t - 2 >= 0 else 0.0,
            3: delta_ch4[t - 3] if t - 3 >= 0 else 0.0,
            4: delta_ch4[t - 4] if t - 4 >= 0 else 0.0,
            5: delta_ch4[t - 5] if t - 5 >= 0 else 0.0,
        }
        df_ch4_lagged = rounded_aoa.replace(timelag_map)
        swv = 2 * alpha * df_ch4_lagged  # ppbv

        # calculate average concentration
        number_density = Atmosphere(heights).number_density
        swv_parts_mat = volume * number_density[:, np.newaxis] * swv * 1e-9
        tot_parts = np.nansum(
            (
                volume
                * np.where(np.isnan(swv_parts_mat), np.nan, 1)
                * number_density[:, np.newaxis]
            )
        )  # to make sure only stratospheric volume is taken
        average_conc = np.nansum(swv_parts_mat) / tot_parts * 1e9  # ppbv

        # calculate total swv mass
        swv_mass_mat = swv * 10**-9 * M_H2O / M_AIR * mass_mat  # kg
        swv_mass = np.nansum(swv_mass_mat) / 1e9  # Tg

        # store data
        delta_mass_swv[t] = swv_mass  # Tg
        delta_conc_swv[t] = average_conc  # ppbv

    final_swv_distribution = swv

    if display_distribution:
        plot_swv_distribution(latitudes, heights, final_swv_distribution)

    return delta_mass_swv, delta_conc_swv, final_swv_distribution


def plot_swv_distribution(latitudes, heights, final_swv_distribution):
    """
    Plot the SWV distribution as a latitude–pressure heatmap.

    Args:
        latitudes: A 1D array of latitude values in degrees north.
        heights: A 1D array of the heights in meters
        final_swv_distribution: A 2D array of SWV distribution values

    Returns:
        None: This function displays a plot and does not return any value.
    """
    plt.figure(figsize=(10, 6))
    heatmap = plt.pcolormesh(
        latitudes,
        Atmosphere(heights).pressure / 100,  # make it hPa
        final_swv_distribution,
        shading="auto",
        cmap="viridis",
    )
    plt.colorbar(heatmap, label="Value")
    plt.yscale("log")
    plt.gca().invert_yaxis()  # invert so low pressure is at the top
    plt.xlabel("Latitude [deg]")
    plt.ylabel("pressure level [hPa]")
    plt.title("SWV distribution")
    plt.tight_layout()
    plt.show()


def plot_alpha_aoa(latitudes, heights, alpha, rounded_aoa):
    """
    Plot the alpha distribution as a latitude–pressure heatmap.
    Plot the alpha distribution as a latitude–pressure contour plot.
    Plot the rounded age-of-air distribution as a latitude–pressure heatmap.

    Args:
        latitudes: A 1D array of latitude values in degrees north.
        heights: A 1D array of the heights in meters
        alpha: A 2D array of alpha values
        rounded_aoa: A 2D array of age-of-air values rounded to whole integer years

    Returns:
        None: This function displays a plot and does not return any value.
    """

    plt.figure(figsize=(10, 6))
    heatmap = plt.pcolormesh(
        latitudes,
        Atmosphere(heights).pressure / 100,  # make it hPa
        alpha,
        shading="auto",
        cmap="viridis",
    )
    plt.colorbar(heatmap, label="Value")
    plt.yscale("log")
    plt.gca().invert_yaxis()  # invert so low pressure is at the top
    plt.xlabel("Latitude [deg]")
    plt.ylabel("pressure level [hPa]")
    plt.title("alpha")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    contour_levels = np.arange(0.0, 1.1, 0.1)
    contours = plt.contour(
        latitudes,
        Atmosphere(heights).pressure / 100,
        alpha,
        levels=contour_levels,
        colors="k",
        linewidths=0.8,
    )
    plt.xlim([-90, 90])
    plt.ylim([1, 1000])
    plt.clabel(contours, inline=True, fmt="%.1f", fontsize=12)
    plt.yscale("log")
    plt.gca().invert_yaxis()  # invert so low pressure is at the top
    plt.xlabel("Latitude [deg]", fontsize=14)
    plt.ylabel("Pressure level [hPa]", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    heatmap = plt.pcolormesh(
        latitudes,
        Atmosphere(heights).pressure / 100,  # make it hPa
        rounded_aoa,
        shading="auto",
        cmap="viridis",
    )
    plt.colorbar(heatmap, label="Value")
    plt.yscale("log")
    plt.gca().invert_yaxis()  # invert so low pressure is at the top
    plt.xlabel("Latitude [deg}")
    plt.ylabel("pressure level [hPa]")
    plt.title("rounded age-of-air")
    plt.tight_layout()
    plt.show()
