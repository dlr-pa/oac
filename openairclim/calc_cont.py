"""Calculates the contrail response."""

import numpy as np

# CONSTANTS
R_EARTH = 6371.  # [km] radius of Earth


def calc_cont_grid_areas(lat, dlon_deg=3.75):
    """Calculate the cell area of the contrail grid using a simplified method.
    This assume a regular grid spacing of `dlon_deg` in longitudinal direction. 
    
    Args:
        lat (np.ndarray): Latitudes of the grid cells [deg].
        dlon_deg (float): Longitude increment [deg]. Defaults to 3.75 deg.
    
    Returns:
        np.ndarray : Contrail grid cell areas as a function of latitude [km^2].
    """
    
    # calculate dlon (assumes regular longitudinal grid spacing)
    dlon = dlon_deg / 360. * 2 * np.pi * R_EARTH

    # calculate dlat 
    lat_padded = np.concatenate(([90], lat, [-90]))  # add +/-90 deg 
    lat_between = lat_padded[:-1] + 0.5 * (lat_padded[1:] - lat_padded[:-1])
    dlat_deg = np.diff(lat_between)
    dlat = R_EARTH * np.abs(np.sin(np.deg2rad(lat_between[:-1])) - 
                            np.sin(np.deg2rad(lat_between[1:])))
    
    return dlon * dlat


