"""Calculates the contrail response.

_author_: Liam Megill
_email_: liam.megill@dlr.de
"""

import numpy as np
from openairclim.read_netcdf import open_netcdf
from openairclim.interpolate_time import apply_evolution

# CONSTANTS
R_EARTH = 6371.  # [km] radius of Earth
KAPPA = 287. / 1003.5  # TBD

# DEFINITION OF CONTRAIL GRID (from AirClim 2.1)
nlon = 96; nlat = 48; nlev = 39
cc_lon_vals = np.arange(0, 360, 3.75)
cc_lat_vals = np.array([
    87.1591, 83.47892, 79.77705, 76.07024, 72.36156, 68.65202,
    64.94195, 61.23157, 57.52099, 53.81027, 50.09945, 46.38856, 42.6776,
    38.96661, 35.25558, 31.54452, 27.83344, 24.12235, 20.41124, 16.70012,
    12.98899, 9.277853, 5.566714, 1.855572, -1.855572, -5.566714, -9.277853,
    -12.98899, -16.70012, -20.41124, -24.12235, -27.83344, -31.54452,
    -35.25558, -38.96661, -42.6776, -46.38856, -50.09945, -53.81027,
    -57.52099, -61.23157, -64.94195, -68.65202, -72.36156, -76.07024,
    -79.77705, -83.47892, -87.1591
])
cc_plev_vals = np.array([
    1014., 996., 968., 921., 865., 809., 755., 704.,
    657., 613., 573., 535., 499., 466., 434., 405., 377., 350., 325., 301.,
    278., 256., 236., 216., 198., 180., 163., 147., 131., 117., 103.0, 89.,
    76.0, 64.0, 52.0, 41.0, 30.0, 20.0, 10.0
])


def calc_cont_grid_areas(lat, dlon_deg=3.75):
    """Calculate the cell area of the contrail grid using a simplified method.
    This assumes a regular grid spacing of `dlon_deg` in longitudinal direction. 
    
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


def calc_cont_weighting(config, val):
    """Calculate weighting functions for the contrail grid developed by 
    Ludwig Hüttenhofer (Bachelorarbeit LMU, 2013). This assumes the 
    contrail grid developed for AirClim 2.1 (Dahlmann et al., 2016).

    Args:
        config (dict): Configuration dictionary from config file.
        val (str): Weighting value to calculate. Choice of "w1", "w2" or "w3"

    Raises:
        ValueError: if invalid value is passed for "val".

    Returns:
        np.ndarray: Array of size (nlat) with weighting values for each 
            latitude value
    """
    
    # Eq. 3.3.4 of Hüttenhofer (2013); "rel" in AirClim 2.1
    if val == "w1":
        idxs = (cc_lat_vals > 68) | (cc_lat_vals < -53)  # as in AirClim 2.1
        res = np.where(idxs, 1, 0.863 * np.cos(np.pi * cc_lat_vals / 50.) + 1.615)
    
    # "fkt_g" in AirClim 2.1
    elif val == "w2":
        eff_fac = config["responses"]["cont"]["eff_fac"]
        res = 1. + 15. * np.abs(0.045 * np.cos(cc_lat_vals * 0.045) + 0.045) * (eff_fac - 1.)
    
    # Eq. 3.3.10 of Hüttenhofer (2013); RF weighting in AirClim 2.1
    elif val == "w3":
        res = 1. + 0.24 * np.cos(cc_lat_vals * np.pi / 23.)
    
    # raise error in case val invalid
    else:
        raise ValueError(f"Contrail weighting parameter {val} is invalid.")
    
    return res


def calc_cfdd(config: dict, inv_dict: dict) -> dict:
    """Calculate the Contrail Flight Distance Density (CFDD) for each year in
    inv_dict. This function uses the p_SAC data calculated during the
    development of AirClim 2.1 (Dahlmann et al., 2016).
    
    Args:
        config (dict): Configuration dictionary from config file.
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
    
    Returns:
        dict: Dictionary with CFDD values [km/km2], keys are inventory years
    """
    
    # load data calculated during the development of AirClim 2.1
    ds_cont = open_netcdf("repository/resp_cont.nc")["resp_cont"]
    
    # calculate p_SAC for aircraft G
    G_comp = config["responses"]["cont"]["G_comp"]
    G_comp_con = 0.04  # EIH2O 1.25, Q 43.6e6, eta 0.3
    G_comp_lh2 = 0.12  # EIH2O 8.94, Q 120.9e6, eta 0.4
    x = (G_comp - G_comp_con) / ( G_comp_lh2 - G_comp)
    p_SAC = (1. - x) * ds_cont.SAC_CON + x * ds_cont.SAC_LH2
    
    # calculate contrail grid areas
    areas = calc_cont_grid_areas(cc_lat_vals)
    
    # calculate CFDD
    # p_SAC is interpolated using a power law over pressure level and using 
    # a nearest neighbour for latitude and longitude.
    cfdd_dict = {}
    for year, inv in inv_dict.items():
        
        # initialise arrays for storage
        intrp_sigma_vals = np.empty_like(inv.index.data, dtype=np.float64)
        p_sac_avg = np.empty_like(inv.index.data, dtype=np.float64)
        sum_km = np.zeros((nlat, nlon))
        
        # find indices
        lat_idxs = np.abs(cc_lat_vals[:, np.newaxis] - inv.lat.data).argmin(axis=0)
        lon_idxs = np.abs(cc_lon_vals[:, np.newaxis] - inv.lon.data).argmin(axis=0)
        plev_idxs = len(cc_plev_vals) - np.searchsorted(cc_plev_vals[::-1],
                                                        inv.plev.data, side="right")
        
        # interpolate over plev using power law between upper and lower bounds
        plev_ub = cc_plev_vals[plev_idxs]
        plev_lb = cc_plev_vals[plev_idxs-1]
        sigma_plev = 1 - ((inv.plev.data ** KAPPA - plev_lb ** KAPPA) /
                          (plev_ub ** KAPPA - plev_lb ** KAPPA))
        
        # calculate p_SAC
        p_SAC_ub = p_SAC.values[lat_idxs, lon_idxs, plev_idxs]
        p_SAC_lb = p_SAC.values[lat_idxs, lon_idxs, plev_idxs-1]
        p_SAC_intrp = sigma_plev * p_SAC_lb + (1 - sigma_plev) * p_SAC_ub
        
        # calculate and store CFDD
        # 1800s since ISS & p_SAC were developed in 30min intervals
        # 3153600s in one year
        sum_contrib = inv.distance.data * p_SAC_intrp * 1800. / 31536000. 
        np.add.at(sum_km, (lat_idxs, lon_idxs), sum_contrib)
        cfdd = sum_km / areas[:, np.newaxis]
        cfdd_dict[year] = cfdd
    
    return cfdd_dict


def calc_cccov(config, cfdd_dict):
    """Calculate contrail cirrus coverage using the relationship developed for 
    AirClim 2.1 (Dahlmann et al., 2016).

    Args:
        config (dict): Configuration dictionary from config file.
        cfdd_dict (dict): Dictionary with CFDD values [km/km2], keys are
            inventory years.

    Returns:
        dict: Dictionary with cccov values, keys are inventory years
    """
    
    # load data calculated during the development of AirClim 2.1
    ds_cont = open_netcdf("repository/resp_cont.nc")["resp_cont"]
    eff_fac = config["responses"]["cont"]["eff_fac"]
    w1 = calc_cont_weighting(config, "w1")
    
    # calculate cccov
    cccov_dict = {}
    for year, cfdd in cfdd_dict.items():
        cccov = 0.128 * ds_cont.ISS.data * np.arctan(97.7 * cfdd / ds_cont.ISS.data) * eff_fac * w1[:, np.newaxis] 
        cccov_dict[year] = cccov
        
    return cccov_dict


def calc_cccov_tot(config, cccov_dict):
    """Calculate total, area-weighted contrail cirrus coverage using the
    relationship developed for AirClim 2.1 (Dahlmann et al., 2016). 

    Args:
        config (dict): Configuration dictionary from config file.
        cccov_dict (dict): Dictionary with cccov values, keys are inventory
            years.

    Returns:
        dict: Dictionary with total, area-weighted contrail cirrus coverage, 
            keys are inventory years.
    """

    # calculate contril grid cell areas
    areas = calc_cont_grid_areas(cc_lat_vals)
    w2 = calc_cont_weighting(config, "w2")
    w3 = calc_cont_weighting(config, "w3")
    
    # calculate total (area-weighted) cccov
    cccov_tot_dict = {}
    for year, cccov in cccov_dict.items():
        cccov_tot = cccov.sum(axis=1) * areas * w2 * w3 / (np.sum(areas) * nlon)
        cccov_tot_dict[year] = cccov_tot
    
    return cccov_tot_dict


def calc_cont_RF(config, cccov_tot_dict, inv_dict):
    """Calculate contrail Radiative Forcing (RF) using the relationship
    developed for AirClim 2.1 (Dahlmann et al., 2016).
    
    Args:
        config (dict): Configuration dictionary from config file.
        cccov_tot_dict (dict): Dictionary with total, area-weighted contrail
            cirrus coverage, keys are inventory years
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.

    Returns:
        dict: Dictionary with contrail RF values interpolated for all years
            between the simulation start and end years.
    """
    
    # calculate contrail RF
    cont_RF_at_inv = []  # RF at inventory years
    for year, cccov_tot in cccov_tot_dict.items():
        cont_RF = np.sum(14.9 * cccov_tot)
        cont_RF_at_inv.append(cont_RF) 
    
    # interpolate RF to all simulation years
    _, rf_cont_dict = apply_evolution(config, {"cont": np.array(cont_RF_at_inv)}, inv_dict)
    
    return rf_cont_dict