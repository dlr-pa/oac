"""
Calculates the contrail response.
"""

__author__ = "Liam Megill"
__email__ = "liam.megill@dlr.de"
__license__ = "Apache License 2.0"


import logging
import numpy as np
import xarray as xr
from openairclim.interpolate_time import apply_evolution

# CONSTANTS
R_EARTH = 6371.0  # [km] radius of Earth
KAPPA = 287.0 / 1003.5


def get_cont_grid(ds_cont: xr.Dataset) -> tuple:
    """Get contrail grid from `ds_cont`.

    Args:
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.

    Returns:
        tuple: Contrail grid of shape (lon, lat, plev).
    """
    cc_lon_vals = ds_cont.lon.data
    cc_lat_vals = ds_cont.lat.data
    cc_plev_vals = ds_cont.plev.data
    return (cc_lon_vals, cc_lat_vals, cc_plev_vals)


def check_cont_input(config, ds_cont, inv_dict, base_inv_dict):
    """Checks the input data for the contrail module.

    Args:
        config (dict): Configuration dictionary from config file.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
        base_inv_dict (dict): Dictionary of base emission inventory
            xarrays, keys are inventory years.
    """

    # check resp_cont
    if "method" not in config["responses"]["cont"]:
        raise KeyError("Missing 'method' key in config['responses']['cont'].")
    cont_method = config["responses"]["cont"]["method"]
    if cont_method not in ["AirClim", "Megill_2025"]:
        raise ValueError(
            "Unknown contrail method in config['responses']['cont']. "
            "Options are 'AirClim' and 'Megill_2025' (default)."
        )

    if cont_method == "AirClim":
        required_vars = ["ISS", "SAC_CON", "SAC_LH2"]
        required_coords = ["lat", "lon", "plev"]
        required_units = ["degrees_north", "degrees_east", "hPa"]

    else:  # Megill_2025 method
        required_vars = [
            "ppcf",
            "ISS",
            "g_250",
            "l_1",
            "k_1",
            "x0_1",
            "d_1",
            "l_2",
            "k_2",
            "x0_2",
        ]
        required_coords = ["lat", "lon", "plev", "AC"]
        required_units = ["degrees_north", "degrees_east", "hPa", "None"]

    for var in required_vars:
        if var not in ds_cont:
            raise KeyError(
                f"Missing required variable '{var}' in `resp_cont.nc`."
            )

    for coord, unit in zip(required_coords, required_units):
        if coord not in ds_cont:
            raise KeyError(
                f"Missing required coordinate '{coord}' in `resp_cont.nc`."
            )
        got_unit = ds_cont[coord].attrs.get("units")
        if got_unit != unit:
            raise ValueError(
                f"Incorrect unit for coordinate '{coord}'. Got '{got_unit}', "
                "should be '{unit}'."
            )

    # ensure that lat values descend and lon values ascend
    if not np.all(ds_cont.lat.values == np.sort(ds_cont.lat.values)[::-1]):
        raise ValueError("Latitude values must descend in `resp_cont.nc`.")
    if not np.all(ds_cont.lon.values == np.sort(ds_cont.lon.values)):
        raise ValueError("Longitude values must ascend in `resp_cont.nc`.")
    if not np.all(ds_cont.lon.values == ds_cont.lon.values % 360.0):
        raise ValueError(
            "Longitude values must be defined between 0 and 360 in "
            "`resp_cont.nc`"
        )

    # check years of inventories
    if base_inv_dict:
        if min(base_inv_dict.keys()) > min(inv_dict.keys()):
            raise ValueError(
                f"The inv_dict key {min(inv_dict.keys())} is less than the "
                f"earliest base_inv_dict key {min(base_inv_dict.keys())}."
            )
        if max(base_inv_dict.keys()) < max(inv_dict.keys()):
            raise ValueError(
                f"The inv_dict key {max(inv_dict.keys())} is larger than the "
                f"largest base_inv_dict key {max(base_inv_dict.keys())}."
            )


def calc_cont_grid_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate the cell area of the contrail grid using a simplified method.

    Args:
        lat (np.ndarray): Latitudes of the grid cells [deg].
        lon (np.ndarray): Longitudes of the grid cells [deg].

    Returns:
        np.ndarray : Contrail grid cell areas as a function of latitude [km^2].
    """

    # pre-conditions
    if len(lat) == 0:
        raise ValueError("Latitudes array cannot be empty.")
    if len(lon) == 0:
        raise ValueError("Longitudes array cannot be empty.")
    if len(lat) != len(np.unique(lat)):
        raise ValueError(
            "Duplicate latitude values detected. Latitudes must be unique."
        )
    if len(lon) != len(np.unique(lon)):
        raise ValueError(
            "Duplicate longitude values detected. Longitudes must be unique."
        )
    if not np.all((lat > -90.0) & (lat < 90.0)):
        raise ValueError(
            "Latitude values must be strictly between -90 and +90 degrees."
        )
    if not np.all((lon >= 0.0) & (lon <= 360.0)):
        raise ValueError("Longitude values must be between 0 and 360 degrees.")
    if np.all((0.0 in lon) & (360.0 in lon)):
        raise ValueError(
            "Longitude grid must not include both 0 and 360 degrees."
        )
    if not np.all(lat == np.sort(lat)[::-1]):
        raise ValueError("Latitude values must be sorted in descending order.")
    if not np.all(lon == np.sort(lon)):
        raise ValueError("Longitude values must be sorted in ascending order.")

    # calculate dlon
    lon_padded = np.concatenate(([lon[-1] - 360.0], lon, [lon[0] + 360.0]))
    lon_midpoints = 0.5 * (lon_padded[1:] + lon_padded[:-1])
    dlon_deg = np.diff(lon_midpoints)
    dlon = np.deg2rad(dlon_deg) * R_EARTH

    # calculate dlat
    lat_padded = np.concatenate(([90], lat, [-90]))  # add +/-90 deg
    lat_midpoints = 0.5 * (lat_padded[1:] + lat_padded[:-1])
    dlat = R_EARTH * np.abs(
        np.sin(np.deg2rad(lat_midpoints[:-1]))
        - np.sin(np.deg2rad(lat_midpoints[1:]))
    )

    # calculate areas
    areas = np.outer(dlat, dlon)

    # post-conditions
    assert np.all(areas) > 0.0, "Not all calculated areas are positive."
    sphere_area = 4 * np.pi * R_EARTH**2
    relative_error = abs(areas.sum() - sphere_area) / sphere_area
    if relative_error >= 1e-3:
        raise ValueError(
            "Total area check failed: computed area differs from Earth's "
            f"surface area by {relative_error:.4%}, which exceeds acceptable "
            "tolerance. Please check the contrail grid in `resp_cont.nc`."
        )

    return areas


def interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars, cont_grid):
    """Create base emission inventories for years in `inv_dict` that do not
    exist in `base_inv_dict`.

    Args:
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
        base_inv_dict (dict): Dictionary of base emission inventory
            xarrays, keys are inventory years.
        intrp_vars (list): List of strings of data variables in
            base_inv_dict that are to be included in the missing base
            inventories, e.g. ["distance", "fuel"].
        cont_grid (tuple): Precalculated contrail grid.

    Returns:
        dict: Dictionary of base emission inventory xarrays including any
            missing years compared to inv_dict, keys are inventory years.

    Note:
        A custom nearest neighbour method is used for regridding and a linear
        interpolation method for calculating data in missing years. In future
        versions, the user will be able to select methods for both.
    """

    # if base_inv_dict is empty, then return the empty dictionary.
    if not base_inv_dict:
        return {}

    # pre-conditions
    if not inv_dict:
        raise ValueError("inv_dict cannot be empty.")
    if not intrp_vars:
        raise ValueError("intrp_vars cannot be empty.")
    if base_inv_dict:
        if min(base_inv_dict.keys()) > min(inv_dict.keys()):
            raise ValueError(
                f"The inv_dict key {min(inv_dict.keys())} is less than the "
                f"earliest base_inv_dict key {min(base_inv_dict.keys())}."
            )
        if max(base_inv_dict.keys()) < max(inv_dict.keys()):
            raise ValueError(
                f"The inv_dict key {max(inv_dict.keys())} is larger than the "
                f"largest base_inv_dict key {max(base_inv_dict.keys())}."
            )
        for intrp_var in intrp_vars:
            for year in base_inv_dict.keys():
                if intrp_var not in base_inv_dict[year]:
                    raise KeyError(
                        f"Variable '{intrp_var}' is missing from base_inv_dict "
                        f"for year {year}."
                    )

    # get years that need to be calculated
    inv_yrs = list(inv_dict.keys())
    base_yrs = list(base_inv_dict.keys())
    intrp_yrs = sorted(set(inv_yrs) - set(base_yrs))

    # initialise output
    full_base_inv_dict = base_inv_dict.copy()

    # get contrail grid
    cc_lon_vals, cc_lat_vals, cc_plev_vals = cont_grid

    # if there are years in inv_dict that do not exist in base_inv_dict
    if intrp_yrs:
        # find upper and lower neighbouring base_inv_dict years
        intrp_yr_idx = np.searchsorted(base_yrs, intrp_yrs)
        yrs_lb = [base_yrs[idx - 1] for idx in intrp_yr_idx]
        yrs_ub = [base_yrs[idx] for idx in intrp_yr_idx]
        yrs_regrid = np.unique(yrs_lb + yrs_ub)

        # regrid base inventories to contrail grid
        regrid_base_inv_dict = {}
        for yr in yrs_regrid:
            base_inv = base_inv_dict[yr]

            # find nearest neighbour indices
            lon_idxs = np.abs(
                cc_lon_vals[:, np.newaxis] - base_inv.lon.data
            ).argmin(axis=0)
            lat_idxs = np.abs(
                cc_lat_vals[:, np.newaxis] - base_inv.lat.data
            ).argmin(axis=0)
            plev_idxs = np.abs(
                cc_plev_vals[:, np.newaxis] - base_inv.plev.data
            ).argmin(axis=0)

            # create DataArray for yr
            regrid_base_inv = {}
            for intrp_var in intrp_vars:
                intrp_arr = np.zeros(
                    (len(cc_lon_vals), len(cc_lat_vals), len(cc_plev_vals))
                )
                np.add.at(
                    intrp_arr,
                    (lon_idxs, lat_idxs, plev_idxs),
                    base_inv[intrp_var].data.flatten(),
                )
                regrid_base_inv[intrp_var] = xr.DataArray(
                    data=intrp_arr,
                    dims=["lon", "lat", "plev"],
                    coords={
                        "lon": cc_lon_vals,
                        "lat": cc_lat_vals,
                        "plev": cc_plev_vals,
                    },
                )

            # create dataset
            regrid_base_inv_dict[yr] = xr.Dataset(regrid_base_inv)

        # linearly interpolate base_inv
        for i, yr in enumerate(intrp_yrs):
            # linear weighting
            w = (yr - yrs_lb[i]) / (yrs_ub[i] - yrs_lb[i])
            ds_i = (
                regrid_base_inv_dict[yrs_lb[i]] * (1 - w)
                + regrid_base_inv_dict[yrs_ub[i]] * w
            )

            # reset index to match input inventories
            ds_i_flat = ds_i.stack(index=["lon", "lat", "plev"])
            ds_i_flat = ds_i_flat.reset_index("index")
            full_base_inv_dict[yr] = ds_i_flat

        # sort full_base_inv_dict
        full_base_inv_dict = dict(sorted(full_base_inv_dict.items()))

    # post-conditions
    if intrp_yrs:
        for yr in intrp_yrs:
            assert yr in full_base_inv_dict, (
                "Missing years not included in " "output dictionary."
            )

    return full_base_inv_dict


def calc_cont_weighting(
    config: dict, val: str, cont_grid: tuple, ac: str
) -> np.ndarray:
    """Calculate weighting functions for the contrail grid developed by
    Ludwig Hüttenhofer (Bachelorarbeit LMU, 2013). This assumes the
    contrail grid developed for AirClim 2.1 (Dahlmann et al., 2016).

    Args:
        config (dict): Configuration dictionary from config file.
        val (str): Weighting value to calculate. Choice of "w1", "w2" or "w3"
        cont_grid (tuple): Precalculated contrail grid.
        ac (str): Aircraft identifier from config

    Raises:
        ValueError: if invalid value is passed for "val".

    Returns:
        np.ndarray: Array of size (nlat) with weighting values for each
            latitude value
    """

    # get latitude values
    cc_lat_vals = cont_grid[1]

    # Eq. 3.3.4 of Hüttenhofer (2013); "rel" in AirClim 2.1
    # fmt: off
    if val == "w1":
        idxs = (cc_lat_vals > 68.0) | (cc_lat_vals < -53.0)  # as in AirClim 2.1
        res = np.where(idxs, 1.0, 0.863 * np.cos(np.pi * cc_lat_vals / 50.0) + 1.615)
    # fmt: on

    # "fkt_g" in AirClim 2.1
    elif val == "w2":
        # pre-conditions
        if "eff_fac" not in config["aircraft"][ac]:
            raise KeyError(
                f"Missing 'eff_fac' key in config['aircraft']['{ac}']."
            )

        eff_fac = config["aircraft"][ac]["eff_fac"]
        res = 1.0 + 15.0 * np.abs(
            0.045 * np.cos(cc_lat_vals * 0.045) + 0.045
        ) * (eff_fac - 1.0)

    # Eq. 3.3.10 of Hüttenhofer (2013); RF weighting in AirClim 2.1
    elif val == "w3":
        res = 1.0 + 0.24 * np.cos(cc_lat_vals * np.pi / 23.0)

    # raise error in case val invalid
    else:
        raise ValueError(f"Contrail weighting parameter {val} is invalid.")

    return res


def calc_ppcf(config: dict, ds_cont: xr.Dataset, ac: str) -> xr.DataArray:
    """Calculate Potential Persistent Contrail Formation (p_PCF) using the
    precalculated contrail data, either from the Limiting Factors study (Megill
    & Grewe, 2025; default) or using the legacy AirClim 2.1 method (Dahlmann
    et al., 2016). The terms p_SAC (AirClim) and p_PCF (OpenAirClim) are
    equivalent.

    Args:
        config (dict): Configuration dictionary from config file.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        ac (str): Aircraft identifier from config.

    Returns:
        xr.DataArray: Interpolated p_PCF on precalculated contrail data grid
    """

    # pre-conditions
    if "method" not in config["responses"]["cont"]:
        raise KeyError("Missing 'method' key in config['responses']['cont'].")
    cont_method = config["responses"]["cont"]["method"]
    if cont_method not in ["AirClim", "Megill_2025"]:
        raise ValueError(
            "Unknown contrail method in config['responses']['cont']. "
            "Options are 'AirClim' and 'Megill_2025' (default)."
        )

    if cont_method == "AirClim":
        return calc_psac_airclim(config, ds_cont, ac)

    # Megill_2025 method (default)
    return calc_ppcf_megill(config, ds_cont, ac)


def calc_psac_airclim(config: dict, ds_cont: xr.Dataset, ac: str) -> xr.DataArray:
    """Calculate the probability that the Schmidt-Appleman Criterion is met
    using the legacy AirClim 2.1 method (Dahlmann et al., 2016) and simulations
    performed for the AHEAD project (Grewe et al., 2017).

    Args:
        config (dict): Configuration dictionary from config file.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        ac (str): Aircraft identifier from config.

    Returns:
        xr.DataArray: Interpolated p_SAC on precalculated contrail data grid.
    """

    # get G_comp
    if "G_comp" not in config["aircraft"][ac]:
        raise KeyError(f"Missing 'G_comp' key in config['aircraft']['{ac}'].")
    g_comp = config["aircraft"][ac]["G_comp"]
    g_comp_con = 0.04  # EIH2O 1.25, Q 43.6e6, eta 0.333
    g_comp_lh2 = 0.12  # EIH2O 8.94, Q 120.9e6, eta 0.4
    if not np.all((g_comp >= g_comp_con) & (g_comp <= g_comp_lh2)):
        raise ValueError("Invalid G_comp value. Expected range: [0.04, 0.12].")

    # calculate p_sac using linear interpolation between CON and LH2
    x = (g_comp - g_comp_con) / (g_comp_lh2 - g_comp_con)
    p_sac = (1.0 - x) * ds_cont.SAC_CON + x * ds_cont.SAC_LH2

    return p_sac


def calc_ppcf_megill(config: dict, ds_cont: xr.Dataset, ac: str) -> xr.DataArray:
    """Calculate the Potential Persistent Contrail Formation (p_PCF) using the
    Megill & Grewe (2025) method and precalculated data from ERA5.

    Args:
        config (dict): Configuration dictionary from config file.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        ac (str): Aircraft identifier from config.

    Returns:
        xr.DataArray: Interpolated p_PCF on precalculated contrail data grid.
    """

    # get G value at 250 hPa
    if "G_250" not in config["aircraft"][ac]:
        raise KeyError(f"Missing 'G_250' key in config['aircraft']['{ac}'].")
    g_in = config["aircraft"][ac]["G_250"]
    precal_g_vals = ds_cont.g_250.data

    # ensure that G is not lower than lowest pre-calculated G
    if g_in < min(precal_g_vals):
        raise ValueError(
            "Selected G_250 value is below pre-calculated values. OpenAirClim "
            "cannot guarantee the accuracy of the fits here. If This G_250 "
            "value is required, please contact the dev team."
        )

    # find left and right neighbours
    right_idx = np.searchsorted(precal_g_vals, g_in)
    left_idx = max(right_idx - 1, 0)
    right_idx = min(right_idx, len(precal_g_vals) - 1)
    g_nbrs = precal_g_vals[[left_idx, right_idx]]

    # find p_pcf values for input and neighbours at 250 hPa
    params_1 = np.array(
        [ds_cont.sel(plev=250)[var] for var in ["l_1", "k_1", "x0_1", "d_1"]]
    )
    params_2 = np.array(
        [ds_cont.sel(plev=250)[var] for var in ["l_2", "k_2", "x0_2"]]
    )
    y_in = logistic_gen(g_in, *params_1) + logistic(g_in, *params_2)
    y_nbrs = logistic_gen(g_nbrs, *params_1) + logistic(g_nbrs, *params_2)

    # if G > largest pre-calculated G
    if left_idx == len(precal_g_vals) - 1:
        logging.warning(
            "Selected G is above pre-calculated values. Use results with "
            "caution."
        )
        p_pcf = ds_cont.isel(AC=-1).ppcf
    # if G is between two pre-calculated values
    else:
        x = (y_in - y_nbrs[0]) / (y_nbrs[1] - y_nbrs[0])
        p_pcf = (1 - x) * ds_cont.isel(AC=left_idx).ppcf + x * ds_cont.isel(
            AC=right_idx
        ).ppcf

    return p_pcf


def logistic(x, l, k, x0):
    """Computes the logistic function, a sigmoid curve, commonly used to model
    growth or decay. Function from Megill & Grewe (2025):
    https://github.com/liammegill/contrail-limiting-factors

    Args:
        x (float or np.ndarray): The input values for which the logistic
            function will be computed.
        l (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half
            of `l`.

    Returns:
        float or array-like: The logistic function values for the given
            input `x`.
    """
    np.seterr(all="raise")
    try:
        return l / (1 + np.exp(-k * (x - x0)))
    except FloatingPointError:  # protect the exponential
        return np.nan


def logistic_gen(x, l, k, x0, d):
    """Computes a generalized logistic function with an additional vertical
    shift. Function from Megill & Grewe (2025):
    https://github.com/liammegill/contrail-limiting-factors

    Args:
        x (float or np.ndarray): The input values for which the logistic
            function will be computed.
        l (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half
            of `l`.
        d (float): The vertical shift applied to the function.

    Returns:
        float or array-like: The values of the shifted logistic function for
            the input `x`.
    """
    np.seterr(all="raise")
    try:
        return l / (1 + np.exp(-k * (x - x0))) + d
    except FloatingPointError:  # protect the exponential
        return np.nan


def calc_cfdd(
    config: dict, inv_dict: dict, ds_cont: xr.Dataset, cont_grid: tuple, ac: str
) -> dict:
    """Calculate the Contrail Flight Distance Density (CFDD) for each year in
    inv_dict. This function uses the p_pcf data calculated using ERA5
    (Megill & Grewe, 2025) or replicates the legacy AirClim 2.1 using the p_sac
    data calculated for the AHEAD project (Dahlmann et al., 2016; Grewe et al.,
    2017).

    Args:
        config (dict): Configuration dictionary from config file.
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        cont_grid (tuple): Precalculated contrail grid.
        ac (str): Aircraft identifier from config.

    Returns:
        dict: Dictionary with CFDD values [km/km2], keys are inventory years
    """

    # calculate ppcf and ensure that it is of shape (lat, lon, plev)
    p_pcf = calc_ppcf(config, ds_cont, ac)
    p_pcf = p_pcf.T.transpose("lat", "lon", "plev")

    # calculate contrail grid areas
    cc_lon_vals, cc_lat_vals, cc_plev_vals = cont_grid
    areas = calc_cont_grid_areas(cc_lat_vals, cc_lon_vals)

    # calculate CFDD
    # p_pcf is interpolated using a power law over pressure level and using
    # a nearest neighbour for latitude and longitude.
    cfdd_dict = {}
    for year, inv in inv_dict.items():

        # initialise arrays for storage
        sum_km = np.zeros((len(cc_lat_vals), len(cc_lon_vals)))

        # find indices
        lat_idxs = np.abs(cc_lat_vals[:, np.newaxis] - inv.lat.data).argmin(axis=0)
        lon_idxs = np.abs(cc_lon_vals[:, np.newaxis] - inv.lon.data).argmin(axis=0)
        plev_idxs = len(cc_plev_vals) - np.searchsorted(
            cc_plev_vals[::-1], inv.plev.data, side="right"
        )

        # interpolate over plev using power law between upper and lower bounds
        plev_ub = cc_plev_vals[plev_idxs]
        plev_lb = cc_plev_vals[plev_idxs - 1]
        sigma_plev = 1 - (
            (inv.plev.data**KAPPA - plev_lb**KAPPA)
            / (plev_ub**KAPPA - plev_lb**KAPPA)
        )

        # calculate p_pcf
        p_pcf_ub = p_pcf.values[lat_idxs, lon_idxs, plev_idxs]
        p_pcf_lb = p_pcf.values[lat_idxs, lon_idxs, plev_idxs - 1]
        p_pcf_intrp = sigma_plev * p_pcf_lb + (1 - sigma_plev) * p_pcf_ub

        # calculate and store CFDD
        # 1800s since the CFDD method was developed using 30min intervals
        # 3153600s in one year
        sum_contrib = inv.distance.data * p_pcf_intrp * 1800.0 / 31536000.0
        np.add.at(sum_km, (lat_idxs, lon_idxs), sum_contrib)
        cfdd = sum_km / areas
        cfdd_dict[year] = cfdd

    # post-conditions
    for year, cfdd in cfdd_dict.items():
        assert cfdd.shape == (len(cc_lat_vals), len(cc_lon_vals)), (
            "Shape " f"of CFDD array for year {year} is not correct."
        )

    return cfdd_dict


def calc_cccov(
    config: dict, cfdd_dict: dict, ds_cont: xr.Dataset, cont_grid: tuple, ac: str
) -> dict:
    """Calculate contrail cirrus coverage using the relationship developed for
    AirClim 2.1 (Dahlmann et al., 2016).

    Args:
        config (dict): Configuration dictionary from config file.
        cfdd_dict (dict): Dictionary with CFDD values [km/km2], keys are
            inventory years.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        cont_grid (tuple): Precalculated contrail grid.
        ac (str): Aircraft identifier from config.

    Returns:
        dict: Dictionary with cccov values, keys are inventory years
    """

    # pre-conditions
    if "eff_fac" not in config["aircraft"][ac]:
        raise KeyError(f"Missing 'eff_fac' key in config['aircraft']['{ac}'].")
    for year, cfdd in cfdd_dict.items():
        assert cfdd.shape == (len(cont_grid[1]), len(cont_grid[0])), (
            "Shape " f"of CFDD array for year {year} is not correct."
        )

    # load weighting function
    eff_fac = config["aircraft"][ac]["eff_fac"]
    w1 = calc_cont_weighting(config, "w1", cont_grid, ac)

    # calculate cccov
    cccov_dict = {}
    iss = ds_cont.ISS.T.transpose("lat", "lon").data  # ensure correct shape
    for year, cfdd in cfdd_dict.items():
        cccov = 0.128 * iss * np.arctan(97.7 * cfdd / iss)
        cccov = cccov * eff_fac * w1[:, np.newaxis]  # add corrections
        cccov_dict[year] = cccov

    # post-conditions
    for year, cccov in cccov_dict.items():
        assert cccov.shape == (len(cont_grid[1]), len(cont_grid[0])), (
            "Shape " f"of cccov array for year {year} is not correct."
        )

    return cccov_dict


def calc_weighted_cccov(comb_cccov_dict, cfdd_dict, comb_cfdd_dict):
    """Calculate the contrail cirrus coverage cccov weighted by the difference
    in the contrail flight distance densities CFDD between the input inventory
    and the base inventory. This function is used when `rel_to_base`
    is TRUE. The keys of all dictionaries must match.

    Args:
        comb_cccov_dict (dict): Dictionary with cccov values of the inventory
            and base summed together, keys are years.
        cfdd_dict (dict): Dictionary with CFDD values of the inventory (without
            base), keys are years.
        comb_cfdd_dict (dict): Dictionary with CFDD values of the inventory
            and base summed together, keys are years.

    Returns:
        dict: Dictionary with weighted cccov values, keys are years.
    """

    # pre-conditions
    assert set(comb_cccov_dict.keys()) == set(cfdd_dict.keys()), (
        "Keys of " "comb_cccov_dict and cfdd_dict do not match."
    )
    assert set(comb_cccov_dict.keys()) == set(comb_cfdd_dict.keys()), (
        "Keys " "of comb_cccov_dict and comb_cfdd_dict do not match."
    )

    weighted_cccov_dict = {}
    for year in comb_cccov_dict.keys():
        weighted_cccov_dict[year] = np.divide(
            comb_cccov_dict[year] * cfdd_dict[year],
            comb_cfdd_dict[year],
            where=comb_cfdd_dict[year] != 0,
        )
        weighted_cccov_dict[year][comb_cfdd_dict[year] == 0] = 0.0

    # post conditions
    for year in comb_cccov_dict.keys():
        assert np.all(weighted_cccov_dict[year] >= 0.0), (
            "Negative weighted " "cccov values detected."
        )

    return weighted_cccov_dict


def calc_cccov_tot(config, cccov_dict, cont_grid, ac):
    """Calculate total, area-weighted contrail cirrus coverage using the
    relationship developed for AirClim 2.1 (Dahlmann et al., 2016).

    Args:
        config (dict): Configuration dictionary from config file.
        cccov_dict (dict): Dictionary with cccov values, keys are inventory
            years.
        cont_grid (tuple): Precalculated contrail grid.
        ac (str): Aircraft identifier from config.


    Returns:
        dict: Dictionary with total, area-weighted contrail cirrus coverage,
            keys are inventory years.
    """

    cc_lon_vals, cc_lat_vals, _ = cont_grid
    for year, cccov in cccov_dict.items():
        assert cccov.shape == (len(cc_lat_vals), len(cc_lon_vals)), (
            "Shape " f"of cccov array for year {year} is not correct."
        )

    # calculate contril grid cell areas
    areas = calc_cont_grid_areas(cc_lat_vals, cc_lon_vals)
    w2 = calc_cont_weighting(config, "w2", cont_grid, ac)
    w3 = calc_cont_weighting(config, "w3", cont_grid, ac)

    # calculate total (area-weighted) cccov
    cccov_tot_dict = {}
    for year, cccov in cccov_dict.items():
        cccov_tot = (cccov * areas).sum(axis=1) * w2 * w3 / areas.sum()
        cccov_tot_dict[year] = cccov_tot

    for year, cccov_tot in cccov_tot_dict.items():
        assert cccov_tot.shape == (len(cc_lat_vals),), (
            "Shape of cccov_tot " f"array for year {year} is not correct."
        )

    return cccov_tot_dict


def calc_cont_rf(config, cccov_tot_dict, inv_dict, cont_grid, ac):
    """Calculate contrail Radiative Forcing (RF) using the relationship
    developed for AirClim 2.1 (Dahlmann et al., 2016).

    Args:
        config (dict): Configuration dictionary from config file.
        cccov_tot_dict (dict): Dictionary with total, area-weighted contrail
            cirrus coverage, keys are inventory years
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
        cont_grid (tuple): Precalculated contrail grid.
        ac (str): Aircraft identifier from config.

    Returns:
        dict: Dictionary with contrail RF values interpolated for all years
            between the simulation start and end years.
    """

    # pre-conditions: check config
    if "PMrel" not in config["aircraft"][ac]:
        raise KeyError(f"Missing 'PMrel' key in config['aircraft']['{ac}'].")
    if not inv_dict:
        raise ValueError("inv_dict cannot be empty.")
    assert len(cccov_tot_dict) > 0, "cccov_tot_dict cannot be empty."
    assert np.all(cccov_tot_dict.keys() == inv_dict.keys()), (
        "Keys of " "cccov_dict do not match those of inv_dict."
    )
    for year, cccov_tot in cccov_tot_dict.items():
        assert cccov_tot.shape == (len(cont_grid[1]),), (
            f"Shape of cccov_tot " f"array for year {year} is not correct."
        )

    # calculate RF factor due to PM reduction, from AirClim 2.1
    pm_rel = config["aircraft"][ac]["PMrel"]
    if pm_rel >= 0.033:
        pm_factor = 0.92 * np.arctan(1.902 * pm_rel**0.74)
    else:
        pm_factor = 0.92 * np.arctan(1.902 * 0.033**0.74)

    # calculate contrail RF
    cont_rf_at_inv = []  # RF at inventory years
    for year, cccov_tot in cccov_tot_dict.items():
        cont_rf = 14.9 * np.sum(cccov_tot) * pm_factor
        cont_rf_at_inv.append(cont_rf)

    # interpolate RF to all simulation years
    _, rf_cont_dict = apply_evolution(
        config,
        {"cont": np.array(cont_rf_at_inv)},
        inv_dict,
        inventories_adjusted=True,
    )

    return rf_cont_dict


def add_inv_to_base(inv_dict, base_inv_dict):
    """Adds the inventory dictionary to the base inventory dictionary.
    Currently, the keys of the inventory dictionary must be a subset of the
    keys of the base inventory dictionary. In other words, the inventories must
    align to at least one year. This function is used when `rel_to_base` is
    TRUE.

    Args:
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
        base_inv_dict (dict): Dictionary of base emission inventory
            xarrays, keys are inventory years.

    Returns:
        dict: Summed dictionary of input inventories
    """

    # check that inv_dict is a subset of base_inv_dict
    if not set(inv_dict.keys()).issubset(base_inv_dict.keys()):
        raise KeyError("inv_dict keys are not a subset of base_inv_dict keys.")

    combined_dict = {}
    for key in inv_dict.keys():
        combined_dict[key] = inv_dict[key] + base_inv_dict[key]

    return combined_dict
