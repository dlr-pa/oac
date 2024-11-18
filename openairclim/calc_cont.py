"""
Calculates the contrail response.
Currently implemented: AirClim 2.1 contrail module.
"""

__author__ = "Liam Megill"
__email__ = "liam.megill@dlr.de"
__license__ = "Apache License 2.0"


import numpy as np
import xarray as xr
from openairclim.interpolate_time import apply_evolution

# CONSTANTS
R_EARTH = 6371.  # [km] radius of Earth
KAPPA = 287. / 1003.5

# DEFINITION OF CONTRAIL GRID (from AirClim 2.1)
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


def check_cont_input(ds_cont, inv_dict, base_inv_dict):
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
    required_vars = ["ISS", "SAC_CON", "SAC_LH2"]
    required_coords = ["lat", "lon", "plev"]
    required_units = ["degrees_north", "degrees_east", "hPa"]
    for var in required_vars:
        assert var in ds_cont, f"Missing required variable '{var}' in " \
            "resp_cont.nc."
    for coord, unit in zip(required_coords, required_units):
        assert coord in ds_cont, f"Missing required coordinate '{coord}' in " \
            "resp_cont.nc."
        got_unit = ds_cont[coord].attrs.get("units")
        assert got_unit == unit, f"Incorrect unit for coordinate '{coord}'. " \
            f"Got '{got_unit}', should be '{unit}'."

    # check years of inventories
    if base_inv_dict:
        assert min(base_inv_dict.keys()) <= min(inv_dict.keys()), "The " \
            f"inv_dict key {min(inv_dict.keys())} is less than the earliest " \
            f"base_inv_dict key {min(base_inv_dict.keys())}."
        assert max(base_inv_dict.keys()) >= max(inv_dict.keys()), "The " \
            f"inv_dict key {max(inv_dict.keys())} is larger than the largest "\
            f"base_inv_dict key {max(base_inv_dict.keys())}."


def calc_cont_grid_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate the cell area of the contrail grid using a simplified method.
    
    Args:
        lat (np.ndarray): Latitudes of the grid cells [deg].
        lon (np.ndarray): Longitudes of the grid cells [deg].
    
    Returns:
        np.ndarray : Contrail grid cell areas as a function of latitude [km^2].
    """

    # pre-conditions
    assert len(lat) > 0, "Latitudes cannot be empty."
    assert len(lon) > 0, "Longitudes cannot be empty."
    assert len(lat) == len(np.unique(lat)), "Duplicate latitude values."
    assert len(lon) == len(np.unique(lon)), "Duplicate longitude values."
    assert np.all((lat > -90.) & (lat < 90.)),  "Latitudes values must be "\
        "between, but not equal to, -90 and +90 degrees."
    assert np.all((lon >= 0.) & (lon <= 360.)), "Longitude values must vary " \
        "between 0 and 360 degrees."
    assert (0. in lon) != (360. in lon), "Longitude values must not include " \
        "both 0 and 360 deg values."

    # ensure that lat values descend and lon values ascend
    lat = np.sort(lat)[::-1]
    lon = np.sort(lon)

    # calculate dlon
    lon_padded = np.concatenate(([lon[-1] - 360.], lon, [lon[0] + 360.]))
    lon_midpoints = 0.5 * (lon_padded[1:] + lon_padded[:-1])
    dlon_deg = np.diff(lon_midpoints)
    dlon = np.deg2rad(dlon_deg) * R_EARTH

    # calculate dlat
    lat_padded = np.concatenate(([90], lat, [-90]))  # add +/-90 deg
    lat_midpoints = 0.5 * (lat_padded[1:] + lat_padded[:-1])
    dlat = R_EARTH * np.abs(np.sin(np.deg2rad(lat_midpoints[:-1])) -
                            np.sin(np.deg2rad(lat_midpoints[1:])))

    # calculate areas
    areas = np.outer(dlat, dlon)

    # post-conditions
    assert np.all(areas) > 0., "Not all calculated areas are positive."
    sphere_area = 4 * np.pi * R_EARTH ** 2
    assert abs(areas.sum() - sphere_area) / sphere_area < 1e-3, "Total area " \
        "calculation is insufficiently accurate."

    return areas


def interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars):
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
    assert inv_dict, "inv_dict cannot be empty."
    assert intrp_vars, "intrp_vars cannot be empty."
    if base_inv_dict:
        assert min(base_inv_dict.keys()) <= min(inv_dict.keys()), "The " \
            f"inv_dict key {min(inv_dict.keys())} is less than the earliest " \
            f"base_inv_dict key {min(base_inv_dict.keys())}."
        assert max(base_inv_dict.keys()) >= max(inv_dict.keys()), "The " \
            f"inv_dict key {max(inv_dict.keys())} is larger than the largest "\
            f"base_inv_dict key {max(base_inv_dict.keys())}."
        for intrp_var in intrp_vars:
            for yr in base_inv_dict.keys():
                assert intrp_var in base_inv_dict[yr], "Variable " \
                    f"'{intrp_var}' not present in base_inv_dict."

    # get years that need to be calculated
    inv_yrs = list(inv_dict.keys())
    base_yrs = list(base_inv_dict.keys())
    intrp_yrs = sorted(set(inv_yrs) - set(base_yrs))

    # initialise output
    full_base_inv_dict = base_inv_dict.copy()

    # if there are years in inv_dict that do not exist in base_inv_dict
    if intrp_yrs:
        # find upper and lower neighbouring base_inv_dict years
        intrp_yr_idx = np.searchsorted(base_yrs, intrp_yrs)
        yrs_lb = [base_yrs[idx-1] for idx in intrp_yr_idx]
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
                intrp_arr = np.zeros((
                    len(cc_lon_vals),
                    len(cc_lat_vals),
                    len(cc_plev_vals)
                ))
                np.add.at(
                    intrp_arr,
                    (lon_idxs, lat_idxs, plev_idxs),
                    base_inv[intrp_var].data.flatten()
                )
                regrid_base_inv[intrp_var] = xr.DataArray(
                    data=intrp_arr,
                    dims=["lon", "lat", "plev"],
                    coords={
                        "lon": cc_lon_vals,
                        "lat": cc_lat_vals,
                        "plev": cc_plev_vals,
                    }
                )

            # create dataset
            regrid_base_inv_dict[yr] = xr.Dataset(regrid_base_inv)

        # linearly interpolate base_inv
        for i, yr in enumerate(intrp_yrs):
            # linear weighting
            w = (yr - yrs_lb[i]) / (yrs_ub[i] - yrs_lb[i])
            ds_i = regrid_base_inv_dict[yrs_lb[i]] * (1 - w) + \
                   regrid_base_inv_dict[yrs_ub[i]] * w

            # reset index to match input inventories
            ds_i_flat = ds_i.stack(index=["lon", "lat", "plev"])
            ds_i_flat = ds_i_flat.reset_index("index")
            full_base_inv_dict[yr] = ds_i_flat

        # sort full_base_inv_dict
        full_base_inv_dict = dict(sorted(full_base_inv_dict.items()))

    # post-conditions
    if intrp_yrs:
        for yr in intrp_yrs:
            assert yr in full_base_inv_dict, "Missing years not included in " \
        "output dictionary."

    return full_base_inv_dict


def calc_cont_weighting(config: dict, val: str) -> np.ndarray:
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
        idxs = (cc_lat_vals > 68.) | (cc_lat_vals < -53.)  # as in AirClim 2.1
        res = np.where(idxs, 1., 0.863 * np.cos(np.pi * cc_lat_vals / 50.) + 1.615)

    # "fkt_g" in AirClim 2.1
    elif val == "w2":
        # pre-conditions
        assert "responses" in config, "Missing 'responses' key in config."
        assert "cont" in config["responses"], "Missing 'cont' key in" \
            "config['responses']."
        assert "eff_fac" in config["responses"]["cont"], "Missing eff_fac " \
            "key in config['responses']['cont']." 

        eff_fac = config["responses"]["cont"]["eff_fac"]
        res = 1. + 15. * np.abs(0.045 * np.cos(cc_lat_vals * 0.045) + 0.045) * (eff_fac - 1.)

    # Eq. 3.3.10 of Hüttenhofer (2013); RF weighting in AirClim 2.1
    elif val == "w3":
        res = 1. + 0.24 * np.cos(cc_lat_vals * np.pi / 23.)

    # raise error in case val invalid
    else:
        raise ValueError(f"Contrail weighting parameter {val} is invalid.")

    return res


def calc_cfdd(config: dict, inv_dict: dict, ds_cont: xr.Dataset) -> dict:
    """Calculate the Contrail Flight Distance Density (CFDD) for each year in
    inv_dict. This function uses the p_sac data calculated during the
    development of AirClim 2.1 (Dahlmann et al., 2016).
    
    Args:
        config (dict): Configuration dictionary from config file.
        inv_dict (dict): Dictionary of emission inventory xarrays,
            keys are inventory years.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
    
    Returns:
        dict: Dictionary with CFDD values [km/km2], keys are inventory years
    """

    # pre-conditions
    assert "responses" in config, "Missing 'responses' key in config."
    assert "cont" in config["responses"], "Missing 'cont' key in" \
        "config['responses']."
    assert "G_comp" in config["responses"]["cont"], "Missing G_comp key in " \
        "config['responses']['cont']." 

    # calculate p_sac for aircraft G
    g_comp = config["responses"]["cont"]["G_comp"]
    g_comp_con = 0.04  # EIH2O 1.25, Q 43.6e6, eta 0.3
    g_comp_lh2 = 0.12  # EIH2O 8.94, Q 120.9e6, eta 0.4
    assert ((g_comp >= g_comp_con) & (g_comp <= g_comp_lh2)), "Invalid " \
        "G_comp value. Expected range: [0.04, 0.12]."

    x = (g_comp - g_comp_con) / ( g_comp_lh2 - g_comp)
    p_sac = (1. - x) * ds_cont.SAC_CON + x * ds_cont.SAC_LH2

    # calculate contrail grid areas
    areas = calc_cont_grid_areas(cc_lat_vals, cc_lon_vals)

    # calculate CFDD
    # p_sac is interpolated using a power law over pressure level and using
    # a nearest neighbour for latitude and longitude.
    cfdd_dict = {}
    for year, inv in inv_dict.items():

        # initialise arrays for storage
        sum_km = np.zeros((len(cc_lat_vals), len(cc_lon_vals)))

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

        # calculate p_sac
        p_sac_ub = p_sac.values[lat_idxs, lon_idxs, plev_idxs]
        p_sac_lb = p_sac.values[lat_idxs, lon_idxs, plev_idxs-1]
        p_sac_intrp = sigma_plev * p_sac_lb + (1 - sigma_plev) * p_sac_ub

        # calculate and store CFDD
        # 1800s since ISS & p_sac were developed in 30min intervals
        # 3153600s in one year
        sum_contrib = inv.distance.data * p_sac_intrp * 1800.0 / 31536000.0
        np.add.at(sum_km, (lat_idxs, lon_idxs), sum_contrib)
        cfdd = sum_km / areas
        cfdd_dict[year] = cfdd

    # post-conditions
    for year, cfdd in cfdd_dict.items():
        assert cfdd.shape == (len(cc_lat_vals), len(cc_lon_vals)), "Shape " \
            f"of CFDD array for year {year} is not correct."

    return cfdd_dict


def calc_cccov(config: dict, cfdd_dict: dict, ds_cont: xr.Dataset) -> dict:
    """Calculate contrail cirrus coverage using the relationship developed for 
    AirClim 2.1 (Dahlmann et al., 2016).

    Args:
        config (dict): Configuration dictionary from config file.
        cfdd_dict (dict): Dictionary with CFDD values [km/km2], keys are
            inventory years.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.

    Returns:
        dict: Dictionary with cccov values, keys are inventory years
    """

    # pre-conditions
    assert "responses" in config, "Missing 'responses' key in config."
    assert "cont" in config["responses"], "Missing 'cont' key in" \
        "config['responses']."
    assert "eff_fac" in config["responses"]["cont"], "Missing eff_fac key " \
        "in config['responses']['cont']." 
    for year, cfdd in cfdd_dict.items():
        assert cfdd.shape == (len(cc_lat_vals), len(cc_lon_vals)), "Shape " \
            f"of CFDD array for year {year} is not correct."

    # load weighting function
    eff_fac = config["responses"]["cont"]["eff_fac"]
    w1 = calc_cont_weighting(config, "w1")

    # calculate cccov
    cccov_dict = {}
    for year, cfdd in cfdd_dict.items():
        cccov = 0.128 * ds_cont.ISS.data * np.arctan(97.7 * cfdd /
                                                     ds_cont.ISS.data)
        cccov = cccov * eff_fac * w1[:, np.newaxis]  # add corrections
        cccov_dict[year] = cccov

    # post-conditions
    for year, cccov in cccov_dict.items():
        assert cccov.shape == (len(cc_lat_vals), len(cc_lon_vals)), "Shape " \
            f"of cccov array for year {year} is not correct."

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
    assert set(comb_cccov_dict.keys()) == set(cfdd_dict.keys()), "Keys of " \
        "comb_cccov_dict and cfdd_dict do not match."
    assert set(comb_cccov_dict.keys()) == set(comb_cfdd_dict.keys()), "Keys " \
        "of comb_cccov_dict and comb_cfdd_dict do not match."

    weighted_cccov_dict = {}
    for year in comb_cccov_dict.keys():
        weighted_cccov_dict[year] = np.divide(
            comb_cccov_dict[year] * cfdd_dict[year],
            comb_cfdd_dict[year],
            where=comb_cfdd_dict[year] != 0
        )
        weighted_cccov_dict[year][comb_cfdd_dict[year] == 0] = 0.

    # post conditions
    for year in comb_cccov_dict.keys():
        assert np.all(weighted_cccov_dict[year] >= 0.), "Negative weighted " \
            "cccov values detected."

    return weighted_cccov_dict


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

    for year, cccov in cccov_dict.items():
        assert cccov.shape == (len(cc_lat_vals), len(cc_lon_vals)), "Shape " \
            f"of cccov array for year {year} is not correct."

    # calculate contril grid cell areas
    areas = calc_cont_grid_areas(cc_lat_vals, cc_lon_vals)
    w2 = calc_cont_weighting(config, "w2")
    w3 = calc_cont_weighting(config, "w3")

    # calculate total (area-weighted) cccov
    cccov_tot_dict = {}
    for year, cccov in cccov_dict.items():
        cccov_tot = (cccov * areas).sum(axis=1) * w2 * w3 / areas.sum()
        cccov_tot_dict[year] = cccov_tot

    for year, cccov_tot in cccov_tot_dict.items():
        assert cccov_tot.shape == (len(cc_lat_vals),), "Shape of cccov_tot " \
            f"array for year {year} is not correct."

    return cccov_tot_dict


def calc_cont_rf(config, cccov_tot_dict, inv_dict):
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

    # pre-conditions: check config
    assert "responses" in config, "Missing 'responses' key in config."
    assert "cont" in config["responses"], "Missing 'cont' key in" \
        "config['responses']."
    assert "PMrel" in config["responses"]["cont"], "Missing 'PMrel' key in " \
        "config['responses']['cont']." 
    assert "time" in config, "Missing 'time' key in config."
    assert "range" in config["time"], "Missing 'range' key in config['time']."
    # pre-conditions: check input dicts
    assert len(inv_dict) > 0, "inv_dict cannot be empty."
    assert len(cccov_tot_dict) > 0, "cccov_tot_dict cannot be empty."
    assert np.all(cccov_tot_dict.keys() == inv_dict.keys()), "Keys of " \
        "cccov_dict do not match those of inv_dict."
    for year, cccov_tot in cccov_tot_dict.items():
        assert cccov_tot.shape == (len(cc_lat_vals),), f"Shape of cccov_tot " \
            f"array for year {year} is not correct."

    # calculate RF factor due to PM reduction, from AirClim 2.1
    pm_rel = config["responses"]["cont"]["PMrel"]
    if pm_rel >= 0.033:
        pm_factor = 0.92 * np.arctan(1.902 * pm_rel ** 0.74)
    else:
        pm_factor = 0.92 * np.arctan(1.902 * 0.033 ** 0.74)

    # calculate contrail RF
    cont_rf_at_inv = []  # RF at inventory years
    for year, cccov_tot in cccov_tot_dict.items():
        cont_rf = 14.9 * np.sum(cccov_tot) * pm_factor
        cont_rf_at_inv.append(cont_rf)

    # interpolate RF to all simulation years
    _, rf_cont_dict = apply_evolution(config,
                                      {"cont": np.array(cont_rf_at_inv)},
                                      inv_dict)

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
    assert set(inv_dict.keys()).issubset(base_inv_dict.keys()), "inv_dict "\
        "keys are not a subset of base_inv_dict keys."

    combined_dict = {}
    for key in inv_dict.keys():
        combined_dict[key] = inv_dict[key] + base_inv_dict[key]

    return combined_dict
