"""
Calculates the contrail response.
"""

__author__ = "Liam Megill"
__email__ = "liam.megill@dlr.de"
__license__ = "Apache License 2.0"


from typing import Iterable, Mapping, MutableMapping, Any
from collections import defaultdict
import logging
import numpy as np
import numpy.typing as npt
import xarray as xr
from openairclim.read_netcdf import open_inventories, split_inventory_by_aircraft

# CONSTANTS
R_EARTH = 6371.0  # [km] radius of Earth
KAPPA = 287.0 / 1003.5

# TYPE HINTS
ContGrid = tuple[
    npt.NDArray[np.floating],  # lon
    npt.NDArray[np.floating],  # lat
    npt.NDArray[np.floating],  # plev
]

def get_cont_grid(ds_cont: xr.Dataset) -> ContGrid:
    """Get contrail grid from `ds_cont`.

    Args:
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.

    Returns:
        ContGrid: Tuple ``(lon, lat, plev)``; each is 1-D float array with
            shapes ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``.
            Units: lon [deg], lat [deg], plev [hPa].
    """
    cc_lon_vals = ds_cont.lon.data
    cc_lat_vals = ds_cont.lat.data
    cc_plev_vals = ds_cont.plev.data
    return (cc_lon_vals, cc_lat_vals, cc_plev_vals)


def check_cont_input(config: Mapping[str, Any], ds_cont: xr.Dataset) -> None:
    """Checks the input data for the contrail module.

    Args:
        config (Mapping[str, Any]): Configuration dictionary from config file.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
    """

    # check resp_cont
    if "method" not in config["responses"]["cont"]:
        raise KeyError("Missing 'method' key in config['responses']['cont'].")
    cont_method = config["responses"]["cont"]["method"]
    if cont_method not in ["Megill_2025"]:
        raise ValueError(
            "Unknown contrail method in config['responses']['cont']. "
            "Options are currently only 'Megill_2025' (default)."
        )

    # required variables for Megill et al. (2025) method
    required_vars = [
        "ppcf", "g_250", "l_1", "k_1", "x0_1", "d_1", "l_2", "k_2", "x0_2",
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


def calc_cont_grid_areas(
    lat: Iterable[float], lon: Iterable[float]
) -> np.ndarray:
    """Calculate the cell area of the contrail grid using a simplified method.

    Args:
        lat (Iterable[float]): Latitudes of the grid cells [deg].
        lon (Iterable[float]): Longitudes of the grid cells [deg].

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


def load_base_inventories(
    config: Mapping[str, Any],
    inv_yrs: Iterable[float],
    cont_grid: ContGrid,
) -> dict[str, dict[int, xr.Dataset]]:
    """Load the base emission inventories. These must at least span the same
    time range as the input emission inventories, but can also be wider. The
    base emission inventories are linearly interpolated onto years that are
    defined by the input emission inventories if those years do not otherwise
    exist.

    Args:
        config (Mapping[str, Any]): Configuration dictionary from config file.
        inv_yrs (Iterable[float]): Years for which the input emission
            inventories are defined.
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].

    Raises:
        ValueError: If the base emission inventories do not at least span the
            input emission inventories (given by `inv_yrs`).

    Returns:
        dict[str, dict[int, xr.Dataset]]: Full base emission inventory. First-
            level keys are "ac", second-level years.
    """

    # load base inventories
    base_inv_dict = open_inventories(config, base=True)
    base_yrs = list(sorted(base_inv_dict.keys()))

    # check base inventories
    if min(base_yrs) > min(inv_yrs):
        raise ValueError(
            f"The inv_dict key {min(inv_yrs)} is less than the "
            f"earliest base_inv_dict key {min(base_yrs)}."
        )
    if max(base_yrs) < max(inv_yrs):
        raise ValueError(
            f"The inv_dict key {max(inv_yrs)} is larger than the "
            f"largest base_inv_dict key {max(base_yrs)}."
        )

    # split base inventories by aircraft identifiers
    full_base_inv_dict = split_inventory_by_aircraft(
        config, base_inv_dict, base=True
    )
    base_ac_lst = list(full_base_inv_dict.keys())

    # if necessary, augment the base_inv_dict with years in inv_dict
    for base_ac in base_ac_lst:
        # add zero arrays if aircraft not defined
        full_base_inv_dict[base_ac] = pad_inv_dict(
            base_yrs,
            full_base_inv_dict[base_ac],
            ["distance"],
            cont_grid,
            base_ac
        )
        # interpolate between inventories to missing years
        full_base_inv_dict[base_ac] = interp_base_inv_dict(
            inv_yrs,
            full_base_inv_dict[base_ac],
            ["distance"],
            cont_grid
        )

    return full_base_inv_dict


def pad_inv_dict(
    inv_yrs: Iterable[int],
    inv_dict: dict[int, xr.Dataset],
    pad_vars: Iterable[str],
    cont_grid: ContGrid,
    ac: str,
) -> dict:
    """This function checks whether all years given in `inv_yrs` are present in
    the input emission inventory `inv_dict`. If a year is missing, a zero
    dataset is added to `inv_dict` for each variable in `pad_vars` on the
    pre-calculated contrail grid `cont_grid`. The `ac` variable is also added
    since this is necessary for other functions in the contrail module.
    
    This functionality can be necessary if a specific aircraft identifier is not
    included in an emission inventory passed to OpenAirClim, for example because
    the aircraft newly enters service at a later time.

    Args:
        inv_yrs (Iterable[int]): Years for which the `inv_dict` emission
            inventory should be defined.
        inv_dict (dict[int, xr.Dataset]): Dictionary of emission inventory
            xarrays, keys are inventory years.
        pad_vars (Iterable[str]): Variables to be included in the xarrays.
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].
        ac (str): Aircraft identifier from config.

    Returns:
        dict: `inv_dict` modified in-place with zero arrays in missing years.
    """

    # pre-conditions
    if "ac" in pad_vars:
        raise ValueError(
            "The 'ac' data variable is automatically added to the output "
            "xarrays and cannot be included in `pad_vars`."
        )

    # determine which years to add zero arrays to
    inp_yrs = sorted(inv_dict.keys())
    new_yrs = sorted(set(inv_yrs) - set(inp_yrs))

    # if all years are present, return the sorted inv_dict with no changes
    if not new_yrs:
        return dict(sorted(inv_dict.items()))

    # otherwise, create the zero arrays
    cc_lon_vals, cc_lat_vals, cc_plev_vals = cont_grid
    grid_shape = (len(cc_lon_vals), len(cc_lat_vals), len(cc_plev_vals))
    zero_arr = np.zeros(grid_shape)
    ac_arr = np.full(grid_shape, ac)

    # for loop over new years
    for new_yr in new_yrs:
        zero_inv = {}

        # add each variable and "ac"
        for var in pad_vars + ["ac"]:
            zero_inv[var] = xr.DataArray(
                data=ac_arr if var == "ac" else zero_arr,
                dims=["lon", "lat", "plev"],
                coords={
                    "lon": cc_lon_vals,
                    "lat": cc_lat_vals,
                    "plev": cc_plev_vals,
                },
            )

        # create dataset
        ds_i = xr.Dataset(zero_inv)
        ds_i_flat = ds_i.stack(index=["lon", "lat", "plev"])
        ds_i_flat = ds_i_flat.reset_index("index")
        inv_dict[new_yr] = ds_i_flat

    # post-conditions
    for new_yr in new_yrs:
        assert new_yr in inv_dict.keys(), (
            "Missing years not included in output dictionary."
        )

    # add message to log
    logging.info(
        "Zero-value xarrays have been created for aircraft identifier %s "
        "for the years %s", ac, new_yrs
    )

    return dict(sorted(inv_dict.items()))


def interp_base_inv_dict(
    inv_yrs: Iterable[int],
    base_inv_dict: MutableMapping[int, xr.Dataset],
    intrp_vars: Iterable[str],
    cont_grid: ContGrid,
) -> dict[int, xr.Dataset]:
    """Create base emission inventories for years in `inv_yrs` that do not
    exist in `base_inv_dict`.

    Args:
        inv_yrs (Iterable[int]): Dictionary of emission inventory xarrays,
            keys are inventory years.
        base_inv_dict (MutableMapping[int, xr.Dataset]): Dictionary of base
            emission inventory xarrays. Keys are inventory years.
        intrp_vars (Iterable[str]): List of strings of data variables in
            base_inv_dict that are to be included in the missing base
            inventories, e.g. ["distance"] (for contrail calculations).
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].

    Returns:
        dict[int, xr.Dataset]: Dictionary of base emission inventory xarrays
            including any missing years compared to inv_dict. Keys are inventory
            years.

    Note:
        A custom nearest neighbour method is used for regridding and a linear
        interpolation method for calculating data in missing years. In future
        versions, the user will be able to select methods for both.
    """

    # if base_inv_dict is empty, then return the empty dictionary.
    if not base_inv_dict:
        return {}

    # pre-conditions
    if inv_yrs is None or np.size(inv_yrs) == 0:
        raise ValueError("inv_yrs cannot be empty.")
    if not intrp_vars:
        raise ValueError("intrp_vars cannot be empty.")
    for intrp_var in intrp_vars:
        for year in base_inv_dict.keys():
            if intrp_var not in base_inv_dict[year]:
                raise KeyError(
                    f"Variable '{intrp_var}' is missing from base_inv_dict "
                    f"for year {year}."
                )

    # get years that need to be calculated
    base_yrs = sorted(base_inv_dict.keys())
    inv_yrs = sorted(set(inv_yrs))
    intrp_yrs = sorted(set(inv_yrs) - set(base_yrs))

    # initialise output
    intrp_base_inv_dict = base_inv_dict.copy()

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
            intrp_base_inv_dict[yr] = ds_i_flat

        # sort intrp_base_inv_dict
        intrp_base_inv_dict = dict(sorted(intrp_base_inv_dict.items()))

    # post-conditions
    if intrp_yrs:
        for yr in intrp_yrs:
            assert yr in intrp_base_inv_dict, (
                "Missing years not included in output dictionary."
            )

    # only return values for years defined in inv_dict
    return {yr: intrp_base_inv_dict[yr] for yr in inv_yrs}


def calc_ppcf(
    config: Mapping[str, Any],
    ds_cont: xr.Dataset,
    ac: str,
) -> xr.DataArray:
    """Calculate Potential Persistent Contrail Formation (p_PCF) using the
    precalculated contrail data from the Limiting Factors study (Megill & Grewe,
    2025; default).

    Args:
        config (Mapping[str, Any]): Configuration dictionary from config file.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        ac (str): Aircraft identifier from config.

    Returns:
        xr.DataArray: Interpolated p_PCF on precalculated contrail data grid
    """

    # pre-conditions
    if "method" not in config["responses"]["cont"]:
        raise KeyError("Missing 'method' key in config['responses']['cont'].")
    cont_method = config["responses"]["cont"]["method"]
    if cont_method not in ["Megill_2025"]:
        raise ValueError(
            "Unknown contrail method in config['responses']['cont']. "
            "Options are currently only 'Megill_2025' (default)."
        )

    # Megill_2025 method (default)
    return calc_ppcf_megill(config, ds_cont, ac)


def calc_ppcf_megill(
    config: Mapping[str, Any],
    ds_cont: xr.Dataset,
    ac: str,
) -> xr.DataArray:
    """Calculate the Potential Persistent Contrail Formation (p_PCF) using the
    Megill & Grewe (2025) method and precalculated data from ERA5.

    Args:
        config (Mapping[str, Any]): Configuration dictionary from config file.
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
            "cannot guarantee the accuracy of the fits here. If this G_250 "
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


def logistic(x: npt.ArrayLike, l: float, k: float, x0: float) -> np.ndarray:
    """Computes the logistic function, a sigmoid curve, commonly used to model
    growth or decay. Function from Megill & Grewe (2025):
    https://github.com/liammegill/contrail-limiting-factors

    Args:
        x (npt.ArrayLike): The input values for which the logistic
            function will be computed.
        l (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half
            of `l`.

    Returns:
        np.ndarray: The logistic function values for the given input `x`.
    """
    np.seterr(all="raise")
    x = np.asarray(x)
    try:
        return l / (1 + np.exp(-k * (x - x0)))
    except FloatingPointError:  # protect the exponential
        return np.nan


def logistic_gen(
    x: npt.ArrayLike, l: float, k: float, x0: float, d: float
) -> np.ndarray:
    """Computes a generalized logistic function with an additional vertical
    shift. Function from Megill & Grewe (2025):
    https://github.com/liammegill/contrail-limiting-factors

    Args:
        x (npt.ArrayLike): The input values for which the logistic
            function will be computed.
        l (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half
            of `l`.
        d (float): The vertical shift applied to the function.

    Returns:
        np.ndarray: The values of the shifted logistic function for the input
            `x`.
    """
    np.seterr(all="raise")
    x = np.asarray(x)
    try:
        return l / (1 + np.exp(-k * (x - x0))) + d
    except FloatingPointError:  # protect the exponential
        return np.nan


def calc_cfdd(
    config: Mapping[str, Any],
    inv_dict: Mapping[int, xr.Dataset],
    ds_cont: xr.Dataset,
    cont_grid: ContGrid,
    ac: str
) -> dict[int, xr.Dataset]:
    """Calculate the Contrail Flight Distance Density (CFDD) for each year in
    inv_dict. This function uses the p_pcf data calculated using ERA5
    (Megill & Grewe, 2025).

    Args:
        config (Mapping[str, Any]): Configuration dictionary from config file.
        inv_dict (Mapping[int, xr.Dataset]): Dictionary of emission inventory
            xarray datasets. Keys are inventory years.
        ds_cont (xr.Dataset): Dataset of precalculated contrail data.
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].
        ac (str): Aircraft identifier from config.

    Returns:
        dict[int, xr.Dataset]: Dictionary with CFDD values [km/km2], keys are
            inventory years
    """

    # calculate ppcf and ensure that it is of shape (lat, lon, plev)
    p_pcf = calc_ppcf(config, ds_cont, ac)
    p_pcf = p_pcf.T.transpose("lat", "lon", "plev")

    # calculate contrail grid areas
    cc_lon_vals, cc_lat_vals, cc_plev_vals = cont_grid
    areas = calc_cont_grid_areas(cc_lat_vals, cc_lon_vals)

    # cut inv_dict at pre-calculated contrail grid extremes
    inv_dict = check_plev_range(inv_dict.copy(), cont_grid)

    # calculate CFDD
    # p_pcf is interpolated using a power law over pressure level and using
    # a nearest neighbour for latitude and longitude.
    cfdd_dict = {}
    for year, inv in inv_dict.items():

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
        sum_km = np.zeros(
            (len(cc_plev_vals), len(cc_lat_vals), len(cc_lon_vals))
        )
        np.add.at(sum_km, (plev_idxs, lat_idxs, lon_idxs), sum_contrib)
        cfdd = sum_km / areas
        cfdd_dict[year] = cfdd

    # post-conditions
    for year, cfdd in cfdd_dict.items():
        assert cfdd.shape == (
            len(cont_grid[2]),
            len(cont_grid[1]),
            len(cont_grid[0]),
        ), f"Shape of CFDD for year {year} is not correct."

    return cfdd_dict


def check_plev_range(
    inv_dict: MutableMapping[int, xr.Dataset],
    cont_grid: ContGrid,
    clamp: bool = True
) -> dict[int, xr.Dataset]:
    """Checks whether all pressure level values in `inv_dict` are within the
    bounds of the pre-calculated contrail grid. Logs a warning if any values
    are found and automatically clamps values into the allowed range.

    Args:
        inv_dict (MutableMapping[int, xr.Dataset]): Dictionary of emission
            inventory xarray datasets. Keys are inventory years.
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].
        clamp (bool, optional): Whether the values should be clamped to within
            the allowed range. Defaults to True.

    Returns:
        dict[int, xr.Dataset]: Dictionary of emission inventory xarray datasets
            clamped to within the allowed plev range.
    """

    # get pre-calculated contrail plev values
    cc_plev_vals = np.asarray(cont_grid[2])
    pmin = float(cc_plev_vals.min())
    pmax = float(cc_plev_vals.max())

    # loop over inventories
    n_bad = 0
    for year, inv in inv_dict.items():
        # calculate number of values outside of range
        plev_inv = inv["plev"].values
        finite = np.isfinite(plev_inv)
        bad_low = finite & (plev_inv < pmin)
        bad_high = finite & (plev_inv > pmax)
        bad_mask = bad_low | bad_high
        n_bad += int(bad_mask.sum())

        # get maximum/minimum values for logger warning
        min_val = plev_inv.min()
        max_val = plev_inv.max()

        if clamp:
            inv_dict[year]["plev"] = np.clip(inv["plev"], pmin, pmax)

    if n_bad > 0:
        logging.warning(
            "Found %d 'plev' values outside the allowed range [%g, %g]. "
            "Observed plev values min=%g, max=%g. Values were automatically "
            "clamped into the allowed range. Use results with caution.",
            n_bad, pmin, pmax, min_val, max_val
        )

    return inv_dict


def cfdd_to_1d(
    cfdd_dict: Mapping[int, np.ndarray],
    cont_grid: ContGrid
) -> dict[int, np.ndarray]:
    """Convert 3D CFDD to 1D (lon axis) CFDD to match contrail cirrus coverage
    using a vertical sum and area-weighting to remove latitude-dependence.

    Args:
        cfdd_dict (Mapping[int, np.ndarray]): Dictionary with CFDD values
            [km/km2] in 3D (plev, lat, lon). Keys are inventory years.
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].

    Returns:
        dict[int, np.ndarray]: Dictionary with CFDD values in 1D (lon).
    """

    # get contrail grid areas
    cc_lon_vals, cc_lat_vals, _ = cont_grid
    areas = calc_cont_grid_areas(cc_lat_vals, cc_lon_vals)

    cfdd_1d = {ac: {} for ac in cfdd_dict.keys()}
    for ac in cfdd_dict.keys():
        for yr, val in cfdd_dict[ac].items():
            cfdd_2d = val.sum(axis=0)
            cfdd_1d[ac][yr] = (cfdd_2d * areas).sum(axis=0) / areas.sum(axis=0)

    return cfdd_1d


def pm_factor_high(x: float, params: tuple) -> float:
    """Calculate nvPM factor in the high-soot regime.

    Args:
        x (float): relative nvPM emissions with respect to 1.5e15 kg^-1
        params (tuple): fit parameters

    Returns:
        float: nvPM factor
    """
    a, b, c = params
    return a * np.arctan(b * x**c)


def pm_factor_high_prime(x: float, params: tuple) -> float:
    """Calculate the first-order derivative of the nvPM factor in the high-soot
    regime.

    Args:
        x (float): relative nvPM emissions with respect to 1.5e15 kg^-1
        params (tuple): fit parameters

    Returns:
        float: derivative of the nvPM factor
    """
    a, b, c = params
    return a * (1.0 / (1.0 + (b * x**c) ** 2)) * b * c * x ** (c - 1.0)


def pm_factor_low(x: float, case_params: tuple, params: tuple) -> float:
    """Calculate the nvPM factor in the low-soot regime.

    Args:
        x (float): relative nvPM emissions with respect to 1.5e15 kg^-1
        case_params (tuple): parameters for low-soot case
        params (tuple): fit parameters

    Returns:
        float: nvPM factor
    """
    x0, c0, k, p = case_params
    y0 = pm_factor_high(x0, params)
    m0 = pm_factor_high_prime(x0, params)
    d = (m0 - (y0 - c0) * (k / (2.0 * x0))) * x0

    # create case
    u = x / x0
    w = 2.0 * u**k / (1.0 + u**k)
    h = (u**p) * (1.0 - u)
    return c0 + (y0 - c0) * w - d * h


def pm_factor(x: float, ls_case: str = "case1") -> float:
    """Calculate the nvPM factor depending on the relative nvPM emissions.
    Note that the factor is not validated in the low-soot region (x < 0.1).

    Args:
        x (float): relative nvPM emissions with respect to 1.5e15 kg^-1
        ls_case (str, optional): Descriptor of pre-calculated case.
            One of: "case1" (mid), "case2" (low) or "case3" (high).
            Defaults to "case1".

    Raises:
        ValueError: For invalid nvPM emissions.

    Returns:
        float: nvPM factor
    """

    # pre-conditions
    if x < 0.0:
        raise ValueError("nvPM emissions must be positive.")
    if 0.0 < x < 0.1:
        logging.warning(
            "Selected nvPM emissions are in the low-soot regime, which is not"
            "validated. Use contrail results with caution."
        )
    if ls_case not in ["case1", "case2", "case3"]:
        raise ValueError(f"Unknown low_soot_case {ls_case}.")

    # predefined parameters
    pmfac_cases = {
        "case1": (0.1, 0.385, 2.0, 2.5),
        "case2": (0.1, 0.08, 1.0, 0.3),
        "case3": (0.1, 0.8, 3.5, 1.2),
    }
    params = (0.91, 1.96, 0.58)

    # choose regime for calculation
    conditions = [(x == 0.0), ((0.0 < x) & (x < 0.1))]
    choices = [0.385, pm_factor_low(x, pmfac_cases[ls_case], params)]
    result = np.select(conditions, choices, default=pm_factor_high(x, params))
    return result


def calc_cccov_alltau(
    cfdd_dict: Mapping[int, np.ndarray],
    cont_grid: ContGrid,
) -> dict[int, np.ndarray]:
    """Calculate contrail cirrus coverage (all tau) using the
    Megill et al. (2025) method.

    Args:
        cfdd_dict (Mapping[int, np.ndarray]): Dictionary with 1D (lon) CFDD
            values [km/km2]. Keys are inventory years.
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].

    Returns:
        dict[int, np.ndarray]: Dictionary with 1D (lon) cccov (all tau) values.
            Keys are inventory years
    """

    # pre-conditions
    cc_lon_vals, cc_lat_vals, cc_plev_vals = cont_grid
    for year, cfdd in cfdd_dict.items():
        assert cfdd.shape == (
            len(cc_plev_vals), len(cc_lat_vals), len(cc_lon_vals),
        ), f"Shape of CFDD array for year {year} is not correct."

    # calculate areas
    areas = calc_cont_grid_areas(cc_lat_vals, cc_lon_vals)

    # calculate cccov
    cccov_dict = {}
    for year, cfdd in cfdd_dict.items():
        cov_3d = 7.722e-2 * np.tanh(1.323e2 * cfdd)
        cov_2d = 1.0 - np.prod(1.0 - cov_3d, axis=0)
        cov_1d = (cov_2d * areas).sum(axis=0) / areas.sum(axis=0)
        cccov_dict[year] = cov_1d

    # post-conditions
    for year, cccov in cccov_dict.items():
        assert cccov.shape == (len(cc_lon_vals),), (
            f"Shape of cccov array for year {year} is not correct."
        )

    return cccov_dict


def calc_cccov_taup05(
    config: Mapping[str, Any],
    cccov_dict: Mapping[int, np.ndarray],
    ac: str
) -> dict[int, np.ndarray]:
    """Convert contrail cirrus coverage (all tau) to optically thick contrail
    cirrus coverage (tau > 0.05).

    Args:
        config (Mapping[str, Any]): Configuration dictionary from config file.
        cccov_dict (Mapping[int, np.ndarray]): Dictionary with 1D (lon) contrail
            cirrus coverage (all optical thicknesses). Keys are inventory years.
        ac (str): Aircraft identifier from config.

    Raises:
        KeyError: If "PMrel" not defined in config.
        KeyError: If "low_soot_case" not defined in config (OpenAirClim will
            default to "case1").

    Returns:
        dict[int, np.ndarray]: Dictionary with 1D (lon) cccov (tau > 0.05)
            values. Keys are inventory years.
    """

    # pre-conditions
    if "PMrel" not in config["aircraft"][ac]:
        raise KeyError(f"Missing 'PMrel' key in config['aircraft']['{ac}'].")
    if "low_soot_case" not in config["responses"]["cont"]:
        raise KeyError(
            "Missing 'low_soot_case' key in config['responses']['cont']."
        )
    pm_rel = config["aircraft"][ac]["PMrel"]
    ls_case = config["responses"]["cont"]["low_soot_case"]

    # convert all tau -> tau > 0.05
    cccov_dict_taup05 = {}
    for year, cccov in cccov_dict.items():
        cov_p05 = cccov * pm_factor(pm_rel, ls_case)
        cccov_dict_taup05[year] = cov_p05

    # post-conditions
    for year, cov_p05 in cccov_dict_taup05.items():
        assert cov_p05.shape == cccov_dict[year].shape, (
            f"Shape of cccov array for year {year} is not correct."
        )

    return cccov_dict_taup05


def proportional_attribution(
    input_dict: dict[int, np.ndarray],
    ac_dict: dict[int, np.ndarray],
    total_dict: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """
    Use proportional attribution to split the input into the contribution from
    ac_dict (single aircraft identifier). The keys of all inputs must match.

    Args:
        input_dict (dict[int, np.ndarray]): Dictionary to be split, keys are
            inventory years.
        ac_dict (dict[int, np.ndarray]): Dictionary with values for a single
            aircraft identifier, could be CFDD or coverage values for example.
            Keys are inventory years.
        total_dict (dict[int, np.ndarray]): Dictionary with values for all
            aircraft identifiers, could be CFDD or coverage values for example
            (must match ac_dict). Keys are inventory years

    Returns:
        dict[int, np.ndarray]: Dictionary with proportionally attributed values.
            Keys are years.
    """

    # pre-conditions
    assert set(input_dict.keys()) == set(total_dict.keys()), (
        "Keys of input_dict and total_dict do not match."
    )
    assert set(input_dict.keys()) == set(ac_dict.keys()), (
        "Keys of input_dict and ac_dict do not match."
    )

    att_dict = {}
    for year in input_dict.keys():
        att_dict[year] = np.divide(
            input_dict[year] * ac_dict[year],
            total_dict[year],
            where=total_dict[year] != 0,
        )
        att_dict[year][total_dict[year] == 0] = 0.0

    # post conditions
    for year in total_dict.keys():
        assert np.all(att_dict[year] >= 0.0), (
            "Negative attributed values detected."
        )

    return att_dict


def calc_cont_rf(
    cccov_dict: dict[int, np.ndarray],
    cont_grid: ContGrid
) -> dict[int, np.ndarray]:
    """Calculate contrail Radiative Forcing (RF) using the Megill et al. (2025)
    method.

    Args:
        cccov_dict (dict[int, np.ndarray]): Dictionary with 1D (lon) contrail
            cirrus coverage (tau > 0.05; shape: lon), keys are inventory years
        cont_grid (tuple): Precalculated contrail grid.
            Shape ``(lon, lat, plev)``; each is 1-D float array with shapes
            ``(n_lon,)``, ``(n_lat,)``, ``(n_plev,)``. Units: lon [deg],
            lat [deg], plev [hPa].

    Returns:
        dict[int, np.ndarray]: Dictionary with contrail RF values for all
            inventory years.
    """

    # pre-conditions: check config
    assert len(cccov_dict) > 0, "cccov_dict cannot be empty."
    for year, cccov in cccov_dict.items():
        assert cccov.shape == (
            len(cont_grid[0]),
        ), f"Shape of cccov_tot array for year {year} is not correct."

    # set up longitude-dependent regions
    rf_fit_arr = [[9.64, 1.02], [10.81, 1.0], [1.93, 0.81], [20.33, 1.14]]
    lon_bins = [-30, 60, 160, 235, 330]
    shift = lon_bins[0]
    lon_shifted = (cont_grid[0] - shift) % 360.0
    lon_bins_shifted = np.array(lon_bins) - shift
    lon_idxs = np.digitize(lon_shifted, lon_bins_shifted) - 1

    # calculate RF
    rf_dict = {}
    for year, cccov in cccov_dict.items():
        rf_arr = np.empty((cont_grid[0].shape))
        for i in range(len(lon_bins) - 1):
            mask = lon_idxs == i
            rf_arr[mask] = rf_fit_arr[i][0] * cccov[mask] ** rf_fit_arr[i][1]
        rf_dict[year] = rf_arr

    return rf_dict


def apply_wingspan_correction(
    config: Mapping[str, Any], rf_arr: Iterable[float], ac: str
) -> np.ndarray:
    """Apply wingspan correction to an array of RF values. The function is
    applied against a reference wingspan of 35 m.
    References: Bruder et al. (2025) and Megill et al. (2025).

    Args:
        config (Mapping[str, Any]): Configuration dictionary from config file.
        rf_arr (Iterable[float]): RF values
        ac (str): Aircraft identifier from config.

    Raises:
        KeyError: If wingspan "b" not defined for aircraft "ac" in config.
        ValueError: If wingspan "b" is outside of valid range [20 m, 80 m].

    Returns:
        np.ndarray: RF values with wingspan correction
    """

    # pre-conditions
    if "b" not in config["aircraft"][ac]:
        raise KeyError(f"Missing 'b' key in config['aircraft']['{ac}'].")
    if not 20.0 < config["aircraft"][ac]["b"] < 80.0:
        raise ValueError(
            f"Invalid wingspan {config['aircraft'][ac]['b']}. Must be "
            "within [20 m, 80 m]."
        )

    v = 0.396
    w = 0.0287
    b_ref = 35.0
    b = config["aircraft"][ac]["b"]
    corr = (v + w * b) / (v + w * b_ref)
    return rf_arr * corr


def calc_total_over_ac(
    data: Mapping[str, Mapping[int, Any]], ac_lst: Iterable[str],
) -> dict[str, dict[int, Any]]:
    """Add a "TOTAL" entry to `data` by summing the per-year values across all
    aircraft identifiers.

    Args:
        data (Mapping[str, Mapping[int, Any]]): Nested dictionary with keys ac,
            then yr
        ac_lst (Iterable[str]): List of aircraft identifiers to be used

    Returns:
        dict[str, dict[int, Any]]: Shallow copy of data with TOTAL added or
            overwritten
    """

    out: dict[str, dict[int, Any]] = {k: dict(v) for k, v in data.items()}
    total = defaultdict(float)
    for ac in ac_lst:
        for yr, val in data[ac].items():
            total[yr] += val
    out["TOTAL"] = dict(total)
    return out
