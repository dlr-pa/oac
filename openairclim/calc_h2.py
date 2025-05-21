"""
Calculates the climate response of hydrogen fugitive emissions.
"""

import copy
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import xarray as xr

# from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import tensorflow as tf

from openairclim.read_netcdf import open_netcdf

Y_TO_S = 365.25 * 24 * 60 * 60
M_TO_S = 60 * 60 * 24 * 365.25 / 12
PPB_TO_CM3 = 2.46e10


def convert_mass_to_concentration(mass, m_x):
    """
    Convert mass in [Tg] to concentration in [ppb].
    :param mass: mass of species in [Tg]
    :param m_x: molar mass of species [g/mol]
    :return: concentration of species in [ppb]
    """
    mass_trop = 4.22e18  # tropospheric mass [kg]
    m_air = 29  # molar mass of air [g/mol]
    return mass * 1e9 / m_x / mass_trop * m_air / 1e-9


def load_data(wd, scenarios, load_2d=False):
    """
    Load data from netCDF files for the given scenarios.
    Args:
        wd: working directory
        scenarios: list of scenario names
        load_2d: whether to load 2D data
        species: list of species to load (only used when load_2d=True)
    Returns:
        dict: dictionary of datasets keyed by scenario name
    """
    d = {}
    for s, sc in enumerate(scenarios):
        if load_2d:
            # Find all files for the scenario
            fn = glob.glob(wd + f"\\repository\\h2\\SSP_scenarios\\*{sc}*.nc")
            if not fn:
                print(f"Warning: No files found for scenario {sc}")
                continue

            # Load the first file to get the dataset
            ds = open_netcdf(fn[0])

            # Get all keys from the dataset
            dataset_keys = list(ds.keys())

            # Process each key in the dataset
            lst = []
            for key in dataset_keys:
                # Get the data for this key
                data = ds[key]

                # Handle longitude values (ensure 360° is included)
                if "lon" in data.dims and 360 not in data["lon"].values:
                    # Duplicate the 0° longitude data
                    data_360 = data.sel(lon=0).copy()
                    data_360["lon"] = 360  # Set longitude to 360°

                    # Append the 360° data to the original DataArray
                    data = xr.concat([data, data_360], dim="lon")

                # Add to the list
                lst.append(data)

            # Merge the datasets if there are multiple keys
            if len(lst) > 1:
                ds = xr.merge(lst)
            elif len(lst) == 1:
                ds = lst[0]
            else:
                print(f"Warning: No data found for scenario {sc}")
                continue

            d[sc] = ds
        else:
            fn = glob.glob(wd + f"\\repository\\h2\\SSP_scenarios\\bg_{sc}.nc")
            print(fn)
            d[sc] = open_netcdf(fn)[f"bg_{sc}"].mean(dim=["lat", "lon"])

        d[sc]["scenario"] = s
    return d


class EmissionModel:
    """
    Model for calculating hydrogen emissions from aviation.
    """

    def __init__(
        self,
        working_directory,
        consumption_scenario,
        t_midpoint,
        m_adoption,
        f_application,
        f_delivery,
        f_production,
    ):
        # Meta
        self.wd = working_directory  # working directory from project root define in __init__.py (path)
        self.cs = "BAU"  # name of consumption scenario (string)
        self.cs_ds = (
            self.read_aviation_fuel_consumption()
        )  # consumption scenario data (xr.Dataset)
        self.eh2_ds = None  # equivalent hydrogen consumption data (xr.Dataset)

        # Parameters
        self.t_mid = t_midpoint  # year of midpoint of adoption curve (int)
        self.m_adp = m_adoption  # adoption rate - slope of adoption curve (float)
        self.f_adp_ds = (
            self.calculate_hydrogen_adoption_fraction()
        )  # temporal evolution of h2 adoption (xr.Dataset)
        self.f_app = f_application  # leakage fraction during application (float)
        self.f_del = f_delivery  # leakage fraction during delivery (float)
        self.f_prd = f_production  # leakage fraction during production (float)

        # Constants
        self.lwh_ker = 43.15  # lower heating potential of kerosene (MJ/kg)
        self.lwh_h2 = 120  # lower heating potential of h2 (MJ/kg)

    def read_aviation_fuel_consumption(self):
        """Read aviation fuel consumption data from csv files and return as xarray dataset
        Returns:
            xr.Dataset: aviation fuel consumption data
        """
        d = {}
        if self.cs is None:
            raise ValueError("No consumption scenario provided")
        else:
            fn = glob.glob(
                self.wd + f"\\repository\\h2\\fuel_consumption_scenarios\\{self.cs}*"
            )
            d[self.cs] = pd.read_csv(fn[0])
            ds = xr.Dataset(d)
            ds = ds.rename({"dim_0": "time"}).isel(dim_1=1).drop_vars("dim_1")
            ds["time"] = xr.date_range("2025-01-01", periods=ds["time"].size, freq="YS")
            return ds

    def calculate_equivalent_hydrogen_consumption(self):
        """Calculate equivalent hydrogen consumption
        Returns:
            xr.Dataset: equivalent hydrogen consumption
        """
        return self.cs_ds * self.lwh_ker / self.lwh_h2

    def calculate_hydrogen_adoption_fraction(self, t_mid=None, m_adp=None):
        """Calculate hydrogen adoption fraction
        Args:
            t_mid (int): midpoint year
            m_adp (float): adoption rate (slope of curve)
        Returns:
            xr.Dataset: hydrogen adoption fraction
        """
        y = self.cs_ds["time"].dt.year
        if t_mid is not None and m_adp is not None:
            return 1 / (1 + np.exp(-m_adp * (y - t_mid)))
        elif t_mid is None and m_adp is None:
            return 1 / (1 + np.exp(-self.m_adp * (y - self.t_mid)))
        else:
            raise ValueError("Provide both t_mid and m_adp or None")

    def calculate_hydrogen_application_mass(self):
        """ " Calculate the total hydrogen mass required for application (flight) using the adoption fraction and leakage
        fraction during application
        Returns:
            xr.Dataset: hydrogen mass required for application
        """
        return (
            self.f_adp_ds
            * self.calculate_equivalent_hydrogen_consumption()
            / (1 - self.f_app)
        )

    def calculate_hydrogen_delivery_mass(self):
        """Calculate the total hydrogen mass required for delivery using the application mass and leakage fraction
        during delivery
        Returns:
            xr.Dataset: hydrogen mass required for delivery
        """
        return self.calculate_hydrogen_application_mass() / (1 - self.f_del)

    def calculate_hydrogen_production_mass(self):
        """Calculate the total hydrogen mass required for production using the delivery mass and leakage fraction
        during production
        Returns:
            xr.Dataset: hydrogen mass required for production
        """
        return self.calculate_hydrogen_delivery_mass() / (1 - self.f_prd)

    def calculate_hydrogen_emission_rate(self):
        """Calculate the total fugitive emissions
        Returns:
            xr.Dataset: fugitive emissions (emih2)
        """
        ds = (
            self.f_app * self.calculate_hydrogen_application_mass()
            + self.f_del * self.calculate_hydrogen_delivery_mass()
            + self.f_prd * self.calculate_hydrogen_production_mass()
        )
        if len(ds.data_vars) == 1:
            ds = ds.rename({list(ds.data_vars)[0]: "emih2"})
        return ds


class BoxModel:
    """
    Box model for calculating atmospheric chemistry changes due to hydrogen emissions.
    """

    def __init__(self, data, rate_of_deposition, spinup_time, start_year):
        # Meta
        self.data = data
        self.y_start = pd.to_datetime(start_year, format="%Y")
        self.y_end = pd.to_datetime(self.data["time"][-1].values)
        self.horizon = self.y_end.year - (self.y_start.year - 1)
        self.t_spin = np.arange(0, spinup_time, 1)
        self.t_span = (0, spinup_time + self.horizon)
        self.t_eval = np.arange(spinup_time, self.t_span[1], 1)

        # Parameters
        self.kd = rate_of_deposition  # hydrogen deposition rate
        self.ks = 0.02  # methane sink rate
        self.alpha = 0.37  # methane feedback yield

        # Constants - rate constants
        self.k1 = 3.17e-15 * PPB_TO_CM3 * Y_TO_S  # CH4 + OH -> products
        self.k2 = 3.80e-15 * PPB_TO_CM3 * Y_TO_S  # H2 + OH -> products
        self.k3 = 1.90e-13 * PPB_TO_CM3 * Y_TO_S  # CO + OH -> products
        self.k4 = 0.3 * Y_TO_S  # OH loss

        # Background production rates (ppb/yr)
        self.p_ch4 = 60  # methane production
        self.p_oh = 1333  # hydroxyl production
        self.p_co = 200  # carbon monoxide production
        self.p_h2 = 265  # hydrogen production

        # Initial hydrogen concentration (ppb)
        self.c_h2 = 500

    def interpolate_sources(self, sources):
        """
        Interpolate source terms for the ODE solver.
        Args:
            sources: Dictionary of source terms
        Returns:
            tuple: Interpolation functions for each source
        """
        S_ch4_func = interp1d(
            self.t_eval, sources["emich4"], kind="linear", fill_value="extrapolate"
        )
        S_co_func = interp1d(
            self.t_eval, sources["emico"], kind="linear", fill_value="extrapolate"
        )
        S_h2_func = interp1d(
            self.t_eval, sources["emih2"], kind="linear", fill_value="extrapolate"
        )
        S_oh_func = interp1d(
            self.t_eval, sources["emioh"], kind="linear", fill_value="extrapolate"
        )
        return S_ch4_func, S_co_func, S_h2_func, S_oh_func

    def system_of_odes(self, t, y, sources):
        """
        System of ODEs for the box model.
        Args:
            t: time
            y: state vector [CH4, CO, OH, H2]
            sources: Dictionary of source terms
        Returns:
            list: Derivatives [dCH4_dt, dCO_dt, dOH_dt, dH2_dt]
        """
        S_ch4_func, S_co_func, S_h2_func, S_oh_func = self.interpolate_sources(sources)

        S_ch4 = S_ch4_func(self.t_eval[0])
        S_co = S_co_func(self.t_eval[0])
        S_h2 = S_h2_func(self.t_eval[0])
        S_oh = S_oh_func(self.t_eval[0])

        if t >= self.t_eval[0]:
            S_ch4 = S_ch4_func(t)
            S_co = S_co_func(t)
            S_h2 = S_h2_func(t)

        CH4, CO, OH, H2 = y

        R_ch4 = self.k1 * OH * CH4
        R_h2 = self.k2 * OH * H2
        R_co = self.k3 * OH * CO
        R_x = self.k4 * OH
        R_d = self.kd * H2
        R_s = self.ks * CH4

        dCH4_dt = S_ch4 - R_ch4 - R_s
        dH2_dt = S_h2 + (self.alpha * R_ch4) - R_h2 - R_d
        dCO_dt = S_co + R_ch4 - R_co
        dOH_dt = S_oh - R_ch4 - R_h2 - R_co - R_x

        return [dCH4_dt, dCO_dt, dOH_dt, dH2_dt]

    def solver(self, ds):
        """
        Solve the system of ODEs.
        Args:
            ds: Dataset containing initial conditions and source terms
        Returns:
            tuple: Solution and time array
        """
        t = self.t_eval
        y_0 = [
            ds["ch4_trop"][0] * 1e9,
            ds["co_trop"][0] * 1e9,
            ds["oh_trop"][0] * 1e9,
            self.c_h2,
        ]
        sources = {
            "emich4": ds["emich4"],
            "emico": ds["emico"],
            "emih2": ds["emih2"],
            "emioh": ds["emioh"],
        }
        sol = solve_ivp(
            self.system_of_odes,
            self.t_span,
            y_0,
            method="Radau",
            rtol=1e-6,
            atol=1e-6,
            dense_output=True,
            args=(sources,),
        )
        return sol.sol(t), t

    def prepare_data(self, ds, ds_eh2, perturbation=False):
        """
        Prepare data for the box model.
        Args:
            ds: Dataset to prepare
            ds_eh2: Hydrogen emissions dataset
            perturbation: Whether to include perturbation
        Returns:
            xr.Dataset: Prepared dataset
        """
        if perturbation:
            ds["emih2"] = ds_eh2["emih2"]
            ds["emih2"] = ds["emih2"].fillna(0)
            ds["emih2"] = convert_mass_to_concentration(ds["emih2"], 2) + self.p_h2
        else:
            ds["emih2"] = self.p_h2
            ds["emih2"] = ds["emih2"].broadcast_like(ds["emico"])
        ds["emich4"] = (
            convert_mass_to_concentration(ds["emich4"] * 1e-9, 16) + self.p_ch4
        )
        ds["emico"] = convert_mass_to_concentration(ds["emico"] * 1e-9, 28) + self.p_co
        ds["emioh"] = self.p_oh
        ds["emioh"] = ds["emioh"].broadcast_like(ds["emico"])
        return ds

    def get_perturbation(self, ds_eh2):
        """
        Calculate perturbations to atmospheric chemistry due to hydrogen emissions.
        Args:
            ds_eh2: Hydrogen emissions dataset
        Returns:
            xr.Dataset: Perturbation dataset
        """
        ds = self.data.sel(time=slice(self.y_start, None))
        ds = ds[["ch4_trop", "co_trop", "oh_trop", "emico", "emich4"]]

        ds_base = self.prepare_data(copy.deepcopy(ds), ds_eh2)
        ds_pert = self.prepare_data(copy.deepcopy(ds), ds_eh2, perturbation=True)

        y_base, t_base = self.solver(ds_base)
        y_pert, t_pert = self.solver(ds_pert)

        ds = copy.deepcopy(self.data.sel(time=slice(self.y_start, None)))
        ds = ds[["ch4_trop", "co_trop", "oh_trop", "emico", "emich4"]]

        ds["h2"] = xr.DataArray(y_base[3], coords=[ds["time"]], dims=["time"]) * 1e-9
        ds = ds.bfill(dim="time")
        ds["dch4"] = xr.DataArray(
            (y_pert[0] - y_base[0]) * 1e-9, coords=[ds["time"]], dims=["time"]
        )
        ds["dco"] = xr.DataArray(
            (y_pert[1] - y_base[1]) * 1e-9, coords=[ds["time"]], dims=["time"]
        )
        ds["doh"] = xr.DataArray(
            (y_pert[2] - y_base[2]) * 1e-9, coords=[ds["time"]], dims=["time"]
        )
        ds["dh2"] = xr.DataArray(
            (y_pert[3] - y_base[3]) * 1e-9, coords=[ds["time"]], dims=["time"]
        )
        ds = ds.fillna(0)

        return ds


class AutoregressiveForecastingModel:
    """
    Model for making autoregressive forecasts based on LSTM neural networks.
    """

    def __init__(
        self, working_directory, test_scenario, start_year, labels, features, diff=None
    ):
        self.test_sc = test_scenario
        self.start = start_year - 1970

        self.y = labels
        self.X = features
        self.g = ["scenario"]
        if diff is None:
            self.diff = self.y + self.X
        else:
            self.diff = diff
        self.length = 15
        self.n_step = 5

        self.t_scale_params = None
        self.s_scale_params = None

        self.scale = "minmax"
        self.diff_params = {}
        self.dims = {}

        self.wd = working_directory

    def scale_spatial(self, ds):
        """
        Scale spatial dimensions of the dataset.
        Args:
            ds: Dataset to scale
        Returns:
            xr.Dataset: Scaled dataset
        """
        scale_lst = []
        for var in self.y + self.X:
            if "time" in ds[var].dims:
                mean = ds[var].mean(dim=["time", "scenario"])
                std = ds[var].std(dim=["time", "scenario"])
                std = std.where(std != 0, other=1)
                ds[var] = (ds[var] - mean) / std
                ds_ss = xr.Dataset({"mean": mean, "std": std})
                scale_lst.append(ds_ss)
        self.s_scale_params = xr.concat(scale_lst, dim="variable")
        self.s_scale_params = self.s_scale_params.assign_coords(
            variable=self.y + self.X
        )
        return ds

    def rescale_spatial(self, ds):
        """
        Rescale spatial dimensions of the dataset.
        Args:
            ds: Dataset to rescale
        Returns:
            xr.Dataset: Rescaled dataset
        """
        for var in self.y + self.X:
            if var in self.s_scale_params["variable"]:
                mean = self.s_scale_params.sel(variable=var)["mean"]
                std = self.s_scale_params.sel(variable=var)["std"]
                ds[var] = ds[var] * std + mean
        return ds

    def differentiate(self, ds):
        """
        Differentiate the dataset with respect to time.
        Args:
            ds: Dataset to differentiate
        Returns:
            xr.Dataset: Differentiated dataset
        """
        for var in self.diff:
            self.diff_params[var] = {
                f"{var}_init": ds[var].isel(time=0, scenario=0).values.tolist()
            }
            ds[var] = ds[var].diff(dim="time", label="upper")
        ds = ds.dropna(dim="time")
        return ds

    def integrate(self, ds):
        """
        Integrate the dataset with respect to time.
        Args:
            ds: Dataset to integrate
        Returns:
            xr.Dataset: Integrated dataset
        """
        for var in self.diff:
            init_value = self.diff_params[var][f"{var}_init"]
            non_time_dims = [dim for dim in ds[var].dims if dim != "time"]
            coords = {dim: ds[var][dim] for dim in non_time_dims}
            init_da = xr.DataArray(data=init_value, dims=non_time_dims, coords=coords)

            init_da = init_da.expand_dims(
                dim={"time": [ds["time"].values[0] - np.timedelta64(1, "D")]}
            )
            full_da = xr.concat([init_da, ds[var]], dim="time")
            full_da = full_da.cumsum(dim="time")
            full_da = full_da.sel(time=ds["time"])
            ds[var] = full_da
        return ds

    def scaler(self, ds):
        """
        Scale the dataset.
        Args:
            ds: Dataset to scale
        Returns:
            xr.Dataset: Scaled dataset
        """
        if self.scale == "standard":
            for var in self.y + self.X:
                mean = self.t_scale_params[var]["mean"]
                std = self.t_scale_params[var]["std"]
                ds[var] = (ds[var] - mean) / std
        if self.scale == "minmax":
            for var in self.y + self.X:
                max_p = self.t_scale_params[var]["max_p"]
                min_p = self.t_scale_params[var]["min_p"]
                ds[var] = 2 * (ds[var] - min_p) / (max_p - min_p) - 1
        if self.scale == "minmax-0":
            for var in self.y + self.X:
                max_p = ds[var].max()
                min_p = ds[var].min()
                ds[var] = (ds[var] - min_p) / (max_p - min_p)
        return ds

    def reverse_scaler(self, ds):
        """
        Reverse the scaling of the dataset.
        Args:
            ds: Dataset to reverse scale
        Returns:
            xr.Dataset: Reverse scaled dataset
        """
        if self.scale == "standard":
            for var in self.y + self.X:
                mean = self.t_scale_params[var]["mean"]
                std = self.t_scale_params[var]["std"]
                ds[var] = ds[var] * std + mean
        if self.scale == "minmax":
            for var in self.y + self.X:
                max_p = self.t_scale_params[var]["max_p"]
                min_p = self.t_scale_params[var]["min_p"]
                ds[var] = (ds[var] + 1) * (max_p - min_p) / 2 + min_p
        if self.scale == "minmax-0":
            for var in self.y + self.X:
                max_p = ds[var].max()
                min_p = ds[var].min()
                ds[var] = ds[var] * (max_p - min_p) + min_p
        return ds

    def dataset_to_array(self, ds):
        """
        Convert dataset to numpy array.
        Args:
            ds: Dataset to convert
        Returns:
            np.ndarray: Array representation of dataset
        """
        for dim in ds.dims:
            self.dims[dim] = ds[dim]
        return np.stack([ds[var].values for var in self.y + self.X], axis=-1)

    def array_to_dataset(self, arr):
        """
        Convert numpy array to dataset.
        Args:
            arr: Array to convert
        Returns:
            xr.Dataset: Dataset representation of array
        """
        if len(arr.shape) == 5:
            return xr.Dataset(
                {
                    var: (["scenario", "time", "lat", "lon"], arr[:, :, :, :, i])
                    for i, var in enumerate(self.y + self.X)
                },
                coords={
                    "scenario": self.dims["scenario"],
                    "time": self.dims["time"],
                    "lat": self.dims["lat"],
                    "lon": self.dims["lon"],
                },
            )
        elif len(arr.shape) == 4:
            return xr.Dataset(
                {
                    var: (["time", "lat", "lon"], arr[:, :, :, i])
                    for i, var in enumerate(self.y + self.X)
                },
                coords={
                    "time": self.dims["time"],
                    "lat": self.dims["lat"],
                    "lon": self.dims["lon"],
                },
            )
        elif len(arr.shape) == 3:
            return xr.Dataset(
                {
                    var: (["scenario", "time"], arr[:, :, i])
                    for i, var in enumerate(self.y + self.X)
                },
                coords={"scenario": self.dims["scenario"], "time": self.dims["time"]},
            )
        else:
            return xr.Dataset(
                {var: (["time"], arr[:, i]) for i, var in enumerate(self.y + self.X)},
                coords={"time": self.dims["time"]},
            )

    def prepare_data_test(self, d):
        """
        Prepare test data for prediction.
        Args:
            d: Dictionary of datasets
        Returns:
            np.ndarray: Prepared test data
        """
        test_arr = None, None, None

        test_datasets = []

        for key, dataset in d.items():
            dataset = dataset[self.y + self.X + self.g]
            dataset = dataset.set_coords("scenario")
            if len(dataset.dims) > 2:
                dataset = dataset.assign_coords(lon=(dataset["lon"] + 180) % 360 - 180)
                dataset = dataset.sortby("lon")
                data_180 = dataset.sel(lon=-180).copy()
                data_180["lon"] = 180
                dataset = xr.concat([dataset, data_180], dim="lon")
            dataset = dataset.drop_vars("scenario")
            if key in self.test_sc:
                test_datasets.append(dataset)

        if test_datasets:
            ds_test = xr.concat(test_datasets, dim="scenario")
            ds_test = self.differentiate(ds_test)
            ds_test = self.scaler(ds_test)
            test_arr = self.dataset_to_array(ds_test)
        return test_arr

    def autoregressor(self, model, arr_test):
        """
        Run autoregressive forecasting.
        Args:
            model: LSTM model
            arr_test: Test data array
        Returns:
            np.ndarray: Forecast array
        """
        if len(arr_test.shape) == 3:
            arr_test = arr_test[0, :, :]
            arr_roll = arr_test[self.start - self.length : self.start]
            arr_pred = arr_test[: self.start]
            for i in range(self.start, len(arr_test), self.n_step):
                inp = arr_roll[self.n_step :, :]
                inp = np.expand_dims(inp, axis=0)
                pred = model.predict(inp, verbose=0)[0, :, :]
                # Only use the first column (label prediction) and concatenate with features
                pred_label = pred[:, 0:1]  # Take only the label prediction
                pred = np.concatenate(
                    [pred_label, arr_test[i : i + self.n_step, 1:]], axis=-1
                )
                arr_roll = np.concatenate([arr_roll, pred], axis=0)[self.n_step :, :]
                arr_pred = np.concatenate(
                    [arr_pred, arr_roll[-self.n_step :, :]], axis=0
                )
        else:
            arr_test = arr_test[0, :, :, :, :]
            arr_roll = arr_test[self.start - self.length : self.start]
            arr_pred = arr_test[: self.start]
            for i in range(self.start, len(arr_test), self.n_step):
                inp = arr_roll[self.n_step :, :, :, :]
                inp = np.expand_dims(inp, axis=0)
                pred = model.predict(inp, verbose=0)[0, :, :, :, :]
                # Only use the first channel (label prediction) and concatenate with features
                pred_label = pred[:, :, :, 0:1]  # Take only the label prediction
                pred = np.concatenate(
                    [pred_label, arr_test[i : i + self.n_step, :, :, 1:]], axis=-1
                )
                arr_roll = np.concatenate([arr_roll, pred], axis=0)[
                    self.n_step :, :, :, :
                ]
                arr_pred = np.concatenate(
                    [arr_pred, arr_roll[-self.n_step :, :, :, :]], axis=0
                )
        return arr_pred

    def load_lstm(self, d, sp, max_runs=4, return_fn=False, get_fn=None):
        """
        Load LSTM model and make predictions.
        Args:
            d: Dictionary of datasets
            sp: Species name
            max_runs: Maximum number of model runs
            return_fn: Whether to return filenames
            get_fn: List of filenames to use
        Returns:
            xr.Dataset: Predictions
        """
        results = []
        lst_fn = []
        files = glob.glob(self.wd + f"\\repository\\h2\\surrogate_models\\{sp}\\model*")
        random.shuffle(files)
        for f in range(max_runs):
            print(f)
            if f >= len(files):
                break
            if return_fn:
                fn = files[f]
                lst_fn.append(fn)
            else:
                fn = get_fn[f]
            with open(
                self.wd + f"\\repository\\h2\\surrogate_models\\{sp}\\scale.json", "r"
            ) as file:
                self.t_scale_params = json.load(file)
            if len(d[list(d.keys())[0]].dims) > 3:
                d[list(d.keys())[0]] = d[list(d.keys())[0]].mean(dim=["lat", "lon"])
            arr_test = self.prepare_data_test(
                d=d,
            )
            model = tf.keras.models.load_model(fn)
            arr_pred = self.autoregressor(model, arr_test)
            ds_pred = self.array_to_dataset(arr_pred)
            ds_pred = self.reverse_scaler(ds_pred)
            ds_pred = self.integrate(ds_pred)
            results.append(ds_pred)
        if return_fn:
            return xr.concat(results, dim="run"), lst_fn
        else:
            return xr.concat(results, dim="run")

    def periodic_padding(self, data, axis=2):
        left_pad = data.take(indices=range(-2, 0), axis=axis)
        right_pad = data.take(indices=range(0, 2), axis=axis)
        return np.concatenate([left_pad, data, right_pad], axis=axis)

    def crop_to_original_size(self, data, original_size, axis=3):
        # Calculate te size difference
        current_size = data.shape[axis]
        crop_size = current_size - original_size

        if crop_size % 2 != 0:
            raise ValueError(
                "The size difference is not divisible by 2. Padding or cropping might be misaligned."
            )

        # Calculate the number of values to crop from each side
        crop_each_side = crop_size // 2

        # Generate slices for each axis
        slices = [slice(None)] * data.ndim  # Initialize slices as [:, :, :, ...]
        slices[axis] = slice(
            crop_each_side, current_size - crop_each_side
        )  # Apply crop to the specified axis

        # Apply the slicing and return the cropped data
        return data[tuple(slices)]

    def load_o3(self, d, sp, max_runs=1, return_fn=False, get_fn=None):
        results = []
        self.n_step = 1
        lst_fn = []
        files = glob.glob(self.wd + f"\\repository\\h2\\surrogate_models\\{sp}\\model*")
        random.shuffle(files)
        for f in range(max_runs):
            if f >= len(files):
                break
            if return_fn:
                fn = files[f]
                lst_fn.append(fn)
            else:
                fn = get_fn[f]
            with open(
                glob.glob(
                    self.wd + f"\\repository\\h2\\surrogate_models\\{sp}\\t_scale*"
                )[0],
                "r",
            ) as file:
                self.t_scale_params = json.load(file)
            with open(
                glob.glob(
                    self.wd + f"\\repository\\h2\\surrogate_models\\{sp}\\s_scale*"
                )[0],
                "r",
            ) as file:
                self.s_scale_params = json.load(file)
            arr_test = self.prepare_data_test(d)
            arr_test = self.periodic_padding(arr_test, 3)
            model = tf.keras.models.load_model(fn)
            arr_pred = self.autoregressor(model, arr_test)
            arr_pred = self.crop_to_original_size(arr_pred, original_size=130, axis=2)
            ds_pred = self.array_to_dataset(arr_pred)
            ds_pred = self.reverse_scaler(ds_pred)
            ds_pred = self.integrate(ds_pred)
            results.append(ds_pred)
        if return_fn:
            return xr.concat(results, dim="run"), lst_fn
        else:
            return xr.concat(results, dim="run")


class RadiativeForcingModel:
    """
    Model for calculating radiative forcing from atmospheric composition changes.
    """

    def __init__(self, t, t_0, dh2, r_o3=0.042, r_h2o=1e-4, r_ch4=0.000389):
        self.r_ch4 = r_ch4
        self.r_o3 = r_o3
        self.r_h2o = r_h2o
        self.t = str(t) + "-01-01"
        self.t_0 = str(t_0) + "-01-01"
        self.dh2 = dh2

    def compute_methane_radiative_forcing(self, ds):
        """
        Compute radiative forcing due to methane changes.
        Args:
            ds: Dataset containing methane concentrations
        Returns:
            float: Radiative forcing (mW/m^2)
        """
        return (
            1.14
            * self.r_ch4
            * (
                ds["ch4_trop"].sel(time=self.t) * 1e9
                - ds["ch4_trop"].sel(time=self.t_0) * 1e9
            )
            * 1000
        )

    def compute_tropospheric_ozone_radiative_forcing(self, ds):
        """
        Compute radiative forcing due to tropospheric ozone changes.
        Args:
            ds: Dataset containing ozone concentrations
        Returns:
            float: Radiative forcing (mW/m^2)
        """
        return (
            self.r_o3
            * (
                ds["o3_trop"].sel(time=self.t) * 1e9
                - ds["o3_trop"].sel(time=self.t_0) * 1e9
            )
            * 1000
        )

    def compute_stratospheric_water_vapour_radiative_forcing(self, ds):
        """
        Compute radiative forcing due to stratospheric water vapor changes.
        Args:
            ds: Dataset containing stratospheric water vapor concentrations
        Returns:
            float: Radiative forcing (mW/m^2)
        """
        return (
            self.r_h2o
            * (
                ds["h2o_strat"].sel(time=self.t) * 1e9
                - ds["h2o_strat"].sel(time=self.t_0) * 1e9
            )
            * 1000
        )


def run_h2_case(
    wd,
    scenario,
    start_year=2035,
    t_mid=2060,
    m_adp=0.36,
    f_app=0.01,
    f_del=0.01,
    f_prod=0.01,
    kd=0.38,
):
    """
    Run a case with the provided parameters.
    Args:
        wd: working directory
        scenario: SSP scenario
        start_year: year to start the simulation
        t_mid: year of midpoint of adoption curve
        m_adp: adoption rate - slope of adoption curve
        f_app: leakage fraction during application
        f_del: leakage fraction during delivery
        f_prod: leakage fraction during production
        kd: rate of deposition
    Returns:
        xr.Dataset: results of the simulation, including perturbations and radiative forcing
    """
    # --- 1. Load Baseline Data ---
    bg_dic = load_data(wd, [scenario])
    bg_dic_2d = load_data(wd, [scenario], load_2d=True)

    # --- 2. Calculate H2 Emissions ---
    em = EmissionModel(
        working_directory=wd,
        consumption_scenario="BAU",  # Use a consistent scenario name, ensure file exists
        t_midpoint=t_mid,
        m_adoption=m_adp,
        f_application=f_app,
        f_delivery=f_del,
        f_production=f_prod,
    )
    eh2_ds = em.calculate_hydrogen_emission_rate()

    # --- 3. Calculate Atmospheric Perturbations (Box Model) ---
    bm = BoxModel(
        data=bg_dic[scenario],
        rate_of_deposition=kd,
        spinup_time=10,
        start_year=start_year,
    )
    perturbation_ds = bm.get_perturbation(eh2_ds)  # Contains doh, dh2, dch4, dco

    # --- 4. Prepare Perturbed Datasets ---
    pert_dic = copy.deepcopy(bg_dic)
    pert_dic_2d = copy.deepcopy(bg_dic_2d)

    # Align perturbation time index with main dataset time index
    # Reindex perturbation variables onto the full time dimension, filling pre-start_year with 0
    doh_pert = perturbation_ds["doh"]
    doh_full_time = doh_pert.reindex_like(
        pert_dic[scenario]["oh_trop"], method="nearest", fill_value=0
    )

    # Apply initial OH perturbation
    pert_dic[scenario]["oh_trop"] = pert_dic[scenario]["oh_trop"] + doh_full_time
    # Ensure relative perturbation is also calculated using the zero-filled full time array
    relative_pert_oh = (
        doh_full_time
        / pert_dic_2d[scenario]["oh_trop"].where(pert_dic_2d[scenario]["oh_trop"] != 0)
    ).fillna(0)
    pert_dic_2d[scenario]["oh_trop"] = pert_dic_2d[scenario]["oh_trop"] * (
        1 + relative_pert_oh
    )

    # --- 5. Initialize Forecasting Models ---
    # Shared parameters
    model_params = {
        "working_directory": wd,
        "test_scenario": scenario,
        "start_year": start_year,
    }
    # CH4 model
    msg = AutoregressiveForecastingModel(
        **model_params,
        labels=["ch4_trop"],
        features=["oh_trop", "co_trop", "emich4", "emico"],
    )
    # Stratospheric water vapor model
    wsg = AutoregressiveForecastingModel(
        **model_params,
        labels=["h2o_strat"],
        features=["ch4_trop", "h2o_trop", "ho2_trop"],
    )
    # Tropospheric ozone model
    osg = AutoregressiveForecastingModel(
        **model_params,
        labels=["o3_trop"],
        features=["ch4_trop", "oh_trop", "ho2_trop", "no_trop", "no2_trop"],
    )

    # --- 6. Run Baseline Forecasts ---
    print("Running baseline forecasts...")
    base_ch4_ds, fn_msg = msg.load_lstm(bg_dic, "ch4_trop", max_runs=1, return_fn=True)
    base_o3_ds, fn_osg = osg.load_o3(bg_dic_2d, "o3_trop", max_runs=1, return_fn=True)
    base_swv_ds, fn_wsg = wsg.load_lstm(bg_dic, "h2o_strat", max_runs=1, return_fn=True)

    # Select only the predicted variable and average runs immediately if needed later
    base_ch4_ds = base_ch4_ds[["ch4_trop"]]
    base_o3_ds = base_o3_ds[["o3_trop"]]  # Keep lat/lon for now
    base_swv_ds = base_swv_ds[["h2o_strat"]]

    # --- 7. Run Perturbed Forecasts Sequentially ---
    print("Running perturbed forecasts...")
    # 7a. CH4 Forecast
    pert_ch4_ds = msg.load_lstm(pert_dic, "ch4_trop", max_runs=1, get_fn=fn_msg)[
        ["ch4_trop"]
    ]

    # 7b. Update Perturbed Datasets with CH4 Forecast Results
    # Update 2D dataset (required for O3 forecast)
    # Ensure alignment: average run dim, match time coords
    ch4_forecast_avg = pert_ch4_ds["ch4_trop"].mean(dim="run")
    ch4_forecast_aligned = ch4_forecast_avg.reindex_like(
        pert_dic_2d[scenario]["ch4_trop"], method="nearest"
    )
    # Expand dims to match lat/lon if needed (assuming ch4_forecast_avg is 1D)
    if "lat" in pert_dic_2d[scenario]["ch4_trop"].dims:
        ch4_forecast_aligned = ch4_forecast_aligned.expand_dims(
            {"lat": pert_dic_2d[scenario]["lat"], "lon": pert_dic_2d[scenario]["lon"]},
            axis=[1, 2],
        )  # Adjust axis numbers if necessary

    pert_dic_2d[scenario]["ch4_trop"] = ch4_forecast_aligned

    # 7c. O3 Forecast
    pert_o3_ds = osg.load_o3(pert_dic_2d, "o3_trop", max_runs=1, get_fn=fn_osg)[
        ["o3_trop"]
    ]

    # 7d. Update Perturbed Dataset (1D averaged) with O3 Forecast Results
    # Update 1D dataset (required for H2O forecast)
    o3_forecast_avg = pert_o3_ds["o3_trop"].mean(dim=["run", "lat", "lon"])
    o3_forecast_aligned = o3_forecast_avg.reindex_like(
        pert_dic[scenario]["o3_trop"], method="nearest"
    )
    pert_dic[scenario]["o3_trop"] = o3_forecast_aligned
    # Also update the 1D CH4 value based on the perturbed forecast
    pert_dic[scenario]["ch4_trop"] = ch4_forecast_aligned.mean(
        dim=["lat", "lon"]
    )  # Average the updated 2D CH4

    # 7e. Stratospheric H2O Forecast
    pert_swv_ds = wsg.load_lstm(pert_dic, "h2o_strat", max_runs=1, get_fn=fn_wsg)[
        ["h2o_strat"]
    ]

    # --- 8. Merge Final Forecast Results (Averaging runs) ---
    ds_base = xr.merge(
        [base_ch4_ds, base_o3_ds.mean(dim=["lat", "lon"]), base_swv_ds]
    ).mean(dim="run")
    ds_pert = xr.merge(
        [pert_ch4_ds, pert_o3_ds.mean(dim=["lat", "lon"]), pert_swv_ds]
    ).mean(dim="run")

    # --- 9. Calculate Radiative Forcing ---
    print("Calculating radiative forcing...")
    rfm = RadiativeForcingModel(
        t=2100,  # Evaluation year
        t_0=start_year,  # Comparison year
        # Select dh2 directly from the original perturbation dataset at the evaluation time
        dh2=perturbation_ds["dh2"].sel(time=str(2100) + "-01-01", method="nearest"),
    )

    drf_ch4 = rfm.compute_methane_radiative_forcing(
        ds_pert
    ) - rfm.compute_methane_radiative_forcing(ds_base)
    drf_o3 = rfm.compute_tropospheric_ozone_radiative_forcing(
        ds_pert
    ) - rfm.compute_tropospheric_ozone_radiative_forcing(ds_base)
    drf_h2o = (
        rfm.compute_stratospheric_water_vapour_radiative_forcing(ds_pert)
        - rfm.compute_stratospheric_water_vapour_radiative_forcing(ds_base)
        +
        # Use the same selected dh2 value for the direct effect
        perturbation_ds["dh2"].sel(time=str(2100) + "-01-01", method="nearest")
        * 1e9
        * 0.03
    )

    # --- 10. Compile Final Results Dataset ---
    print("Compiling results...")
    result_ds = ds_pert.copy()

    # Add control values for comparison
    result_ds["ch4_cont"] = ds_base["ch4_trop"]
    result_ds["o3_cont"] = ds_base["o3_trop"]
    result_ds["h2o_strat_cont"] = ds_base["h2o_strat"]

    # Add perturbation values (calculated from final forecasts)
    result_ds["dch4"] = ds_pert["ch4_trop"] - ds_base["ch4_trop"]
    result_ds["do3"] = ds_pert["o3_trop"] - ds_base["o3_trop"]
    result_ds["dh2o_strat"] = ds_pert["h2o_strat"] - ds_base["h2o_strat"]

    # Add original H2 emissions and Box Model perturbations
    # Reindex these onto the result dataset's time axis, filling missing with 0
    result_ds["emih2"] = eh2_ds["emih2"].reindex_like(
        result_ds, method="nearest", fill_value=0
    )
    result_ds["h2"] = perturbation_ds["h2"].reindex_like(
        result_ds, method="nearest", fill_value=0
    )  # Box model base H2
    result_ds["dh2"] = perturbation_ds["dh2"].reindex_like(
        result_ds, method="nearest", fill_value=0
    )  # Box model H2 perturbation
    # Use the doh_full_time calculated earlier which is already aligned and zero-filled
    result_ds["doh"] = doh_full_time.reindex_like(
        result_ds, method="nearest"
    )  # Box model OH perturbation

    # Add OH concentration from perturbed simulation (before forecasting)
    result_ds["oh"] = pert_dic[scenario]["oh_trop"].reindex_like(
        result_ds, method="nearest"
    )

    # Add radiative forcing values (as attributes since they are scalar)
    result_ds.attrs["drf_ch4"] = drf_ch4.item()
    result_ds.attrs["drf_o3"] = drf_o3.item()
    result_ds.attrs["drf_h2o"] = drf_h2o.item()
    result_ds.attrs["drf_total"] = (drf_ch4 + drf_o3 + drf_h2o).item()

    # Add parameters used for this run (as attributes)
    result_ds.attrs["scenario"] = scenario
    result_ds.attrs["start_year"] = start_year
    result_ds.attrs["t_mid"] = t_mid
    result_ds.attrs["m_adp"] = m_adp
    result_ds.attrs["f_app"] = f_app
    result_ds.attrs["f_del"] = f_del
    result_ds.attrs["f_prod"] = f_prod
    result_ds.attrs["kd"] = kd

    print("Run case finished.")
    return result_ds


if __name__ == "__main__":
    # Example use case
    working_directory = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # Fix ROOT_DIR issue
    scenario = "ssp126"

    # Run a case with default parameters
    result = run_h2_case(
        wd=working_directory,
        scenario=scenario,
        start_year=2035,
        t_mid=2060,
        m_adp=0.27,
        f_app=0.0300,
        f_del=0.0133,
        f_prod=0.0178,
        kd=0.38,
    )

    print(result)
