import copy
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import openairclim
from openairclim import open_netcdf

y_to_s = 365.25 * 24 * 60 * 60
m_to_s = 60 * 60 * 24 * 365.25 / 12
ppb_to_cm3 = 2.46e10

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

def load_data(wd, scenarios):
    d = {}
    for sc in scenarios:
        fn = glob.glob(wd + f'\\example\\repository\\h2\\SSP_scenarios\\bg_{sc}.nc')
        d[sc] = open_netcdf(fn)[f'bg_{sc}']
    return d

class EmissionModel:
    def __init__(self, working_directory, consumption_scenario, t_midpoint, m_adoption, f_application, f_delivery,
                 f_production):
        # Meta
        self.wd = working_directory  # working directory from project root define in __init__.py (path)
        self.cs = consumption_scenario  # name of consumption scenario (string)
        self.cs_ds = self.read_aviation_fuel_consumption() # consumption scenario data (xr.Dataset)
        self.eh2_ds = None  # equivalent hydrogen consumption data (xr.Dataset)

        # Parameters
        self.t_mid = t_midpoint  # year of midpoint of adoption curve (int)
        self.m_adp = m_adoption  # adoption rate - slope of adoption curve (float)
        self.f_adp_ds = self.calculate_hydrogen_adoption_fraction()  # temporal evolution of h2 adoption (xr.Dataset)
        self.f_app = f_application  # leakage fraction during application (float)
        self.f_del = f_delivery  # leakage fraction during delivery (float)
        self.f_prd = f_production  # leakage fraction during production (float)

        # Constants
        self.lwh_ker = 43.15  # lower heating potential of kerosene (MJ/kg)
        self.lwh_h2 = 120  # lower heating potential of h2 (MJ/kg)

    def read_aviation_fuel_consumption(self):
        """ Read aviation fuel consumption data from csv files and return as xarray dataset
        Returns:
            xr.Dataset: aviation fuel consumption data
        """
        d = {}
        if self.cs is None:
            raise ValueError('No consumption scenario provided')
        else:
            fn = glob.glob(self.wd + f'\\example\\repository\\h2\\fuel_consumption_scenarios\\{self.cs}*')
            d[self.cs] = pd.read_csv(fn[0])
            ds = xr.Dataset(d)
            ds = ds.rename({'dim_0': 'time'}).isel(dim_1=1).drop_vars('dim_1')
            ds['time'] = xr.date_range('2025-01-01', periods=ds['time'].size, freq='YS')
            return ds

    def calculate_equivalent_hydrogen_consumption(self):
        """ Calculate equivalent hydrogen consumption
        Returns:
            xr.Dataset: equivalent hydrogen consumption
        """
        return self.cs_ds * self.lwh_ker / self.lwh_h2

    def calculate_hydrogen_adoption_fraction(self, t_mid=None, m_adp=None):
        """ Calculate hydrogen adoption fraction
        Args:
            t_mid (int): midpoint year
            m_adp (float): adoption rate (slope of curve)
        Returns:
            xr.Dataset: hydrogen adoption fraction
        """
        y = self.cs_ds['time'].dt.year
        if t_mid is not None and m_adp is not None:
            return 1 / (1 + np.exp(-m_adp * (y - t_mid)))
        elif t_mid is None and m_adp is None:
            return 1 / (1 + np.exp(-self.m_adp * (y - self.t_mid)))
        else:
            raise ValueError('Provide both t_mid and m_adp or None')

    def calculate_hydrogen_application_mass(self):
        """" Calculate the total hydrogen mass required for application (flight) using the adoption fraction and leakage
        fraction during application
        Returns:
            xr.Dataset: hydrogen mass required for application
        """
        return self.f_adp_ds * self.calculate_equivalent_hydrogen_consumption() / (1 - self.f_app)

    def calculate_hydrogen_delivery_mass(self):
        """ Calculate the total hydrogen mass required for delivery using the application mass and leakage fraction
        during delivery
        Returns:
            xr.Dataset: hydrogen mass required for delivery
        """
        return self.calculate_hydrogen_application_mass() / (1 - self.f_del)

    def calculate_hydrogen_production_mass(self):
        """ Calculate the total hydrogen mass required for production using the delivery mass and leakage fraction
        during production
        Returns:
            xr.Dataset: hydrogen mass required for production
        """
        return self.calculate_hydrogen_delivery_mass() / (1 - self.f_prd)

    def calculate_hydrogen_emission_rate(self):
        """ Calculate the total fugitive emissions
        Returns:
            xr.Dataset: fugitive emissions (emih2)
        """
        ds = (
                self.f_app * self.calculate_hydrogen_application_mass()
                + self.f_del * self.calculate_hydrogen_delivery_mass()
                + self.f_prd * self.calculate_hydrogen_production_mass()
        )
        if len(ds.data_vars) == 1:
            ds = ds.rename({list(ds.data_vars)[0]: 'emih2'})
        return ds

class BoxModel:
    def __init__(self, data, rate_of_deposition, spinup_time, start_year):
        # Meta
        self.data = data
        self.y_start = pd.to_datetime(start_year, format='%Y')
        self.y_end = pd.to_datetime(self.data['time'][-1].values)
        self.horizon = self.y_end.year - (self.y_start.year-1)
        self.t_spin = np.arange(0, spinup_time, 1)
        self.t_span = (0, spinup_time + self.horizon)
        self.t_eval = np.arange(spinup_time, self.t_span[1], 1)

        # Parameters
        self.kd = rate_of_deposition

        # Constants
        self.k1 = 3.17e-15 * ppb_to_cm3 * y_to_s
        self.k2 = 3.80e-15 * ppb_to_cm3 * y_to_s
        self.k3 = 1.90e-13 * ppb_to_cm3 * y_to_s
        self.k4 = 0.3 * y_to_s
        self.ks = 0.02
        self.alpha = 0.37

        self.p_ch4 = 60
        self.p_oh = 1333
        self.p_co = 200
        self.p_h2 = 265

        self.c_h2 = 500
        pass

    def interpolate_sources(self, sources):
        S_ch4_func = interp1d(self.t_eval, sources['emich4'], kind='linear', fill_value="extrapolate")
        S_co_func = interp1d(self.t_eval, sources['emico'], kind='linear', fill_value="extrapolate")
        S_h2_func = interp1d(self.t_eval, sources['emih2'], kind='linear', fill_value="extrapolate")
        S_oh_func = interp1d(self.t_eval, sources['emioh'], kind='linear', fill_value="extrapolate")
        return S_ch4_func, S_co_func, S_h2_func, S_oh_func

    def system_of_odes(self, t, y, sources):
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
        t = self.t_eval
        y_0 = [ds['ch4_trop'][0]*1e9, ds['co_trop'][0]*1e9, ds['oh_trop'][0]*1e9, self.c_h2]
        sources = {
            'emich4': ds['emich4'],
            'emico': ds['emico'],
            'emih2': ds['emih2'],
            'emioh': ds['emioh']
        }
        sol = solve_ivp(
            self.system_of_odes,
            self.t_span,
            y_0,
            method='Radau',
            rtol=1e-6,
            atol=1e-6,
            dense_output=True,
            args=(sources,)
        )
        return sol.sol(t), t

    def prepare_data(self, ds, ds_eh2, perturbation=False):
        if perturbation:
            ds['emih2'] = ds_eh2['emih2']
            ds['emih2'] = ds['emih2'].fillna(0)
            ds['emih2'] = convert_mass_to_concentration(ds['emih2'], 2) + self.p_h2
        else:
            ds['emih2'] = self.p_h2
            ds['emih2'] = ds['emih2'].broadcast_like(ds['emico'])
        ds['emich4'] = convert_mass_to_concentration(ds['emich4'] * 1e-9, 16) + self.p_ch4
        ds['emico'] = convert_mass_to_concentration(ds['emico'] * 1e-9, 28) + self.p_co
        ds['emioh'] = self.p_oh
        ds['emioh'] = ds['emioh'].broadcast_like(ds['emico'])
        return ds

    def get_perturbation(self, ds_eh2):
        ds = self.data.sel(time=slice(self.y_start, None))
        ds = ds.mean(dim=['lat', 'lon'])
        ds = ds[['ch4_trop', 'co_trop', 'oh_trop', 'emico', 'emich4']]

        ds_base = self.prepare_data(copy.deepcopy(ds), ds_eh2)
        ds_pert = self.prepare_data(copy.deepcopy(ds), ds_eh2, perturbation=True)

        y_base, t_base = self.solver(ds_base)
        y_pert, t_pert = self.solver(ds_pert)

        ds['h2_trop'] = xr.DataArray(y_base[3], coords=[ds['time']], dims=['time']) * 1e-9
        ds['doh_trop'] = xr.DataArray(y_pert[2] - y_base[2], coords=[ds['time']], dims=['time']) * 1e-9
        ds['dh2_trop'] = xr.DataArray(y_pert[3] - y_base[3], coords=[ds['time']], dims=['time']) * 1e-9

        # fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        # ax[0, 0].plot(t_base, y_base[0], label='CH4')
        # ax[0, 0].plot(t_pert, y_pert[0], label='CH4 perturbed')
        # ax[0, 1].plot(t_base, y_base[1], label='CO')
        # ax[0, 1].plot(t_pert, y_pert[1], label='CH4 perturbed')
        # ax[1, 0].plot(t_base, y_base[2], label='OH')
        # ax[1, 0].plot(t_pert, y_pert[2], label='CH4 perturbed')
        # ax[1, 1].plot(t_base, y_base[3], label='H2')
        # ax[1, 1].plot(t_pert, y_pert[3], label='CH4 perturbed')
        # plt.show()
        return ds[['h2_trop', 'doh_trop', 'dh2_trop']]


class AutoregressiveForecastingModel:
    pass

class RadiativeForcingModel:
    pass


if __name__ == '__main__':
    wd = os.path.dirname(openairclim.ROOT_DIR)
    em = EmissionModel(
        working_directory=wd,
        consumption_scenario='BAU',
        t_midpoint=2060,
        m_adoption=0.27,
        f_application=0.01,
        f_delivery=0.01,
        f_production=0.01
    )

    emih2 = em.calculate_hydrogen_emission_rate()
    scenarios = ['ssp126', 'ssp585']
    test_scenario = 'ssp126'

    bm = BoxModel(
        data=load_data(wd, scenarios)[test_scenario],
        rate_of_deposition=0.38,
        spinup_time=10,
        start_year=2035
    )

    ds = bm.get_perturbation(emih2)
    print(ds)
