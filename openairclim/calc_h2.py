import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

import openairclim


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
    pass

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

