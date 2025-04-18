"""Create emission inventories with random values"""

# import numpy as np
# from scipy.stats import truncnorm
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# CONSTANTS
EI_CO2 = 3.16  # Lee et al. 2010, Table 1, doi:10.1016/j.atmosenv.2009.06.005
EI_H2O = 1.24  # Lee et al. 2010, Table 1, doi:10.1016/j.atmosenv.2009.06.0

# OUTPUT options
#
# Number of samples in output emission inventory
OUT_SIZE = 10000
#
OUT_PATH = "../example/input/"
#
# Coordinate ranges
# lon, lat ranges in deg
# plev_range in hPa
LON_RANGE = [0.0, 360.0]
LAT_RANGE = [-90.0, 90.0]
PLEV_RANGE = [200.0, 1000.0]
#
# Years of output emission inventories
YEAR_ARR = [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120]
# Yearly (linear) increase rate of emissions
DELTA = 0.03

# Statistical input data based on inventory from DEPA 2050 project
# https://elib.dlr.de/142185/
#
# Mean values
MEAN_DICT = {
    "lon": 163.00254821777344,
    "lat": 27.598291397094727,
    "plev": 438.63165283203125,
    "fuel": 432343.28125,
    "CO2": 1346749.125,
    "H2O": 540429.125,
    "NOx": 6783.677734375,
    "distance": 87872.8671875,
}
# Standard deviation
STD_DICT = {
    "lon": 109.76872253417969,
    "lat": 29.54259490966797,
    "plev": 227.44091796875,
    "fuel": 1379145.125,
    "CO2": 4294265.0,
    "H2O": 1722916.375,
    "NOx": 20044.056640625,
    "distance": 299754.8125,
}
# Maximum values
MAX_DICT = {
    "lon": 359.0,
    "lat": 88.0,
    "plev": 1013.25,
    "fuel": 114132048.0,
    "CO2": 355521344.0,
    "H2O": 142665056.0,
    "NOx": 1440964.125,
    "distance": 17989950.0,
}
# Sums
SUM_DICT = {
    "lon": 98807088.0,
    "lat": 16729229.0,
    "plev": 265884912.0,
    "fuel": 262073090048.0,
    "CO2": 816357572608.0,
    "H2O": 327591395328.0,
    "NOx": 4112055296.0,
    "distance": 53265809408.0,
}
# Size / length of DEPA inventory
STAT_SIZE = 606169


class ArtificialInventory:
    """Class for generating artificial emission inventories."""

    # Read statistical data
    mean_dict = MEAN_DICT
    stat_size = STAT_SIZE

    def __init__(
        self,
        year,
        ac_lst=None,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        plev_range=PLEV_RANGE,
        scaling=1.0,
        size=OUT_SIZE,
    ):
        self.year = year
        self.ac_lst = ac_lst
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.plev_range = plev_range
        self.scaling = scaling
        self.size = size
        self.norm = self.stat_size / size
        self.fuel_mean = self.mean_dict["fuel"] * self.norm
        self.nox_mean = self.mean_dict["NOx"] * self.norm
        self.dist_mean = self.mean_dict["distance"] * self.norm
        self.df = None
        self.inv = None

    def create_uniform_dist(self):
        """Create an emission inventory with a uniform distribution."""
        lon_samples = np.random.uniform(
            low=self.lon_range[0], high=self.lon_range[-1], size=self.size
        )
        lat_samples = np.random.uniform(
            low=self.lat_range[0], high=self.lat_range[-1], size=self.size
        )
        plev_samples = np.random.uniform(
            low=self.plev_range[0], high=self.plev_range[-1], size=self.size
        )
        fuel_samples = (
            self.fuel_mean * np.random.rand(self.size) * self.scaling
        )
        co2_samples = fuel_samples * EI_CO2
        h2o_samples = fuel_samples * EI_H2O
        nox_samples = self.nox_mean * np.random.rand(self.size) * self.scaling
        dist_samples = (
            self.dist_mean * np.random.rand(self.size) * self.scaling
        )
        data = {
            "lon": lon_samples.astype("float32"),
            "lat": lat_samples.astype("float32"),
            "plev": plev_samples.astype("float32"),
            "fuel": fuel_samples.astype("float32"),
            "CO2": co2_samples.astype("float32"),
            "H2O": h2o_samples.astype("float32"),
            "NOx": nox_samples.astype("float32"),
            "distance": dist_samples.astype("float32"),
        }
        # only add "ac" if aircraft types are provided
        if self.ac_lst:
            ac_values = np.random.choice(self.ac_lst, size=self.size)
            data["ac"] = ac_values.astype("str")
        self.df = pd.DataFrame(data)
        return self

    def create_normal_dist(self):
        """Create an inventory with a normal distribution."""
        # TODO implement function
        pass

    def convert_df_to_xr(self):
        """
        Convert the pandas dataframe to an xarray dataset.

        Returns:
            xarray.Dataset: The xarray dataset with the emission inventory data.
        """
        inv = self.df.to_xarray()
        inv.attrs = dict(
            Title="Artificial emission inventory",
            Convention="CF-XXX",
            Inventory_Year=self.year,
        )
        inv.lon.attrs = dict(
            standard_name="longitude",
            long_name="longitude",
            units="degrees_east",
            axis="X",
        )
        inv.lat.attrs = dict(
            standard_name="latitude",
            long_name="latitude",
            units="degrees_north",
            axis="Y",
        )
        inv.plev.attrs = dict(
            standard_name="air_pressure",
            long_name="pressure",
            units="hPa",
            positive="down",
            axis="Z",
        )
        inv.fuel.attrs = {"long_name": "fuel", "units": "kg"}
        inv.NOx.attrs = {"long_name": "NOx", "units": "kg"}
        inv.distance.attrs = {"long_name": "distance flown", "units": "km"}
        inv.CO2.attrs = {"long_name": "CO2", "units": "kg"}
        inv.H2O.attrs = {"long_name": "H2O", "units": "kg"}
        if self.ac_lst:
            inv.ac.attrs = {"long_name": "aircraft identifier", "units": "-"}
        self.inv = inv
        return self

    def create(self, distribution: str = "uniform") -> xr.Dataset:
        """
        Create an emission inventory with the specified distribution.

        Args:
            distribution (str, optional): The distribution to use for creating the
                emission inventory. Can be either "uniform" or "normal". Defaults to
                "uniform".

        Returns:
            xr.Dataset: The emission inventory as an xarray dataset.
        """
        if distribution == "uniform":
            self.inv = self.create_uniform_dist().convert_df_to_xr().inv
        else:
            raise ValueError("Invalid distribution argument!")
        return self.inv


class ArtificialInventoryDict:
    """
    Class for generating artificial emission inventories.

    Args:
        year_arr (list): List of inventory years.
        delta (float, optional): Linear increase rate of emissions. Defaults to DELTA.
        ac_lst (list, optional): List of aircraft identifiers (strings).
            Defaults to None (ac coordinate not generated).

    Attributes:
        year_arr (list): List of inventory years.
        year_0 (int): First year in the list.
        delta (float): Linear increase rate of emissions.
        inv_dict (dict): Dictionary of xarray datasets, where the keys are the
            inventory years and the values are the datasets for that year.
    """

    def __init__(self, year_arr, delta=DELTA, ac_lst=None):
        self.year_arr = year_arr
        self.year_0 = year_arr[0]
        self.delta = delta
        self.inv_dict = None
        self.ac_lst = ac_lst

    def create_linear_increase(self):
        """
        Create an inventory with a linear increase in emissions.

        Returns:
            None: None
        """
        inv_dict = {}
        for year in self.year_arr:
            scaling = 1.0 + (year - self.year_0) * self.delta
            inv_dict[year] = ArtificialInventory(
                year=year,
                scaling=scaling,
                ac_lst=self.ac_lst,
            ).create()
        self.inv_dict = inv_dict
        return self

    def create(self, evolution="increment"):
        """
        Create an emission inventory with the specified evolution.

        Args:
            evolution (str, optional): Evolution method. Can be either "increment" or
                "uniform". Defaults to "increment".

        Returns:
            None: None
        """
        if evolution == "increment":
            self.inv_dict = self.create_linear_increase().inv_dict
        else:
            raise ValueError("Invalid evolution argument!")
        return self.inv_dict


def convert_xr_dict_to_nc(inv_dict: dict, prefix: str = "rnd_inv"):
    """
    Convert a dictionary of xarray datasets to netCDF files and write to OUT_PATH.
    Create OUT_PATH if not existing.

    Args:
        inv_dict (dict): Dictionary of xarray datasets, where the keys are the
            inventory years and the values are the datasets for that year.
        prefix (str, optional): Prefix for the output netCDF files.
            Defaults to "rnd_inv".

    Returns:
        None: None
    """
    os.makedirs(OUT_PATH, exist_ok=True)
    for year, inv in inv_dict.items():
        out_file = OUT_PATH + prefix + "_" + str(year) + ".nc"
        inv.to_netcdf(out_file)


def plot_sample_emission_inventory(rnd_inv_dict):
    """
    Plots a sample emission inventory from the provided dictionary of xarray datasets.

    Args:
        rnd_inv_dict (dict): Dictionary of xarray datasets, where the keys are the
            inventory years and the values are the datasets for that year.

    Returns:
        None: None
    """
    # Plot first emission inventory
    rnd_inv = next(iter(rnd_inv_dict.values()))
    rnd_inv.plot.scatter(x="lon", y="lat", hue="fuel")
    # plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    import openairclim as oac

    art_inv_dict = ArtificialInventoryDict(year_arr=YEAR_ARR).create()
    convert_xr_dict_to_nc(art_inv_dict)
    oac.plot_inventory_vertical_profiles(art_inv_dict)
    plot_sample_emission_inventory(art_inv_dict)
