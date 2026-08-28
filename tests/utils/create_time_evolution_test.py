"""
Provides tests for module openairclim.utils.create_time_evolution
"""

import numpy as np
import xarray as xr

from openairclim.utils import create_time_evolution as cte


class TestCreateTimeScalingXr:
    """Tests function create_time_scaling_xr(scaling_time, scaling_arr)"""

    def test_builds_expected_dataset(self):
        """The dataset has a scaling data variable indexed by time, with
        the expected attrs."""
        time = np.array([2020, 2021, 2022])
        scaling = np.array([1.0, 1.1, 1.2], dtype="float32")
        ds = cte.create_time_scaling_xr(time, scaling)
        assert isinstance(ds, xr.Dataset)
        assert list(ds["scaling"].values) == list(scaling)
        assert ds.attrs["Type"] == "scaling"
        assert ds.time.attrs["units"] == "years"


class TestCreateTimeNormalizationXr:
    """Tests function create_time_normalization_xr(time_arr, fuel_arr,
    ei_co2_arr, ei_h2o_arr, dis_per_fuel_arr)"""

    def test_builds_expected_dataset(self):
        """The dataset has fuel/EI_CO2/EI_H2O/dis_per_fuel data variables
        indexed by time, with the expected attrs."""
        time = np.array([2020, 2021])
        fuel = np.array([100.0, 110.0], dtype="float32")
        ei_co2 = np.array([3.1, 3.1], dtype="float32")
        ei_h2o = np.array([1.2, 1.2], dtype="float32")
        dis_per_fuel = np.array([0.3, 0.3], dtype="float32")
        ds = cte.create_time_normalization_xr(time, fuel, ei_co2, ei_h2o, dis_per_fuel)
        assert isinstance(ds, xr.Dataset)
        assert {"fuel", "EI_CO2", "EI_H2O", "dis_per_fuel"}.issubset(ds.data_vars)
        assert ds.attrs["Type"] == "norm"
