"""
Provides tests for module calc_h2
"""

import sys
import numpy as np
import pytest
import xarray as xr

# calc_h2 must be executed in separate oac_h2 environment due to tensorflow and keras dependencies.
# These tests run only in an evironment with tensorflow and keras packages properly installed.
from openairclim.calc_h2 import EmissionModel, BoxModel

# Disable tests within this file if tensorflow and keras are not installed
pytestmark = pytest.mark.skipif(
    "keras" not in sys.modules, reason="packages tensorflow and keras not available"
)


class TestCalculateHydrogenAdoptionFraction:
    """Tests function calculate_hydrogen_adoption_fraction(self, t_mid=None, m_adp=None)"""

    @pytest.fixture
    def setup(self):
        """Setup sample xarray Dataset with consumption scenario, and default parameters

        Returns:
            TestingClass.instance: consumption scenario and default parameters
        """
        time = xr.date_range("2020-01-01", "2030-01-01", freq="YS")
        # H2 consumption scenario
        cs_ds = xr.Dataset(
            {"BAU": ("time", np.linspace(0.0, 10.0, len(time)))}, coords={"time": time}
        )
        # instance of dynamically created class TestingClass
        self.instance = type(
            "TestingClass", (object,), {"cs_ds": cs_ds, "m_adp": 2.0, "t_mid": 2025}
        )()
        return self.instance

    def test_calculate_hydrogen_adoption_fraction_with_params(self, setup):
        """Tests function with costumized parameters"""
        t_mid = 2025
        m_adp = 0.5
        expected = 1 / (
            1 + np.exp(-m_adp * (np.array(setup.cs_ds["time"].dt.year) - t_mid))
        )
        result = EmissionModel.calculate_hydrogen_adoption_fraction(setup, t_mid, m_adp)
        assert np.allclose(result, expected)

    def test_calculate_hydrogen_adoption_fraction_with_default_params(self, setup):
        """Tests function with default parameters"""
        expected = 1 / (
            1
            + np.exp(
                -setup.m_adp * (np.array(setup.cs_ds["time"].dt.year) - setup.t_mid)
            )
        )
        result = EmissionModel.calculate_hydrogen_adoption_fraction(setup)
        assert np.allclose(result, expected)

    def test_calculate_hydrogen_adoption_fraction_exception(self, setup):
        """If only one of the two parameters are passed, a ValueError is raised"""
        with pytest.raises(ValueError):
            EmissionModel.calculate_hydrogen_adoption_fraction(setup, t_mid=2025)
        with pytest.raises(ValueError):
            EmissionModel.calculate_hydrogen_adoption_fraction(setup, m_adp=0.5)


class TestPrepareData:
    """Tests function prepare_data(self, ds, ds_eh2, perturbation=False)"""

    @pytest.fixture
    def setup(self):
        """Setup sample xarray Datasets ds and ds_ehs, and further parameters

        Returns:
            TestingClass.instance: ds and ds_ehs, and further parameters
        """
        time = xr.date_range("2020-01-01", "2022-01-01", freq="YS")
        ds = xr.Dataset(
            {
                "emico": (("time"), np.array([1, 2, 3])),
                "emich4": (("time"), np.array([4, 5, 6])),
            },
            coords={"time": time},
        )
        ds_eh2 = xr.Dataset(
            {
                "emih2": (("time"), np.array([10.0, 11.0, 12.0])),
            },
            coords={"time": np.arange(3)},
        )
        # Background production rates (ppb/yr)
        p_h2 = 265  # hydrogen production
        p_ch4 = 60  # methane production
        p_co = 200  # carbon monoxide production
        p_oh = 1333  # hydrogen production
        spinup_time = 0
        start_year = 2020
        kd = 0.38  # hydrogen deposition rate
        # instance of dynamically created class TestingClass
        self.instance = type(
            "TestingClass",
            (object,),
            {
                "ds": ds,
                "ds_eh2": ds_eh2,
                "p_h2": p_h2,
                "p_ch4": p_ch4,
                "p_co": p_co,
                "p_oh": p_oh,
                "kd": kd,
                "spinup_time": spinup_time,
                "start_year": start_year,
            },
        )()
        return self.instance

    def test_prepare_data_with_perturbation(self, setup):
        """Test for correct return values, perturbation included"""
        ds = setup.ds
        ds_eh2 = setup.ds_eh2
        box_model = BoxModel(
            data=ds,
            rate_of_deposition=setup.kd,
            spinup_time=setup.spinup_time,
            start_year=setup.start_year,
        )
        result = box_model.prepare_data(ds, ds_eh2, perturbation=True)
        assert np.allclose(result["emioh"].values, setup.p_oh)

    def test_prepare_data_without_perturbation(self, setup):
        """Test for correct return values, without perturbation"""
        ds = setup.ds
        ds_eh2 = setup.ds_eh2
        box_model = BoxModel(
            data=ds,
            rate_of_deposition=setup.kd,
            spinup_time=setup.spinup_time,
            start_year=setup.start_year,
        )
        result = box_model.prepare_data(ds, ds_eh2, perturbation=False)
        assert np.allclose(result["emih2"].values, setup.p_h2)
        assert np.allclose(result["emioh"].values, setup.p_oh)

    def test_prepare_data_with_missing_data(self, setup):
        """Test with missing data in ds_eh2 which should be filled"""
        ds = setup.ds
        ds_eh2 = setup.ds_eh2
        ds_eh2["emih2"].data[1] = np.nan
        box_model = BoxModel(
            data=ds,
            rate_of_deposition=setup.kd,
            spinup_time=setup.spinup_time,
            start_year=setup.start_year,
        )
        result = box_model.prepare_data(ds, ds_eh2, perturbation=True)
        assert np.allclose(result["emih2"].values[1], setup.p_h2)
