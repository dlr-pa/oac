"""
Provides tests for module interpolate_time
"""

import numpy as np
import pytest
import openairclim as oac
from utils.create_test_data import create_test_inv


@pytest.fixture(name="setup_valid_arguments", scope="class")
def fixture_setup_valid_arguments():
    """Setup valid arguments for interp_linear(config, years, val_dict)

    Returns dict, np.ndarray, dict: configuration dictionary, numpy array of years,
        dictionary of time series numpy arrays with species names as keys
    """
    config = {"time": {"range": [2000, 2011, 1]}}
    years = np.array([2000, 2010])
    val_dict = {"fuel": np.array([100, 150])}
    return config, years, val_dict


@pytest.fixture(name="setup_invalid_arguments", scope="class")
def fixture_setup_invalid_arguments():
    """Setup invalid arguments for interp_linear(config, years, val_dict)

    Returns dict, np.ndarray, dict: configuration dictionary, numpy array of years,
        dictionary of time series numpy arrays with species names as keys
    """
    config = {"time": {"range": [2000, 2011, 1]}}
    years = np.array([])
    val_dict = {"fuel": np.array([100, 150])}
    return config, years, val_dict


@pytest.mark.usefixtures("setup_valid_arguments", "setup_invalid_arguments")
class TestInterpLinear:
    """Tests function interp_linear(config, years, vald_dict)"""

    def test_correct_input(self, setup_valid_arguments):
        """Valid input returns time_range (np.ndarray), interp_dict (dict of np.ndarray)"""
        config, years, vald_dict = setup_valid_arguments
        time_range, interp_dict = oac.interp_linear(config, years, vald_dict)
        # Test for correct output types
        assert isinstance(time_range, np.ndarray)
        assert isinstance(interp_dict, dict)
        # Test for non-empty array of interpolated values
        assert interp_dict["fuel"].size

    def test_incorrect_input(self, setup_invalid_arguments):
        """Invalid input returns IndexError"""
        config, years, val_dict = setup_invalid_arguments
        with pytest.raises(IndexError):
            oac.interp_linear(config, years, val_dict)


@pytest.mark.usefixtures("setup_valid_arguments", "setup_invalid_arguments")
class TestInterpolate:
    """Tests function interpolate(config, years, val_dict)"""

    def test_correct_input(self, setup_valid_arguments):
        """Valid input returns time_range (np.ndarray), interp_dict (dict of np.ndarray)"""
        config, years, vald_dict = setup_valid_arguments
        time_range, interp_dict = oac.interpolate(config, years, vald_dict)
        # Test for correct output types
        assert isinstance(time_range, np.ndarray)
        assert isinstance(interp_dict, dict)
        # Test for non-empty array of interpolated values
        assert interp_dict["fuel"].size

    def test_incorrect_input(self, setup_invalid_arguments):
        """Invalid input returns IndexError"""
        config, years, val_dict = setup_invalid_arguments
        with pytest.raises(IndexError):
            oac.interpolate(config, years, val_dict)


class TestFilterToInvYears:
    """Tests function filter_to_inv_years(inv_years, time_range, interp_dict)"""

    def test_correct_input(self):
        """Valid input returns dictionary of arrays, filtered to inventory years"""
        inv_years = np.array([2010])
        time_range = np.arange(2000, 2021, 1, dtype=int)
        interp_dict = {"fuel": np.arange(0.0, 21.0, 1.0)}
        filtered_dict = oac.filter_to_inv_years(
            inv_years, time_range, interp_dict
        )
        # Test for correct output type
        assert isinstance(filtered_dict, dict)
        # Test for correct output value
        assert filtered_dict["fuel"] == np.array([10.0])


class TestCalcNorm:
    """Tests function calc_norm(evo_dict, ei_inv_dict)"""

    def test_correct_input(self):
        "Valid input returns dictionary"
        evo_dict = {"fuel": np.array([100.0]), "EI_CO2": np.array([1.0])}
        ei_inv_dict = {"fuel": np.array([200.0]), "EI_CO2": np.array([2.0])}
        norm_dict = oac.calc_norm(evo_dict, ei_inv_dict)
        assert isinstance(norm_dict, dict)

    def test_correct_normalization(self):
        "Test for correct normalization"
        evo_dict = {"fuel": np.array([100.0]), "EI_CO2": np.array([1.0])}
        ei_inv_dict = {"fuel": np.array([200.0]), "EI_CO2": np.array([2.0])}
        norm_dict = oac.calc_norm(evo_dict, ei_inv_dict)
        expected_norm_dict = {"fuel": np.array(0.5), "CO2": np.array(0.25)}
        np.testing.assert_equal(norm_dict["fuel"], expected_norm_dict["fuel"])
        np.testing.assert_equal(norm_dict["CO2"], expected_norm_dict["CO2"])

    def test_incorrect_input(self):
        "Invalid ei_inv_dict (no fuel key, empty dict) returns KeyError"
        evo_dict = {"fuel": np.array([100.0]), "EI_CO2": np.array([1.0])}
        ei_inv_dict = {}
        with pytest.raises(KeyError):
            oac.calc_norm(evo_dict, ei_inv_dict)


@pytest.fixture(name="inv_dict", scope="class")
def fixture_inv_dict():
    """Fixture to create an example inv_dict"""
    return {2020: create_test_inv(year=2020), 2050: create_test_inv(year=2050)}


@pytest.mark.usefixtures("inv_dict")
class TestCalcInvQuantities:
    """Tests function calc_inv_quantities(config, inv_dict)"""

    def test_correct_input(self, inv_dict):
        "Valid input returns np.ndarray, dict, dict"
        # Input
        config = {"species": {"inv": ["CO2", "H2O"]}}
        # Output
        inv_years, inv_sum_dict, ei_inv_dict = oac.calc_inv_quantities(
            config, inv_dict
        )
        assert isinstance(inv_years, np.ndarray)
        assert isinstance(inv_sum_dict, dict)
        assert isinstance(ei_inv_dict, dict)

    def test_correct_years(self, inv_dict):
        "Test for correct output years"
        # Input
        inp_years = np.array(list(inv_dict.keys()))
        config = {"species": {"inv": ["CO2", "H2O"]}}
        # Output
        inv_years, _inv_sum_dict, _ei_inv_dict = oac.calc_inv_quantities(
            config, inv_dict
        )
        np.testing.assert_equal(inv_years, inp_years)

    def test_correct_sums(self, inv_dict):
        "Test for correct sums"
        # Input
        config = {"species": {"inv": ["CO2", "H2O"]}}
        # Expected sums
        expected_2020_fuel = inv_dict[2020].fuel.sum().values.item()
        expected_2020_co2 = inv_dict[2020].CO2.sum().values.item()
        expected_2050_fuel = inv_dict[2050].fuel.sum().values.item()
        expected_2050_co2 = inv_dict[2050].CO2.sum().values.item()
        # Output
        _inv_years, inv_sum_dict, _ei_inv_dict = oac.calc_inv_quantities(
            config, inv_dict
        )
        # Test if arrays of computed fuels sums are as expected
        np.testing.assert_equal(
            inv_sum_dict["fuel"],
            np.array([expected_2020_fuel, expected_2050_fuel]),
        )
        # Test if arrays of computed CO2 sums are as expected
        np.testing.assert_equal(
            inv_sum_dict["CO2"],
            np.array([expected_2020_co2, expected_2050_co2]),
        )


@pytest.mark.usefixtures("inv_dict")
class TestNormInv:
    """Tests function norm_inv(inv_dict, norm_dict)"""

    def test_correct_input(self, inv_dict):
        "Valid input returns dictionary of xr.Dataset, keys are inventory years"
        norm_dict = {"fuel": np.array([1.0, 2.0])}
        years = list(inv_dict.keys())
        out_dict = oac.norm_inv(inv_dict, norm_dict)
        # Test for correct output type
        assert isinstance(out_dict, dict)
        # Test for correct dictionary keys (inventory years)
        assert list(out_dict.keys()) == years

    def test_correct_normalization(self, inv_dict):
        "Test for correct normalization of inventories"
        # Input
        norm_dict = {"fuel": np.array([1.0, 2.0])}
        inp_2020_fuel_arr = inv_dict[2020].fuel.values
        inp_2050_fuel_arr = inv_dict[2050].fuel.values
        inp_2050_lon_arr = inv_dict[2050].lon.values
        inp_2050_lat_arr = inv_dict[2050].lat.values
        inp_2050_plev_arr = inv_dict[2050].plev.values
        # Output
        out_dict = oac.norm_inv(inv_dict, norm_dict)
        out_2020_fuel_arr = out_dict[2020].fuel.values
        out_2050_fuel_arr = out_dict[2050].fuel.values
        out_2050_lon_arr = out_dict[2050].lon.values
        out_2050_lat_arr = out_dict[2050].lat.values
        out_2050_plev_arr = out_dict[2050].plev.values
        # Test for correct scaling of fuel data variable
        np.testing.assert_equal(out_2020_fuel_arr, inp_2020_fuel_arr)
        np.testing.assert_equal(out_2050_fuel_arr, (2.0 * inp_2050_fuel_arr))
        # Test that coordinates remain unchanged
        np.testing.assert_equal(out_2050_lon_arr, inp_2050_lon_arr)
        np.testing.assert_equal(out_2050_lat_arr, inp_2050_lat_arr)
        np.testing.assert_equal(out_2050_plev_arr, inp_2050_plev_arr)

    def test_incorrect_input(self, inv_dict):
        "Invalid norm_dict (no fuel key, empty dict) returns KeyError"
        norm_dict = {}
        with pytest.raises(KeyError):
            oac.norm_inv(inv_dict, norm_dict)


@pytest.mark.usefixtures("inv_dict")
class TestScaleInv:
    """Tests function scale_inv(inv_dict, scale_dict)"""

    def test_correct_input(self, inv_dict):
        "Valid input returns dictionary of xr.Dataset, keys are inventory years"
        scale_dict = {"scaling": np.array([1.0, 2.0])}
        years = list(inv_dict.keys())
        out_dict = oac.scale_inv(inv_dict, scale_dict)
        # Test for correct output type
        assert isinstance(out_dict, dict)
        # Test for correct dictionary keys (inventory years)
        assert list(out_dict.keys()) == years

    def test_correct_scaling(self, inv_dict):
        "Test for correct scaling of inventories"
        # Input
        scale_dict = {"scaling": np.array([1.0, 2.0])}
        inp_2020_fuel_arr = inv_dict[2020].fuel.values
        inp_2050_fuel_arr = inv_dict[2050].fuel.values
        inp_2050_lon_arr = inv_dict[2050].lon.values
        inp_2050_lat_arr = inv_dict[2050].lat.values
        inp_2050_plev_arr = inv_dict[2050].plev.values
        # Output
        out_dict = oac.scale_inv(inv_dict, scale_dict)
        out_2020_fuel_arr = out_dict[2020].fuel.values
        out_2050_fuel_arr = out_dict[2050].fuel.values
        out_2050_lon_arr = out_dict[2050].lon.values
        out_2050_lat_arr = out_dict[2050].lat.values
        out_2050_plev_arr = out_dict[2050].plev.values
        # Test for correct scaling of fuel data variable
        np.testing.assert_equal(out_2020_fuel_arr, inp_2020_fuel_arr)
        np.testing.assert_equal(out_2050_fuel_arr, (2.0 * inp_2050_fuel_arr))
        # Test that coordinates remain unchanged
        np.testing.assert_equal(out_2050_lon_arr, inp_2050_lon_arr)
        np.testing.assert_equal(out_2050_lat_arr, inp_2050_lat_arr)
        np.testing.assert_equal(out_2050_plev_arr, inp_2050_plev_arr)

    def test_incorrect_input(self, inv_dict):
        "Invalid scale_dict (invalid key) returns KeyError"
        scale_dict = {"invalid_key": np.array([1.0, 2.0])}
        with pytest.raises(KeyError):
            oac.scale_inv(inv_dict, scale_dict)
