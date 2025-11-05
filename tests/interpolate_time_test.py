"""
Provides tests for module interpolate_time
"""

import numpy as np
import pytest
import openairclim as oac
from utils.create_test_data import create_test_inv


@pytest.fixture(name="setup_valid_arguments", scope="module")
def fixture_setup_valid_arguments():
    """Setup valid arguments for interp_linear(config, years, val_dict)

    Returns dict, np.ndarray, dict: configuration dictionary, numpy array of years,
        dictionary of time series numpy arrays with species names as keys
    """
    config = {"time": {"range": [2000, 2011, 1]}}
    years = np.array([2000, 2010])
    val_dict = {"fuel": np.array([100, 150])}
    return config, years, val_dict


@pytest.fixture(name="setup_invalid_arguments", scope="module")
def fixture_setup_invalid_arguments():
    """Setup invalid arguments for interp_linear(config, years, val_dict)

    Returns dict, np.ndarray, dict: configuration dictionary, numpy array of years,
        dictionary of time series numpy arrays with species names as keys
    """
    config = {"time": {"range": [2000, 2011, 1]}}
    years = np.array([])
    val_dict = {"fuel": np.array([100, 150])}
    return config, years, val_dict


@pytest.fixture(name="inv_dict", scope="module")
def fixture_inv_dict():
    """Fixture to create an example inv_dict"""
    return {2020: create_test_inv(year=2020), 2050: create_test_inv(year=2050)}


@pytest.mark.usefixtures("setup_valid_arguments", "setup_invalid_arguments")
class TestInterpLinear:
    """Tests function interp_linear(config, years, val_dict)"""

    def test_correct_input(self, setup_valid_arguments):
        """Valid input returns time_range (np.ndarray), interp_dict (dict of np.ndarray)"""
        config, years, val_dict = setup_valid_arguments
        time_range, interp_dict = oac.interp_linear(config, years, val_dict)
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
        """Valid input returns dictionary"""
        evo_dict = {"fuel": np.array([100.0]), "EI_CO2": np.array([1.0])}
        ei_inv_dict = {"fuel": np.array([200.0]), "EI_CO2": np.array([2.0])}
        norm_dict = oac.calc_norm(evo_dict, ei_inv_dict)
        assert isinstance(norm_dict, dict)

    def test_correct_normalization(self):
        """Test for correct normalization"""
        evo_dict = {"fuel": np.array([100.0]), "EI_CO2": np.array([1.0])}
        ei_inv_dict = {"fuel": np.array([200.0]), "EI_CO2": np.array([2.0])}
        norm_dict = oac.calc_norm(evo_dict, ei_inv_dict)
        expected_norm_dict = {"fuel": np.array(0.5), "CO2": np.array(0.25)}
        np.testing.assert_equal(norm_dict["fuel"], expected_norm_dict["fuel"])
        np.testing.assert_equal(norm_dict["CO2"], expected_norm_dict["CO2"])

    def test_incorrect_input(self):
        """Invalid ei_inv_dict (no fuel key, empty dict) returns KeyError"""
        evo_dict = {"fuel": np.array([100.0]), "EI_CO2": np.array([1.0])}
        ei_inv_dict = {}
        with pytest.raises(KeyError):
            oac.calc_norm(evo_dict, ei_inv_dict)


@pytest.mark.usefixtures("inv_dict")
class TestCalcInvQuantities:
    """Tests function calc_inv_quantities(config, inv_dict)"""

    def test_correct_input(self, inv_dict):
        """Valid input returns np.ndarray, dict, dict"""
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
        """Test for correct output years"""
        # Input
        inp_years = np.array(list(inv_dict.keys()))
        config = {"species": {"inv": ["CO2", "H2O"]}}
        # Output
        inv_years, _inv_sum_dict, _ei_inv_dict = oac.calc_inv_quantities(
            config, inv_dict
        )
        np.testing.assert_equal(inv_years, inp_years)

    def test_correct_sums(self, inv_dict):
        """Test for correct sums"""
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
        """Valid input returns dictionary of xr.Dataset, keys are inventory years"""
        norm_dict = {"fuel": np.array([1.0, 2.0])}
        years = list(inv_dict.keys())
        out_dict = oac.norm_inv(inv_dict, norm_dict)
        # Test for correct output type
        assert isinstance(out_dict, dict)
        # Test for correct dictionary keys (inventory years)
        assert list(out_dict.keys()) == years

    def test_correct_normalization(self, inv_dict):
        """Test for correct normalization of inventories"""
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
        """Invalid norm_dict (no fuel key, empty dict) returns KeyError"""
        norm_dict = {}
        with pytest.raises(KeyError):
            oac.norm_inv(inv_dict, norm_dict)


@pytest.mark.usefixtures("inv_dict")
class TestScaleInv:
    """Tests function scale_inv(inv_dict, scale_dict)"""

    def test_correct_input(self, inv_dict):
        """Valid input returns dictionary of xr.Dataset, keys are inventory years"""
        scale_dict = {
            "scaling": np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0], 
                [1.0, 1.2, 1.4, 1.6, 1.8]
            ]),
            "species": ['fuel', 'CO2', 'H2O', 'NOx', 'distance']
        }
        years = list(inv_dict.keys())
        out_dict = oac.scale_inv(inv_dict, scale_dict)
        # Test for correct output type
        assert isinstance(out_dict, dict)
        # Test for correct dictionary keys (inventory years)
        assert list(out_dict.keys()) == years

    def test_correct_scaling(self, inv_dict):
        """Test for correct scaling of inventories"""
        # Input
        scale_dict = {
            "scaling": np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0], 
                [1.0, 1.2, 1.4, 1.6, 1.8]
            ]),
            "species": ['fuel', 'CO2', 'H2O', 'NOx', 'distance']
        }
        inp_2020_arrs = {}
        inp_2050_arrs = {}
        out_2020_arrs = {}
        out_2050_arrs = {}
        for specie in scale_dict["species"]:
            inp_2020_arrs[specie] = inv_dict[2020][specie].values
            inp_2050_arrs[specie] = inv_dict[2050][specie].values
        inp_2050_lon_arr = inv_dict[2050].lon.values
        inp_2050_lat_arr = inv_dict[2050].lat.values
        inp_2050_plev_arr = inv_dict[2050].plev.values
        # Output
        out_dict = oac.scale_inv(inv_dict, scale_dict)
        for specie in scale_dict["species"]:
            out_2020_arrs[specie] = out_dict[2020][specie].values
            out_2050_arrs[specie] = out_dict[2050][specie].values
        out_2050_lon_arr = out_dict[2050].lon.values
        out_2050_lat_arr = out_dict[2050].lat.values
        out_2050_plev_arr = out_dict[2050].plev.values
        # Test for correct scaling of fuel data variable
        for i, specie in enumerate(scale_dict["species"]):
            np.testing.assert_equal(out_2020_arrs[specie], inp_2020_arrs[specie])
            expected = scale_dict["scaling"][1, i] * inp_2050_arrs[specie]
            np.testing.assert_allclose(out_2050_arrs[specie], expected, rtol=1e-12)
        
        # Test that coordinates remain unchanged
        np.testing.assert_equal(out_2050_lon_arr, inp_2050_lon_arr)
        np.testing.assert_equal(out_2050_lat_arr, inp_2050_lat_arr)
        np.testing.assert_equal(out_2050_plev_arr, inp_2050_plev_arr)

    def test_incorrect_input(self, inv_dict):
        """Invalid scale_dict (invalid key) returns KeyError"""
        scale_dict = {
            "invalid_key": np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0], 
                [1.0, 1.2, 1.4, 1.6, 1.8]
            ]),
            "species": ['fuel', 'CO2', 'H2O', 'NOx', 'distance']
        }
        with pytest.raises(KeyError):
            oac.scale_inv(inv_dict, scale_dict)




@pytest.mark.usefixtures("inv_dict")
class TestApplyScaling:
    """Tests for apply_scaling function"""

    def test_basic_scaling(self, inv_dict):
        """Test that apply_scaling correctly multiplies by scaling factors"""

        # --- Config ---
        config = {
            "time": {
                "dir": "../oac/example/input/",
                "file": "time_scaling_example.nc",
                "range": [2020, 2051, 1],
            }
        }

        # --- Load scaling file to get species order ---
        evolution = xr.load_dataset(config["time"]["dir"] + config["time"]["file"])
        species_order = evolution.species.values.tolist()
        scaling = evolution.scaling.values
        scaling_years = evolution.time.values

        # --- Prepare dummy val_dict ---
        years = np.array(list(inv_dict.keys()))
        val_dict = {spec: np.ones(len(years)) * (i + 1) for i, spec in enumerate(species_order)}

        # --- Run apply_scaling ---
        time_range, out_dict = oac.apply_scaling(config, val_dict, inv_dict, inventories_adjusted=False)

        # --- Basic checks ---
        assert isinstance(time_range, np.ndarray)
        assert isinstance(out_dict, dict)
        assert set(out_dict.keys()) == set(val_dict.keys())
        assert all(out_dict[spec].shape == time_range.shape for spec in val_dict)

        # --- Check scaling applied correctly ---
        for idx, spec in enumerate(species_order):
            expected = np.interp(time_range, scaling_years, scaling[:, idx]) * val_dict[spec][0]
            np.testing.assert_allclose(out_dict[spec], expected, rtol=1e-12)
