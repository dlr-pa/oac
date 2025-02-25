"""
Provides tests for module interpolate_time
"""

# import os
# import xarray as xr
import numpy as np
import pytest
import openairclim as oac

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)


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


class TestFilterToInvYears:
    """Tests function filter_to_inv_years(inv_years, time_range, interp_dict)"""
    def test_correct_input(self):
        """Valid input returns dictionary of arrays, filtered to inventory years"""
        inv_years = np.array([2010])
        time_range = np.arange(2000, 2021, 1, dtype=int)
        interp_dict = {"fuel": np.arange(0.0, 21.0, 1.0)}
        filtered_dict = oac.filter_to_inv_years(inv_years, time_range, interp_dict)
        # Test for correct output type
        assert isinstance(filtered_dict, dict)
        # Test for correct output value
        assert filtered_dict["fuel"] == np.array([10.0])
