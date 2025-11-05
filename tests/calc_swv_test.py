"""
Provides tests for module calc_swv
"""

import numpy as np
import xarray as xr
import pytest
import openairclim as oac

# TODO this is just ch4 test copied for reference, first cleanup the code to properly test it


class TestCalcSwvRf:
    # TODO """Tests function calc_swv_rf(total_swv_dict)"""
    def test_calc_swv_rf(self):
        """
        Tests if calc_swv_rf is working properly when inputting correct values.:
        """
        total_swv_mass = {"SWV": np.array([-10, 100, 160])}
        rf_swv_dict = oac.calc_swv_rf(total_swv_mass)
        assert np.allclose(
            rf_swv_dict["SWV"],
            np.array([-0.00390254, 0.03782624, 0.05252204]),
            rtol=1e-08,
            atol=1e-11,
        )
        with pytest.raises(ValueError):
            rf_swv_dict = oac.calc_swv_rf({"SWV": np.array([170])})
        with pytest.raises(ValueError):
            rf_swv_dict = oac.calc_swv_rf({"SWV": np.array([1.5])})

    def test_invalid_entry(self):
        """
        Invalid input type returns TypeError
        """
        with pytest.raises(TypeError):
            total_swv_mass = [10, 100]
            rf_swv_dict = oac.calc_swv_rf(total_swv_mass)


# SOne test that raises an error when there is no methane concentration available/in OAC


import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# from your_module_name import get_volume_matrix, get_griddata, get_alpha_AOA
# Replace `your_module_name` with the actual filename (without .py)


class TestGetVolumeMatrix:
    """Tests the function get_volume_matrix(heights, latitudes, delta_h, delta_deg)"""

    def test_volume_shape_and_values(self):
        """Tests if the proper shape is returned, also checks for increasing values for increasing altitude and decreasing latitude"""
        heights = np.array([0, 1000])  # 2 altitude levels
        latitudes = np.array([0, 10, 20])  # 3 latitude levels
        delta_h = 1000  # meters
        delta_deg = 10  # degrees

        vol_matrix = oac.get_volume_matrix(heights, latitudes, delta_h, delta_deg)

        # Check shape
        assert vol_matrix.shape == (2, 3)

        # Check positivity of volumes
        assert np.all(vol_matrix > 0)

        # Check that volume increases with altitude (R+h)^2 factor
        assert np.all(vol_matrix[1, :] > vol_matrix[0, :])

        # Check that volume gets smaller closer to the poles
        assert np.all(vol_matrix[:, 1] < vol_matrix[:, 0])

        # TODO think about left,right or center value for the box: Is negligible

    def test_summed_volume(self):
        """Checks if the sum of all volumes corresponds with the atmospheric volume"""
        # calculated total atmospheric volume from the volume_matrix
        delta_h = 1000.0  # height increment in meters
        delta_deg = 1.0  # latitude increment
        heights = np.arange(0, 100000 + delta_h, delta_h)  # 0 to 60 km
        latitudes = np.arange(-90, 91, delta_deg)
        volume_matrix = oac.get_volume_matrix(heights, latitudes, delta_h, delta_deg)

        # Calculate total atmospheric volume using 2 spheres
        R = 6371000
        outer_radius = R + 100000 + delta_h
        volume_atm = 4 / 3 * np.pi * (outer_radius**3 - R**3)
        assert volume_atm == pytest.approx(np.sum(volume_matrix), rel=1e-3)


class TestGetGridData:
    """Thest the function get_grid_data(df, heights, latitudes)"""

    def test_griddata_shape(self):
        """Checks if the proper shape is returned, a visual check is performed on the interpolation"""
        # Create simple DataFrame
        data = {
            "latitude": [0, 10, 20, -50],
            "altitude": [0, 1000, 2000, 1000],
            "value": [1.0, 2.0, 3.0, 2.0],
        }
        df = pd.DataFrame(data)
        heights = np.array([0, 1000, 2000, 3000])
        latitudes = np.array([0, 10, 20])

        grid = oac.get_griddata(df, heights, latitudes, plot_data=False)

        # Should return grid with shape (len(heights), len(latitudes))
        assert grid.shape == (4, 3)


class TestGetAlphaAoa:
    """Test the function get_alpha_AOA(heights,latitudes)"""

    @patch("openairclim.construct_myhre_1m_df")
    @patch("openairclim.get_griddata")
    def test_alpha_aoa_output_shape(self, mock_get_griddata, mock_construct):
        """Checks the proper shape of alpha and AOA matrix"""
        # Mock the functions to avoid actual interpolation / plotting
        mock_construct.return_value = pd.DataFrame(
            {"latitude": [0], "altitude": [0], "value": [1.0]}
        )
        mock_get_griddata.return_value = (
            np.ones((3, 2)) * 1.0
        )  # grid with constant value

        heights = np.array([0, 1000, 2000])
        latitudes = np.array([0, 10])

        alpha, AoA = oac.get_alpha_AOA(heights, latitudes)

        # Alpha should have same shape as grid
        assert alpha.shape == (3, 2)
        # AoA should be a DataFrame with matching shape
        assert AoA.shape == (3, 2)

    # @patch("openairclim.construct_myhre_1m_df")
    # @patch("openairclim.get_griddata")
    # TODO it now will call the functons for griddata and myhre 1m, see if that causes problems...
    def test_alpha_aoa_value_range(self):
        """Checks that values in alpha are ebtween 0 and 1,"""
        # Return smaller grid values than tp_value (1.778)
        # mock_construct.return_value = pd.DataFrame(
        #     {"latitude": [0, 10], "altitude": [0, 1000], "value": [1.0, 1.0]}
        # )
        # mock_get_griddata.return_value = 0.5 * np.ones((2, 2))

        heights = np.array([0, 30000])
        latitudes = np.array([0, 10])

        alpha, AoA = oac.get_alpha_AOA(heights, latitudes)
        AoA_values = np.asarray(AoA, dtype=float)

        # alpha should be between 0 and 1
        # Check only non-NaN entries
        mask = ~np.isnan(alpha)
        assert np.all(
            (alpha[mask] >= 0) & (alpha[mask] <= 1)
        ), "All non-NaN values in alpha must be in range [0, 1]"

        # Check that all non-NaN values are integer
        mask = ~np.isnan(AoA_values)
        assert np.allclose(
            AoA_values[mask], np.round(AoA_values[mask])
        ), "Matrix contains non-integer values"
        # TODO do a small scale dummy calculation on alpha, aoa and AOA
