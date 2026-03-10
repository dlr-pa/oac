"""
Provides tests for module calc_swv
"""

from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
import openairclim as oac


class TestCalcSwvRf:
    """Tests function calc_swv_rf(total_swv_dict)"""

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

    def test_invalid_entry(self):
        """
        Invalid input type returns TypeError
        """
        with pytest.raises(TypeError):
            total_swv_mass = [10, 100]
            rf_swv_dict = oac.calc_swv_rf(total_swv_mass)


class TestGetVolumeMatrix:
    """Tests the function get_volume_matrix(heights, latitudes, delta_h, delta_deg)"""

    def test_volume_shape_and_values(self):
        """Tests if the proper shape is returned, also checks for
        increasing values for increasing altitude and decreasing latitude"""
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

    def test_summed_volume(self):
        """Checks if the sum of all volumes corresponds with the atmospheric volume"""
        # calculated total atmospheric volume from the volume_matrix
        delta_h = 1000.0  # height increment in meters
        delta_deg = 1.0  # latitude increment
        heights = np.arange(0, 100000 + delta_h, delta_h)  # 0 to 60 km
        latitudes = np.arange(-90, 91, delta_deg)
        volume_matrix = oac.get_volume_matrix(heights, latitudes, delta_h, delta_deg)

        # Calculate total atmospheric volume using 2 spheres
        earth_radius = 6371000
        outer_radius = earth_radius + 100000 + delta_h
        volume_atm = 4 / 3 * np.pi * (outer_radius**3 - earth_radius**3)
        assert volume_atm == pytest.approx(np.sum(volume_matrix), rel=1e-3)


class TestGetGridData:
    """Test the function get_grid_data(df, heights, latitudes)"""

    def test_griddata_shape(self):
        """Checks if the proper shape is returned, a visual check is
        performed on the interpolation"""
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

    @patch("openairclim.calc_swv.construct_myhre_1m_df")
    @patch("openairclim.calc_swv.get_griddata")
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

        alpha, aoa = oac.get_alpha_aoa(heights, latitudes)

        # Alpha should have same shape as grid
        assert alpha.shape == (3, 2)
        # AoA should be a DataFrame with matching shape
        assert aoa.shape == (3, 2)

    def test_alpha_aoa_value_range(self):
        """Checks that values in alpha are between 0 and 1,"""
        heights = np.array([0, 30000])
        latitudes = np.array([0, 10])

        alpha, aoa = oac.get_alpha_aoa(heights, latitudes)
        aoa_values = np.asarray(aoa, dtype=float)

        # alpha should be between 0 and 1
        # Check only non-NaN entries
        mask = ~np.isnan(alpha)
        assert np.all(
            (alpha[mask] >= 0) & (alpha[mask] <= 1)
        ), "All non-NaN values in alpha must be in range [0, 1]"

        # Check that all non-NaN values are integer
        mask = ~np.isnan(aoa_values)
        assert np.allclose(
            aoa_values[mask], np.round(aoa_values[mask])
        ), "Matrix contains non-integer values"


class TestCalcSWV:
    """
    Tests the function calc_swv_mass_conc(delta_ch4, display_distribution=False)
    """

    @pytest.mark.parametrize(
        "delta_ch4, expected_mass",
        [
            ([0, 0, 0, 0], [0, 0, 0, 0]),
            ([10, 20, 30, 40, 50, 60, 70, 90], [0, 4, 24, 50, 94, 138, 182, 226]),
        ],
    )
    @patch("openairclim.calc_swv.get_alpha_aoa")
    @patch("openairclim.calc_swv.get_volume_matrix")
    @patch("openairclim.calc_swv.Atmosphere")
    def test_calc_swv_mass_conc_basic(
        self,
        mock_atmosphere,
        mock_get_volume,
        mock_get_alpha_aoa,
        delta_ch4,
        expected_mass,
    ):
        """
        Test to verify the function calc_swv_mass_conc().
        Mocking of other functions is done for simplicity
        """
        # Mock get_volume_matrix
        mock_get_volume.return_value = np.ones((2, 2))  # simple 2x2 grid of 1.0

        # Mock Atmosphere
        mock_atm_instance = MagicMock()
        mock_atm_instance.density = np.ones(2) * 1.0  # constant density
        mock_atm_instance.number_density = np.ones(2) * 1e25  # arbitrary number density
        mock_atmosphere.return_value = mock_atm_instance

        # Mock get_alpha_AOA
        alpha = pd.DataFrame([[0.9, 0.8], [0.2, 0.3]])  # fractional release factor
        aoa = pd.DataFrame([[4, 2], [1, 3]])  # years as lags
        mock_get_alpha_aoa.return_value = alpha, aoa

        # Run
        delta_mass_swv, delta_conc_swv, _ = oac.calc_swv_mass_conc(
            delta_ch4, display_distribution=False
        )

        # Assertions
        assert isinstance(delta_mass_swv, np.ndarray)
        assert isinstance(delta_conc_swv, np.ndarray)
        assert delta_mass_swv.shape == (len(delta_ch4),)
        assert delta_conc_swv.shape == (len(delta_ch4),)

        # No NaNs or infs in output
        assert np.all(np.isfinite(delta_mass_swv))
        assert np.all(np.isfinite(delta_conc_swv))

        # Since everything mocked is constant, outputs should be > 0

        m_h2o = 18.01528 * 10**-3  # kg/mol
        m_air = 28.97 * 10**-3  # kg/mol
        for i in range(len(delta_ch4)):
            assert delta_mass_swv[i] * 1e18 == pytest.approx(
                expected_mass[i] * m_h2o / m_air, rel=1e-8
            )
