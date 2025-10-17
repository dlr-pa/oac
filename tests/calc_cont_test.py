"""
Provides tests for module calc_cont
"""

__author__ = "Liam Megill"
__email__ = "liam.megill@dlr.de"
__license__ = "Apache License 2.0"


import numpy as np
import pytest
import openairclim as oac
from utils.create_test_data import create_test_inv, create_test_resp_cont


class TestCheckContInput:
    """Tests function check_cont_input(ds_cont, full_inv_dict, full_base_inv_dict)"""

    @pytest.mark.parametrize("method", ["Megill_2025"])
    def test_missing_ds_cont_vars(self, method):
        """Tests ds_cont with missing data variable."""
        config = {"responses": {"cont": {"method": method}}}
        ds_cont = create_test_resp_cont(method=method)
        ds_cont_incorrect = ds_cont.drop_vars(["g_250"])
        with pytest.raises(KeyError, match=r".* variable 'g_250' .*"):
            oac.check_cont_input(config, ds_cont_incorrect)

    @pytest.mark.parametrize("method", ["Megill_2025"])
    def test_incorrect_ds_cont_coord_unit(self, method):
        """Tests ds_cont with incorrect coordinates and units."""
        config = {"responses": {"cont": {"method": method}}}
        ds_cont = create_test_resp_cont(method=method)
        ds_cont_incorrect1 = ds_cont.copy()
        ds_cont_incorrect1.lat.attrs["units"] = "deg"
        with pytest.raises(ValueError, match=r".* unit .*"):
            oac.check_cont_input(config, ds_cont_incorrect1)
        ds_cont_incorrect2 = ds_cont.copy()
        ds_cont_incorrect2 = ds_cont_incorrect2.rename({"lat": "latitude"})
        with pytest.raises(KeyError, match=r".* coordinate 'lat' .*"):
            oac.check_cont_input(config, ds_cont_incorrect2)


class TestCalcContGridAreas:
    """Tests function calc_cont_grid_areas(lat, lon)"""

    def test_unsorted_latitudes(self):
        """Ensures that the latitude order does not affect results."""
        rnd_lat_vals = np.arange(-89.0, 89.0, 3.0)[::-1]
        np.random.shuffle(rnd_lat_vals)
        lon_vals = np.arange(0, 360, 3.75)
        with pytest.raises(ValueError, match=r".*descend.*"):
            oac.calc_cont_grid_areas(rnd_lat_vals, lon_vals)

    def test_unsorted_longitudes(self):
        """Ensures that the longitude order does not affect results."""
        lat_vals = np.arange(-89.0, 89.0, 3.0)[::-1]
        rnd_lon_vals = np.arange(0, 360, 3.75)
        np.random.shuffle(rnd_lon_vals)
        with pytest.raises(ValueError, match=r".*ascend.*"):
            oac.calc_cont_grid_areas(lat_vals, rnd_lon_vals)

    def test_longitude_edge_cases(self):
        """Checks that longitude edge cases are properly considered."""
        lat_vals = np.arange(-89.0, 89.0, 3.0)[::-1]
        lon_vals = np.arange(0.0, 363.0, 3.0)
        with pytest.raises(ValueError, match=r".* both 0 and 360 .*"):
            oac.calc_cont_grid_areas(lat_vals, lon_vals)


class TestInterpBaseInvDict:
    """Tests function interp_base_inv_dict(inv_yrs, base_inv_dict,
    intrp_vars, cont_grid)"""

    @pytest.fixture(scope="class")
    def cont_grid(self):
        """Fixture to create an example cont_grid."""
        cc_lon_vals = np.arange(0.0, 363.0, 3.0)
        cc_lat_vals = np.arange(-89.0, 89.0, 3.0)[::-1]
        cc_plev_vals = np.arange(150, 350, 50)[::-1]
        return (cc_lon_vals, cc_lat_vals, cc_plev_vals)

    def test_empty_base_inv_dict(self, cont_grid):
        """Tests an empty base_inv_dict."""
        base_inv_dict = {}
        inv_yrs = np.array([2020, 2030, 2040, 2050])
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_yrs, base_inv_dict, intrp_vars, cont_grid)
        assert not result, "Expected empty output when base_inv_dict is empty."

    def test_empty_inv_dict(self, cont_grid):
        """Tests an empty inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["distance"]
        with pytest.raises(ValueError, match="inv_yrs cannot be empty."):
            oac.interp_base_inv_dict([], base_inv_dict, intrp_vars, cont_grid)

    def test_no_missing_years(self, cont_grid):
        """Tests behaviour when all keys in inv_dict are in base_inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        inv_yrs = np.array([2020, 2050])
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_yrs, base_inv_dict, intrp_vars, cont_grid)
        assert result == base_inv_dict, "Expected no change to base_inv_dict."

    def test_missing_years(self, cont_grid):
        """Tests behaviour when there is a key in inv_dict that is not in
        base_inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        inv_yrs = np.array([2020, 2030, 2040, 2050])
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_yrs, base_inv_dict, intrp_vars, cont_grid)
        assert 2030 in result, "Missing year 2030 should have been calculated."

        # compare the sum of the distances
        tot_dist_2020 = base_inv_dict[2020]["distance"].data.sum()
        tot_dist_2050 = base_inv_dict[2050]["distance"].data.sum()
        exp_tot_dist_2030 = tot_dist_2020 + (tot_dist_2050 - tot_dist_2020) / 3
        act_tot_dist_2030 = result[2030]["distance"].data.sum()
        np.testing.assert_allclose(act_tot_dist_2030, exp_tot_dist_2030)

    def test_incorrect_intrp_vars(self, cont_grid):
        """Tests behaviour when the list of values to be interpolated includes
        a value not in inv_dict or base_inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        inv_yrs = np.array([2020, 2030, 2040, 2050])
        intrp_vars = ["wrong-value"]
        with pytest.raises(KeyError, match=r"Variable 'wrong-value' .*"):
            oac.interp_base_inv_dict(inv_yrs, base_inv_dict, intrp_vars, cont_grid)


class TestCalcPPCFMegill:
    """Tests function calc_ppcf_megill(config, ds_cont, ac)."""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont(method="Megill_2025")

    def test_preconditions(self, ds_cont):
        """Tests pre-conditions of function."""
        config = {"aircraft": {"LR": {}}}
        with pytest.raises(KeyError, match="Missing 'G_250'"):
            oac.calc_ppcf_megill(config, ds_cont, "LR")

    def test_linear_interpolation(self, ds_cont):
        """Tests functionality."""
        config = {"aircraft": {"LR": {"G_250": 1.75}}}
        ds_cont = ds_cont.sel(AC=["oac0", "oac1"])
        ds_cont.g_250.loc[{"AC": "oac0"}] = 1.0
        ds_cont.g_250.loc[{"AC": "oac1"}] = 2.0
        ds_cont.ppcf.loc[{"AC": "oac1"}] += 1.0
        result = oac.calc_ppcf_megill(config, ds_cont, "LR")
        a = ds_cont.sel(AC="oac0")["ppcf"].mean().data
        b = result.mean().data
        c = ds_cont.sel(AC="oac1")["ppcf"].mean().data
        assert a < b < c, "Interpolation was unsuccessful."

    def test_lower_bound(self, ds_cont):
        """Tests when G_250 is less than the lower pre-calculaed value."""
        config = {"aircraft": {"LR": {"G_250": 0.2}}}
        ds_cont = ds_cont.sel(AC=["oac0", "oac1"])
        ds_cont.g_250.loc[{"AC": "oac0"}] = 0.5
        ds_cont.g_250.loc[{"AC": "oac1"}] = 2.0
        ds_cont.ppcf.loc[{"AC": "oac1"}] += 1.0
        with pytest.raises(ValueError, match="below pre-calculated"):
            oac.calc_ppcf_megill(config, ds_cont, "LR")

    def test_higher_bound(self, ds_cont):
        """Tests when G_250 is greater than the highest pre-calculated value."""
        config = {"aircraft": {"LR": {"G_250": 5.0}}}
        ds_cont = ds_cont.sel(AC=["oac0", "oac1"])
        ds_cont.g_250.loc[{"AC": "oac0"}] = 0.5
        ds_cont.g_250.loc[{"AC": "oac1"}] = 1.0
        ds_cont.ppcf.loc[{"AC": "oac1"}] += 1.0
        result = oac.calc_ppcf_megill(config, ds_cont, "LR")
        b = result.sel(plev=250).mean().data
        c = ds_cont.sel(AC="oac1", plev=250)["ppcf"].mean().data
        assert b == c, "Interpolation was unsuccessful."


class TestCalcCFDD:
    """Tests function calc_cfdd(config, inv_dict, ds_cont, cont_grid, ac)"""

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020)}

    @pytest.mark.parametrize("method", ["Megill_2025"])
    def test_output_structure(self, inv_dict, method):
        """Tests the output structure."""
        config = {"responses": {"cont": { "method": method}},
                  "aircraft": {"LR": {"G_250": 1.70}},
        }
        ds_cont = create_test_resp_cont(method=method, iss_dim="3D")
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        result = oac.calc_cfdd(config, inv_dict, ds_cont, cont_grid, "LR")

        # run tests
        assert isinstance(result, dict), "Output is not a dictionary."
        assert set(result.keys()) == set(inv_dict.keys()), "Output keys " \
            "do not match input keys."
        for year, cfdd in result.items():
            assert isinstance(cfdd, np.ndarray), "CFDD is not an array."
            assert cfdd.shape == (
                len(cont_grid[2]), len(cont_grid[1]), len(cont_grid[0])
                ), f"CFDD array has incorrect shape for year {year}."

    def test_empty_inventory(self):
        """Tests the handling of an empty input inventory."""
        config = {"responses": {"cont": {"method": "Megill_2025"}},
                  "aircraft": {"LR": {"G_250": 1.70}}}
        ds_cont = create_test_resp_cont(method="Megill_2025")
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        inv_dict = {}  # empty inventory
        result = oac.calc_cfdd(config, inv_dict, ds_cont, cont_grid, "LR")
        assert not result, "Result should be an empty dictionary for an " \
            "empty inventory."


class TestCheckPlevRange:
    """Tests function check_plev_range(inv_dict, cont_grid, clamp=True)"""

    def test_clamping(self):
        """Tests that the clamping to pre-calculated plev values works."""

        # create inv_dict
        inv_year = 2020
        inv_dict = {inv_year: create_test_inv(year=inv_year)}

        # create example contrail grid
        pmin = 150
        pmax = 350
        cont_grid = (None, None, np.arange(pmin, pmax, 50)[::-1])

        # run tests
        inv_dict_out = oac.check_plev_range(inv_dict.copy(), cont_grid)
        assert inv_year in inv_dict, "Incorrect output shape"
        assert np.all(inv_dict_out[inv_year]["plev"] >= pmin), (
            "Clamping above minimum plev did not work."
        )
        assert np.all(inv_dict_out[inv_year]["plev"] <= pmax), (
            "Clamping below maximum plev did not work."
        )


class TestCalcCccovAlltau:
    """Tests function calc_cccov_alltau(cfdd_dict, cont_grid)."""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont()

    def test_output_structure(self, ds_cont):
        """Tests the output structure."""
        len_lon = len(ds_cont.lon.data)
        len_lat = len(ds_cont.lat.data)
        len_plev = len(ds_cont.plev.data)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cfdd_dict = {2020: np.random.rand(len_plev, len_lat, len_lon),
                     2050: np.random.rand(len_plev, len_lat, len_lon)}
        result = oac.calc_cccov_alltau(cfdd_dict, cont_grid)

        # run assertions
        assert isinstance(result, dict), "Output is not a dictionary."
        assert set(result.keys()) == set(cfdd_dict.keys()), "Output keys " \
            "do not match input keys."
        for year, cccov in result.items():
            assert isinstance(cccov, np.ndarray), "cccov is not an array."
            assert cccov.shape == (len_lon,), "cccov array has " \
                f"incorrect shape for year {year}."

    def test_incorrect_cfdd_shape(self, ds_cont):
        """Tests incorrect shape of each cfdd array within cfdd_dict."""
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cfdd_dict = {2020: np.random.rand(10, 10),
                     2050: np.random.rand(10, 10)}
        with pytest.raises(AssertionError, match="Shape"):
            oac.calc_cccov_alltau(cfdd_dict, cont_grid)

    def test_empty_cfdd_dict(self, ds_cont):
        """Tests the output for an empty cfdd_dict."""
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cfdd_dict = {}
        result = oac.calc_cccov_alltau(cfdd_dict, cont_grid)
        assert not result, "Result should be an empty dictionary for an " \
            "empty cfdd_dict."


class TestCalcCccovTaup05:
    """Tests function calc_cccov_taup05(config, cccov_dict, ac)."""

    def test_output_structure(self):
        """Tests the output structure."""
        config = {
            "responses": {"cont": {"low_soot_case": "case1"}},
            "aircraft": {"LR": {"PMrel": 1.0}}
        }
        len_lon = 96
        cccov_dict = {2020: np.random.rand(len_lon)}
        result = oac.calc_cccov_taup05(config, cccov_dict, "LR")

        # run assertions
        assert isinstance(result, dict), "Output is not a dictionary"
        assert set(result.keys()) == set(cccov_dict.keys()), (
            "Output keys do not match input keys."
        )
        for year, cccov_p05 in result.items():
            assert isinstance(cccov_p05, np.ndarray), "cccov_p05 is not an array"
            assert cccov_p05.shape == (len_lon,), (
                f"cccov_p05 array has incorrect shape for year {year}"
            )

    def test_missing_pmrel(self):
        """Tests missing PMrel key."""
        config = {
            "responses": {"cont": {"low_soot_case": "case1"}},
            "aircraft": {"LR": {}}
        }
        len_lon = 96
        cccov_dict = {2020: np.random.rand(len_lon)}
        with pytest.raises(KeyError, match="'PMrel'"):
            oac.calc_cccov_taup05(config, cccov_dict, "LR")

    def test_missing_ls_case(self):
        """Tests missing low_soot_case key."""
        config = {
            "responses": {"cont": {}},
            "aircraft": {"LR": {"PMrel": 1.0}}
        }
        len_lon = 96
        cccov_dict = {2020: np.random.rand(len_lon)}
        with pytest.raises(KeyError, match="'low_soot_case'"):
            oac.calc_cccov_taup05(config, cccov_dict, "LR")


class TestProportionalAttribution:
    """Tests function proportional_attribution(input_dict, ac_dict, total_dict)"""

    def test_key_mismatch(self):
        """Tests mismatched keys in dictionaries."""
        input_dict = {2020: np.array([1.0, 2.0])}
        ac_dict = {2050: np.array([1.0, 2.0])}
        total_dict = {2020: np.array([1.0, 2.0])}
        with pytest.raises(AssertionError, match=r"Keys.*match.*"):
            oac.proportional_attribution(input_dict, ac_dict, total_dict)

    def test_empty_inputs(self):
        """Tests empty input dictionaries."""
        input_dict = {}
        ac_dict = {}
        total_dict = {}
        result = oac.proportional_attribution(
            input_dict, ac_dict, total_dict
        )
        assert not result, "Expected empty result for empty input dictionaries."


class TestCalcContRF:
    """Tests function calc_cont_RF(cccov_dict, cont_grid)."""

    @pytest.fixture(scope="class")
    def cont_grid(self):
        """Fixture to create an example cont_grid."""
        cc_lon_vals = np.arange(0.0, 363.0, 3.0)
        cc_lat_vals = np.arange(-89.0, 89.0, 3.0)[::-1]
        cc_plev_vals = np.arange(150, 350, 50)[::-1]
        return (cc_lon_vals, cc_lat_vals, cc_plev_vals)

    def test_output_structure(self, cont_grid):
        """Tests the output structure."""
        len_lon = len(cont_grid[0])
        cccov_dict = {
            2020: np.random.rand(len_lon),
            2050: np.random.rand(len_lon)}
        result = oac.calc_cont_rf(cccov_dict, cont_grid)

        # run assertions
        assert isinstance(result, dict), "Output should be a dictionary"
        assert set(result.keys()) == set(cccov_dict.keys()), (
            "Output keys do not match input keys."
        )
        for year, rf in result.items():
            assert isinstance(rf, np.ndarray), "rf is not an array"
            assert rf.shape == (len_lon,), (
                f"rf array has incorrect shape for year {year}"
            )

    def test_empty_input_dict(self, cont_grid):
        """Tests empty input dict."""
        with pytest.raises(AssertionError, match="empty"):
            oac.calc_cont_rf({}, cont_grid)


class TestWingspanCorrection:
    """Tests function apply_wingspan_correction(config, rf_arr, ac)."""

    @pytest.mark.parametrize("b", [10.0, 90.0])
    def test_invalid_b(self, b):
        """Tests invalid wingspan."""
        config = {"aircraft": {"LR": {"b": b}}}
        rf_arr = np.random.rand(10)
        with pytest.raises(ValueError, match="Invalid"):
            oac.apply_wingspan_correction(config, rf_arr, "LR")
