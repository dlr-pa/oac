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
    """Tests function check_cont_input(ds_cont, inv_dict, base_inv_dict)"""

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020),
                2030: create_test_inv(year=2030),
                2040: create_test_inv(year=2040),
                2050: create_test_inv(year=2050)}

    @pytest.mark.parametrize("method", ["AirClim", "Megill_2025"])
    def test_year_out_of_range(self, inv_dict, method):
        """Tests behaviour when inv_dict includes a year that is out of range
        of the years in base_inv_dict."""
        config = {"responses": {"cont": {"method": method}}}
        ds_cont = create_test_resp_cont(method=method)
        # test year too low
        base_inv_dict = {2030: create_test_inv(year=2030),
                         2050: create_test_inv(year=2050)}
        with pytest.raises(ValueError, match=r".* inv_dict key .* earliest .*"):
            oac.check_cont_input(config, ds_cont, inv_dict, base_inv_dict)
        # test year too high
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2040: create_test_inv(year=2040)}
        with pytest.raises(ValueError, match=r".* inv_dict key .* largest .*"):
            oac.check_cont_input(config, ds_cont, inv_dict, base_inv_dict)

    @pytest.mark.parametrize("method", ["AirClim", "Megill_2025"])
    def test_missing_ds_cont_vars(self, inv_dict, method):
        """Tests ds_cont with missing data variable."""
        config = {"responses": {"cont": {"method": method}}}
        ds_cont = create_test_resp_cont(method=method)
        base_inv_dict = inv_dict
        ds_cont_incorrect = ds_cont.drop_vars(["ISS"])
        with pytest.raises(KeyError, match=r".* variable 'ISS' .*"):
            oac.check_cont_input(config, ds_cont_incorrect, inv_dict, base_inv_dict)

    @pytest.mark.parametrize("method", ["AirClim", "Megill_2025"])
    def test_incorrect_ds_cont_coord_unit(self, inv_dict, method):
        """Tests ds_cont with incorrect coordinates and units."""
        config = {"responses": {"cont": {"method": method}}}
        ds_cont = create_test_resp_cont(method=method)
        base_inv_dict = inv_dict
        ds_cont_incorrect1 = ds_cont.copy()
        ds_cont_incorrect1.lat.attrs["units"] = "deg"
        with pytest.raises(ValueError, match=r".* unit .*"):
            oac.check_cont_input(config, ds_cont_incorrect1, inv_dict, base_inv_dict)
        ds_cont_incorrect2 = ds_cont.copy()
        ds_cont_incorrect2 = ds_cont_incorrect2.rename({"lat": "latitude"})
        with pytest.raises(KeyError, match=r".* coordinate 'lat' .*"):
            oac.check_cont_input(config, ds_cont_incorrect2, inv_dict, base_inv_dict)


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
    """Tests function interp_base_inv_dict(inv_dict, base_inv_dict,
    intrp_vars)"""

    @pytest.fixture(scope="class")
    def cont_grid(self):
        """Fixture to create an example cont_grid."""
        cc_lon_vals = np.arange(0.0, 363.0, 3.0)
        cc_lat_vals = np.arange(-89.0, 89.0, 3.0)[::-1]
        cc_plev_vals = np.arange(150, 350, 50)[::-1]
        return (cc_lon_vals, cc_lat_vals, cc_plev_vals)

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020),
                2030: create_test_inv(year=2030),
                2040: create_test_inv(year=2040),
                2050: create_test_inv(year=2050)}

    def test_empty_base_inv_dict(self, inv_dict, cont_grid):
        """Tests an empty base_inv_dict."""
        base_inv_dict = {}
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars, cont_grid)
        assert not result, "Expected empty output when base_inv_dict is empty."

    def test_empty_inv_dict(self, cont_grid):
        """Tests an empty inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["distance"]
        with pytest.raises(ValueError, match="inv_dict cannot be empty."):
            oac.interp_base_inv_dict({}, base_inv_dict, intrp_vars, cont_grid)

    def test_no_missing_years(self, inv_dict, cont_grid):
        """Tests behaviour when all keys in inv_dict are in base_inv_dict."""
        base_inv_dict = inv_dict.copy()
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars, cont_grid)
        assert result == base_inv_dict, "Expected no change to base_inv_dict."

    def test_missing_years(self, inv_dict, cont_grid):
        """Tests behaviour when there is a key in inv_dict that is not in
        base_inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars, cont_grid)
        assert 2030 in result, "Missing year 2030 should have been calculated."

        # compare the sum of the distances
        tot_dist_2020 = base_inv_dict[2020]["distance"].data.sum()
        tot_dist_2050 = base_inv_dict[2050]["distance"].data.sum()
        exp_tot_dist_2030 = tot_dist_2020 + (tot_dist_2050 - tot_dist_2020) / 3
        act_tot_dist_2030 = result[2030]["distance"].data.sum()
        np.testing.assert_allclose(act_tot_dist_2030, exp_tot_dist_2030)

    def test_incorrect_intrp_vars(self, inv_dict, cont_grid):
        """Tests behaviour when the list of values to be interpolated includes
        a value not in inv_dict or base_inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["wrong-value"]
        with pytest.raises(KeyError, match=r"Variable 'wrong-value' .*"):
            oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars, cont_grid)

    def test_year_out_of_range(self, inv_dict, cont_grid):
        """Tests behaviour when inv_dict includes a year that is out of range
        of the years in base_inv_dict."""
        # test year too low
        base_inv_dict = {2030: create_test_inv(year=2030),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["distance"]
        with pytest.raises(ValueError, match=r".*inv_dict.*less.*"):
            oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars, cont_grid)
        # test year too high
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2040: create_test_inv(year=2040)}
        with pytest.raises(ValueError, match=r".*inv_dict.*larger.*"):
            oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars, cont_grid)


class TestCalcContWeighting:
    """Tests function calc_cont_weighting(config, val, cont_grid, ac)"""

    @pytest.fixture(scope="class")
    def cont_grid(self):
        """Fixture to create an example cont_grid."""
        cc_lon_vals = np.arange(0.0, 363.0, 3.0)
        cc_lat_vals = np.linspace(-89.0, 89.0, 60)[::-1]
        cc_plev_vals = np.arange(150, 350, 50)[::-1]
        return (cc_lon_vals, cc_lat_vals, cc_plev_vals)

    def test_invalid_value(self, cont_grid):
        """Tests an invalid weighting value."""
        config = {"aircraft": {"LR": {"eff_fac": 1.0}}}
        with pytest.raises(ValueError):
            oac.calc_cont_weighting(config, "invalid_value", cont_grid, "LR")

    @pytest.mark.parametrize(
        "val,len_val", [("w1", 60), ("w2", 60), ("w3", 60)]
    )
    def test_weighting_size(self, val, len_val, cont_grid):
        """Tests that calculated weightings are of size (nlat)."""
        config = {"aircraft": {"LR": {"eff_fac": 1.0}}}
        assert len(oac.calc_cont_weighting(config, val, cont_grid, "LR")) == len_val

    @pytest.mark.parametrize(
        "config", [{}, {"aircraft": {}}, {"aircraft": {"LR": {}}}]
    )
    def test_missing_config_values(self, config, cont_grid):
        """Tests missing config values. Only w2 uses config, so only this
        weighting factor is tested."""
        with pytest.raises(KeyError):
            oac.calc_cont_weighting(config, "w2", cont_grid, "LR")


class TestCalcPSACAirclim:
    """Tests function calc_psac_airclim(config, ds_cont, ac)."""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont(method="AirClim")

    def test_invalid_g_comp(self, ds_cont):
        """Tests an invalid G_comp value."""
        config = {"aircraft": {"LR": {"G_comp": 0.2}}}
        with pytest.raises(ValueError, match="Invalid G_comp"):
            oac.calc_psac_airclim(config, ds_cont, "LR")
        # test lower bound
        config = {"aircraft": {"LR": {"G_comp": 0.02}}}
        with pytest.raises(ValueError, match="Invalid G_comp"):
            oac.calc_psac_airclim(config, ds_cont, "LR")

    def test_linear_interpolation(self, ds_cont):
        """Tests the linear interpolation function."""
        config = {"aircraft": {"LR": {"G_comp": 0.07}}}
        ds_cont["SAC_LH2"] += 1.0
        b = oac.calc_psac_airclim(config, ds_cont, "LR").mean().data
        a = ds_cont.SAC_CON.mean().data
        c = ds_cont.SAC_LH2.mean().data
        assert a < b < c, "Linear interpolation was unsuccessful."


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

    @pytest.mark.parametrize("method", ["AirClim", "Megill_2025"])
    def test_output_structure(self, inv_dict, method):
        """Tests the output structure."""
        config = {"responses": {"cont": { "method": method}},
                  "aircraft": {"LR": {"G_comp": 0.1, "G_250": 1.70}},
        }
        ds_cont = create_test_resp_cont(method=method)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        result = oac.calc_cfdd(config, inv_dict, ds_cont, cont_grid, "LR")

        # run tests
        assert isinstance(result, dict), "Output is not a dictionary."
        assert set(result.keys()) == set(inv_dict.keys()), "Output keys " \
            "do not match input keys."
        for year, cfdd in result.items():
            assert isinstance(cfdd, np.ndarray), "CFDD is not an array."
            assert cfdd.shape == (len(cont_grid[1]), len(cont_grid[0])),\
                f"CFDD array has incorrect shape for year {year}."

    def test_empty_inventory(self):
        """Tests the handling of an empty input inventory."""
        config = {"responses": {"cont": {"method": "AirClim"}},
                  "aircraft": {"LR": {"G_comp": 0.1}}}
        ds_cont = create_test_resp_cont(method="AirClim")
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        inv_dict = {}  # empty inventory
        result = oac.calc_cfdd(config, inv_dict, ds_cont, cont_grid, "LR")
        assert not result, "Result should be an empty dictionary for an " \
            "empty inventory."


class TestCalcCccov:
    """Tests function calc_cccov(config, cfdd_dict, ds_cont, cont_grid, ac)."""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont()

    def test_output_structure(self, ds_cont):
        """Tests the output structure."""
        config = {"aircraft": {"LR": {"eff_fac": 0.5}}}
        len_lon = len(ds_cont.lon.data)
        len_lat = len(ds_cont.lat.data)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cfdd_dict = {2020: np.random.rand(len_lat, len_lon),
                     2050: np.random.rand(len_lat, len_lon)}
        result = oac.calc_cccov(config, cfdd_dict, ds_cont, cont_grid, "LR")

        # run assertions
        assert isinstance(result, dict), "Output is not a dictionary."
        assert set(result.keys()) == set(cfdd_dict.keys()), "Output keys " \
            "do not match input keys."
        for year, cccov in result.items():
            assert isinstance(cccov, np.ndarray), "cccov is not an array."
            assert cccov.shape == (len_lat, len_lon), "cccov array has " \
                f"incorrect shape for year {year}."

    def test_incorrect_cfdd_shape(self, ds_cont):
        """Tests incorrect shape of each cfdd array within cfdd_dict."""
        config = {"aircraft": {"LR": {"eff_fac": 0.5}}}
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cfdd_dict = {2020: np.random.rand(10, 10),
                     2050: np.random.rand(10, 10)}
        with pytest.raises(AssertionError, match="Shape"):
            oac.calc_cccov(config, cfdd_dict, ds_cont, cont_grid, "LR")

    def test_missing_config_values(self, ds_cont):
        """Tests missing config values."""
        config = {"aircraft": {"LR": {}}}
        len_lon = len(ds_cont.lon.data)
        len_lat = len(ds_cont.lat.data)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cfdd_dict = {2020: np.random.rand(len_lat, len_lon),
                     2050: np.random.rand(len_lat, len_lon)}
        with pytest.raises(KeyError, match="Missing"):
            oac.calc_cccov(config, cfdd_dict, ds_cont, cont_grid, "LR")

    def test_empty_cfdd_dict(self, ds_cont):
        """Tests the output for an empty cfdd_dict."""
        config = {"aircraft": {"LR": {"eff_fac": 0.5}}}
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cfdd_dict = {}
        result = oac.calc_cccov(config, cfdd_dict, ds_cont, cont_grid, "LR")
        assert not result, "Result should be an empty dictionary for an " \
            "empty cfdd_dict."


class TestCalcWeightedCccov:
    """Tests function calc_weighted_cccov(comb_cccov_dict, cfdd_dict,
    comb_cfdd_dict)"""

    def test_key_mismatch(self):
        """Tests mismatched keys in dictionaries."""
        comb_cccov_dict = {2020: np.array([1.0, 2.0])}
        cfdd_dict = {2050: np.array([1.0, 2.0])}
        comb_cfdd_dict = {2020: np.array([1.0, 2.0])}
        with pytest.raises(AssertionError, match=r"Keys.*match.*"):
            oac.calc_weighted_cccov(comb_cccov_dict, cfdd_dict, comb_cfdd_dict)

    def test_empty_inputs(self):
        """Tests empty input dictionaries."""
        comb_cccov_dict = {}
        cfdd_dict = {}
        comb_cfdd_dict = {}
        result = oac.calc_weighted_cccov(
            comb_cccov_dict, cfdd_dict, comb_cfdd_dict
        )
        assert not result, "Expected empty result for empty input dictionaries."


class TestCalcCccovTot:
    """Tests function calc_cccov_tot(config, cccov_dict, cont_grid, ac)"""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont()

    def test_output_structure(self, ds_cont):
        """Tests output structure."""
        len_lon = len(ds_cont.lon.data)
        len_lat = len(ds_cont.lat.data)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cccov_dict = {2020: np.random.rand(len_lat, len_lon),
                      2050: np.random.rand(len_lat, len_lon)}
        config = {"aircraft": {"LR": {"eff_fac": 0.5}}}
        result = oac.calc_cccov_tot(config, cccov_dict, cont_grid, "LR")

        # run assertions
        assert isinstance(result, dict), "Output is not a dictionary."
        assert set(result.keys()) == set(cccov_dict.keys()), "Output keys " \
            "do not match input keys."
        for _, cccov_tot in result.items():
            assert isinstance(cccov_tot, np.ndarray), "cccov_tot should be " \
                "an array."
            assert cccov_tot.shape == (len_lat,), "cccov_tot should be a " \
                "function of latitude only."

    def test_incorrect_cccov_shape(self, ds_cont):
        """Tests incorrect shape of each cccov array within cfdd_dict."""
        config = {"aircraft": {"LR": {"eff_fac": 0.5}}}
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cccov_dict = {2020: np.random.rand(10, 10),
                      2050: np.random.rand(10, 10)}
        with pytest.raises(AssertionError):
            oac.calc_cccov_tot(config, cccov_dict, cont_grid, "LR")

    def test_empty_cccov_dict(self, ds_cont):
        """Tests the output for an empty cccov_dict."""
        config = {"aircraft": {"LR": {"eff_fac": 0.5}}}
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cccov_dict = {}
        result = oac.calc_cccov_tot(config, cccov_dict, cont_grid, "LR")
        assert not result, "Result should be an empty dictionary for an " \
            "empty cccov_dict."


class TestCalcContRF:
    """Tests function calc_cont_RF(config, cccov_tot_dict, inv_dict, cont_grid, ac)."""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont()

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020),
                2050: create_test_inv(year=2050)}

    def test_output_structure(self, inv_dict, ds_cont):
        """Tests the output structure."""
        config = {"aircraft": {"LR": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        len_lat = len(ds_cont.lat.data)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        years = list(inv_dict.keys())
        cccov_tot_dict = {years[0]: np.random.rand(len_lat),
                          years[1]: np.random.rand(len_lat)}
        result = oac.calc_cont_rf(config, cccov_tot_dict, inv_dict, cont_grid, "LR")

        # run assertions
        assert isinstance(result, dict), "Output should be a dictionary"
        assert "cont" in result, "Output does not include 'cont'."
        assert len(result["cont"]) == 31, "Output length does not match the " \
            " number of years in inv_dict."

    def test_incorrect_keys(self, inv_dict, ds_cont):
        """Tests differing keys in inv_dict and cccov_tot_dict."""
        config = {"aircraft": {"LR": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        len_lat = len(ds_cont.lat.data)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        cccov_tot_dict = {2021: np.random.rand(len_lat),
                          2049: np.random.rand(len_lat)}
        with pytest.raises(AssertionError, match="Keys"):
            oac.calc_cont_rf(config, cccov_tot_dict, inv_dict, cont_grid, "LR")

    def test_missing_config_values(self, inv_dict, ds_cont):
        """Tests missing config values."""
        config = {"aircraft": {"LR": {}}}
        len_lat = len(ds_cont.lat.data)
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        years = list(inv_dict.keys())
        cccov_tot_dict = {years[0]: np.random.rand(len_lat),
                          years[1]: np.random.rand(len_lat)}
        with pytest.raises(KeyError, match="Missing"):
            oac.calc_cont_rf(config, cccov_tot_dict, inv_dict, cont_grid, "LR")

    def test_empty_input_dicts(self, ds_cont):
        """Tests empty input dicts."""
        config = {"aircraft": {"LR": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        cont_grid = (ds_cont.lon.data, ds_cont.lat.data, ds_cont.plev.data)
        with pytest.raises(ValueError, match="empty"):
            oac.calc_cont_rf(config, {}, {}, cont_grid, "LR")


class TestAddInvToBase:
    """Tests function add_inv_to_base(inv_dict, base_inv_dict)"""

    def test_key_mismatch(self):
        """Tests mismatched keys in input dictionaries."""
        inv_dict = {2020: np.array([1.0, 2.0])}
        base_inv_dict = {2050: np.array([1.0, 2.0])}
        with pytest.raises(KeyError, match="keys"):
            oac.add_inv_to_base(inv_dict, base_inv_dict)

    def test_addition(self):
        """Tests function with simple inputs."""
        inv_dict = {2020: np.array([1.0, 2.0])}
        base_inv_dict = {2020: np.array([2.0, 3.0])}
        expected = {2020: np.array([3.0, 5.0])}
        result = oac.add_inv_to_base(inv_dict, base_inv_dict)
        np.testing.assert_array_equal(
            result[2020], expected[2020],
            err_msg="Addition fails for simple input."
        )

    def test_empty_inputs(self):
        """Tests empty input dictionaries."""
        inv_dict = {}
        base_inv_dict = {}
        result = oac.add_inv_to_base(inv_dict, base_inv_dict)
        assert not result, "Expected empty result for empty input dictionaries."
