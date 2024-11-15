"""
Provides tests for module calc_cont
"""
import numpy as np
import pytest
import openairclim as oac
from utils.create_test_data import create_test_inv, create_test_resp_cont


class TestInputContrailGrid:
    """Tests the input contrail grid, defined by `cc_lat_vals`, `cc_lon_vals`
    and `cc_plev_vals`."""

    def test_lon_vals(self):
        """Tests longitude values."""
        lon = oac.cc_lon_vals
        assert len(lon) > 0, "Contrail grid longitudes cannot be empty."
        assert len(lon) == len(np.unique(lon)), "Duplicate longitude values " \
            "in contrail grid."
        assert np.all((lon >= 0.) & (lon <= 360.)), "Longitude values must " \
            "vary between 0 and 360 degrees"
        assert (0. in lon) != (360. in lon), "Longitude values must not " \
            "include both 0 and 360 deg values."
        assert np.all(lon == np.sort(lon)), "Contrail grid longitudes must " \
            "be sorted in ascending order."

    def test_lat_vals(self):
        """Tests latitude values."""
        lat = oac.cc_lat_vals
        assert len(lat) > 0, "Contrail grid latitudes cannot be empty."
        assert len(lat) == len(np.unique(lat)), "Duplicate latitude values " \
            "in contrail grid."
        assert np.all((lat >= -90.) & (lat <= 90)),  "Latitude values must " \
            "be between, but not equal to, -90 and +90 degrees."
        assert np.all(lat == np.sort(lat)[::-1]), "Contrail grid latitudes " \
            "must be sorted in descending order."

    def test_plev_vals(self):
        """Tests pressure level values."""
        plev = oac.cc_plev_vals
        assert len(plev) > 0, "Contrail grid pressure levels cannot be empty."
        assert len(plev) == len(np.unique(plev)), "Duplicate pressure level " \
            "values in contrail grid."
        assert np.all(plev == np.sort(plev)[::-1]), "Contrail grid pressure " \
            "level values must be sorted in descending order."
        assert np.all(plev <= 1014.), "Contrail grid pressure levels must " \
            "be at altitudes above ground level - defined as 1014 hPa."


class TestCheckContInput:
    """Tests function check_cont_input(ds_cont, inv_dict, base_inv_dict)"""

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020),
                2030: create_test_inv(year=2030),
                2040: create_test_inv(year=2040),
                2050: create_test_inv(year=2050)}

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont()

    def test_year_out_of_range(self, ds_cont, inv_dict):
        """Tests behaviour when inv_dict includes a year that is out of range
        of the years in base_inv_dict."""
        # test year too low
        base_inv_dict = {2030: create_test_inv(year=2030),
                         2050: create_test_inv(year=2050)}
        with pytest.raises(AssertionError):
            oac.check_cont_input(ds_cont, inv_dict, base_inv_dict)
        # test year too high
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2040: create_test_inv(year=2040)}
        with pytest.raises(AssertionError):
            oac.check_cont_input(ds_cont, inv_dict, base_inv_dict)

    def test_missing_ds_cont_vars(self, ds_cont, inv_dict):
        """Tests ds_cont with missing data variable."""
        base_inv_dict = inv_dict
        ds_cont_incorrect = ds_cont.drop_vars(["ISS"])
        with pytest.raises(AssertionError):
            oac.check_cont_input(ds_cont_incorrect, inv_dict, base_inv_dict)

    def test_incorrect_ds_cont_coord_unit(self, ds_cont, inv_dict):
        """Tests ds_cont with incorrect coordinates and units."""
        base_inv_dict = inv_dict
        ds_cont_incorrect1 = ds_cont.copy()
        ds_cont_incorrect1.lat.attrs["units"] = "deg"
        ds_cont_incorrect2 = ds_cont.copy()
        ds_cont_incorrect2 = ds_cont_incorrect2.rename({"lat": "latitude"})
        for ds_cont_incorrect in [ds_cont_incorrect1, ds_cont_incorrect2]:
            with pytest.raises(AssertionError):
                oac.check_cont_input(ds_cont_incorrect, inv_dict, base_inv_dict)


class TestCalcContGridAreas:
    """Tests function calc_cont_grid_areas(lat, lon)"""

    def test_unsorted_latitudes(self):
        """Ensures that the latitude order does not affect results."""
        lat_vals = np.arange(-89.0, 89.0, 3.0)
        rnd_lat_vals = np.arange(-89.0, 89.0, 3.0)
        np.random.shuffle(rnd_lat_vals)
        lon_vals = np.arange(0, 360, 3.75)
        res_unsorted = oac.calc_cont_grid_areas(rnd_lat_vals, lon_vals)
        res_sorted = oac.calc_cont_grid_areas(lat_vals, lon_vals)
        assert np.all(res_unsorted == res_sorted), "Sorting of latitudes " \
            "unsuccessful."

    def test_unsorted_longitudes(self):
        """Ensures that the longitude order does not affect results."""
        lat_vals = np.arange(-89.0, 89.0, 3.0)
        lon_vals = np.arange(0, 360, 3.75)
        rnd_lon_vals = np.arange(0, 360, 3.75)
        np.random.shuffle(rnd_lon_vals)
        res_unsorted = oac.calc_cont_grid_areas(lat_vals, rnd_lon_vals)
        res_sorted = oac.calc_cont_grid_areas(lat_vals, lon_vals)
        assert np.all(res_unsorted == res_sorted), "Sorting of longitudes " \
            "unsuccessful."


class TestInterpBaseInvDict:
    """Tests function interp_base_inv_dict(inv_dict, base_inv_dict,
    intrp_vars)"""

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020),
                2030: create_test_inv(year=2030),
                2040: create_test_inv(year=2040),
                2050: create_test_inv(year=2050)}

    def test_empty_base_inv_dict(self, inv_dict):
        """Tests an empty base_inv_dict."""
        base_inv_dict = {}
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars)
        assert not result, "Expected empty output when base_inv_dict is empty."

    def test_empty_inv_dict(self):
        """Tests an empty inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["distance"]
        with pytest.raises(AssertionError):
            oac.interp_base_inv_dict({}, base_inv_dict, intrp_vars)

    def test_no_missing_years(self, inv_dict):
        """Tests behaviour when all keys in inv_dict are in base_inv_dict."""
        base_inv_dict = inv_dict.copy()
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars)
        assert result == base_inv_dict, "Expected no change to base_inv_dict."

    def test_missing_years(self, inv_dict):
        """Tests behaviour when there is a key in inv_dict that is not in
        base_inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["distance"]
        result = oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars)
        assert 2030 in result, "Missing year 2030 should have been calculated."

        # compare the sum of the distances
        tot_dist_2020 = base_inv_dict[2020]["distance"].data.sum()
        tot_dist_2050 = base_inv_dict[2050]["distance"].data.sum()
        exp_tot_dist_2030 = tot_dist_2020 + (tot_dist_2050 - tot_dist_2020) / 3
        act_tot_dist_2030 = result[2030]["distance"].data.sum()
        np.testing.assert_allclose(act_tot_dist_2030, exp_tot_dist_2030)

    def test_incorrect_intrp_vars(self, inv_dict):
        """Tests behaviour when the list of values to be interpolated includes
        a value not in inv_dict or base_inv_dict."""
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["wrong-value"]
        with pytest.raises(AssertionError):
            oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars)

    def test_year_out_of_range(self, inv_dict):
        """Tests behaviour when inv_dict includes a year that is out of range
        of the years in base_inv_dict."""
        # test year too low
        base_inv_dict = {2030: create_test_inv(year=2030),
                         2050: create_test_inv(year=2050)}
        intrp_vars = ["distance"]
        with pytest.raises(AssertionError):
            oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars)
        # test year too high
        base_inv_dict = {2020: create_test_inv(year=2020),
                         2040: create_test_inv(year=2040)}
        with pytest.raises(AssertionError):
            oac.interp_base_inv_dict(inv_dict, base_inv_dict, intrp_vars)


class TestCalcContWeighting:
    """Tests function calc_cont_weighting(config, val)"""

    def test_invalid_value(self):
        """Tests an invalid weighting value."""
        config = {"responses": {"cont":{"eff_fac": 1.0}}}
        with pytest.raises(ValueError):
            oac.calc_cont_weighting(config, "invalid_value")

    nlat = len(oac.cc_lat_vals)
    @pytest.mark.parametrize("val,len_val", [("w1", nlat),
                                             ("w2", nlat),
                                             ("w3", nlat)])
    def test_weighting_size(self, val, len_val):
        """Tests that calculated weightings are of size (nlat)."""
        config = {"responses": {"cont": {"eff_fac": 1.0}}}
        assert len(oac.calc_cont_weighting(config, val)) == len_val

    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}}])
    def test_missing_config_values(self, config):
        """Tests missing config values."""
        with pytest.raises(AssertionError):
            oac.calc_cont_weighting(config, "w2")  # only w2 uses config


class TestCalcCFDD:
    """Tests function calc_cfdd(config, inv_dict)"""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont()

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020)}

    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}}])
    def test_missing_config_values(self, config, inv_dict, ds_cont):
        """Tests missing config values."""
        with pytest.raises(AssertionError):
            oac.calc_cfdd(config, inv_dict, ds_cont)

    def test_invalid_g_comp(self, inv_dict, ds_cont):
        """Tests an invalid G_comp value."""
        config = {"responses": {"cont": {"G_comp": 0.2}}}
        with pytest.raises(AssertionError):
            oac.calc_cfdd(config, inv_dict, ds_cont)
        # test lower bound
        config["responses"]["cont"]["G_comp"] = 0.02
        with pytest.raises(AssertionError):
            oac.calc_cfdd(config, inv_dict, ds_cont)

    def test_output_structure(self, inv_dict, ds_cont):
        """Tests the output structure."""
        config = {"responses": {"cont": {"G_comp": 0.1}}}
        result = oac.calc_cfdd(config, inv_dict, ds_cont)

        # run tests
        assert isinstance(result, dict), "Output is not a dictionary."
        assert set(result.keys()) == set(inv_dict.keys()), "Output keys " \
            "do not match input keys."
        for year, cfdd in result.items():
            assert isinstance(cfdd, np.ndarray), "CFDD is not an array."
            assert cfdd.shape == (len(oac.cc_lat_vals), len(oac.cc_lon_vals)),\
                f"CFDD array has incorrect shape for year {year}."

    def test_empty_inventory(self, ds_cont):
        """Tests the handling of an empty input inventory."""
        config = {"responses": {"cont": {"G_comp": 0.1}}}
        inv_dict = {}  # empty inventory
        result = oac.calc_cfdd(config, inv_dict, ds_cont)
        assert not result, "Result should be an empty dictionary for an " \
            "empty inventory."


class TestCalcCccov:
    """Tests function calc_cccov(config, cfdd_dict)"""

    @pytest.fixture(scope="class")
    def ds_cont(self):
        """Fixture to load an example ds_cont file."""
        return create_test_resp_cont()

    def test_output_structure(self, ds_cont):
        """Tests the output structure."""
        config = {"responses": {"cont": {"eff_fac": 0.5}}}
        len_lon = len(oac.cc_lon_vals)
        len_lat = len(oac.cc_lat_vals)
        cfdd_dict = {2020: np.random.rand(len_lat, len_lon),
                     2050: np.random.rand(len_lat, len_lon)}
        result = oac.calc_cccov(config, cfdd_dict, ds_cont)

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
        config = {"responses": {"cont": {"eff_fac": 0.5}}}
        cfdd_dict = {2020: np.random.rand(10, 10),
                     2050: np.random.rand(10, 10)}
        with pytest.raises(AssertionError):
            oac.calc_cccov(config, cfdd_dict, ds_cont)

    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}}])
    def test_missing_config_values(self, config, ds_cont):
        """Tests missing config values."""
        len_lon = len(oac.cc_lon_vals)
        len_lat = len(oac.cc_lat_vals)
        cfdd_dict = {2020: np.random.rand(len_lat, len_lon),
                        2050: np.random.rand(len_lat, len_lon)}
        with pytest.raises(AssertionError):
            oac.calc_cccov(config, cfdd_dict, ds_cont)

    def test_empty_cfdd_dict(self, ds_cont):
        """Tests the output for an empty cfdd_dict."""
        config = {"responses": {"cont": {"eff_fac": 0.5}}}
        cfdd_dict = {}
        result = oac.calc_cccov(config, cfdd_dict, ds_cont)
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
        with pytest.raises(AssertionError):
            oac.calc_weighted_cccov(comb_cccov_dict, cfdd_dict, comb_cfdd_dict)

    def test_empty_inputs(self):
        """Tests empty input dictionaries."""
        comb_cccov_dict = {}
        cfdd_dict = {}
        comb_cfdd_dict = {}
        result = oac.calc_weighted_cccov(comb_cccov_dict,
                                         cfdd_dict,
                                         comb_cfdd_dict)
        assert not result, "Expected empty result for empty input dictionaries."


class TestCalcCccovTot:
    """Tests function calc_cccov_tot(config, cccov_dict)"""

    def test_output_structure(self):
        """Tests output structure."""
        len_lon = len(oac.cc_lon_vals)
        len_lat = len(oac.cc_lat_vals)
        cccov_dict = {2020: np.random.rand(len_lat, len_lon),
                      2050: np.random.rand(len_lat, len_lon)}
        config = {"responses": {"cont": {"eff_fac": 0.5}}}
        result = oac.calc_cccov_tot(config, cccov_dict)

        # run assertions
        assert isinstance(result, dict), "Output is not a dictionary."
        assert set(result.keys()) == set(cccov_dict.keys()), "Output keys " \
            "do not match input keys."
        for _, cccov_tot in result.items():
            assert isinstance(cccov_tot, np.ndarray), "cccov_tot should be " \
                "an array."
            assert cccov_tot.shape == (len_lat,), "cccov_tot should be a " \
                "function of latitude only."

    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}}])
    def test_missing_config_values(self, config):
        """Tests missing config values."""
        len_lon = len(oac.cc_lon_vals)
        len_lat = len(oac.cc_lat_vals)
        cccov_dict = {2020: np.random.rand(len_lat, len_lon),
                      2050: np.random.rand(len_lat, len_lon)}
        with pytest.raises(AssertionError):
            oac.calc_cccov_tot(config, cccov_dict)

    def test_incorrect_cccov_shape(self):
        """Tests incorrect shape of each cccov array within cfdd_dict."""
        config = {"responses": {"cont": {"eff_fac": 0.5}}}
        cccov_dict = {2020: np.random.rand(10, 10),
                      2050: np.random.rand(10, 10)}
        with pytest.raises(AssertionError):
            oac.calc_cccov_tot(config, cccov_dict)

    def test_empty_cccov_dict(self):
        """Tests the output for an empty cccov_dict."""
        config = {"responses": {"cont": {"eff_fac": 0.5}}}
        cccov_dict = {}
        result = oac.calc_cccov_tot(config, cccov_dict)
        assert not result, "Result should be an empty dictionary for an " \
            "empty cccov_dict."


class TestCalcContRF:
    """Tests function calc_cont_RF(config, cccov_tot_dict, inv_dict)"""

    @pytest.fixture(scope="class")
    def inv_dict(self):
        """Fixture to create an example inv_dict."""
        return {2020: create_test_inv(year=2020),
                2050: create_test_inv(year=2050)}

    def test_output_structure(self, inv_dict):
        """Tests the output structure."""
        config = {"responses": {"cont": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        len_lat = len(oac.cc_lat_vals)
        years = list(inv_dict.keys())
        cccov_tot_dict = {years[0]: np.random.rand(len_lat),
                          years[1]: np.random.rand(len_lat)}
        result = oac.calc_cont_rf(config, cccov_tot_dict, inv_dict)

        # run assertions
        assert isinstance(result, dict), "Output should be a dictionary"
        assert "cont" in result, "Output does not include 'cont'."
        assert len(result["cont"]) == 31, "Output length does not match the " \
            " number of years in inv_dict."

    def test_incorrect_keys(self, inv_dict):
        """Tests differing keys in inv_dict and cccov_tot_dict."""
        config = {"responses": {"cont": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        len_lat = len(oac.cc_lat_vals)
        cccov_tot_dict = {2021: np.random.rand(len_lat),
                          2049: np.random.rand(len_lat)}
        with pytest.raises(AssertionError):
            oac.calc_cont_rf(config, cccov_tot_dict, inv_dict)

    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}},
                                        {"time": {}}])
    def test_missing_config_values(self, config, inv_dict):
        """Tests missing config values."""
        len_lat = len(oac.cc_lat_vals)
        years = list(inv_dict.keys())
        cccov_tot_dict = {years[0]: np.random.rand(len_lat),
                          years[1]: np.random.rand(len_lat)}
        with pytest.raises(AssertionError):
            oac.calc_cont_rf(config, cccov_tot_dict, inv_dict)

    def test_empty_input_dicts(self):
        """Tests empty input dicts."""
        config = {"responses": {"cont": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        with pytest.raises(AssertionError):
            oac.calc_cont_rf(config, {}, {})


class TestAddInvToBase:
    """Tests function add_inv_to_base(inv_dict, base_inv_dict)"""

    def test_key_mismatch(self):
        """Tests mismatched keys in input dictionaries."""
        inv_dict = {2020: np.array([1.0, 2.0])}
        base_inv_dict = {2050: np.array([1.0, 2.0])}
        with pytest.raises(AssertionError):
            oac.add_inv_to_base(inv_dict, base_inv_dict)

    def test_addition(self):
        """Tests function with simple inputs."""
        inv_dict = {2020: np.array([1.0, 2.0])}
        base_inv_dict = {2020: np.array([2.0, 3.0])}
        expected = {2020: np.array([3.0, 5.0])}
        result = oac.add_inv_to_base(inv_dict, base_inv_dict)
        np.testing.assert_array_equal(result[2020], expected[2020],
                                      err_msg="Addition fails for simple input.")

    def test_empty_inputs(self):
        """Tests empty input dictionaries."""
        inv_dict = {}; base_inv_dict = {}
        result = oac.add_inv_to_base(inv_dict, base_inv_dict)
        assert not result, "Expected empty result for empty input dictionaries."
