"""
Provides tests for module calc_cont
"""

import numpy as np
import pytest
import openairclim as oac 
from contextlib import chdir
import xarray as xr


def create_synthetic_inv_dict(n_keys, n_values, seed=None):
    """Creates synthetic inventory dictionary of datasets for testing.
    
    Args:
        n_keys (int): number of keys in the dictionary
        n_values (int): number of values in each dataset
        seed (int, optional): set random seed, defaults to "None"
    
    Returns:
        dict: synthetic inventory dictionary, keys are random years between
            2020 and 2050
    """
    
    # set random seed
    np.random.seed(seed)
    
    # create random values
    keys = sorted(np.random.choice(range(2020, 2051), n_keys, replace=False))
    lat_values = np.random.uniform(-90, 90, n_values)
    lon_values = np.random.uniform(0, 360, n_values)
    plev_values = np.random.uniform(10, 1014, n_values)
    distance_values = np.random.uniform(1e3, 1e6, n_values)

    # create dictionary
    inv_dict = {}
    for key in keys:
        ds_key = xr.Dataset(
            {
                "lat": ("index", lat_values),
                "lon": ("index", lon_values),
                "plev": ("index", plev_values),
                "distance": ("index", distance_values),
            },
            coords={"index": np.arange(n_values)},
        )
        inv_dict[key] = ds_key
    
    return inv_dict


class TestInputContrailGrid:
    """Tests the input contrail grid, defined by `cc_lat_vals`, `cc_lon_vals`
    and `cc_plev_vals`."""
    
    def test_lon_vals(self):
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
        lat = oac.cc_lat_vals
        assert len(lat) > 0, "Contrail grid latitudes cannot be empty."
        assert len(lat) == len(np.unique(lat)), "Duplicate latitude values " \
            "in contrail grid."
        assert np.all((lat >= -90.) & (lat <= 90)),  "Latitude values must " \
            "be between, but not equal to, -90 and +90 degrees."
        assert np.all(lat == np.sort(lat)[::-1]), "Contrail grid latitudes " \
            "must be sorted in descending order."
    
    def test_plev_vals(self):
        plev = oac.cc_plev_vals
        assert len(plev) > 0, "Contrail grid pressure levels cannot be empty."
        assert len(plev) == len(np.unique(plev)), "Duplicate pressure level " \
            "values in contrail grid."
        assert np.all(plev == np.sort(plev)[::-1]), "Contrail grid pressure " \
            "level values must be sorted in descending order."
        assert np.all(plev <= 1014.), "Contrail grid pressure levels must " \
            "be at altitudes above ground level - defined as 1014 hPa."


class TestContrailInputValues:
    """Tests the input values for the contrail calculations stored as
    `resp_cont.nc`."""
    
    @pytest.fixture(scope="class")
    def ds_resp_cont(self):
        """Fixture to load the `resp_cont.nc` file."""
        with chdir("example/"):
            return xr.open_dataset("repository/resp_cont.nc")
    
    def test_required_variables(self, ds_resp_cont):
        """Test that `resp_cont` includes data variables ISS, SAC_CON and
        SAC_LH2."""
        required_vars = ["ISS", "SAC_CON", "SAC_LH2"]
        for var in required_vars:
            assert var in ds_resp_cont, f"Missing required variable '{var}' " \
                "in resp_cont."
    
    def test_required_coords(self, ds_resp_cont):
        """Test required coordinates and their respective units."""
        required_coords = ["lat", "lon", "plev"]
        required_units = ["deg", "deg", "hPa"]
        for coord, unit in zip(required_coords, required_units):
            assert coord in ds_resp_cont, f"Missing required coordinate " \
                "'{coord}' in resp_cont."
            assert ds_resp_cont[coord].attrs.get("units") == unit, f"Incorrect " \
                "unit for coordinate '{coord}'. Should be '{unit}'."


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
    
    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}}])
    def test_missing_config_values(self, config):
        """Tests missing config values."""
        with chdir("example/"): # TODO ds_comp is currently saved in example/directory
            inv_dict = create_synthetic_inv_dict(2, 100)
            with pytest.raises(AssertionError):
                oac.calc_cfdd(config, inv_dict)
    
    def test_invalid_Gcomp(self):
        """Tests an invalid G_comp value."""
        with chdir("example/"):  # TODO ds_comp is currently saved in example/directory
            # test upper bound
            config = {"responses": {"cont": {"G_comp": 0.2}}}
            inv_dict = {} 
            with pytest.raises(AssertionError):
                oac.calc_cfdd(config, inv_dict)
            # test lower bound
            config["responses"]["cont"]["G_comp"] = 0.02
            with pytest.raises(AssertionError):
                oac.calc_cfdd(config, inv_dict)
    
    def test_output_structure(self):
        """Tests the output structure."""
        with chdir("example/"):  # TODO ds_comp is currently saved in example/directory
            config = {"responses": {"cont": {"G_comp": 0.1}}}
            inv_dict = create_synthetic_inv_dict(2, 100)
            result = oac.calc_cfdd(config, inv_dict)
            
            # run tests
            assert isinstance(result, dict), "Output is not a dictionary."
            assert set(result.keys()) == set(inv_dict.keys()), "Output keys " \
                "do not match input keys."
            for year, cfdd in result.items():
                assert isinstance(cfdd, np.ndarray), "CFDD is not an array."
                assert cfdd.shape == (len(oac.cc_lat_vals), len(oac.cc_lon_vals)), \
                    "CFDD array has incorrect shape."
    
    def test_empty_inventory(self):
        """Tests the handling of an empty input inventory."""
        with chdir("example/"):  # TODO ds_comp is currently saved in example/directory
            config = {"responses": {"cont": {"G_comp": 0.1}}}
            inv_dict = {}  # empty inventory
            result = oac.calc_cfdd(config, inv_dict)
            assert result == {}, "Result should be an empty dictionary for an " \
                "empty inventory."
            

class TestCalcCccov:
    """Tests function calc_cccov(config, cfdd_dict)"""
    
    def test_output_structure(self):
        """Tests the output structure."""
        with chdir("example/"):  # TODO ds_comp is currently saved in example/directory
            config = {"responses": {"cont": {"eff_fac": 0.5}}}
            len_lon = len(oac.cc_lon_vals)
            len_lat = len(oac.cc_lat_vals)
            cfdd_dict = {2020: np.random.rand(len_lat, len_lon),
                         2050: np.random.rand(len_lat, len_lon)}
            result = oac.calc_cccov(config, cfdd_dict)
            
            # run assertions
            assert isinstance(result, dict), "Output is not a dictionary."
            assert set(result.keys()) == set(cfdd_dict.keys()), "Output keys " \
                "do not match input keys."
            for year, cccov in result.items():
                assert isinstance(cccov, np.ndarray), "cccov is not an array."
                assert cccov.shape == (len_lat, len_lon), "cccov array has " \
                    "incorrect shape"

    def test_incorrect_cfdd_shape(self):
        """Tests incorrect shape of each cfdd array within cfdd_dict."""
        with chdir("example/"):  # TODO ds_comp is currently saved in example/directory
            config = {"responses": {"cont": {"eff_fac": 0.5}}}
            cfdd_dict = {2020: np.random.rand(10, 10),
                         2050: np.random.rand(10, 10)}
            with pytest.raises(AssertionError):
                oac.calc_cccov(config, cfdd_dict)
    
    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}}])
    def test_missing_config_values(self, config):
        """Tests missing config values."""
        with chdir("example/"):  # TODO ds_comp is currently saved in example/directory
            len_lon = len(oac.cc_lon_vals)
            len_lat = len(oac.cc_lat_vals)
            cfdd_dict = {2020: np.random.rand(len_lat, len_lon),
                         2050: np.random.rand(len_lat, len_lon)}
            with pytest.raises(AssertionError):
                oac.calc_cccov(config, cfdd_dict)
    
    def test_empty_cfdd_dict(self):
        """Tests the output for an empty cfdd_dict."""
        with chdir("example/"):  # TODO ds_comp is currently saved in example/directory
            config = {"responses": {"cont": {"eff_fac": 0.5}}}
            cfdd_dict = {}
            result = oac.calc_cccov(config, cfdd_dict)
            assert result == {}, "Result should be an empty dictionary for an " \
                "empty cfdd_dict."
    

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
        for year, cccov_tot in result.items():
            assert isinstance(cccov_tot, np.ndarray), "cccov_tot is not an array."
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
        assert result == {}, "Result should be an empty dictionary for an " \
            "empty cccov_dict."


class TestCalcContRF:
    """Tests function calc_cont_RF(config, cccov_tot_dict, inv_dict)"""
    
    def test_output_structure(self):
        """Tests the output structure."""
        config = {"responses": {"cont": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        len_lat = len(oac.cc_lat_vals)
        inv_dict = create_synthetic_inv_dict(2, 100)
        years = list(inv_dict.keys())
        cccov_tot_dict = {years[0]: np.random.rand(len_lat),
                          years[1]: np.random.rand(len_lat)}
        result = oac.calc_cont_RF(config, cccov_tot_dict, inv_dict)
        
        # run assertions
        assert isinstance(result, dict), "Output should be a dictionary"
        assert "cont" in result, "Output does not include 'cont'."
        assert len(result["cont"]) == 31, "Output length does not match the " \
            " number of years in inv_dict."
    
    def test_incorrect_keys(self):
        """Tests differing keys in inv_dict and cccov_tot_dict."""
        config = {"responses": {"cont": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        len_lat = len(oac.cc_lat_vals)
        inv_dict = create_synthetic_inv_dict(2, 100, 42)  # keys: 2035, 2047
        cccov_tot_dict = {2020: np.random.rand(len_lat),
                          2050: np.random.rand(len_lat)}
        with pytest.raises(AssertionError):
            oac.calc_cont_RF(config, cccov_tot_dict, inv_dict)

    @pytest.mark.parametrize("config", [{},
                                        {"responses": {}},
                                        {"responses": {"cont": {}}},
                                        {"time": {}}])
    def test_missing_config_values(self, config):
        """Tests missing config values."""
        len_lat = len(oac.cc_lat_vals)
        inv_dict = create_synthetic_inv_dict(2, 100)
        years = list(inv_dict.keys())
        cccov_tot_dict = {years[0]: np.random.rand(len_lat),
                          years[1]: np.random.rand(len_lat)}
        with pytest.raises(AssertionError):
            oac.calc_cont_RF(config, cccov_tot_dict, inv_dict)

    def test_empty_input_dicts(self):
        """Tests empty input dicts."""
        config = {"responses": {"cont": {"PMrel": 1.0}},
                  "time": {"range": [2020, 2051, 1]}}
        with pytest.raises(AssertionError):
            oac.calc_cont_RF(config, {}, {})
        