"""
Provides tests for module openairclim.utils.create_test_data
"""

import xarray as xr

from openairclim.utils import create_test_data as ctd


class TestCreateTestConcResp:
    """Tests function create_test_conc_resp()"""

    def test_builds_expected_dataset(self):
        """The dataset has the expected data variables/coords and
        resp_type attribute for a 'conc' response file."""
        resp = ctd.create_test_conc_resp()
        assert isinstance(resp, xr.Dataset)
        assert resp.attrs["resp_type"] == "conc"
        assert {"p10_250", "p40_250", "p10_500", "p40_500"}.issubset(resp.data_vars)
        assert "lat" in resp.coords and "plev" in resp.coords


class TestCreateTestRfResp:
    """Tests function create_test_rf_resp()"""

    def test_builds_expected_dataset(self):
        """The dataset has an H2O data variable and resp_type 'rf'."""
        resp = ctd.create_test_rf_resp()
        assert isinstance(resp, xr.Dataset)
        assert resp.attrs["resp_type"] == "rf"
        assert "H2O" in resp.data_vars


class TestCreateTestRespCont:
    """Tests function create_test_resp_cont(n_lat, n_lon, n_plev, seed)"""

    def test_default_shape_and_variables(self):
        """The dataset has the expected coordinate sizes and data
        variables for a contrail response file."""
        ds = ctd.create_test_resp_cont(n_lat=4, n_lon=6, n_plev=5, seed=0)
        assert ds.sizes["lat"] == 4
        assert ds.sizes["lon"] == 6
        assert ds.sizes["plev"] == 5
        assert "ppcf" in ds.data_vars
        assert "g_250" in ds.data_vars

    def test_seed_is_reproducible(self):
        """The same seed produces identical output."""
        ds1 = ctd.create_test_resp_cont(n_lat=4, n_lon=6, n_plev=5, seed=42)
        ds2 = ctd.create_test_resp_cont(n_lat=4, n_lon=6, n_plev=5, seed=42)
        xr.testing.assert_identical(ds1, ds2)
