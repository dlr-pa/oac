"""Provides tests for module openairclim.gui.components.utils"""

import numpy as np
import xarray as xr
import pytest

from openairclim.gui.components import utils


class TestAutoScale:
    """Tests function auto_scale(max_val)"""

    @pytest.mark.parametrize("max_val", [0, np.inf, np.nan, 0.1, 999])
    def test_unscaled_cases(self, max_val):
        """Tests cases where the output should not be scaled."""
        assert utils.auto_scale(max_val) == (1.0, "")

    def test_large_value_scaled(self):
        """Tests that a large value is scaled."""
        divisor, _ = utils.auto_scale(3e8)
        assert divisor == 1e8

    def test_small_value_scaled(self):
        """Tests that a small value is scaled"""
        divisor, _ = utils.auto_scale(0.05)
        assert divisor == 1e-2


class TestGetNumericVars:
    """Tests function get_numeric_vars(ds)"""

    def test_returns_float_vars_excluding_skip_list(self):
        """Tests that variables are skipped and only numeric values are kept."""
        ds = xr.Dataset(
            {
                "emis_CO2": ("x", np.array([1.0, 2.0])),
                "ac": ("x", np.array([1.0, 2.0])),
                "plev": ("x", np.array([1.0, 2.0])),
                "label": ("x", np.array(["a", "b"])),
                "count": ("x", np.array([1, 2])),
            }
        )
        result = utils.get_numeric_vars(ds)
        assert result == ["emis_CO2", "count"]

    def test_empty_dataset_returns_empty_list(self):
        """Tests that a dataset with no data variables returns an empty list."""
        assert utils.get_numeric_vars(xr.Dataset()) == []


class TestLoadInventory:
    """Tests function load_inventory(working_dir, inv_dir, inv_file)"""

    def test_promotes_spatial_fields_to_coords(self, tmp_path):
        """Tests that plev/lat/lon data variables become coordinates."""
        (tmp_path / "inv").mkdir()
        ds = xr.Dataset(
            {
                "plev": ("index", np.array([250.0, 500.0])),
                "lat": ("index", np.array([10.0, 20.0])),
                "lon": ("index", np.array([30.0, 40.0])),
                "fuel": ("index", np.array([1.0, 2.0])),
            }
        )
        ds.to_netcdf(tmp_path / "inv" / "test.nc")

        loaded = utils.load_inventory(str(tmp_path), "inv", "test.nc")

        assert "plev" in loaded.coords
        assert "lat" in loaded.coords
        assert "lon" in loaded.coords
        assert "fuel" in loaded.data_vars
