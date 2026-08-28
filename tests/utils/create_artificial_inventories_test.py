"""
Provides tests for module openairclim.utils.create_artificial_inventories
"""

import pytest
import xarray as xr

from openairclim.utils import create_artificial_inventories as cai


class TestArtificialInventory:
    """Tests class ArtificialInventory"""

    def test_create_uniform_dist_columns_and_size(self):
        """A uniform-distribution inventory has the expected columns and
        row count, with no aircraft column when ac_lst isn't given."""
        inv = cai.ArtificialInventory(year=2020, size=50)
        inv.create_uniform_dist()
        expected_cols = {"lon", "lat", "plev", "fuel", "CO2", "H2O", "NOx", "distance"}
        assert expected_cols.issubset(set(inv.df.columns))
        assert "ac" not in inv.df.columns
        assert len(inv.df) == 50

    def test_create_uniform_dist_with_aircraft_list(self):
        """An aircraft column is added, drawn only from ac_lst, when given."""
        ac_lst = ["RJ", "NB", "WB"]
        inv = cai.ArtificialInventory(year=2020, size=30, ac_lst=ac_lst)
        inv.create_uniform_dist()
        assert "ac" in inv.df.columns
        assert set(inv.df["ac"].unique()).issubset(set(ac_lst))

    def test_values_within_configured_ranges(self):
        """Sampled lon/lat/plev stay within the configured ranges."""
        lon_range = [10.0, 20.0]
        lat_range = [-5.0, 5.0]
        plev_range = [300.0, 400.0]
        inv = cai.ArtificialInventory(
            year=2020,
            size=100,
            lon_range=lon_range,
            lat_range=lat_range,
            plev_range=plev_range,
        )
        inv.create_uniform_dist()
        assert inv.df["lon"].between(*lon_range).all()
        assert inv.df["lat"].between(*lat_range).all()
        assert inv.df["plev"].between(*plev_range).all()

    def test_convert_df_to_xr_sets_attrs(self):
        """The converted xr.Dataset carries the inventory year and
        coordinate attributes."""
        inv = cai.ArtificialInventory(year=2035, size=10)
        ds = inv.create_uniform_dist().convert_df_to_xr().inv
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["Inventory_Year"] == 2035
        assert ds.lon.attrs["units"] == "degrees_east"
        assert ds.lat.attrs["units"] == "degrees_north"

    def test_create_uniform_returns_dataset(self):
        """create(distribution='uniform') returns an xr.Dataset."""
        inv = cai.ArtificialInventory(year=2020, size=5)
        result = inv.create()
        assert isinstance(result, xr.Dataset)

    def test_create_invalid_distribution_raises(self):
        """An unsupported distribution argument raises ValueError."""
        inv = cai.ArtificialInventory(year=2020, size=5)
        with pytest.raises(ValueError):
            inv.create(distribution="bogus")


class TestArtificialInventoryDict:
    """Tests class ArtificialInventoryDict"""

    def test_create_linear_increase_keys_match_years(self):
        """The resulting dict has exactly one entry per year in year_arr."""
        year_arr = [2020, 2030, 2040]
        inv_dict = cai.ArtificialInventoryDict(year_arr=year_arr).create()
        assert set(inv_dict.keys()) == set(year_arr)
        assert all(isinstance(v, xr.Dataset) for v in inv_dict.values())

    def test_create_invalid_evolution_raises(self):
        """An unsupported evolution argument raises ValueError."""
        with pytest.raises(ValueError):
            cai.ArtificialInventoryDict(year_arr=[2020]).create(evolution="bogus")


class TestConvertXrDictToNc:
    """Tests function convert_xr_dict_to_nc(inv_dict, prefix, out_path)"""

    def test_writes_one_file_per_year(self, tmp_path):
        """Each year in inv_dict is written to its own prefixed netCDF
        file in out_path."""
        inv_dict = cai.ArtificialInventoryDict(year_arr=[2020, 2021]).create()
        cai.convert_xr_dict_to_nc(inv_dict, prefix="test_inv", out_path=str(tmp_path))
        assert (tmp_path / "test_inv_2020.nc").is_file()
        assert (tmp_path / "test_inv_2021.nc").is_file()

    def test_creates_output_dir_if_missing(self, tmp_path):
        """out_path is created if it doesn't already exist."""
        out_path = tmp_path / "nested" / "dir"
        inv_dict = cai.ArtificialInventoryDict(year_arr=[2020]).create()
        cai.convert_xr_dict_to_nc(inv_dict, out_path=str(out_path))
        assert out_path.is_dir()
        assert (out_path / "rnd_inv_2020.nc").is_file()
