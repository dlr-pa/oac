"""
Provides tests for module write_output
"""

import numpy as np
import xarray as xr
import pytest
import openairclim as oac

# CONSTANTS
REPO_PATH = "repository/"
CACHE_PATH = "repository/cache/weights/"
INV_NAME = "test_inv.nc"
RESP_NAME = "test_resp.nc"
CACHE_NAME = "000.nc"


class TestWriteOutputDictToNetcdf:
    """Tests function write_output_dict_to_netcdf(config, output_dict)."""

    @pytest.fixture
    def mock_save(self, monkeypatch):
        """Prevents .to_netcdf() from writing to file."""
        monkeypatch.setattr(
            xr.Dataset, "to_netcdf",
            lambda self, *args, **kwargs: None
        )

    @pytest.fixture
    def config(self, tmp_path):
        """Fixture to create a valid config."""
        return {
            "output": {
                "dir": str(tmp_path) + "/",
                "name": "test_output"
            },
            "time": {"range": [2000, 2020, 1]},
            "aircraft": {"types": ["LR", "REG"]},
        }

    @pytest.fixture
    def output_dict(self):
        """Fixture to create a valid output_dict."""
        time_len = 20
        return {
            "LR": {
                "RF_CO2": np.full(time_len, 1),
                "RF_CH4": np.full(time_len, 2),
            },
            "REG": {
                "RF_CO2": np.full(time_len, 3),
                "RF_CH4": np.full(time_len, 4),
            }
        }

    @pytest.mark.usefixtures("mock_save")
    def test_valid_write(self, config, output_dict):
        """Tests valid config and dictionary."""
        ds = oac.write_output_dict_to_netcdf(config, output_dict)
        assert isinstance(ds, xr.Dataset)
        assert "RF_CO2" in ds
        assert "RF_CH4" in ds
        assert "ac" in ds.dims
        assert "time" in ds.dims
        assert ds.dims["ac"] == 2
        assert ds.dims["time"] == 20
