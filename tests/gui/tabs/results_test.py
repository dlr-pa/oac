"""Provides tests for module openairclim.gui.tabs.results"""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

from pathlib import Path

import numpy as np
import xarray as xr
import pytest

from openairclim.gui.state import AppState
from openairclim.gui.tabs import results


class TestGetTimeCoord:
    """Tests function _get_time_coord(ds)"""

    @pytest.mark.parametrize("name", ["time", "year", "custom"])
    def test_finds_time_coord(self, name):
        """Tests that time coordinates can be identified."""
        ds = xr.Dataset(coords={name: np.arange(5)})
        assert results._get_time_coord(ds) == name

    def test_no_numeric_coord_returns_none(self):
        """Tests that a lack of numeric coordinates returns `None`."""
        ds = xr.Dataset(coords={"label": ("x", ["a", "b"])})
        assert results._get_time_coord(ds) is None


class TestCategoriseVariables:
    """Tests function _categorise_variables(ds)"""

    def test_groups_by_known_prefix(self):
        """Tests correct functioning."""
        ds = xr.Dataset(
            {
                "dT_CO2": ("t", [1.0]),
                "RF_CH4": ("t", [1.0]),
                "AGWP_CO2": ("t", [1.0]),
                "conc_CO2": ("t", [1.0]),
                "unknown_var": ("t", [1.0]),
            }
        )
        categories = results._categorise_variables(ds)
        assert categories["Temperature response"] == ["dT_CO2"]
        assert categories["Radiative forcing"] == ["RF_CH4"]
        assert categories["AGWP"] == ["AGWP_CO2"]
        assert categories["Concentration"] == ["conc_CO2"]
        assert categories["Other"] == ["unknown_var"]

    def test_empty_categories_dropped(self):
        """Tests that empty categories are dropped."""
        ds = xr.Dataset({"unknown_var": ("t", [1.0])})
        categories = results._categorise_variables(ds)
        assert list(categories.keys()) == ["Other"]


class TestCandidateResultsPath:
    """Tests function _candidate_results_path(state)"""

    def test_no_config_returns_none(self):
        """Tests that no config returns `None`."""
        state = AppState()
        assert results._candidate_results_path(state) is None

    def test_incomplete_output_section_returns_none(self):
        """Tests an incomplete config output section."""
        state = AppState()
        state.edited_config = {"output": {"dir": "", "name": ""}}
        assert results._candidate_results_path(state) is None

    def test_resolved_against_working_dir(self, tmp_path):
        """Tests that the working directory and file can be resolved."""
        state = AppState()
        state.working_dir = str(tmp_path)
        state.edited_config = {"output": {"dir": "out", "name": "example"}}
        result = results._candidate_results_path(state)
        assert result == (tmp_path / "out" / "example.nc").resolve()

    def test_without_working_dir_uses_dir_as_is(self):
        """Tests a relative directory."""
        state = AppState()
        state.edited_config = {"output": {"dir": "/abs/out", "name": "example"}}
        result = results._candidate_results_path(state)
        assert result == Path("/abs/out/example.nc")


class TestLoadResults:
    """Tests function _load_results(filepath)"""

    def test_promotes_ac_data_var_to_coord(self, tmp_path):
        """Tests that `ac` is promoted to a coordinate."""
        ds = xr.Dataset(
            {
                "ac": ("ac_dim", ["AC1", "AC2"]),
                "RF_CO2": (("ac_dim", "time"), [[1.0, 2.0], [3.0, 4.0]]),
            }
        )
        filepath = tmp_path / "results.nc"
        ds.to_netcdf(filepath)
        loaded = results._load_results(filepath)
        assert "ac" in loaded.coords
