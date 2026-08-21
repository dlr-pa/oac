"""Provides tests for module openairclim.gui.tabs.config"""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

from copy import deepcopy

import pytest

from openairclim.gui import config_io
from openairclim.gui.tabs import config


class TestGetPath:
    """Tests function _get_path(edited, path)"""

    def test_resolves_dotted_path(self):
        """Tests that a dotted path resolves correctly."""
        edited = {"a": {"b": {"c": 42}}}
        assert config._get_path(edited, "a.b.c") == 42

    def test_missing_intermediate_key_returns_none(self):
        """Tests that a missing intermediate key returns `None`."""
        edited = {"a": {}}
        assert config._get_path(edited, "a.b.c") is None

    def test_non_dict_intermediate_returns_none(self):
        """Tests that an intermediate non-dict key returns `None`."""
        edited = {"a": "not a dict"}
        assert config._get_path(edited, "a.b") is None


class TestRequiredFieldsStatus:
    """Tests function _required_fields_status(edited, paths)"""

    def test_all_present(self):
        """Tests valid input."""
        edited = {"a": 1, "b": 2}
        assert config._required_fields_status(edited, ["a", "b"]) is None

    def test_one_blank_flags_warning(self):
        """Tests that a blank value triggers a warning."""
        edited = {"a": 1, "b": ""}
        assert config._required_fields_status(edited, ["a", "b"]) == "⚠️"

    def test_no_paths_never_warns(self):
        """Tests an empty dictionary."""
        assert config._required_fields_status({}, []) is None


class TestMissingLabel:
    """Tests functions _missing_label(filename) and
    _strip_missing_label(value).
    """

    @pytest.mark.parametrize(
        "value", [config._missing_label("file.nc"), "file.nc"],
        ids=["decorated", "plain"]
    )
    def test_strip_returns_real_filename(self, value):
        """Tests both the strip and pass-through branches of the function."""
        assert config._strip_missing_label(value) == "file.nc"


class TestCheckTime:
    """Tests function _check_time(edited)"""

    def test_end_before_start_flagged(self):
        """Tests that matching starting and ending dates raise a flag."""
        edited = {"time": {"range": [2020, 2020, 1]}}
        assert config._check_time(edited) == "⚠️"

    def test_valid_range_ok(self):
        """Tests a valid time range."""
        edited = {"time": {"range": [2020, 2030, 1]}}
        assert config._check_time(edited) is None


class TestCheckInventories:
    """Tests function _check_inventories(edited)"""

    def test_missing_dir_flagged(self):
        """Tests missing directory."""
        edited = {
            "inventories": {
                "dir": "", "files": [], "rel_to_base": False, "base": {}
            }
        }
        assert config._check_inventories(edited) == "⚠️"

    def test_missing_file_on_disk_flagged(self, tmp_path):
        """Tests a missing file."""
        edited = {
            "inventories": {
                "dir": str(tmp_path),
                "files": ["missing.nc"],
                "rel_to_base": False,
                "base": {},
            }
        }
        assert config._check_inventories(edited) == "⚠️"

    def test_valid_without_base(self, tmp_path):
        """Tests a valid configuration (without base inventories)."""
        (tmp_path / "a.nc").touch()
        edited = {
            "inventories": {
                "dir": str(tmp_path),
                "files": ["a.nc"],
                "rel_to_base": False,
                "base": {"dir": "", "files": []},
            }
        }
        assert config._check_inventories(edited) is None

    def test_rel_to_base_requires_base_fields(self, tmp_path):
        """Tests missing base files if `rel_to_base == True`."""
        (tmp_path / "a.nc").touch()
        edited = {
            "inventories": {
                "dir": str(tmp_path),
                "files": ["a.nc"],
                "rel_to_base": True,
                "base": {"dir": "", "files": []},
            }
        }
        assert config._check_inventories(edited) == "⚠️"

    def test_rel_to_base_with_missing_base_file_flagged(self, tmp_path):
        """Tests missing base emission inventory."""
        (tmp_path / "a.nc").touch()
        edited = {
            "inventories": {
                "dir": str(tmp_path),
                "files": ["a.nc"],
                "rel_to_base": True,
                "base": {"dir": str(tmp_path), "files": ["missing.nc"]},
            }
        }
        assert config._check_inventories(edited) == "⚠️"


class TestCheckBackground:
    """Tests function _check_background(edited)"""

    def test_missing_dir_flagged(self):
        """Tests missing directory."""
        edited = {"background": {"dir": ""}}
        assert config._check_background(edited) == "⚠️"

    def test_missing_species_file_flagged(self):
        """Tests missing file."""
        edited = {
            "background": {
                "dir": "/x",
                "CO2": {"file": "", "scenario": "SSP2-4.5"},
                "CH4": {"file": "f.nc", "scenario": "SSP2-4.5"},
                "N2O": {"file": "f.nc", "scenario": "SSP2-4.5"},
            }
        }
        assert config._check_background(edited) == "⚠️"

    def test_all_filled_ok(self):
        """Tests valid background section."""
        edited = {
            "background": {
                "dir": "/x",
                "CO2": {"file": "f.nc", "scenario": "SSP2-4.5"},
                "CH4": {"file": "f.nc", "scenario": "SSP2-4.5"},
                "N2O": {"file": "f.nc", "scenario": "SSP2-4.5"},
            }
        }
        assert config._check_background(edited) is None


class TestCheckResponses:
    """Tests function _check_responses(edited)"""

    def test_missing_dir_flagged(self):
        """Tests missing directory."""
        edited = {"responses": {"dir": ""}}
        assert config._check_responses(edited) == "⚠️"

    def test_missing_file_flagged(self):
        """Tests missing file."""
        edited = {
            "responses": {
                "dir": "/x",
                "H2O": {"rf": {"file": ""}},
                "O3": {"rf": {"file": "f.nc"}},
                "CH4": {"tau": {"file": "f.nc"}},
                "cont": {"resp": {"file": "f.nc"}},
            }
        }
        assert config._check_responses(edited) == "⚠️"

    def test_all_filled_ok(self):
        """Tests valid response section."""
        edited = {
            "responses": {
                "dir": "/x",
                "H2O": {"rf": {"file": "f.nc"}},
                "O3": {"rf": {"file": "f.nc"}},
                "CH4": {"tau": {"file": "f.nc"}},
                "cont": {"resp": {"file": "f.nc"}},
            }
        }
        assert config._check_responses(edited) is None


class TestCheckMetrics:
    """Tests function _check_metrics(edited)"""

    def test_run_metrics_off_and_unset_ok(self):
        """Tests that an empty input is fine when `run_metrics == False`."""
        edited = {"metrics": {}, "output": {"run_metrics": False}}
        assert config._check_metrics(edited) is None

    @pytest.mark.parametrize("run_metrics", [True, False])
    def test_incomplete_metrics_setup(self, run_metrics):
        """Tests incomplete metrics setup, regardless of whether metrics are
        to be calculated or not."""
        edited = {
            "metrics": {"types": ["ATR"]},
            "output": {"run_metrics": run_metrics},
        }
        assert config._check_metrics(edited) == "⚠️"

    def test_run_metrics_on_and_complete_ok(self):
        """Tests valid configuration."""
        edited = {
            "metrics": {"types": ["ATR"], "H": [100], "t_0": [2020]},
            "output": {"run_metrics": True},
        }
        assert config._check_metrics(edited) is None


class TestCheckRequiredFields:
    """Tests function check_required_fields(edited_config)"""

    def test_blank_config_reports_multiple_problems(self):
        """Tests that a blank config reports multiple problems to the user."""
        blank = config_io.blank_config()
        problems = config.check_required_fields(blank)
        titles = [title for title, _status in problems]
        assert "Simulation period" in titles
        assert "Species" in titles

    def test_order_follows_card_checks(self):
        """Tests the order of the cards."""
        blank = config_io.blank_config()
        problems = config.check_required_fields(blank)
        titles = [title for title, _status in problems]
        expected_order = [t for t in config.CARD_CHECKS if t in titles]
        assert titles == expected_order

    def test_complete_config_has_no_problems(self, tmp_path):
        """Tests valid config."""
        blank = config_io.blank_config()
        edited = deepcopy(blank)
        edited["species"] = {"inv": ["CO2"], "out": ["CO2"], "nox": "NO"}
        edited["time"]["range"] = [2020, 2030, 1]
        (tmp_path / "inv.nc").touch()
        edited["inventories"].update(
            {"dir": str(tmp_path), "files": ["inv.nc"], "rel_to_base": False}
        )
        edited["output"].update({"dir": str(tmp_path), "name": "out"})
        edited["background"]["dir"] = str(tmp_path)
        for species in ("CO2", "CH4", "N2O"):
            edited["background"][species] = {"file": "bg.nc", "scenario": "SSP2-4.5"}
        edited["responses"]["dir"] = str(tmp_path)
        edited["responses"]["H2O"]["rf"]["file"] = "r.nc"
        edited["responses"]["O3"]["rf"]["file"] = "r.nc"
        edited["responses"]["CH4"]["tau"]["file"] = "r.nc"
        edited["responses"]["cont"]["resp"]["file"] = "r.nc"

        assert not config.check_required_fields(edited)
