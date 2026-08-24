"""Provides tests for module openairclim.gui.config_io"""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access


from copy import deepcopy
from pathlib import Path
import tomllib

import pytest
import xarray as xr

from openairclim.gui import config_io
from openairclim.gui.state import AppState

from ..conftest import REPO_DIR, INV_NAME


class TestStringifyPaths:
    """Tests function _stringify_paths(obj)"""

    def test_converts_nested_paths(self):
        """Tests that path values are converted to str, at any nesting depth."""
        obj = {"a": Path("/tmp/x"), "b": [Path("/tmp/y"), "already_str"]}
        result = config_io._stringify_paths(obj)
        assert result == {
            "a": str(Path("/tmp/x")),
            "b": [str(Path("/tmp/y")), "already_str"]
        }

    def test_dot_and_empty_path_become_empty_string(self):
        """Tests that Path(".")/Path("") collapse to "" rather than "."."""
        assert config_io._stringify_paths(Path(".")) == ""
        assert config_io._stringify_paths(Path("")) == ""

    def test_non_path_values_untouched(self):
        """Tests that non-Path, non-container values pass through unchanged."""
        assert config_io._stringify_paths(5) == 5
        assert config_io._stringify_paths(True) is True
        assert config_io._stringify_paths(None) is None


class TestBlankConfig:
    """Tests function blank_config()"""

    def test_returns_dict_with_unset_time_sentinel(self):
        """Tests that time.range keeps its "not set yet" sentinel
        (start == end), even though core's Config model requires end > start.
        See gui/config_io.py's `blank_config()` docstring and comments."""
        blank = config_io.blank_config()
        assert isinstance(blank, dict)
        assert blank["time"]["range"] == [0, 0, 1]

    def test_defaults_are_filled_in(self):
        """Tests that sections not given explicitly (responses, temperature,
        ...) are filled in by core's validate_config."""
        blank = config_io.blank_config()
        assert "responses" in blank
        assert "temperature" in blank
        assert "metrics" in blank
        assert "parametric" in blank

    def test_flagged_as_missing_by_config_tab(self):
        """Tests that the simulation period card (and others) show a warning
        for a blank config."""
        blank = config_io.blank_config()
        problems = dict(config_io.check_required_fields(blank))
        assert problems["Simulation period"] == "⚠️"


class TestParseAndCheckStructure:
    """Tests function parse_and_check_structure(working_dir, config_path)"""

    def test_valid_file(self, tmp_path, valid_config):
        """Tests that a structurally valid TOML file parses with no errors."""
        config_path = tmp_path / "config.toml"
        config_io.write_toml(valid_config, config_path)

        config, errors = config_io.parse_and_check_structure(
            str(tmp_path), str(config_path)
        )

        assert not errors
        assert isinstance(config, dict)
        assert config["species"]["inv"] == ["CO2"]

    def test_relative_path_resolved_against_working_dir(
            self, tmp_path, valid_config
        ):
        """Tests that a relative config_path is resolved against working_dir."""
        config_path = tmp_path / "config.toml"
        config_io.write_toml(valid_config, config_path)

        config, errors = config_io.parse_and_check_structure(
            str(tmp_path), "config.toml"
        )

        assert not errors
        assert config is not None

    def test_missing_file(self, tmp_path):
        """Tests tha a nonexistent file returns a "not found" error."""
        config, errors = config_io.parse_and_check_structure(
            str(tmp_path), str(tmp_path / "does_not_exist.toml")
        )

        assert config is None
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_malformed_toml(self, tmp_path):
        """Tests that an invalid TOML syntax returns a parse error."""
        config_path = tmp_path / "bad.toml"
        config_path.write_text("not = [valid = toml", encoding="utf-8")

        config, errors = config_io.parse_and_check_structure(
            str(tmp_path), str(config_path)
        )

        assert config is None
        assert len(errors) == 1
        assert "parse" in errors[0].lower()

    def test_structurally_invalid_config(self, tmp_path, valid_config):
        """Tests that a structurally invalid config (wrong type) returns a
        validation error."""
        bad_config = deepcopy(valid_config)
        bad_config["inventories"]["dir"] = 9  # must be a str/Path
        config_path = tmp_path / "config.toml"
        config_io.write_toml(bad_config, config_path)

        config, errors = config_io.parse_and_check_structure(
            str(tmp_path), str(config_path)
        )

        assert config is None
        assert len(errors) == 1
        assert "validation error" in errors[0].lower()


class TestParseTomlText:
    """Tests function parse_toml_text(text)"""

    def test_valid_text(self, valid_config):
        """Tests valid parse."""
        text = config_io.to_toml_string(valid_config)
        config, errors = config_io.parse_toml_text(text)
        assert not errors
        assert config["species"]["inv"] == ["CO2"]

    def test_malformed_text(self):
        """Tests that an invalid TOML syntax returns a parse error."""
        config, errors = config_io.parse_toml_text("not = [valid = toml")
        assert config is None
        assert "failed to parse toml" in errors[0].lower()

    def test_structurally_invalid_text(self):
        """Tests that a structurally invalid config text returns a validation
        error."""
        config, errors = config_io.parse_toml_text('[inventories]\ndir = 9\n')
        assert config is None
        assert "validation error" in errors[0].lower()


class TestCheckFullConfig:
    """Tests function check_full_config(working_dir, config)"""

    def test_valid_config_does_not_raise(self, working_dir, valid_config):
        """Tests that a fully valid config does not raise."""
        config_io.check_full_config(working_dir, valid_config)

    def test_input_config_not_mutated(self, working_dir, valid_config):
        """Tests that validation does not mutate the config stored in memory."""
        original = deepcopy(valid_config)
        config_io.check_full_config(working_dir, valid_config)
        assert valid_config == original

    def test_invalid_config_raises(self, working_dir, valid_config):
        """Tests an invalid config."""
        bad_config = deepcopy(valid_config)
        bad_config["inventories"]["files"] = ["not-a-real-file.nc"]
        with pytest.raises(Exception):
            config_io.check_full_config(working_dir, bad_config)

    def test_restores_cwd_on_success(self, working_dir, valid_config):
        """Tests that a temporary cwd is undone."""
        import os
        before = os.getcwd()
        config_io.check_full_config(working_dir, valid_config)
        assert os.getcwd() == before

    def test_restores_cwd_on_failure(self, working_dir, valid_config):
        """Tests that a temporary cwd is undone, even if an Exception is
        raised."""
        import os
        bad_config = deepcopy(valid_config)
        bad_config["inventories"]["files"] = ["not-a-real-file.nc"]
        before = os.getcwd()
        with pytest.raises(Exception):
            config_io.check_full_config(working_dir, bad_config)
        assert os.getcwd() == before


class TestRunFullValidation:
    """Tests function run_full_validation(state)"""

    def test_no_working_dir(self):
        """Tests missing working directory."""
        state = AppState()
        state.edited_config = {"species": {}}
        valid, message = config_io.run_full_validation(state)
        assert valid is False
        assert "working directory" in message.lower()

    def test_no_config(self):
        """Tests missing config."""
        state = AppState()
        state.working_dir = "."
        valid, message = config_io.run_full_validation(state)
        assert valid is False
        assert "no configuration" in message.lower()

    def test_dirty_aircraft_csv(self, blank_state):
        """Tests warning preventing validation is the aircraft csv is dirty."""
        blank_state.aircraft_csv_dirty = True
        valid, message = config_io.run_full_validation(blank_state)
        assert valid is False
        assert "aircraft csv" in message.lower()

    def test_missing_required_fields(self, blank_state):
        """Test that validation stops if required fields are missing, and that
        the check doesn't reach `check_full_config`, which would otherwise
        return errors."""
        valid, message = config_io.run_full_validation(blank_state)
        assert valid is False
        assert "Fields missing or invalid" in message
        assert "Simulation period" in message

    def test_passes_req_fields_runs_full_check(self, monkeypatch, tmp_path):
        """Tests validation of valid config."""
        config = config_io.blank_config()
        config["species"] = {"inv": ["CO2"], "out": ["CO2"]}
        config["time"]["range"] = [2020, 2030, 1]
        config["inventories"].update(
            {"dir": str(REPO_DIR), "files": [INV_NAME], "rel_to_base": False}
        )
        config["output"].update({"dir": str(tmp_path), "name": "out"})
        for species in ("CO2", "CH4", "N2O"):
            config["background"][species] = {
                "file": "bg.nc", "scenario": "SSP2-4.5"
            }
        config["background"]["dir"] = str(REPO_DIR)
        config["responses"]["dir"] = str(REPO_DIR)
        config["responses"]["H2O"]["rf"]["file"] = "resp.nc"
        config["responses"]["O3"]["rf"]["file"] = "resp.nc"
        config["responses"]["CH4"]["tau"]["file"] = "resp.nc"
        config["responses"]["cont"]["resp"]["file"] = "resp.nc"

        problems = config_io.check_required_fields(config)
        assert not problems

        state = AppState()
        state.working_dir = str(REPO_DIR)
        state.edited_config = config
        calls = []

        monkeypatch.setattr(
            config_io, "check_full_config",
            lambda wd, cfg: calls.append((wd, cfg))
        )
        valid, message = config_io.run_full_validation(state)
        assert calls == [(str(REPO_DIR), config)]
        assert valid is True
        assert "valid" in message.lower()

    def test_check_full_config_error_surfaced(self, monkeypatch, blank_state):
        """Test that errors are shown to the user."""
        monkeypatch.setattr(
            config_io, "check_required_fields", lambda edited: []
        )

        def _raise(*_args):
            raise ValueError("boom")

        monkeypatch.setattr(config_io, "check_full_config", _raise)
        valid, message = config_io.run_full_validation(blank_state)
        assert valid is False
        assert "boom" in message


class TestRunConfig:
    """Tests function run_config(working_dir, config_path)"""

    def test_chdirs_and_calls_core_run(self, monkeypatch, tmp_path):
        """Tests the temporarily switch into working_dir so relative
        paths inside the saved config file resolve correctly."""
        import os

        seen_cwd = {}

        def _fake_run(config_path):
            seen_cwd["cwd"] = os.getcwd()
            seen_cwd["config_path"] = config_path

        monkeypatch.setattr("openairclim.core.run", _fake_run)

        before = os.getcwd()
        config_io.run_config(str(tmp_path), "my_config.toml")

        assert seen_cwd["cwd"] == str(tmp_path.resolve())
        assert seen_cwd["config_path"] == "my_config.toml"
        assert os.getcwd() == before

    def test_restores_cwd_on_failure(self, monkeypatch, tmp_path):
        """Test that cwd is restored even after a run failure."""
        import os

        def _raise(_config_path):
            raise RuntimeError("run failed")

        monkeypatch.setattr("openairclim.core.run", _raise)

        before = os.getcwd()
        with pytest.raises(RuntimeError):
            config_io.run_config(str(tmp_path), "my_config.toml")
        assert os.getcwd() == before


class TestResolveDir:
    """Tests function resolve_dir(working_dir, dir_str)"""

    def test_absolute_path_unchanged(self, tmp_path):
        """Tests that an absolute path remains unchanged."""
        result = config_io.resolve_dir("/anything", str(tmp_path))
        assert result == tmp_path.resolve()

    def test_relative_path_joined_with_working_dir(self, tmp_path):
        """Tests that a relative path is combined with the working directory."""
        result = config_io.resolve_dir(str(tmp_path), "sub/dir")
        assert result == (tmp_path / "sub" / "dir").resolve()


class TestToRelative:
    """Tests function to_relative(working_dir, absolute_path)"""

    def test_inside_working_dir(self, tmp_path):
        """Tests a path to a file within the working directory."""
        target = tmp_path / "sub" / "file.nc"
        result = config_io.to_relative(str(tmp_path), str(target))
        assert result == "sub/file.nc"

    def test_outside_working_dir_returns_absolute_unchanged(self, tmp_path):
        """Tests a path to a folder outside the working directory."""
        outside = "/completely/unrelated/path"
        result = config_io.to_relative(str(tmp_path), outside)
        assert result == outside


class TestListNcFiles:
    """Tests function list_nc_files(directory_path)"""

    def test_lists_and_sorts_nc_files(self, tmp_path):
        """Tests that only nc files are returned."""
        (tmp_path / "b.nc").touch()
        (tmp_path / "a.nc").touch()
        (tmp_path / "ignore.toml").touch()

        result = config_io.list_nc_files(tmp_path)

        assert result == ["a.nc", "b.nc"]

    def test_missing_dir_returns_empty_list(self, tmp_path):
        """Tests missing directory."""
        result = config_io.list_nc_files(tmp_path / "does_not_exist")
        assert result == []


class TestListNcDataVars:
    """Tests function list_nc_data_vars(filepath)"""

    def test_lists_sorted_data_vars(self, tmp_path):
        """Tests that all data variables are identified."""
        ds = xr.Dataset({"b_var": ("x", [1, 2]), "a_var": ("x", [3, 4])})
        filepath = tmp_path / "data.nc"
        ds.to_netcdf(filepath)

        result = config_io.list_nc_data_vars(filepath)

        assert result == ["a_var", "b_var"]

    def test_missing_file_returns_empty_list(self, tmp_path):
        """Tests that a missing file returns an empty list."""
        result = config_io.list_nc_data_vars(tmp_path / "does_not_exist.nc")
        assert result == []


class TestFormatTomlValue:
    """Tests function _format_toml_value(value)"""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, "true"),
            (False, "false"),
            (5, "5"),
            (1.5, "1.5"),
            (Path("a/b"), '"a/b"'),
            ([1, "x", True], '[1, "x", true]'),
        ],
    )
    def test_literal_formatting(self, value, expected):
        """Tests literal formatting."""
        assert config_io._format_toml_value(value) == expected

    def test_string_escaping(self):
        """Tests that backslahses and quotes are escapted in the right order to
        ensure that the TOML file can be properly parsed."""
        assert config_io._format_toml_value('a "quoted" \\ value') == (
            '"a \\"quoted\\" \\\\ value"'
        )

    def test_unsupported_type_raises(self):
        """Tests an unsupported type."""
        with pytest.raises(TypeError):
            config_io._format_toml_value({"a": 1})


class TestFlattenDict:
    """Tests function _flatten_dict(d, parent_key)"""

    def test_flattens_nested_dict(self):
        """Tests correct flattening of a dict."""
        d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        result = config_io._flatten_dict(d)
        assert result == [("a.b", 1), ("a.c.d", 2), ("e", 3)]

    def test_empty_dict(self):
        """Tests an empty dict."""
        assert not config_io._flatten_dict({})


class TestToTomlString:
    """Tests function to_toml_string(config)"""

    def test_round_trips_through_tomllib(self, valid_config):
        """Tests that a valid config written to TOML can subsequently be read
        by `tomllib.loads()`."""
        text = config_io.to_toml_string(valid_config)
        reparsed = tomllib.loads(text)
        assert reparsed == valid_config

    def test_section_headers(self):
        """Tests that section heads are present as expected."""
        text = config_io.to_toml_string({"species": {"inv": ["CO2"]}})
        assert text.startswith("[species]\n")
        assert 'inv = ["CO2"]' in text


class TestWriteToml:
    """Tests function write_toml(config, filepath)"""

    def test_writes_file_matching_to_toml_string(self, tmp_path, valid_config):
        """Tests that the contents of a saved file matches memory."""
        filepath = tmp_path / "out.toml"
        config_io.write_toml(valid_config, filepath)
        assert filepath.read_text(encoding="utf-8") == config_io.to_toml_string(
            valid_config
        )


class TestPrepareForSave:
    """Tests function prepare_for_save(config, working_dir)"""

    def test_makes_dir_fields_relative(self, tmp_path):
        """Tests that directory fiels are made relative."""
        config = {
            "inventories": {
                "dir": str(tmp_path / "inv"),
                "base": {"dir": str(tmp_path / "base")},
            },
            "output": {"dir": str(tmp_path / "out")},
            "background": {"dir": str(tmp_path / "bg")},
            "responses": {"dir": str(tmp_path / "resp")},
            "time": {"dir": str(tmp_path / "time")},
        }

        result = config_io.prepare_for_save(config, str(tmp_path))

        assert result["inventories"]["dir"] == "inv"
        assert result["inventories"]["base"]["dir"] == "base"
        assert result["output"]["dir"] == "out"
        assert result["background"]["dir"] == "bg"
        assert result["responses"]["dir"] == "resp"
        assert result["time"]["dir"] == "time"

    def test_does_not_mutate_input(self, tmp_path):
        """Tests that the config stored in memory is not mutated by preparing
        a copy for saving."""
        config = {"inventories": {"dir": str(tmp_path / "inv")}}
        original = deepcopy(config)
        config_io.prepare_for_save(config, str(tmp_path))
        assert config == original

    def test_blank_dirs_left_untouched(self, tmp_path):
        """Tests that a blank config is left untouched."""
        config = {"inventories": {"dir": ""}}
        result = config_io.prepare_for_save(config, str(tmp_path))
        assert result["inventories"]["dir"] == ""


class TestCheckTime:
    """Tests function _check_time(edited)"""

    def test_end_before_start_flagged(self):
        """Tests that matching starting and ending dates raise a flag."""
        edited = {"time": {"range": [2020, 2020, 1]}}
        assert config_io._check_time(edited) == "⚠️"

    def test_valid_range_ok(self):
        """Tests a valid time range."""
        edited = {"time": {"range": [2020, 2030, 1]}}
        assert config_io._check_time(edited) is None


class TestCheckInventories:
    """Tests function _check_inventories(edited)"""

    def test_missing_dir_flagged(self):
        """Tests missing directory."""
        edited = {
            "inventories": {
                "dir": "", "files": [], "rel_to_base": False, "base": {}
            }
        }
        assert config_io._check_inventories(edited) == "⚠️"

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
        assert config_io._check_inventories(edited) == "⚠️"

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
        assert config_io._check_inventories(edited) is None

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
        assert config_io._check_inventories(edited) == "⚠️"

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
        assert config_io._check_inventories(edited) == "⚠️"


class TestCheckBackground:
    """Tests function _check_background(edited)"""

    def test_missing_dir_flagged(self):
        """Tests missing directory."""
        edited = {"background": {"dir": ""}}
        assert config_io._check_background(edited) == "⚠️"

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
        assert config_io._check_background(edited) == "⚠️"

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
        assert config_io._check_background(edited) is None


class TestCheckResponses:
    """Tests function _check_responses(edited)"""

    def test_missing_dir_flagged(self):
        """Tests missing directory."""
        edited = {"responses": {"dir": ""}}
        assert config_io._check_responses(edited) == "⚠️"

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
        assert config_io._check_responses(edited) == "⚠️"

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
        assert config_io._check_responses(edited) is None


class TestCheckMetrics:
    """Tests function _check_metrics(edited)"""

    def test_run_metrics_off_and_unset_ok(self):
        """Tests that an empty input is fine when `run_metrics == False`."""
        edited = {"metrics": {}, "output": {"run_metrics": False}}
        assert config_io._check_metrics(edited) is None

    @pytest.mark.parametrize("run_metrics", [True, False])
    def test_incomplete_metrics_setup(self, run_metrics):
        """Tests incomplete metrics setup, regardless of whether metrics are
        to be calculated or not."""
        edited = {
            "metrics": {"types": ["ATR"]},
            "output": {"run_metrics": run_metrics},
        }
        assert config_io._check_metrics(edited) == "⚠️"

    def test_run_metrics_on_and_complete_ok(self):
        """Tests valid configuration."""
        edited = {
            "metrics": {"types": ["ATR"], "H": [100], "t_0": [2020]},
            "output": {"run_metrics": True},
        }
        assert config_io._check_metrics(edited) is None


class TestCheckRequiredFields:
    """Tests function check_required_fields(edited_config)"""

    def test_blank_config_reports_multiple_problems(self):
        """Tests that a blank config reports multiple problems to the user."""
        blank = config_io.blank_config()
        problems = config_io.check_required_fields(blank)
        titles = [title for title, _status in problems]
        assert "Simulation period" in titles
        assert "Species" in titles

    def test_order_follows_card_checks(self):
        """Tests the order of the cards."""
        blank = config_io.blank_config()
        problems = config_io.check_required_fields(blank)
        titles = [title for title, _status in problems]
        expected_order = [t for t in config_io.CARD_CHECKS if t in titles]
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

        assert not config_io.check_required_fields(edited)
