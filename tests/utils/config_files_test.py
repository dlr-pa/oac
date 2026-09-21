"""Provides tests for module openairclim.utils.config_files."""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

import tomllib
from copy import deepcopy
from pathlib import Path

import pytest
import xarray as xr

from openairclim.utils import config_files


class TestResolveDir:
    """Tests function resolve_dir(working_dir, dir_str)."""

    def test_absolute_path_unchanged(self, tmp_path):
        """Tests that an absolute path remains unchanged."""
        result = config_files.resolve_dir("/anything", str(tmp_path))
        assert result == tmp_path.resolve()

    def test_relative_path_joined_with_working_dir(self, tmp_path):
        """Tests that a relative path is combined with the working directory."""
        result = config_files.resolve_dir(str(tmp_path), "sub/dir")
        assert result == (tmp_path / "sub" / "dir").resolve()


class TestToRelative:
    """Tests function to_relative(working_dir, absolute_path)."""

    def test_inside_working_dir(self, tmp_path):
        """Tests a path to a file within the working directory."""
        target = tmp_path / "sub" / "file.nc"
        result = config_files.to_relative(str(tmp_path), str(target))
        assert result == "sub/file.nc"

    def test_outside_working_dir_returns_relative_with_dotdot(self, tmp_path):
        """Tests a path to a folder outside the working directory, same drive."""
        import os

        working_dir = tmp_path / "project"
        working_dir.mkdir()
        outside = tmp_path / "sibling"
        outside.mkdir()

        result = config_files.to_relative(str(working_dir), str(outside))
        expected = Path(os.path.relpath(str(outside), str(working_dir))).as_posix()
        assert result == expected

    def test_unrelativisable_path_returns_absolute(self, tmp_path, monkeypatch):
        """Tests the fallback when no relative path can be computed at all.

        (e.g. different drives on Windows).
        """
        import os

        outside = "/completely/unrelated/path"

        def _raise(*_args, **_kwargs):
            raise ValueError("no relative path")

        monkeypatch.setattr(os.path, "relpath", _raise)
        result = config_files.to_relative(str(tmp_path), outside)
        assert result == Path(outside).as_posix()


class TestListNcFiles:
    """Tests function list_nc_files(directory_path)."""

    def test_lists_and_sorts_nc_files(self, tmp_path):
        """Tests that only nc files are returned."""
        (tmp_path / "b.nc").touch()
        (tmp_path / "a.nc").touch()
        (tmp_path / "ignore.toml").touch()

        result = config_files.list_nc_files(tmp_path)

        assert result == ["a.nc", "b.nc"]

    def test_missing_dir_returns_empty_list(self, tmp_path):
        """Tests missing directory."""
        result = config_files.list_nc_files(tmp_path / "does_not_exist")
        assert result == []


class TestListNcDataVars:
    """Tests function list_nc_data_vars(filepath)."""

    def test_lists_sorted_data_vars(self, tmp_path):
        """Tests that all data variables are identified."""
        ds = xr.Dataset({"b_var": ("x", [1, 2]), "a_var": ("x", [3, 4])})
        filepath = tmp_path / "data.nc"
        ds.to_netcdf(filepath)

        result = config_files.list_nc_data_vars(filepath)

        assert result == ["a_var", "b_var"]

    def test_missing_file_returns_empty_list(self, tmp_path):
        """Tests that a missing file returns an empty list."""
        result = config_files.list_nc_data_vars(tmp_path / "does_not_exist.nc")
        assert result == []


class TestFormatTomlValue:
    """Tests function _format_toml_value(value)."""

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
        assert config_files._format_toml_value(value) == expected

    def test_string_escaping(self):
        """Tests that backslashes and quotes are escaped in the right order.

        Ensures that the TOML file can be properly parsed.
        """
        assert config_files._format_toml_value('a "quoted" \\ value') == (
            '"a \\"quoted\\" \\\\ value"'
        )

    def test_unsupported_type_raises(self):
        """Tests an unsupported type."""
        with pytest.raises(TypeError):
            config_files._format_toml_value({"a": 1})


class TestFlattenDict:
    """Tests function _flatten_dict(d, parent_key)."""

    def test_flattens_nested_dict(self):
        """Tests correct flattening of a dict."""
        d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        result = config_files._flatten_dict(d)
        assert result == [("a.b", 1), ("a.c.d", 2), ("e", 3)]

    def test_empty_dict(self):
        """Tests an empty dict."""
        assert not config_files._flatten_dict({})


class TestToTomlString:
    """Tests function to_toml_string(config)."""

    def test_round_trips_through_tomllib(self, valid_config):
        """Tests that a valid config written to TOML can be read back.

        Read back with `tomllib.loads()`.
        """
        text = config_files.to_toml_string(valid_config)
        reparsed = tomllib.loads(text)
        assert reparsed == valid_config

    def test_section_headers(self):
        """Tests that section heads are present as expected."""
        text = config_files.to_toml_string({"species": {"inv": ["CO2"]}})
        assert text.startswith("[species]\n")
        assert 'inv = ["CO2"]' in text


class TestWriteToml:
    """Tests function write_toml(config, filepath)."""

    def test_writes_file_matching_to_toml_string(self, tmp_path, valid_config):
        """Tests that the contents of a saved file matches memory."""
        filepath = tmp_path / "out.toml"
        config_files.write_toml(valid_config, filepath)
        assert filepath.read_text(encoding="utf-8") == config_files.to_toml_string(
            valid_config
        )


class TestPrepareForSave:
    """Tests function prepare_for_save(config, working_dir)."""

    def test_makes_dir_fields_relative(self, tmp_path):
        """Tests that directory fields are made relative."""
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

        result = config_files.prepare_for_save(config, str(tmp_path))

        assert result["inventories"]["dir"] == "inv"
        assert result["inventories"]["base"]["dir"] == "base"
        assert result["output"]["dir"] == "out"
        assert result["background"]["dir"] == "bg"
        assert result["responses"]["dir"] == "resp"
        assert result["time"]["dir"] == "time"

    def test_does_not_mutate_input(self, tmp_path):
        """Tests that the config stored in memory is not mutated.

        Not mutated by preparing a copy for saving.
        """
        config = {"inventories": {"dir": str(tmp_path / "inv")}}
        original = deepcopy(config)
        config_files.prepare_for_save(config, str(tmp_path))
        assert config == original

    def test_blank_dirs_left_untouched(self, tmp_path):
        """Tests that a blank config is left untouched."""
        config = {"inventories": {"dir": ""}}
        result = config_files.prepare_for_save(config, str(tmp_path))
        assert result["inventories"]["dir"] == ""
