"""Provides tests for module openairclim.utils.output_metadata."""

# since we are testing private helpers within the module, we ignore the
# corresponding pylint warning in this file
# pylint: disable=protected-access

import json
import tomllib
from pathlib import Path

import pytest
import xarray as xr
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from openairclim.utils import output_metadata

SAMPLE_CONFIG = {
    "species": {"inv": ["CO2"], "out": ["CO2"]},
    "output": {"dir": ".", "name": "run"},
}

SAMPLE_ATTRS = {
    "created": "2026-01-01T00:00:00+00:00",
    "user": "tester",
    "platform": "TestOS",
    "python_version": "3.12.0",
    "oac_version": "1.2.3",
    "oac_git_commit": "abcdef",
    "config_hash": "aviation01",
    "config_json": json.dumps(SAMPLE_CONFIG, sort_keys=True),
}


def _write_nc(path, attrs=None):
    """Write a minimal netCDF file with the given global attrs."""
    ds = xr.Dataset({"var": ("x", [1, 2, 3])})
    if attrs:
        ds.attrs.update(attrs)
    ds.to_netcdf(path)


def _write_png(path, metadata=None):
    """Write a minimal PNG with the given text chunks (as fig_metadata does)."""
    info = PngInfo()
    for key, value in (metadata or {}).items():
        info.add_text(key, value)
    Image.new("RGB", (1, 1)).save(path, pnginfo=info)


class TestReadNcMetadata:
    """Tests function _read_nc_metadata(path)."""

    def test_reads_metadata_and_parses_config(self, tmp_path):
        """Tests that attrs are read back and config_json parsed into a dict."""
        nc_path = tmp_path / "out.nc"
        _write_nc(nc_path, SAMPLE_ATTRS)

        metadata = output_metadata._read_nc_metadata(nc_path)

        assert metadata["config_hash"] == "aviation01"
        assert metadata["user"] == "tester"
        assert metadata["config"] == SAMPLE_CONFIG

    def test_missing_config_hash_returns_none(self, tmp_path):
        """Tests that a netCDF file without OpenAirClim metadata is skipped."""
        nc_path = tmp_path / "plain.nc"
        _write_nc(nc_path, {"title": "not an oac file"})

        assert output_metadata._read_nc_metadata(nc_path) is None

    def test_unreadable_file_returns_none(self, tmp_path):
        """Tests that a nonexistent file is skipped rather than raising."""
        assert output_metadata._read_nc_metadata(tmp_path / "missing.nc") is None


class TestReadPngMetadata:
    """Tests function _read_png_metadata(path)."""

    def test_reads_known_keys(self, tmp_path):
        """Tests that config_hash/oac_version/created text chunks are read."""
        png_path = tmp_path / "fig.png"
        _write_png(
            png_path,
            {"config_hash": "aviation01", "oac_version": "1.2.3", "created": "now"},
        )

        metadata = output_metadata._read_png_metadata(png_path)

        assert metadata == {
            "config_hash": "aviation01",
            "oac_version": "1.2.3",
            "created": "now",
        }

    def test_png_without_metadata_returns_none(self, tmp_path):
        """Tests a plain PNG with no OpenAirClim text chunks."""
        png_path = tmp_path / "plain.png"
        _write_png(png_path)

        assert output_metadata._read_png_metadata(png_path) is None

    def test_unreadable_file_returns_none(self, tmp_path):
        """Tests that a nonexistent file is skipped rather than raising."""
        assert output_metadata._read_png_metadata(tmp_path / "missing.png") is None


class TestReadMetadata:
    """Tests function read_metadata(path)."""

    def test_dispatches_by_suffix(self, tmp_path):
        """Tests that .nc and .png files are routed to the right reader."""
        nc_path = tmp_path / "out.nc"
        _write_nc(nc_path, SAMPLE_ATTRS)
        png_path = tmp_path / "fig.png"
        _write_png(png_path, {"config_hash": "aviation01"})

        assert output_metadata.read_metadata(nc_path)["config_hash"] == "aviation01"
        assert output_metadata.read_metadata(png_path)["config_hash"] == "aviation01"

    def test_unrecognised_suffix_returns_none(self, tmp_path):
        """Tests that an unsupported file type is skipped rather than raising."""
        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("hello", encoding="utf-8")

        assert output_metadata.read_metadata(txt_path) is None


class TestGetMetadata:
    """Tests function get_metadata(path)."""

    def test_single_file(self, tmp_path):
        """Tests that a single file argument returns a one-entry mapping."""
        nc_path = tmp_path / "out.nc"
        _write_nc(nc_path, SAMPLE_ATTRS)

        result = output_metadata.get_metadata(nc_path, full=True)

        assert set(result) == {"out.nc"}
        assert result["out.nc"]["config_hash"] == "aviation01"

    def test_directory_scans_nc_files_and_skips_non_oac(self, tmp_path):
        """Tests that a directory is scanned for .nc files.

        Only those carrying OpenAirClim metadata are included.
        """
        _write_nc(tmp_path / "run1.nc", SAMPLE_ATTRS)
        other_attrs = dict(SAMPLE_ATTRS, config_hash="aircraft02")
        _write_nc(tmp_path / "run2.nc", other_attrs)
        _write_nc(tmp_path / "not_oac.nc", {"title": "plain"})
        (tmp_path / "ignore.toml").touch()

        result = output_metadata.get_metadata(tmp_path, full=True)

        assert set(result) == {"run1.nc", "run2.nc"}
        assert result["run1.nc"]["config_hash"] == "aviation01"
        assert result["run2.nc"]["config_hash"] == "aircraft02"

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        """Tests a directory with no matching files."""
        assert output_metadata.get_metadata(tmp_path) == {}

    def test_directory_includes_png_figures_alongside_nc_files(self, tmp_path):
        """Tests that .png figures are scanned alongside .nc files."""
        _write_nc(tmp_path / "run.nc", SAMPLE_ATTRS)
        _write_png(tmp_path / "fig.png", {"config_hash": "aviation01"})
        _write_png(tmp_path / "unrelated.png")

        result = output_metadata.get_metadata(tmp_path, full=True)

        assert set(result) == {"run.nc", "fig.png"}
        assert all(m["config_hash"] == "aviation01" for m in result.values())

    def test_full_false_returns_only_hashes(self, tmp_path):
        """Tests that full=False maps filename -> config_hash only."""
        _write_nc(tmp_path / "run1.nc", SAMPLE_ATTRS)
        other_attrs = dict(SAMPLE_ATTRS, config_hash="aircraft02")
        _write_nc(tmp_path / "run2.nc", other_attrs)

        result = output_metadata.get_metadata(tmp_path, full=False)

        assert result == {"run1.nc": "aviation01", "run2.nc": "aircraft02"}

    def test_full_false_omits_files_without_config_hash(self, tmp_path):
        """Tests that full=False skips files with metadata but no config_hash."""
        _write_nc(tmp_path / "run.nc", SAMPLE_ATTRS)
        _write_png(tmp_path / "partial.png", {"oac_version": "1.2.3"})

        result = output_metadata.get_metadata(tmp_path, full=False)

        assert result == {"run.nc": "aviation01"}

    def test_not_recursive_by_default_skips_subdirectories(self, tmp_path):
        """Tests that files in subdirectories are ignored unless recursive."""
        _write_nc(tmp_path / "top.nc", SAMPLE_ATTRS)
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_nc(sub / "nested.nc", SAMPLE_ATTRS)

        result = output_metadata.get_metadata(tmp_path)

        assert set(result) == {"top.nc"}

    def test_recursive_scans_subdirectories_and_keys_by_relpath(self, tmp_path):
        """Tests that recursive=True finds nested files, keyed by relative path."""
        _write_nc(tmp_path / "top.nc", SAMPLE_ATTRS)
        sub = tmp_path / "sub"
        sub.mkdir()
        other_attrs = dict(SAMPLE_ATTRS, config_hash="aircraft02")
        _write_nc(sub / "nested.nc", other_attrs)

        result = output_metadata.get_metadata(tmp_path, full=False, recursive=True)

        assert result == {
            "top.nc": "aviation01",
            str(Path("sub") / "nested.nc"): "aircraft02",
        }

    def test_recursive_distinguishes_same_named_files_in_subdirectories(self, tmp_path):
        """Tests that same-named files in different subdirectories don't collide."""
        sub_a = tmp_path / "a"
        sub_b = tmp_path / "b"
        sub_a.mkdir()
        sub_b.mkdir()
        _write_nc(sub_a / "out.nc", SAMPLE_ATTRS)
        other_attrs = dict(SAMPLE_ATTRS, config_hash="aircraft02")
        _write_nc(sub_b / "out.nc", other_attrs)

        result = output_metadata.get_metadata(tmp_path, full=False, recursive=True)

        assert result == {
            str(Path("a") / "out.nc"): "aviation01",
            str(Path("b") / "out.nc"): "aircraft02",
        }


class TestFormatResults:
    """Tests function _format_results(results)."""

    def test_hash_only_groups_files_under_their_hash(self):
        """Tests full=False style results: filename -> hash."""
        results = {
            "b.nc": "hash1",
            "a.nc": "hash1",
            "c.nc": "hash2",
        }

        text = output_metadata._format_results(results)

        assert text == "hash1:\n  a.nc\n  b.nc\nhash2:\n  c.nc"

    def test_full_lists_metadata_under_each_file(self):
        """Tests full=True style results: filename -> metadata dict."""
        results = {
            "run.nc": {"config_hash": "hash1", "oac_version": "1.2.3"},
        }

        text = output_metadata._format_results(results)

        assert text == "hash1:\n  run.nc\n    oac_version: 1.2.3"

    def test_full_serialises_nested_values_as_json(self):
        """Tests that dict/list metadata values (e.g. config) are JSON-dumped."""
        results = {
            "run.nc": {"config_hash": "hash1", "config": {"a": 1}},
        }

        text = output_metadata._format_results(results)

        assert text == 'hash1:\n  run.nc\n    config: {"a": 1}'

    def test_missing_config_hash_grouped_separately(self):
        """Tests that full=True files without a config_hash get their own group."""
        results = {
            "partial.png": {"oac_version": "1.2.3"},
        }

        text = output_metadata._format_results(results)

        assert text == "(no config_hash):\n  partial.png\n    oac_version: 1.2.3"

    def test_empty_results(self):
        """Tests that an empty mapping formats as an empty string."""
        assert output_metadata._format_results({}) == ""


class TestWriteConfigFromNetcdf:
    """Tests function write_config_from_netcdf(nc_path, toml_path)."""

    def test_recovers_config_and_writes_toml(self, tmp_path):
        """Tests that the embedded config is parsed and written as valid TOML."""
        nc_path = tmp_path / "out.nc"
        _write_nc(nc_path, SAMPLE_ATTRS)
        toml_path = tmp_path / "recovered.toml"

        config = output_metadata.write_config_from_netcdf(nc_path, toml_path)

        assert config == SAMPLE_CONFIG
        reparsed = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        assert reparsed == SAMPLE_CONFIG

    def test_file_without_metadata_raises(self, tmp_path):
        """Tests that a file with no embedded config metadata raises KeyError."""
        nc_path = tmp_path / "plain.nc"
        _write_nc(nc_path, {"title": "plain"})

        with pytest.raises(KeyError):
            output_metadata.write_config_from_netcdf(nc_path, tmp_path / "out.toml")

    def test_without_relative_to_keeps_absolute_dirs(self, tmp_path):
        """Tests that omitting relative_to keeps directory fields absolute.

        Left as the absolute paths embedded in the output file.
        """
        config_with_dirs = {
            "species": {"inv": ["CO2"], "out": ["CO2"]},
            "output": {"dir": str(tmp_path / "out"), "name": "run"},
        }
        attrs = dict(
            SAMPLE_ATTRS, config_json=json.dumps(config_with_dirs, sort_keys=True)
        )
        nc_path = tmp_path / "out.nc"
        _write_nc(nc_path, attrs)

        config = output_metadata.write_config_from_netcdf(
            nc_path, tmp_path / "recovered.toml"
        )

        assert config["output"]["dir"] == str(tmp_path / "out")

    def test_relative_to_relativises_directory_fields(self, tmp_path):
        """Tests that passing relative_to rewrites directory fields.

        Rewritten relative to that directory, via config_files.prepare_for_save.
        """
        config_with_dirs = {
            "species": {"inv": ["CO2"], "out": ["CO2"]},
            "inventories": {"dir": str(tmp_path / "inv"), "files": ["a.nc"]},
            "output": {"dir": str(tmp_path / "out"), "name": "run"},
        }
        attrs = dict(
            SAMPLE_ATTRS, config_json=json.dumps(config_with_dirs, sort_keys=True)
        )
        nc_path = tmp_path / "out.nc"
        _write_nc(nc_path, attrs)
        toml_path = tmp_path / "recovered.toml"

        config = output_metadata.write_config_from_netcdf(
            nc_path, toml_path, relative_to=tmp_path
        )

        assert config["inventories"]["dir"] == "inv"
        assert config["output"]["dir"] == "out"
        reparsed = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        assert reparsed["inventories"]["dir"] == "inv"
        assert reparsed["output"]["dir"] == "out"
