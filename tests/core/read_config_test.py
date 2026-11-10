"""Provides tests for module read_config."""

# accessing _resolve_repository_dirs directly is the point of those tests
# pylint: disable=protected-access

import logging
import os
import tomllib
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from openairclim.core import read_config

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# CONSTANTS
REPO_PATH = "repository/"
CACHE_PATH = "repository/cache/weights/"
INV_NAME = "test_inv.nc"
RESP_NAME = "test_resp.nc"
BG_NAME = "co2_bg.nc"
CACHE_NAME = "000.nc"
TOML_NAME = "test.toml"
TOML_INVALID_NAME = "test_invalid.toml"


class TestLoadConfig:
    """Tests function load_config(file_name)."""

    def test_type(self):
        """Loads correct toml file and checks if output is of type dictionary."""
        config = read_config.load_config(REPO_PATH + TOML_NAME)
        assert isinstance(config, dict)

    def test_invalid(self):
        """Loads incorrect toml file and checks for raising exception."""
        with pytest.raises(tomllib.TOMLDecodeError):
            read_config.load_config(REPO_PATH + TOML_INVALID_NAME)


class TestCheckConfig:
    """Tests function check_config(config)."""

    def test_correct_config(self, valid_config):
        """Correct config returns True."""
        assert isinstance(read_config.check_config(deepcopy(valid_config)), dict)

    def test_incorrect_config(self):
        """Incorrect config raises pydantic.ValidationError."""
        config = {
            "species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]},
            "inventories": {
                "dir": 9,
                "files": [INV_NAME],
                "rel_to_base": 1,
                "base": {"dir": 9, "files": [INV_NAME]},
            },
            "output": {
                "dir": "results/",
                "name": "example",
                "overwrite": True,
            },
            "time": {"range": [2020, 2026, 1]},
            "background": {
                "CO2": {"file": (REPO_PATH + BG_NAME), "scenario": "SSP2-4.5"}
            },
            "responses": {"CO2": {"response_grid": "0D"}},
            "temperature": {"method": "Boucher&Reddy", "CO2": {"lambda": 1.0}},
            "aircraft": {"types": ["DEFAULT"]},
        }
        with pytest.raises(ValidationError):
            read_config.check_config(config)

    def test_incorrect_file_path(self):
        """Incorrect file path of emission inventory returns False."""
        config = {
            "species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]},
            "inventories": {
                "dir": REPO_PATH,
                "files": ["not-existing-example.nc"],
            },
            "output": {
                "dir": "results/",
                "name": "example",
                "overwrite": True,
            },
            "time": {"range": [2020, 2026, 1]},
            "background": {
                "CO2": {"file": (REPO_PATH + BG_NAME), "scenario": "SSP2-4.5"}
            },
            "responses": {"CO2": {"response_grid": "0D"}},
            "temperature": {"method": "Boucher&Reddy", "CO2": {"lambda": 1.0}},
            "aircraft": {"types": ["DEFAULT"]},
        }
        with pytest.raises(ValidationError):
            read_config.check_config(config)

    def test_missing_background_file_raises(self, valid_config):
        """Tests that a missing background file is caught by check_config."""
        config = deepcopy(valid_config)
        config["background"]["CO2"]["file"] = "does-not-exist.nc"
        with pytest.raises(FileNotFoundError):
            read_config.check_config(config)

    def test_missing_files_under_resolved_cache_dir_hint_download_command(
        self, monkeypatch, tmp_path, valid_config
    ):
        """Tests that missing files under the resolved cache dir hint at the CLI.

        When background.dir is left unset and auto-resolves to the shared
        repository data cache, and the files aren't actually there, the
        error message should point the user at oac-download-data.
        """
        monkeypatch.setattr(read_config.repository, "get_cache_dir", lambda: tmp_path)
        config = deepcopy(valid_config)
        config["background"]["dir"] = ""
        with pytest.raises(FileNotFoundError, match="oac-download-data"):
            read_config.check_config(config)


class TestResolveRepositoryDirs:
    """Tests function _resolve_repository_dirs(config)."""

    def test_blank_dirs_resolve_to_cache_dir(self, monkeypatch, tmp_path):
        """Tests that unset background.dir/responses.dir resolve to the cache dir.

        Resolves via repository.get_cache_dir().
        """
        monkeypatch.setattr(read_config.repository, "get_cache_dir", lambda: tmp_path)
        config = {"background": {"dir": "."}, "responses": {"dir": "."}}
        read_config._resolve_repository_dirs(config)
        assert config["background"]["dir"] == str(tmp_path)
        assert config["responses"]["dir"] == str(tmp_path)

    def test_blank_string_also_resolves(self, monkeypatch, tmp_path):
        """Tests that a literal empty-string dir also resolves, not just Path(".")."""
        monkeypatch.setattr(read_config.repository, "get_cache_dir", lambda: tmp_path)
        config = {"background": {"dir": ""}, "responses": {"dir": ""}}
        read_config._resolve_repository_dirs(config)
        assert config["background"]["dir"] == str(tmp_path)
        assert config["responses"]["dir"] == str(tmp_path)

    def test_explicit_dirs_untouched(self, monkeypatch):
        """Tests that an explicitly-set dir is left alone.

        get_cache_dir is never called.
        """

        def _fail(*_args, **_kwargs):
            raise AssertionError("get_cache_dir should not be called")

        monkeypatch.setattr(read_config.repository, "get_cache_dir", _fail)
        config = {
            "background": {"dir": "custom/bg/"},
            "responses": {"dir": "custom/resp/"},
        }
        read_config._resolve_repository_dirs(config)
        assert config["background"]["dir"] == "custom/bg/"
        assert config["responses"]["dir"] == "custom/resp/"


class TestCreateOutputDir:
    """Tests function create_output_dir(config)."""

    @staticmethod
    def _make_config(
        tmp_path: Path,
        name: str = "test",
        overwrite: bool = True,
        run_oac: bool = True,
    ) -> dict:
        """Build a minimal config for create_output_dir.

        Args:
            tmp_path (pathlib.Path): Output directory (pytest fixture)
            name (str): Output name, prefix of this simulation's output files
            overwrite (bool): Output overwrite setting
            run_oac (bool): Output run_oac setting

        Returns:
            dict: Configuration dictionary
        """
        return {
            "output": {
                "run_oac": run_oac,
                "dir": str(tmp_path),
                "name": name,
                "overwrite": overwrite,
            }
        }

    def test_creates_missing_dir(self, tmp_path):
        """Missing output directory is created, regardless of overwrite."""
        config = self._make_config(tmp_path, overwrite=False)
        read_config.create_output_dir(config)
        assert Path(tmp_path).is_dir()

    def test_existing_files_overwrite(self, tmp_path):
        """With "overwrite = True", existing output files are removed."""
        config = self._make_config(tmp_path, name="test", overwrite=True)
        results_file = tmp_path / "test.nc"
        metrics_file = tmp_path / "test_metrics.nc"
        results_file.write_text("old results")
        metrics_file.write_text("old metrics")
        read_config.create_output_dir(config)
        assert not results_file.exists()
        assert not metrics_file.exists()

    def test_existing_files_no_overwrite_raises(self, tmp_path):
        """Existing output files and "overwrite = False" raises OSError."""
        config = self._make_config(tmp_path, name="test", overwrite=False)
        (tmp_path / "test.nc").write_text("old results")
        with pytest.raises(OSError):
            read_config.create_output_dir(config)

    def test_no_existing_files_no_overwrite(self, tmp_path):
        """With "overwrite = False", missing output files do not raise."""
        config = self._make_config(tmp_path, name="test", overwrite=False)
        read_config.create_output_dir(config)
        assert Path(tmp_path).is_dir()

    def test_other_simulations_preserved_overwrite(self, tmp_path):
        """Files of another simulation (different name) are preserved."""
        config = self._make_config(tmp_path, name="test_new", overwrite=True)
        other_results = tmp_path / "example.nc"
        other_metrics = tmp_path / "example_metrics.nc"
        other_results.write_text("previous simulation")
        other_metrics.write_text("previous simulation")
        read_config.create_output_dir(config)
        assert other_results.read_text() == "previous simulation"
        assert other_metrics.read_text() == "previous simulation"

    def test_other_simulations_preserved_no_overwrite(self, tmp_path):
        """Files of another simulation are preserved with "overwrite = False"."""
        config = self._make_config(tmp_path, name="test_new", overwrite=False)
        other_results = tmp_path / "example.nc"
        other_results.write_text("previous simulation")
        read_config.create_output_dir(config)
        assert other_results.read_text() == "previous simulation"

    def test_metrics_only_missing_results_raises(self, tmp_path):
        """run_oac = False and missing results file raises OSError."""
        config = self._make_config(tmp_path, name="test", run_oac=False)
        with pytest.raises(OSError):
            read_config.create_output_dir(config)

    def test_metrics_only_with_results(self, tmp_path, caplog):
        """run_oac = False with existing results file logs metrics-only mode."""
        config = self._make_config(tmp_path, name="test", run_oac=False)
        (tmp_path / "test.nc").write_text("results")
        with caplog.at_level(logging.INFO):
            read_config.create_output_dir(config)
        assert "Compute climate metrics only" in caplog.text


class TestClassifySpecies:
    """Tests function classify_species(config)."""

    def test_classification(self):
        """Tests that species are bucketed correctly.

        Bucketed by response_grid (0D/2D/cont), read live from
        config["responses"], or as a sub-species (SPECIES_SUB_ARR).
        """
        config = {
            "species": {"out": ["CO2", "H2O", "cont", "PMO"]},
            "responses": {
                "CO2": {"response_grid": "0D"},
                "H2O": {"response_grid": "2D"},
                "cont": {"response_grid": "cont"},
            },
        }
        species_0d, species_2d, species_cont, species_sub = (
            read_config.classify_species(config)
        )
        assert species_0d == ["CO2"]
        assert species_2d == ["H2O"]
        assert species_cont == ["cont"]
        assert species_sub == ["PMO"]
