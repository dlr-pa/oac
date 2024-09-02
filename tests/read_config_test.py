"""
Provides tests for module read_config
"""

import os
import tomllib
from unittest.mock import patch
import pytest
import openairclim as oac

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
    """Tests function load_config(file_name)"""

    def test_type(self):
        """Loads correct toml file and checks if output is of type dictionary"""
        config = oac.load_config((REPO_PATH + TOML_NAME))
        assert isinstance(config, dict)

    def test_invalid(self):
        """Loads incorrect toml file and checks for raising exception"""
        with pytest.raises(tomllib.TOMLDecodeError):
            oac.load_config((REPO_PATH + TOML_INVALID_NAME))


class TestCheckConfig:
    """Tests function check_config(config)"""

    def test_correct_config(self):
        """Correct config returns True"""
        config = {
            "species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]},
            "inventories": {
                "dir": REPO_PATH,
                "files": [INV_NAME],
            },
            "output": {
                "full_run": True,
                "dir": "results/",
                "name": "example",
                "overwrite": True,
                "concentrations": False,
            },
            "time": {"dir": REPO_PATH, "range": [2020, 2121, 1]},
            "background": {
                "CO2": {"file": (REPO_PATH + BG_NAME), "scenario": "SSP2-4.5"}
            },
            "responses": {"CO2": {"response_grid": "0D"}},
            "temperature": {"method": "Boucher&Reddy", "CO2": {"lambda": 1.0}},
            "metrics": {"types": ["ATR"], "t_0": [2020], "H": [100]},
        }
        assert oac.check_config(config)

    def test_incorrect_config(self):
        """Incorrect config returns False"""
        config = {
            "species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]},
            "inventories": {
                "dir": 9,
                "files": [INV_NAME],
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
        }
        assert oac.check_config(config) is False

    def test_incorrect_file_path(self):
        """Incorrect file path of emission inventory returns False"""
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
        }
        assert oac.check_config(config) is False


# TODO Instead of creating and removing directories, use patch or monkeypatch
#      fixtures for the simulation of os functionalities (test doubles)
@pytest.fixture(scope="class")
def make_remove_dir(request):
    """Arrange and Cleanup fixture, create an output directory for testing
        and remove it afterwards, setup and the directory name can be reused
        in several test functions of the same class.

    Args:
        request (_pytest.fixtures.FixtureRequest): pytest request parameter
            for injecting objects into test functions
    """
    dir_path = "results/"
    request.cls.dir_path = dir_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    yield
    os.rmdir(dir_path)


@pytest.mark.usefixtures("make_remove_dir")
class TestCreateOutputDir:
    """Tests function create_output_dir(config)"""

    def test_existing_dir_no_overwrite(self):
        """Existing output directory and "overwrite = False" raises OSError"""
        config = {
            "output": {
                "full_run": True,
                "dir": "results/",
                "name": "test",
                "overwrite": False,
            }
        }
        with pytest.raises(OSError):
            oac.create_output_dir(config)

    @patch("os.path.isdir")
    def test_existing_dir_overwrite(self, patch_isdir):
        """Existing output directory and "overwrite = True" creates output dictionary"""
        config = {
            "output": {
                "full_run": True,
                "dir": "results/",
                "name": "test",
                "overwrite": True,
            }
        }
        oac.create_output_dir(config)
        assert patch_isdir("results/")


class TestClassifySpecies:
    """Tests function classify_species(config)"""

    def test_missing_response_species(self):
        """Species defined in "species", but not in "responses" raises KeyError"""
        config = {
            "species": {
                "inv": ["CO2", "H2O"],
                "nox": "NO",
                "out": ["CO2", "H2O"],
            },
            "responses": {"CO2": {"response_grid": "0D"}},
        }
        with pytest.raises(KeyError):
            oac.classify_species(config)

    def test_no_response_grid(self):
        """No response_grid for a species raises KeyError"""
        config = {
            "species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]},
            "responses": {"CO2": {}},
        }
        with pytest.raises(KeyError):
            oac.classify_species(config)
