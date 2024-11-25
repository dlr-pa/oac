"""
Provides tests for module read_config
"""

import os
import tomllib
from collections.abc import Iterable
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


@pytest.fixture(name="setup_arguments", scope="class")
def fixture_setup_arguments():
    """Setup arguments for check_config

    Returns:
        dict, dict: Configuration template and default config
    """
    config_template = {
        "species": {"inv": Iterable, "out": Iterable},
        "inventories": {"dir": str, "files": Iterable, "rel_to_base": bool},
        "output": {
            "full_run": bool,
            "dir": str,
            "name": str,
            "overwrite": bool,
            "concentrations": bool,
        },
        "time": {"range": Iterable},
        "background": {"CO2": {"file": str, "scenario": str}},
        "responses": {"CO2": {"response_grid": str, "rf": {"method": str}}},
        "temperature": {"method": str, "CO2": {"lambda": float}},
        "metrics": {"types": Iterable, "t_0": Iterable, "H": Iterable},
    }
    default_config = {"responses": {"CO2": {"rf": {"method": "Etminan_2016"}}}}
    return config_template, default_config


@pytest.mark.usefixtures("setup_arguments")
class TestCheckConfig:
    """Tests function check_config(config)"""

    def test_correct_config(self, setup_arguments):
        """Correct config returns True"""
        config_template, default_config = setup_arguments
        config = {
            "species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]},
            "inventories": {
                "dir": REPO_PATH,
                "files": [INV_NAME],
                "rel_to_base": False,
                "base": {"dir": REPO_PATH, "files": [INV_NAME]},
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
        assert isinstance(
            oac.check_config(config, config_template, default_config), dict
        )

    def test_incorrect_config(self, setup_arguments):
        """Incorrect config returns TypeError"""
        config_template, default_config = setup_arguments
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
        }
        with pytest.raises(TypeError):
            oac.check_config(config, config_template, default_config)

    def test_incorrect_file_path(self, setup_arguments):
        """Incorrect file path of emission inventory returns False"""
        config_template, default_config = setup_arguments
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
        with pytest.raises(KeyError):
            oac.check_config(config, config_template, default_config)


class TestGetKeysValues:
    """Tests function get_key_values"""

    def test_get_keys_values_empty_dict(self):
        """Tests get_keys_values with an empty dictionary."""
        config = {}
        expected_key_arr = []
        expected_val_arr = []
        # Initialize key, value lists
        actual_key_arr = []
        actual_val_arr = []
        oac.get_keys_values(config, actual_key_arr, actual_val_arr)
        assert expected_key_arr == actual_key_arr
        assert expected_val_arr == actual_val_arr

    def test_get_keys_values_nest_dict(self):
        """Tests get_keys_values with a nested dictionary."""
        # Define a nested dictionary and expected list of keys and values.
        config = {
            "key1": 10,
            "key2": {"sub_key1": "string-value", "sub_key2": 20},
        }
        expected_key_arr_nest_dict = [
            "key1 ",
            "key2 sub_key1 ",
            "key2 sub_key2 ",
        ]
        expected_val_arr_nest_dict = [10, "string-value", 20]
        # Initialize key, value lists
        actual_key_arr = []
        actual_val_arr = []
        oac.get_keys_values(config, actual_key_arr, actual_val_arr)
        assert expected_key_arr_nest_dict == actual_key_arr
        assert expected_val_arr_nest_dict == actual_val_arr


class TestAddDefaultConfig:
    """Tests function add_default_config"""

    def test_add_default_success(self):
        """Tests output of a dictionary when default setting is added successfully"""
        # Create a small test configuration dictionary
        config = {}
        # Define a default configuration with some settings
        default_config = {
            "setting1": True,
            "setting2": {"sub_setting": {"sub_sub_setting": False}},
        }
        # Call add_default_config and assert that the result is as expected
        actual = oac.add_default_config(
            config, "setting2 sub_setting sub_sub_setting ", default_config
        )
        assert isinstance(actual, dict)

    def test_add_default_error(self):
        """ "KeyError is raised if required setting not in default_config"""
        # Create a small test configuration dictionary
        config = {}
        # Define a default configuration with some settings
        default_config = {
            "setting1": True,
            "thing_to_set_option1": {"sub_setting": False},
        }
        # Call add_default_config and assert that the KeyError is raised
        with pytest.raises(KeyError):
            oac.add_default_config(config, "option2 setting3", default_config)


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
