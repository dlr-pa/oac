"""Shared fixtures for GUI tests."""

# pylint doesn't recognise pytest's fixture injection
# pylint: disable=redefined-outer-name

from copy import deepcopy
from pathlib import Path

import pytest

from openairclim.gui import config_io
from openairclim.gui.state import AppState

CORE_TESTS_DIR = Path(__file__).resolve().parents[1] / "core"
REPO_DIR = CORE_TESTS_DIR / "repository"
BG_NAME = "co2_bg.nc"
INV_NAME = "test_inv.nc"


def make_valid_config():
    """Return a complete, valid configuration dict.

    Mirrors `read_config_test.TestCheckConfig.test_correct_config`: every path
    is relative to `CORE_TESTS_DIR`, which is used as the working directory
    wherever this config is exercised.

    Returns:
        dict: A configuration dict that passes `core.read_config.check_config`.
    """
    repo_path = "repository/"
    return {
        "species": {"inv": ["CO2"], "nox": "NO", "out": ["CO2"]},
        "inventories": {
            "dir": repo_path,
            "files": [INV_NAME],
            "rel_to_base": False,
            "base": {"dir": repo_path, "files": [INV_NAME]},
        },
        "output": {
            "run_oac": True,
            "run_metrics": True,
            "run_plots": True,
            "dir": "results/",
            "name": "example",
            "overwrite": True,
            "concentrations": False,
        },
        "time": {"range": [2020, 2121, 1]},
        "background": {
            "dir": repo_path,
            "CO2": {"file": (repo_path + BG_NAME), "scenario": "SSP2-4.5"},
            "CH4": {"file": (repo_path + BG_NAME), "scenario": "SSP2-4.5"},
            "N2O": {"file": (repo_path + BG_NAME), "scenario": "SSP2-4.5"},
        },
        "responses": {"dir": repo_path},
        "temperature": {"method": "Boucher&Reddy", "CO2": {"lambda": 1.0}},
        "metrics": {"types": ["ATR"], "t_0": [2020], "H": [100]},
        "aircraft": {"types": ["DEFAULT"]},
    }


@pytest.fixture
def valid_config():
    """A complete, valid configuration dict."""
    return make_valid_config()


@pytest.fixture
def working_dir():
    """Working directory that `valid_config`'s relative paths resolve
    against."""
    return str(CORE_TESTS_DIR)


@pytest.fixture
def blank_state():
    """An AppState holding a freshly-created blank configuration."""
    state = AppState()
    state.working_dir = "."
    state.edited_config = config_io.blank_config()
    state.config_generation = 1
    return state


@pytest.fixture
def loaded_state(working_dir, valid_config):
    """An AppState holding a complete, valid configuration."""
    from openairclim.core.config_model import validate_config

    state = AppState()
    state.working_dir = working_dir
    state.edited_config = config_io._stringify_paths(  # pylint: disable=protected-access
        validate_config(deepcopy(valid_config))
    )
    state.config_generation = 1
    return state
