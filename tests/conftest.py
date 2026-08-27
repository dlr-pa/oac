"""Shared fixtures for the whole test suite. The core purpose is to
centralise a single valid configuration dict that can be read by pytest
functionality in tests/core and tests/gui. All paths are relative to
`tests/core` since core tests with that as its cwd. GUI tests use their own
`working_dir` fixture, which also reflects how the GUI works.
"""

from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent / "core" / "repository"
INV_NAME = "test_inv.nc"
BG_NAME = "co2_bg.nc"


def make_valid_config():
    """Return a complete, valid configuration dict.

    Passes `core.read_config.check_config` as-is (real referenced files),
    and - once run through `core.config_model.validate_config` - is also a
    valid `state.edited_config` for the GUI.

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
            "CO2": {"file": BG_NAME, "scenario": "SSP2-4.5"},
            "CH4": {"file": BG_NAME, "scenario": "SSP2-4.5"},
            "N2O": {"file": BG_NAME, "scenario": "SSP2-4.5"},
        },
        "responses": {"dir": repo_path},
        "temperature": {"method": "Boucher&Reddy", "CO2": {"lambda": 1.0}},
        "metrics": {"types": ["ATR"], "t_0": [2020], "H": [100]},
        "aircraft": {"types": ["DEFAULT"]},
    }


@pytest.fixture
def valid_config():
    """A complete, valid configuration dict (see :func:`make_valid_config`)."""
    return make_valid_config()


@pytest.fixture
def working_dir():
    """Working directory that ``valid_config``'s relative paths resolve against."""
    return str(REPO_DIR.parent)
