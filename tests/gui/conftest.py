"""Shared fixtures for GUI tests.

`valid_config` and `working_dir` come from the top-level tests/conftest.py -
pytest injects them automatically, no import needed.
"""

# pylint doesn't recognise pytest's fixture injection
# pylint: disable=redefined-outer-name

from copy import deepcopy

import pytest

from openairclim.gui import config_io
from openairclim.gui.state import AppState


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
