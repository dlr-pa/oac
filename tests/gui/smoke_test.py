"""Smoke tests confirming each GUI tab (and the assembled app) can be built
without raising, for both an empty and a fully-populated application state.

Button clicks and widget interactions are deliberately not simulated here.
"""

# pylint doesn't recognise pytest's fixture injection
# pylint: disable=redefined-outer-name

import panel as pn
import pytest

from openairclim.gui import config_io, sidebar
from openairclim.gui.app import build_app
from openairclim.gui.state import AppState
from openairclim.gui.tabs import aircraft, config, config_text, inventories, results, scenario


@pytest.fixture(params=["empty", "blank", "loaded"])
def state(request, blank_state, loaded_state):
    """An AppState in each of the three shapes a tab's panel() must render:
    nothing open yet, a freshly-blanked config, and a fully loaded one."""
    if request.param == "empty":
        return AppState()
    if request.param == "blank":
        return blank_state
    return loaded_state


TAB_MODULES = [config, inventories, scenario, aircraft, results]


@pytest.mark.parametrize("tab_module", TAB_MODULES, ids=lambda m: m.__name__)
def test_tab_panel_builds(tab_module, state):
    """Tests that each tab builds."""
    result = tab_module.panel(state)
    assert isinstance(result, pn.layout.Panel)


def test_config_text_panel_builds(state):
    """Tests that the config_text panel builds."""
    status_panes = sidebar.build_status_panes()
    result = config_text.panel(state, status_panes)
    assert isinstance(result, pn.layout.Panel)


def test_sidebar_panel_builds(state):
    """Tests that the sidebar builds."""
    result = sidebar.panel(state)
    assert isinstance(result, pn.layout.Panel)


def test_build_app_builds():
    """Tests that the app builds with the correct template."""
    template = build_app()
    assert isinstance(template, pn.template.FastListTemplate)


def test_build_app_with_preloaded_config(tmp_path, valid_config):
    """Tests that the app builds with a pre-loaded config file."""
    config_path = tmp_path / "config.toml"
    config_io.write_toml(valid_config, config_path)
    template = build_app(config_path=str(config_path))
    assert isinstance(template, pn.template.FastListTemplate)
