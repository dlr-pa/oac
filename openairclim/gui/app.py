"""Assemble the OpenAirClim GUI application."""

from pathlib import Path

import panel as pn

from . import sidebar
from .state import AppState
from .tabs import aircraft, config, config_text, results, scenario, inventories


def build_app(config_path=None, results_path=None):
    """Return the top-level Panel template.

    Args:
        config_path (str, optional): Config file to pre-load.
        results_path (str, optional): Output file to pre-load.

    Returns:
        pn.template.FastListTemplate: The assembled application.
    """
    state = AppState()

    if config_path:
        state.config_path = str(Path(config_path).resolve())
    if results_path:
        state.results_path = str(Path(results_path).resolve())

    # shared with config (expert) tab
    status_panes = sidebar.build_status_panes()

    tabs = pn.Tabs(
        ("Config", config.panel(state)),
        ("Config (Expert)", config_text.panel(state, status_panes)),
        ("Inventories", inventories.panel(state)),
        ("Scenario", scenario.panel(state)),
        ("Aircraft", aircraft.panel(state)),
        ("Results", results.panel(state)),
        # dynamic=True re-renders a tab's Bokeh view every time it's
        # switched to, rather than building it once. This breaks card
        # expand/collapse toggle after navigating to different tabs. Eager
        # rendering is fine since all tabs already have placeholders.
    )

    template = pn.template.FastListTemplate(
        title="OpenAirClim",
        sidebar=[sidebar.panel(state, status_panes)],
        main=[tabs],
    )
    return template
