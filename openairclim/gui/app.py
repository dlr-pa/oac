"""Assemble the OpenAirClim GUI application."""

from pathlib import Path

import panel as pn

from . import sidebar
from .state import AppState
from .tabs import aircraft, config, results, scenario, inventories


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

    tabs = pn.Tabs(
        ("Config", config.panel(state)),
        ("Inventories", inventories.panel(state)),
        ("Scenario", scenario.panel(state)),
        ("Aircraft", aircraft.panel(state)),
        ("Results", results.panel(state)),
        dynamic=True,
    )

    template = pn.template.FastListTemplate(
        title="OpenAirClim",
        sidebar=[sidebar.panel(state)],
        main=[tabs],
    )
    return template
