"""Scenario tab: explore simulation scenario (placeholder)."""

import panel as pn


def panel(state):
    """Return the scenario tab content.

    Args:
        state (AppState): Shared application state.
    """
    return pn.Column(
        pn.pane.Markdown("## Scenario"),
        pn.pane.Markdown("*Coming soon.*"),
    )
