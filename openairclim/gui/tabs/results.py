"""Results tab: explore simulation output (placeholder)."""

import panel as pn


def panel(state):
    """Return the results tab content.

    Args:
        state (AppState): Shared application state.
    """
    return pn.Column(
        pn.pane.Markdown("## Results"),
        pn.pane.Markdown("*Coming soon.*"),
    )
