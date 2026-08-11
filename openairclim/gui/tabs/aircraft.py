"""Aircraft tab: aircraft and fuel property configuration (placeholder)."""

import panel as pn


def panel(state):
    """Return the aircraft tab content.

    Args:
        state (AppState): Shared application state.
    """
    return pn.Column(
        pn.pane.Markdown("## Aircraft"),
        pn.pane.Markdown("*Coming soon.*"),
        styles={"margin-top": "15px"},
    )
