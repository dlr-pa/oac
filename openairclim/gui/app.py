"""Assemble the OpenAirClim GUI application."""

from pathlib import Path

import panel as pn

from . import sidebar
from .state import AppState
from .tabs import aircraft, config, config_text, results, scenario, inventories


def _wire_config_expert_dirty_notification(tabs, state, expert_index):
    """Warn the user about unapplied Config (Expert) text edits.

    Driven by `state.config_text_dirty` (set by the Config (Expert) tab on
    every keystroke): navigating away from that tab while dirty pops a
    notification pointing at "Apply to Config", which stays open (no
    auto-dismiss timeout) until either the user closes it themselves or
    navigates back to that tab.

    Args:
        tabs (pn.Tabs): The assembled tab bar.
        state (AppState): Shared application state.
        expert_index (int): Config (Expert)'s position within `tabs`.
    """
    # tracks the currently-open warning Notification (if any), so it can
    # be dismissed when the user comes back, and so we don't stack a
    # second toast on top of one that's already showing
    _notification = {"obj": None}

    def _dismiss_notification():
        notif = _notification["obj"]
        if notif is not None:
            notif.destroy()
            _notification["obj"] = None

    def _on_config_text_dirty_changed(event):
        if not event.new:
            _dismiss_notification()

    def _on_active_changed(event):
        entered_config_expert = event.new == expert_index
        left_config_expert = event.old == expert_index and event.new != expert_index

        if entered_config_expert:
            _dismiss_notification()
            return

        if left_config_expert and state.config_text_dirty and _notification["obj"] is None:
            if pn.state.notifications is not None:
                notif = pn.state.notifications.warning(
                    "You have unapplied text edits on the Config (Expert) "
                    "tab. They won't take effect elsewhere until you go "
                    "back and click 'Apply to Config'.",
                    duration=0,
                )
                _notification["obj"] = notif

                def _on_destroyed(destroy_event):
                    # covers the user dismissing the toast manually
                    if destroy_event.new and _notification["obj"] is notif:
                        _notification["obj"] = None

                notif.param.watch(_on_destroyed, "_destroyed")

    state.param.watch(_on_config_text_dirty_changed, "config_text_dirty")
    tabs.param.watch(_on_active_changed, "active")


def build_app(config_path=None, results_path=None, theme="default"):
    """Return the top-level Panel template.

    Args:
        config_path (str, optional): Config file to pre-load.
        results_path (str, optional): Output file to pre-load.
        theme (str, optional): Colour theme, "default" (light) or "dark".
            Defaults to "default".

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

    # tabs are prefixed as "editing" (✏️) and "viewing" (📊)
    config_expert_index = 1  # must match its position in the list below
    tabs = pn.Tabs(
        ("✏️ Config", config.panel(state)),
        ("✏️ Config (Expert)", config_text.panel(state, status_panes)),
        ("✏️ Aircraft", aircraft.panel(state)),
        ("📊 Inventories", inventories.panel(state)),
        ("📊 Scenario", scenario.panel(state)),
        ("📊 Results", results.panel(state)),
        # dynamic=True re-renders a tab's Bokeh view every time it's
        # switched to, rather than building it once. This breaks card
        # expand/collapse toggle after navigating to different tabs. Eager
        # rendering is fine since all tabs already have placeholders.
    )
    _wire_config_expert_dirty_notification(tabs, state, config_expert_index)

    template = pn.template.FastListTemplate(
        title="OpenAirClim",
        sidebar=[sidebar.panel(state, status_panes)],
        main=[tabs],
        theme=theme,
        # The built-in toggle switches theme via a full page reload, which
        # would wipe any unsaved edits. The theme is thus set once at launch
        # instead via "--theme"
        theme_toggle=False,
    )
    return template
