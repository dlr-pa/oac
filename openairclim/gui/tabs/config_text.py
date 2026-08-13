"""Config (Expert) tab: view and hand-edit the config as raw TOML text.

Deliberately does *not* stay in sync with other tabs while the user is
typing — the text box is only rebuilt when ``state.config_generation``
changes (a fresh config was loaded/created, or loaded from this tab's
own validate button), matching how the Config tab's form is rebuilt. Plain
field edits made elsewhere only trigger ``edited_config``, which this
tab ignores, so nothing changes underneath the user while they're
mid-edit here.

- "Reset" discards any text edits, replacing the box with the current
  ``state.edited_config`` re-serialized to TOML.
- "Validate" parses and validates the typed text exactly like the
  sidebar's "Load" button validates a file  and, if structurally valid,
  makes it the new working ``state.edited_config``. The sidebar's own
  load/validate status panes are used for reporting.
"""

import panel as pn

from .. import config_io

TITLE = """
### Edit configuration as text
Manually edit the working configuration below. **Reset** discards any
edits here and reloads this box from the current configuration.
**Validate** performs validation checks and, if successful, makes your
edited text the new working configuration.

**NOTE**: Any edits made here are not immediately reflected throughout
the GUI. Make sure you validate your changes before moving on to other
tabs or making changes elsewhere. Changes in other tabs will override
any unsaved changes here.
"""


def _serialize(state):
    """Return state.edited_config as TOML text, or "" if none is open.

    Args:
        state (AppState): Shared application state.

    Returns:
        str: TOML text.
    """
    if not state.edited_config:
        return ""
    prepared = config_io.prepare_for_save(state.edited_config, state.working_dir)
    return config_io.to_toml_string(prepared)


def panel(state, status_panes):
    """Return the Config (Expert) tab content.

    Args:
        state (AppState): Shared application state.
        status_panes (dict): "load"/"validate" Markdown panes shared
            with the sidebar (see ``sidebar.build_status_panes``).

    Returns:
        pn.Column: Tab layout.
    """
    load_status = status_panes["load"]
    validate_status = status_panes["validate"]

    empty_msg = pn.pane.Markdown(
        "⚠️ Create a new configuration or load an existing one "
        "from the sidebar to get started."
    )

    text_area = pn.widgets.TextAreaInput(
        name="",
        value="",
        placeholder="No configuration open.",
        sizing_mode="stretch_both",
        min_height=600,
        styles={"font-family": "monospace"},
    )

    reset_btn = pn.widgets.Button(name="Reset", button_type="default")
    save_btn = pn.widgets.Button(name="Validate", button_type="success")

    def _refresh(event=None):
        has_config = state.edited_config is not None
        text_area.value = _serialize(state)
        text_area.disabled = not has_config
        reset_btn.disabled = not has_config
        save_btn.disabled = not has_config
        empty_msg.visible = not has_config

    def _on_reset(event=None):
        _refresh()

    def _on_save(event=None):
        if not state.edited_config:
            load_status.object = "⚠️ No configuration open."
            return

        load_status.object = "⏳ Validating…"
        validate_status.object = ""
        config, errors = config_io.parse_toml_text(text_area.value)
        if errors:
            load_status.object = "\n".join(f"❌ {e}" for e in errors)
            return

        state.edited_config = config
        state.dirty = True
        state.config_generation += 1
        load_status.object = "ℹ️ Loaded edits from the Config (Expert) tab."

        # Run the full validation automatically, same as loading a file.
        valid, message = config_io.run_full_validation(state)
        validate_status.object = message

    reset_btn.on_click(_on_reset)
    save_btn.on_click(_on_save)

    # Rebuild only when a fresh config is loaded/created (config_generation) —
    # not on every field edit elsewhere (edited_config), so nothing changes
    # underneath the user while they're editing text here.
    state.param.watch(_refresh, "config_generation")
    _refresh()

    return pn.Column(
        pn.pane.Markdown(TITLE),
        pn.Row(reset_btn, save_btn),
        empty_msg,
        text_area,
        sizing_mode="stretch_both",
        styles={"gap": "10px", "margin-top": "15px"},
    )
