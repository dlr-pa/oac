"""Config (Expert) tab: view and hand-edit the config as raw TOML text.

Deliberately does *not* stay in sync with other tabs while the user is
typing. The text box is only rebuilt when `state.config_generation`
changes (a fresh config was loaded/created, or loaded from this tab's
own "Apply to Config" button), matching how the Config tab's form is
rebuilt. Plain field edits made elsewhere only trigger `edited_config`,
which this tab ignores, so nothing changes underneath the user while
they're mid-edit here.

- "Reload from Config" discards any text edits, replacing the box with
  the current `state.edited_config` re-serialized to TOML.
- "Apply to Config" parses and validates the typed text exactly like the
  sidebar's "Load" button validates a file and, if structurally valid,
  makes it the new working `state.edited_config`. The sidebar's own
  load/validate status panes are used for reporting.
"""

import panel as pn

from .. import config_io

TITLE = """
### Edit configuration as text
This is an **editing** tab, and an alternative to the **Config** tab: instead
of clicking through cards and fields, the raw TOML text can be edited directly.
This can be significantly quicker for power users making many small changes
at once.

Unlike the Config tab, text typed here is a private scratchpad — it is
**not** applied automatically:

- **Apply to Config** parses and validates the text above and, if it's
  valid, makes it the new working configuration, immediately visible on
  every other tab. This runs the same validation as the sidebar's "Load"
  button.
- **Reload from Config** discards whatever you've typed here and reloads
  this box from the current working configuration — i.e. it throws away
  unsaved edits *in this box*, not your saved configuration file.

**⚠️ WARNING: Apply your edits before switching to another tab.** Edits made
elsewhere in the GUI (e.g. the Config or Aircraft tabs) will overwrite whatever
is typed here but not yet applied.
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

    code_editor = pn.widgets.CodeEditor(
        name="",
        value="",
        language="toml",
        sizing_mode="stretch_both",
        min_height=600,
    )

    reset_btn = pn.widgets.Button(name="Reload from Config", button_type="default")
    save_btn = pn.widgets.Button(name="Apply to Config", button_type="success")

    # text last known to match the working configuration (set whenever the
    # box is (re)loaded from state). Compared against on every keystroke to
    # derive config_text_dirty, rather than latching it permanently True on
    # first edit - so it clears itself if the user types back to a no-op.
    _baseline = [""]

    def _refresh(_event=None):
        has_config = state.edited_config is not None
        serialized = _serialize(state)
        # set before assigning code_editor.value so the value-change watcher
        # below sees a matching baseline and doesn't flag this as dirty.
        _baseline[0] = serialized
        code_editor.value = serialized
        code_editor.disabled = not has_config
        reset_btn.disabled = not has_config
        save_btn.disabled = not has_config
        empty_msg.visible = not has_config
        state.config_text_dirty = False

    def _on_text_change(_event=None):
        if state.edited_config is None:
            return
        state.config_text_dirty = code_editor.value != _baseline[0]

    def _on_reset(_event=None):
        _refresh()

    def _on_save(_event=None):
        if not state.edited_config:
            load_status.object = "⚠️ No configuration open."
            return

        load_status.object = "⏳ Validating…"
        validate_status.object = ""
        config, errors = config_io.parse_toml_text(code_editor.value)
        if errors:
            load_status.object = "\n".join(f"❌ {e}" for e in errors)
            return

        state.edited_config = config
        state.dirty = True
        state.config_generation += 1
        load_status.object = "ℹ️ Loaded edits from the Config (Expert) tab."

        # Run the full validation automatically, same as loading a file.
        _valid, message = config_io.run_full_validation(state)
        validate_status.object = message
        state.needs_revalidation = False

    reset_btn.on_click(_on_reset)
    save_btn.on_click(_on_save)
    code_editor.param.watch(_on_text_change, "value")

    # Rebuild only when a fresh config is loaded/created (config_generation) —
    # not on every field edit elsewhere (edited_config), so nothing changes
    # underneath the user while they're editing text here.
    state.param.watch(_refresh, "config_generation")
    _refresh()

    return pn.Column(
        pn.pane.Markdown(TITLE),
        pn.Row(reset_btn, save_btn),
        empty_msg,
        code_editor,
        sizing_mode="stretch_both",
        styles={"gap": "10px", "margin-top": "15px"},
    )
