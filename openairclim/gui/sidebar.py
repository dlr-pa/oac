"""Sidebar: intro text, load/new, validation and run."""

from copy import deepcopy
from pathlib import Path

import panel as pn

from . import config_io
from .tabs import config as config_tab

INTRO_TEXT = """
Welcome to the OpenAirClim GUI! Here, you can create, load and edit config
files, visualise the emission inventories and scenarios, define aircraft
parameters, run OpenAirClim and visualise the results.

We welcome your feedback! Please reach out to openairclim@dlr.de.
"""


def panel(state):
    """Return the sidebar content.

    Args:
        state (AppState): Shared application state.

    Returns:
        pn.Column: Sidebar layout.
    """

    # load/create new config buttons
    load_btn = pn.widgets.Button(name="Load", button_type="primary")
    new_btn = pn.widgets.Button(name="New", button_type="default")
    load_status = pn.pane.Markdown("")

    # confirm yes/no for discarding changes
    confirm_msg = pn.pane.Markdown(
        "⚠️ You have unsaved edits to the current configuration. "
        "Continuing will discard them.",
        margin=(0, 0, 0, 10),
    )
    confirm_yes = pn.widgets.Button(name="Discard", button_type="danger")
    confirm_no = pn.widgets.Button(name="Cancel", button_type="default")
    confirm_row = pn.Column(
        confirm_msg, pn.Row(confirm_yes, confirm_no), visible=False, margin=(0, 0, 0, 0)
    )

    # config file status
    file_status = pn.pane.Markdown("", margin=(0, 0, 0, 10))
    dirty_status = pn.pane.Markdown("", margin=(0, 0, 0, 10))

    def _update_file_status(event=None):
        if state.edited_config is None:
            file_status.object = "*No configuration open.*"
        elif state.config_path:
            file_status.object = f"**File:** `{Path(state.config_path).name}`"
        else:
            file_status.object = "**File:** New (unsaved)"

    def _update_dirty_status(event=None):
        dirty_status.object = "🔴 Unsaved changes" if state.dirty else ""

    state.param.watch(_update_file_status, ["edited_config", "config_path"])
    state.param.watch(_update_dirty_status, "dirty")
    _update_file_status()
    _update_dirty_status()

    run_btn = pn.widgets.Button(name="Run", button_type="primary")
    run_status = pn.pane.Markdown("")

    pending_action = {"type": None}

    # ------------------------------------------------------------------
    # Load / New
    # ------------------------------------------------------------------

    def _do_load(config_path):
        """Validate the given file and load it as the working config.

        The working directory is automatically set to the folder that
        contains the config file, since all relative paths inside the
        config (inventory dir, response dir, etc.) are resolved against
        that location by OpenAirClim's core code.

        Args:
            config_path (str): Path to the config file to load.
        """
        # Derive working dir from the config file location before
        # validating, so that parse_and_check_structure can resolve
        # relative paths inside the file correctly.
        config_dir = str(Path(config_path).parent)
        state.working_dir = config_dir

        # ensure that the file can be parsed
        load_status.object = "⏳ Loading…"
        config, errors = config_io.parse_and_check_structure(
            state.working_dir, config_path
        )

        if errors:
            load_status.object = "\n".join(f"❌ {e}" for e in errors)
            return

        state.config_path = config_path
        state.edited_config = deepcopy(config)
        state.dirty = False
        state.config_generation += 1
        load_status.object = f"ℹ️ Loaded `{Path(config_path).name}`."

        # Run the full validation automatically on load
        _run_validation()

    def _do_new():
        """Start a blank configuration."""
        blank = config_io.blank_config()
        state.config_path = ""
        state.edited_config = deepcopy(blank)
        state.dirty = False
        state.config_generation += 1
        load_status.object = "ℹ️ Started a new blank configuration."

    def _request_load(event=None):
        """Ask user for config file path."""
        import tkinter as tk
        from tkinter import filedialog

        def _open_dialog():
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askopenfilename(
                title="Select configuration file",
                filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
                initialdir=state.working_dir or None,
            )
            root.destroy()
            return selected

        if state.dirty:
            pending_action["type"] = "load"
            pending_action["path_getter"] = _open_dialog
            confirm_row.visible = True
        else:
            selected = _open_dialog()
            if selected:
                _do_load(selected)

    def _request_new(event=None):
        if state.dirty:
            pending_action["type"] = "new"
            confirm_row.visible = True
        else:
            _do_new()

    def _on_confirm_yes(event):
        confirm_row.visible = False
        action = pending_action["type"]
        pending_action["type"] = None
        if action == "load":
            selected = pending_action.pop("path_getter")()
            if selected:
                _do_load(selected)
        elif action == "new":
            _do_new()

    def _on_confirm_no(event):
        confirm_row.visible = False
        pending_action["type"] = None
        pending_action.pop("path_getter", None)

    load_btn.on_click(_request_load)
    new_btn.on_click(_request_new)
    confirm_yes.on_click(_on_confirm_yes)
    confirm_no.on_click(_on_confirm_no)

    # ------------------------------------------------------------------
    # Validate / save
    # ------------------------------------------------------------------

    validate_btn = pn.widgets.Button(name="Validate", button_type="primary")
    save_btn = pn.widgets.Button(name="Save", button_type="success")
    validate_status = pn.pane.Markdown("")

    def _run_validation():
        """Run the full validation pipeline, in order:

        1. Required-field check — the same ⚠️ checks shown on the
           Config tab's card titles, run directly against the config
           dict. Cheap, no I/O. If anything's missing, stop here —
           OpenAirClim's own check below produces far less actionable
           errors (KeyErrors etc.) when basic fields are still empty.
        2. OpenAirClim's own full check (core.read_config.check_config,
           reused as-is via config_io.check_full_config): structure,
           aircraft/contrail setup, and that every referenced file
           actually exists.

        Updates validate_status either way.

        Returns:
            bool: True if the configuration is fully valid.
        """
        if not state.working_dir:
            validate_status.object = "⚠️ Select a working directory first."
            return False
        if not state.edited_config:
            validate_status.object = "⚠️ No configuration to validate yet."
            return False

        problems = config_tab.check_required_fields(state.edited_config)
        if problems:
            lines = ["### ⚠️ Fields missing or invalid\n"]
            lines += [f"- {status} **{title}**" for title, status in problems]
            validate_status.object = "\n".join(lines)
            return False

        try:
            config_io.check_full_config(state.working_dir, state.edited_config)
        except Exception as e:
            validate_status.object = f"### ❌ Configuration invalid\n\n{e}"
            return False

        validate_status.object = "### ✅ Configuration valid"
        return True

    def _on_validate(event=None):
        validate_status.object = "⏳ Validating…"
        _run_validation()

    def _on_save(event=None):
        if not state.edited_config:
            validate_status.object = "⚠️ No configuration to save yet."
            return

        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.asksaveasfilename(
            title="Save configuration as",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialdir=state.working_dir or None,
        )
        root.destroy()

        if not selected:
            return

        try:
            prepared = config_io.prepare_for_save(state.edited_config, state.working_dir)
            config_io.write_toml(prepared, selected)
        except Exception as e:
            validate_status.object = f"❌ Failed to save: {e}"
            return

        state.config_path = selected
        validate_status.object = f"✅ Saved to `{selected}`"
        state.dirty = False

    validate_btn.on_click(_on_validate)
    save_btn.on_click(_on_save)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _on_run(event=None):
        if state.dirty:
            run_status.object = (
                "⚠️ You have unsaved edits — "
                "save the configuration before running."
            )
            return
        if not state.config_path:
            run_status.object = "⚠️ Save the configuration before running."
            return
        if not state.working_dir:
            run_status.object = "⚠️ Select a working directory first."
            return

        run_btn.loading = True
        run_status.object = "⏳ Running OpenAirClim..."
        try:
            config_io.run_config(state.working_dir, state.config_path)
            run_status.object = "✅ Run completed."
        except Exception as e:
            run_status.object = f"❌ Run failed: {e}"
        finally:
            run_btn.loading = False

    run_btn.on_click(_on_run)

    # ------------------------------------------------------------------
    # Restore config status if a config was already loaded at launch
    # (e.g. via --config on the command line, loaded on the Config tab).
    # ------------------------------------------------------------------

    if state.config_path and state.edited_config is None:
        _do_load(state.config_path)

    return pn.Column(
        pn.pane.Markdown("## OpenAirClim"),
        pn.pane.Markdown(INTRO_TEXT),
        pn.layout.Divider(),
        pn.pane.Markdown("### Load / create new configuration"),
        pn.Row(load_btn, new_btn),
        confirm_row,
        load_status,
        pn.layout.Divider(),
        pn.pane.Markdown("### Validate & save configuration"),
        pn.Row(validate_btn, save_btn),
        validate_status,
        dirty_status,
        pn.layout.Divider(),
        pn.pane.Markdown("### Run OpenAirClim"),
        file_status,
        run_btn,
        run_status,
    )
