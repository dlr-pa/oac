"""Config tab: working directory and the full configuration form.

The form is rebuilt from scratch whenever `state.config_generation` changes
(i.e. a fresh config was loaded or created via the sidebar's Load/New buttons) -
individual field edits mutate the same `state.edited_config` dict in place and
only trigger it (via `_notify`), so they don't tear down and rebuild
the widgets.
"""

import panel as pn

from .. import config_io
from ..components.file_picker import FilePicker
from ..components.schema import submodel, literal_choices, field_description
from ...addon._premium import OAC_PREMIUM_AVAILABLE, LOW_SOOT_CASES

TITLE = """
### Edit configuration
This is an **editing** tab. Changes you make here update the shared working
configuration immediately, and are reflected across the rest of the GUI.

All variables are available within the cards below, grouped to match the
sections of the config file. A card's title shows a warning if any of its
required variables are invalid or not yet set. The full configuration's
validity can be checked at any time using the "Validate" button in the
sidebar.

If you prefer editing the config file directly as text, use the
**Config (Expert)** tab instead.
"""


# Sentinel option used by optional single-file Select widgets, since
# Select requires `value` to be one of `options` — there's no built-in
# "nothing selected" state.
_NONE_OPTION = "— none —"

# Suggested response and background filenames, matching the example config.
# These are used to pre-select a sensible default once a folder is chosen, but
# only if that file actually exists there.
RESPONSE_FILE_DEFAULTS = {
    "H2O": "resp_RF.nc",
    "O3": "resp_RF_O3.nc",
    "CH4": "resp_ch4.nc",
    "cont": "resp_cont_lf.nc",
    "SWV": "ch4_for_swv_calc.nc",
}
BACKGROUND_FILE_DEFAULTS = {
    "CO2": "co2_bg.nc",
    "CH4": "ch4_bg.nc",
    "N2O": "n2o_bg.nc",
}

# placeholder for the Background/Responses folder pickers. Unlike inventories,
# leaving this blank is meaningful, because it calls back to the shread data
# repository cache
_REPOSITORY_DIR_PLACEHOLDER = "Leave blank to use oac's data cache..."

# Width/wrap hint used for the sub-columns inside the wide Background and
# Responses cards, so their content forms roughly two columns instead of
# one long vertical stack.
_SUBCOL_STYLES = {"flex": "1 1 45%", "min-width": "260px"}


def _resolve_dir_or_none(working_dir, dir_str):
    """Resolve a directory string, or None if nothing has been chosen yet.

    Used instead of `config_io.resolve_dir` directly wherever a blank
    `dir_str` must stay blank (not silently fall back to
    `working_dir`, or to "." if that's empty too) — otherwise a fresh
    config would appear to have a valid folder selected everywhere,
    when really the user hasn't picked one.

    Args:
        working_dir (str): Project working directory.
        dir_str (str): Directory path, or "" if unset.

    Returns:
        Path or None: Resolved absolute path, or None if dir_str is empty.
    """
    if not dir_str:
        return None
    return config_io.resolve_dir(working_dir, dir_str)


def _resolve_dir_or_default(working_dir, dir_str):
    """Resolve a directory string for the Background/Responses sections,
    falling back to OpenAirClim's shared repository-data cache when
    dir_str is blank.

    Args:
        working_dir (str): Project working directory.
        dir_str (str): Directory path, or "" if unset.

    Returns:
        Path: Resolved absolute path — dir_str resolved against
            working_dir, or the shared repository-data cache directory.
    """
    if dir_str:
        return config_io.resolve_dir(working_dir, dir_str)
    return config_io.default_repository_dir()


def _build_default_dir_hint(dir_picker):
    """Build a hint pane showing where a blank dir will resolve to. Used by the
    background and responses cards, where a blank folder picker resolves to the
    shared data cache.

    Args:
        dir_picker (FilePicker): The section's folder picker widget.

    Returns:
        tuple: (pn.pane.Markdown, callable) — the hint pane, and a
            zero-argument function that updates its text; call this
            whenever dir_picker.path changes.
    """
    hint = pn.pane.Markdown(
        "", styles={"font-size": "0.85em"}, margin=(0, 5, 10, 5), visible=False
    )

    def _refresh():
        if dir_picker.path:
            hint.visible = False
        else:
            hint.object = (
                f"*Using data cache at: `{config_io.default_repository_dir()}`*"
            )
            hint.visible = True

    return hint, _refresh


# A MultiChoice with plain-string options uses the same string as both
# the display label and the stored value — so a selected-but-missing
# file needs its warning marker baked into the "value" for it to show
# up as selected at all. These two helpers decorate a filename for
# display/selection, and strip it back off before writing to the config
# (which must only ever hold real filenames).
_MISSING_PREFIX = "⚠️ "
_MISSING_SUFFIX = " (not found here)"


def _missing_label(filename):
    """Return the decorated option/value string for a missing file."""
    return f"{_MISSING_PREFIX}{filename}{_MISSING_SUFFIX}"


def _strip_missing_label(value):
    """Undo _missing_label — return value unchanged if it wasn't decorated."""
    if value.startswith(_MISSING_PREFIX) and value.endswith(_MISSING_SUFFIX):
        return value[len(_MISSING_PREFIX) : -len(_MISSING_SUFFIX)]
    return value


def _card(title, content):
    """Wrap section content in a collapsible card that fills its row.

    Args:
        title (str): Card title.
        content: Panel object to place inside the card.

    Returns:
        pn.Card: The wrapped section.
    """
    return pn.Card(
        content,
        title=title,
        collapsible=True,
        collapsed=True,
        sizing_mode="stretch_width",
        # force explicit collapsed card width
        styles={"flex": "1 1 0%", "min-width": "0"},
    )


# ======================================================================
# Small shared widget builders
# ======================================================================


def _build_int_list_field(parent, key, label, notify):
    """Build an "add a value" + multi-select widget for a list of ints.

    The MultiChoice's *options* accumulate every value ever added; its
    *value* (the subset currently selected/"checked") is written back to
    ``parent[key]`` directly. Deselecting a tag without re-adding it is
    how the user removes a value from the list.

    Args:
        parent (dict): Dict containing the list (e.g. edited["metrics"]).
        key (str): Key of the list within parent (e.g. "H").
        label (str): Display label.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: The widget group.
    """
    current = [str(v) for v in parent.get(key, [])]
    select = pn.widgets.MultiChoice(
        name=label,
        options=list(current),
        value=list(current),
        sizing_mode="stretch_width",
    )
    add_input = pn.widgets.IntInput(name="Add value", value=0, width=110)
    add_btn = pn.widgets.Button(name="Add", width=50, margin=(25, 0, 0, 6))

    def _sync(_event=None):
        parent[key] = [int(v) for v in select.value]
        notify()

    def _on_add(_event=None):
        val = str(add_input.value)
        opts = list(select.options)
        if val not in opts:
            opts.append(val)
            opts.sort(key=int)
            select.options = opts
        if val not in select.value:
            select.value = select.value + [val]  # triggers _sync via watcher
        else:
            _sync()  # already selected — nothing changed, sync manually

    select.param.watch(_sync, "value")
    add_btn.on_click(_on_add)

    return pn.Column(select, pn.Row(add_input, add_btn))


# ======================================================================
# Section builders (one per config-tab card)
# ======================================================================


def _build_species_section(edited, notify):
    """Build the Species section.

    Args:
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    species = edited["species"]
    species_model = submodel("species")

    inv_select = pn.widgets.MultiChoice(
        name="Input species (from inventories)",
        options=literal_choices(species_model, "inv"),
        value=list(species["inv"]),
        description=field_description(species_model, "inv"),
    )
    out_select = pn.widgets.MultiChoice(
        name="Output species (responses)",
        options=literal_choices(species_model, "out"),
        value=list(species["out"]),
        description=field_description(species_model, "out"),
    )
    nox_select = pn.widgets.Select(
        name="Assumed NOx species in inventories",
        options=literal_choices(species_model, "nox"),
        value=species["nox"],
        description=field_description(species_model, "nox"),
    )

    def _on_inv_changed(event):
        species["inv"] = list(event.new)
        notify()

    def _on_out_changed(event):
        species["out"] = list(event.new)
        notify()

    def _on_nox_changed(event):
        species["nox"] = event.new
        notify()

    inv_select.param.watch(_on_inv_changed, "value")
    out_select.param.watch(_on_out_changed, "value")
    nox_select.param.watch(_on_nox_changed, "value")

    return pn.Column(inv_select, out_select, nox_select)


def _build_time_section(_state, edited, notify):
    """Build the Simulation period section.

    Args:
        _state (AppState): Shared application state (currently unused).
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    time_cfg = edited["time"]
    t_start, t_end, t_step = time_cfg["range"]

    start_input = pn.widgets.IntInput(name="Start year", value=int(t_start))
    end_input = pn.widgets.IntInput(
        name="End year (exclusive)",
        value=int(t_end),
        description="Note that the end year is *exclusive*. For example, if "
        "2051 is selected, the last year simulated will be 2050.",
    )
    # fixed at 1 — core.config_model._TimeConfig rejects any other step,
    # since the response calculations assume annual time steps.
    step_input = pn.widgets.IntInput(
        name="Step",
        value=int(t_step),
        start=1,
        disabled=True,
        description="Fixed at 1 year — other step sizes aren't yet "
        "supported by OpenAirClim's response calculations.",
    )
    warning = pn.pane.Markdown("")

    def _on_time_changed(_event=None):
        start, end, step = start_input.value, end_input.value, step_input.value
        if end <= start:
            warning.object = "⚠️ End year must be after the start year."
            notify()
            return
        warning.object = ""
        time_cfg["range"] = [start, end, step]
        notify()

    start_input.param.watch(_on_time_changed, "value")
    end_input.param.watch(_on_time_changed, "value")
    step_input.param.watch(_on_time_changed, "value")

    return pn.Column(
        start_input,
        end_input,
        step_input,
        warning,
    )


def _build_time_evolution_section(state, edited, notify):
    """Build the time evolution section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    time_cfg = edited["time"]

    dir_picker = FilePicker(
        label="Folder (for time evolution file)",
        directory=True,
        description=field_description(submodel("time"), "dir"),
    )
    if time_cfg["dir"]:
        dir_resolved = config_io.resolve_dir(state.working_dir, time_cfg["dir"])
        time_cfg["dir"] = str(dir_resolved)
        dir_picker.set_path(str(dir_resolved))

    file_select = pn.widgets.Select(
        name="Time evolution file (optional)",
        options=[_NONE_OPTION],
        value=_NONE_OPTION,
        description=field_description(submodel("time"), "file"),
    )
    clear_btn = pn.widgets.Button(name="Clear", width=70, margin=(24, 10, 0, 6))

    def _refresh_time_file():
        resolved = _resolve_dir_or_none(state.working_dir, dir_picker.path)
        files = config_io.list_nc_files(resolved) if resolved is not None else []
        file_select.options = [_NONE_OPTION] + files
        current = time_cfg.get("file")
        file_select.value = current if current in files else _NONE_OPTION

    def _on_dir_changed(event):
        time_cfg["dir"] = event.new
        _refresh_time_file()
        notify()

    def _on_file_changed(event):
        # "time.file" must not be defined at all in the config when
        # nothing is selected — not even as an empty string.
        if event.new and event.new != _NONE_OPTION:
            time_cfg["file"] = event.new
        else:
            time_cfg.pop("file", None)
        notify()

    def _on_clear(_event=None):
        file_select.value = _NONE_OPTION

    dir_picker.param.watch(_on_dir_changed, "path")
    clear_btn.on_click(_on_clear)

    # Populate before wiring the "value changed" watcher below, so
    # setting the initial selection doesn't itself count as an edit
    # (and incorrectly mark the config dirty right after loading it).
    _refresh_time_file()
    file_select.param.watch(_on_file_changed, "value")

    return pn.Column(dir_picker, pn.Row(file_select, clear_btn))


def _build_dir_files_widgets(state, section, label, notify, initial_files=None):
    """Build a (dir picker + files multi-select + status) group.

    Bound to ``section["dir"]`` / ``section["files"]`` — works for any
    dict shaped like ``{"dir": ..., "files": [...]}``, e.g.
    ``edited["inventories"]`` or ``edited["inventories"]["base"]``.

    Args:
        state (AppState): Shared application state.
        section (dict): Sub-dict with "dir" and "files" keys, mutated
            in place.
        label (str): Label for the folder picker.
        notify (callable): Called after every edit.
        initial_files (list, optional): Files to pre-select if present
            among the scanned options.

    Returns:
        pn.Column: The widget group.
    """
    dir_picker = FilePicker(label=label, directory=True)
    existing_dir = section.get("dir", "")
    if existing_dir:
        # Canonicalize to absolute now, so the stored path stays correct
        # even if state.working_dir changes (or is set) later on.
        resolved = config_io.resolve_dir(state.working_dir, existing_dir)
        section["dir"] = str(resolved)
        dir_picker.set_path(str(resolved))

    files_select = pn.widgets.MultiChoice(name="Files", options=[], value=[])
    status = pn.pane.Markdown("")

    def _refresh(initial_selection=None):
        """Rescan the folder and refresh the file list.

        Selected files that aren't found in the folder stay selected —
        shown with a ⚠️ marker — instead of being silently dropped, so
        the widget never disagrees with what's actually recorded in
        `section["files"]` (e.g. right after loading a config whose
        files no longer exist here).

        Args:
            initial_selection (list, optional): Files to select
                initially — used only for the initial build, from the
                loaded/blank config's file list. Later refreshes (e.g.
                after changing the folder) fall back to whatever is
                currently selected in the widget.
        """
        resolved = _resolve_dir_or_none(state.working_dir, dir_picker.path)
        files = config_io.list_nc_files(resolved) if resolved is not None else []

        if initial_selection is not None:
            selected = list(initial_selection)
        else:
            # Widget values for missing files are decorated (see
            # _missing_label) — strip that back off to get real
            # filenames before comparing against the rescanned folder.
            selected = [_strip_missing_label(v) for v in files_select.value]
        missing = [f for f in selected if f not in files]

        files_select.options = list(files) + [_missing_label(f) for f in missing]
        files_select.value = [f if f in files else _missing_label(f) for f in selected]

        if resolved is None:
            status.object = ""
        else:
            status.object = "" if files else "⚠️ No .nc files found."

    def _on_dir_changed(event):
        # Keep the absolute path during editing — converted to relative
        # only at save time, in config_io.prepare_for_save.
        section["dir"] = event.new
        _refresh()
        notify()

    def _on_files_changed(event):
        # event.new may include decorated "missing" values (see
        # _missing_label) — only real filenames get written to config.
        section["files"] = [_strip_missing_label(v) for v in event.new]
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")

    # Populate before wiring the "value changed" watcher below, so
    # setting the initial selection doesn't itself count as an edit
    # (and incorrectly mark the config dirty right after loading it).
    _refresh(initial_selection=list(initial_files or []))
    files_select.param.watch(_on_files_changed, "value")

    return pn.Column(dir_picker, files_select, status)


def _build_inventories_section(state, edited, notify):
    """Build the Emission inventories section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    inv = edited["inventories"]

    main_widgets = _build_dir_files_widgets(
        state, inv, "Folder", notify, initial_files=inv["files"]
    )

    rtb_checkbox = pn.widgets.Checkbox(
        name="Relative to base", value=bool(inv.get("rel_to_base", False))
    )

    base_widgets = _build_dir_files_widgets(
        state, inv["base"], "Base folder", notify, initial_files=inv["base"]["files"]
    )
    # Only relevant when rel_to_base is True — start hidden/shown to match.
    base_section = pn.Column(base_widgets, visible=rtb_checkbox.value)

    def _on_rtb_changed(event):
        inv["rel_to_base"] = event.new
        base_section.visible = event.new
        notify()

    rtb_checkbox.param.watch(_on_rtb_changed, "value")

    return pn.Column(main_widgets, rtb_checkbox, base_section)


def _build_background_section(state, edited, notify):
    """Build the Background section.

    One shared folder, plus per-species (CO2, CH4, N2O) file and
    scenario dropdowns. The scenario dropdown for a species is
    populated by opening its selected NetCDF file and reading the
    data variable names — each scenario (e.g. "SSP2-4.5") is stored as
    a data variable, not declared anywhere in the config itself.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    bg = edited["background"]

    dir_picker = FilePicker(
        label="Folder", directory=True, placeholder=_REPOSITORY_DIR_PLACEHOLDER
    )
    existing_dir = bg.get("dir", "")
    if existing_dir:
        dir_resolved = config_io.resolve_dir(state.working_dir, existing_dir)
        bg["dir"] = str(dir_resolved)
        dir_picker.set_path(str(dir_resolved))

    default_dir_hint, _refresh_default_dir_hint = _build_default_dir_hint(dir_picker)

    refresh_funcs = []
    species_columns = []

    def _make_species_widgets(species_key, label):
        """Build the file + scenario dropdown pair for one species."""
        sub = bg[species_key]
        file_select = pn.widgets.Select(
            name=f"{label} file", options=[_NONE_OPTION], value=_NONE_OPTION
        )
        scenario_select = pn.widgets.Select(
            name=f"{label} scenario", options=[_NONE_OPTION], value=_NONE_OPTION
        )

        def _refresh_scenario():
            # written to `sub` directly because if the widget's value happens
            # to already equal what we're about to set it to, no change event
            # fires
            resolved = _resolve_dir_or_default(state.working_dir, dir_picker.path)
            if file_select.value == _NONE_OPTION:
                sub["scenario"] = ""
                scenario_select.options = [_NONE_OPTION]
                scenario_select.value = _NONE_OPTION
                return
            variables = config_io.list_nc_data_vars(resolved / file_select.value)
            scenario_select.options = [_NONE_OPTION] + variables
            current = sub.get("scenario", "")
            if current in variables:
                scenario_select.value = current
            else:
                sub["scenario"] = ""
                scenario_select.value = _NONE_OPTION

        def _refresh_file():
            # written to `sub` directly for the same reason as
            # _refresh_scenario above
            resolved = _resolve_dir_or_default(state.working_dir, dir_picker.path)
            files = config_io.list_nc_files(resolved)
            file_select.options = [_NONE_OPTION] + files
            current = sub.get("file", "")
            default_filename = BACKGROUND_FILE_DEFAULTS.get(species_key, "")
            if current in files:
                file_select.value = current
            elif not current and default_filename in files:
                # Suggest the example config's filename if it's present
                # and nothing has been chosen yet.
                sub["file"] = default_filename
                file_select.value = default_filename
            else:
                sub["file"] = ""
                file_select.value = _NONE_OPTION
            # _refresh_scenario also fires via the watcher below if the
            # value actually changed; called explicitly too, to cover
            # the case where it doesn't (e.g. value stays _NONE_OPTION).
            _refresh_scenario()

        def _on_file_changed(event):
            sub["file"] = event.new if event.new != _NONE_OPTION else ""
            _refresh_scenario()
            notify()

        def _on_scenario_changed(event):
            sub["scenario"] = event.new if event.new != _NONE_OPTION else ""
            notify()

        # Populate before wiring the "value changed" watchers below, so
        # setting the initial selection doesn't itself count as an edit
        # (and incorrectly mark the config dirty right after loading it).
        _refresh_file()
        file_select.param.watch(_on_file_changed, "value")
        scenario_select.param.watch(_on_scenario_changed, "value")

        refresh_funcs.append(_refresh_file)

        return pn.Column(
            pn.pane.Markdown(f"**{label}**"),
            file_select,
            scenario_select,
            styles=_SUBCOL_STYLES,
        )

    species_columns.append(_make_species_widgets("CO2", "CO₂"))
    species_columns.append(_make_species_widgets("CH4", "CH₄"))
    species_columns.append(_make_species_widgets("N2O", "N₂O"))

    def _on_dir_changed(event):
        bg["dir"] = event.new
        _refresh_default_dir_hint()
        for refresh in refresh_funcs:
            refresh()
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")
    _refresh_default_dir_hint()

    return pn.Column(
        dir_picker,
        default_dir_hint,
        pn.FlexBox(*species_columns, styles={"gap": "10px"}),
    )


def _build_responses_section(state, edited, notify):
    """Build the Responses section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    resp = edited["responses"]

    dir_picker = FilePicker(
        label="Folder", directory=True, placeholder=_REPOSITORY_DIR_PLACEHOLDER
    )
    existing_dir = resp.get("dir", "")
    if existing_dir:
        dir_resolved = config_io.resolve_dir(state.working_dir, existing_dir)
        resp["dir"] = str(dir_resolved)
        dir_picker.set_path(str(dir_resolved))

    default_dir_hint, _refresh_default_dir_hint = _build_default_dir_hint(dir_picker)

    refresh_funcs = []

    def _make_file_select(label, sub_dict, default_filename):
        """Build a single response-file dropdown bound to sub_dict["file"]."""
        select = pn.widgets.Select(
            name=label, options=[_NONE_OPTION], value=_NONE_OPTION
        )

        def _refresh():
            # written to `sub` directly because if the widget's value happens
            # to already equal what we're about to set it to, no change event
            # fires
            resolved = _resolve_dir_or_default(state.working_dir, dir_picker.path)
            files = config_io.list_nc_files(resolved)
            select.options = [_NONE_OPTION] + files
            current = sub_dict.get("file", "")
            if current in files:
                select.value = current
            elif not current and default_filename in files:
                # Suggest the example config's filename if it's present
                # and nothing has been chosen yet.
                sub_dict["file"] = default_filename
                select.value = default_filename
            else:
                sub_dict["file"] = ""
                select.value = _NONE_OPTION

        def _on_change(event):
            sub_dict["file"] = event.new if event.new != _NONE_OPTION else ""
            notify()

        # Populate before wiring the "value changed" watcher below, so
        # setting the initial selection doesn't itself count as an edit
        # (and incorrectly mark the config dirty right after loading it).
        _refresh()
        select.param.watch(_on_change, "value")
        refresh_funcs.append(_refresh)
        return select

    h2o_select = _make_file_select(
        "H₂O response file", resp["H2O"]["rf"], RESPONSE_FILE_DEFAULTS["H2O"]
    )
    o3_select = _make_file_select(
        "O₃ response file", resp["O3"]["rf"], RESPONSE_FILE_DEFAULTS["O3"]
    )
    ch4_select = _make_file_select(
        "CH₄ response file", resp["CH4"]["tau"], RESPONSE_FILE_DEFAULTS["CH4"]
    )
    cont_select = _make_file_select(
        "Contrail response file", resp["cont"]["resp"], RESPONSE_FILE_DEFAULTS["cont"]
    )
    swv_select = _make_file_select(
        "SWV CH₄ profile file", resp["SWV"], RESPONSE_FILE_DEFAULTS["SWV"]
    )

    # Only offer real case names if openairclim_premium is actually
    # installed — otherwise there's nothing valid to compute with, so
    # the dropdown just shows the "none selected" sentinel. If a value
    # was already set (e.g. loaded from a config saved somewhere
    # premium *was* available), it's preserved even though it can't be
    # picked here — only an explicit change through this widget writes
    # to resp["cont"]["low_soot_case"].
    if OAC_PREMIUM_AVAILABLE and LOW_SOOT_CASES:
        low_soot_options = [_NONE_OPTION] + sorted(LOW_SOOT_CASES)
    else:
        low_soot_options = [_NONE_OPTION]

    current_low_soot = resp["cont"].get("low_soot_case", "")
    low_soot_select = pn.widgets.Select(
        name="Low soot case (requires OpenAirClim Premium)",
        options=low_soot_options,
        value=(
            current_low_soot if current_low_soot in low_soot_options else _NONE_OPTION
        ),
        description=field_description(submodel("responses.cont"), "low_soot_case"),
    )

    def _on_low_soot_changed(event):
        # "low_soot_case" must not be defined at all when nothing's
        # selected — not even as an empty string (same convention as
        # "time.file", see _build_time_evolution_section).
        if event.new and event.new != _NONE_OPTION:
            resp["cont"]["low_soot_case"] = event.new
        else:
            resp["cont"].pop("low_soot_case", None)
        notify()

    low_soot_select.param.watch(_on_low_soot_changed, "value")

    def _on_dir_changed(event):
        resp["dir"] = event.new
        _refresh_default_dir_hint()
        for refresh in refresh_funcs:
            refresh()
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")
    _refresh_default_dir_hint()

    # ---- CO2 / CH4 method & attribution dropdowns ----------------------
    # response_grid isn't shown — it's filled in via DEFAULT_CONFIG and
    # isn't something the user needs to set directly.
    # CH4's attribution options are identical to CO2's (see config_model.py),
    # so both dropdowns share the same choice set, read once here.
    rf_attr_options = literal_choices(submodel("responses.CO2.rf"), "attr")
    co2_conc_method = pn.widgets.Select(
        name="CO₂ concentration method",
        options=literal_choices(submodel("responses.CO2.conc"), "method"),
        value=resp["CO2"]["conc"]["method"],
        description=field_description(submodel("responses.CO2.conc"), "method"),
    )
    co2_rf_method = pn.widgets.Select(
        name="CO₂ RF method",
        options=literal_choices(submodel("responses.CO2.rf"), "method"),
        value=resp["CO2"]["rf"]["method"],
        description=field_description(submodel("responses.CO2.rf"), "method"),
    )
    co2_rf_attr = pn.widgets.Select(
        name="CO₂ RF attribution",
        options=rf_attr_options,
        value=resp["CO2"]["rf"]["attr"],
        description=field_description(submodel("responses.CO2.rf"), "attr"),
    )
    ch4_rf_attr = pn.widgets.Select(
        name="CH₄ RF attribution",
        options=rf_attr_options,
        value=resp["CH4"]["rf"]["attr"],
        description=field_description(submodel("responses.CH4.rf"), "attr"),
    )

    def _on_co2_conc_method(event):
        resp["CO2"]["conc"]["method"] = event.new
        notify()

    def _on_co2_rf_method(event):
        resp["CO2"]["rf"]["method"] = event.new
        notify()

    def _on_co2_rf_attr(event):
        resp["CO2"]["rf"]["attr"] = event.new
        notify()

    def _on_ch4_rf_attr(event):
        resp["CH4"]["rf"]["attr"] = event.new
        notify()

    co2_conc_method.param.watch(_on_co2_conc_method, "value")
    co2_rf_method.param.watch(_on_co2_rf_method, "value")
    co2_rf_attr.param.watch(_on_co2_rf_attr, "value")
    ch4_rf_attr.param.watch(_on_ch4_rf_attr, "value")

    return pn.Column(
        dir_picker,
        default_dir_hint,
        pn.FlexBox(
            pn.Column(
                pn.pane.Markdown("**CO₂**"),
                co2_conc_method,
                co2_rf_method,
                co2_rf_attr,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**H₂O**"),
                h2o_select,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**O₃**"),
                o3_select,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**CH₄**"),
                ch4_select,
                ch4_rf_attr,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**Contrails**"),
                cont_select,
                low_soot_select,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**SWV**"),
                swv_select,
                styles=_SUBCOL_STYLES,
            ),
            styles={"gap": "10px"},
        ),
    )


def _build_temperature_section(edited, notify):
    """Build the Temperature section.

    Args:
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    temp = edited["temperature"]
    temperature_model = submodel("temperature")
    efficacy_species = [
        f for f in temperature_model.model_fields if f not in ("method", "CO2")
    ]

    method_select = pn.widgets.Select(
        name="Method",
        options=["Boucher&Reddy"],
        value=temp.get("method", "Boucher&Reddy"),
    )
    lambda_input = pn.widgets.FloatInput(
        name="CO2 climate sensitivity (lambda)",
        value=float(temp["CO2"]["lambda"]),
        description=field_description(submodel("temperature.CO2"), "lambda_"),
    )

    def _on_method_changed(event):
        temp["method"] = event.new
        notify()

    def _on_lambda_changed(event):
        temp["CO2"]["lambda"] = event.new
        notify()

    method_select.param.watch(_on_method_changed, "value")
    lambda_input.param.watch(_on_lambda_changed, "value")

    efficacy_widgets = []
    for species in efficacy_species:
        fi = pn.widgets.FloatInput(
            name=f"{species} efficacy",
            value=float(temp[species]["efficacy"]),
            description=field_description(temperature_model, species),
        )

        def _make_handler(sp):
            def _on_change(event):
                temp[sp]["efficacy"] = event.new
                notify()

            return _on_change

        fi.param.watch(_make_handler(species), "value")
        efficacy_widgets.append(fi)

    return pn.Column(method_select, lambda_input, *efficacy_widgets)


def _build_metrics_section(edited, notify, run_metrics_checkbox):
    """Build the Metrics section.

    Hidden behind an info message unless output.run_metrics is enabled
    (tracked live via run_metrics_checkbox from the Output section).

    Args:
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.
        run_metrics_checkbox (pn.widgets.Checkbox): The "Calculate
            climate metrics" checkbox from the Output section.

    Returns:
        pn.Column: Section content.
    """
    metrics = edited["metrics"]

    types_select = pn.widgets.MultiChoice(
        name="Metric types",
        options=literal_choices(submodel("metrics"), "types"),
        value=list(metrics.get("types", [])),
    )

    def _on_types_changed(event):
        metrics["types"] = list(event.new)
        notify()

    types_select.param.watch(_on_types_changed, "value")

    h_field = _build_int_list_field(metrics, "H", "Time horizon (H)", notify)
    t0_field = _build_int_list_field(metrics, "t_0", "Start time (t_0)", notify)

    metrics_content = pn.Column(
        types_select, h_field, t0_field, visible=run_metrics_checkbox.value
    )
    info_pane = pn.pane.Markdown(
        "ℹ️ Enable **Calculate climate metrics** in the Output section "
        "to configure metrics.",
        visible=not run_metrics_checkbox.value,
    )

    def _on_run_metrics_changed(event):
        metrics_content.visible = event.new
        info_pane.visible = not event.new

    run_metrics_checkbox.param.watch(_on_run_metrics_changed, "value")

    return pn.Column(info_pane, metrics_content)


def _build_parametric_section(edited, notify):
    """Build the Parametric section.

    Args:
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    param_cfg = edited["parametric"]

    # Field order on the model doubles as display order in the form.
    parametric_species = [
        f for f in submodel("parametric").model_fields if f != "enabled"
    ]

    enabled_cb = pn.widgets.Checkbox(
        name="Enabled",
        value=bool(param_cfg["enabled"]),
    )

    ratio_widgets = []
    for species in parametric_species:
        fi = pn.widgets.FloatInput(
            name=f"{species} ATR20 ratio", value=float(param_cfg[species])
        )

        def _make_handler(sp):
            def _on_change(event):
                param_cfg[sp] = event.new
                notify()

            return _on_change

        fi.param.watch(_make_handler(species), "value")
        ratio_widgets.append(fi)

    ratios_col = pn.Column(*ratio_widgets, visible=enabled_cb.value)

    def _on_enabled_changed(event):
        param_cfg["enabled"] = event.new
        ratios_col.visible = event.new
        notify()

    enabled_cb.param.watch(_on_enabled_changed, "value")

    return pn.Column(enabled_cb, ratios_col)


def _build_output_section(state, edited, notify):
    """Build the Output section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        tuple: (pn.Column section content, run_metrics Checkbox widget).
            The checkbox is returned separately so the Metrics section
            can watch it and show/hide accordingly.
    """
    out = edited["output"]

    dir_picker = FilePicker(label="Output folder", directory=True)
    if out["dir"]:
        dir_resolved = config_io.resolve_dir(state.working_dir, out["dir"])
        out["dir"] = str(dir_resolved)
        dir_picker.set_path(str(dir_resolved))

    name_input = pn.widgets.TextInput(name="Output file name", value=out["name"])

    run_oac_cb = pn.widgets.Checkbox(name="Run OpenAirClim", value=bool(out["run_oac"]))
    run_metrics_cb = pn.widgets.Checkbox(
        name="Calculate climate metrics", value=bool(out["run_metrics"])
    )
    run_plots_cb = pn.widgets.Checkbox(
        name="Generate output plots", value=bool(out["run_plots"])
    )
    overwrite_cb = pn.widgets.Checkbox(
        name="Overwrite existing output", value=bool(out["overwrite"])
    )

    def _on_dir_changed(event):
        out["dir"] = event.new
        notify()

    def _on_name_changed(event):
        out["name"] = event.new
        notify()

    def _make_bool_handler(key):
        def _on_change(event):
            out[key] = event.new
            notify()

        return _on_change

    dir_picker.param.watch(_on_dir_changed, "path")
    name_input.param.watch(_on_name_changed, "value")
    run_oac_cb.param.watch(_make_bool_handler("run_oac"), "value")
    run_metrics_cb.param.watch(_make_bool_handler("run_metrics"), "value")
    run_plots_cb.param.watch(_make_bool_handler("run_plots"), "value")
    overwrite_cb.param.watch(_make_bool_handler("overwrite"), "value")

    pn_col = pn.Column(
        dir_picker, name_input, run_oac_cb, run_metrics_cb, run_plots_cb, overwrite_cb
    )
    return pn_col, run_metrics_cb


# ======================================================================
# Full form + tab layout
# ======================================================================


def _build_form(state):
    """Build the full configuration form: one card per section.

    Mutates ``state.edited_config`` in place as the user makes changes.
    Cards are laid out in fixed rows (rather than a free-wrapping grid)
    so that Background and Responses — which have more content — can
    span a wider, two-card row instead of a narrow one.

    Args:
        state (AppState): Shared application state. ``state.edited_config``
            must already hold the dict to edit (set by the sidebar's
            Load/New handlers before bumping ``config_generation``).

    Returns:
        pn.Column: Rows of section cards.
    """
    edited = state.edited_config

    # base_title -> pn.Card — used by _refresh_statuses to append/clear
    # each card's status icon on every edit, looking up its check
    # function from the shared config_io.CARD_CHECKS registry.
    cards = {}

    def _register(title, content):
        card = _card(title, content)
        cards[title] = card
        return card

    def _refresh_statuses():
        for title, card in cards.items():
            status = config_io.CARD_CHECKS[title](edited)
            card.title = f"{title} {status}" if status else title

    def _notify():
        state.dirty = True
        # edited_config is mutated in place, so a plain reassignment
        # wouldn't fire watchers (same object, equal by identity).
        # Trigger explicitly so other tabs can react to live edits.
        state.param.trigger("edited_config")
        _refresh_statuses()

    # Output is built first since Metrics needs to watch its
    # run_metrics checkbox to decide whether to show its own content.
    output_panel, run_metrics_checkbox = _build_output_section(state, edited, _notify)

    row_styles = {"gap": "10px"}

    row1 = pn.Row(
        _register("Species", _build_species_section(edited, _notify)),
        _register("Simulation period", _build_time_section(state, edited, _notify)),
        _register(
            "Time evolution (optional)",
            _build_time_evolution_section(state, edited, _notify),
        ),
        _register(
            "Emission inventories", _build_inventories_section(state, edited, _notify)
        ),
        sizing_mode="stretch_width",
        styles=row_styles,
    )
    row2 = pn.Row(
        _register("Temperature", _build_temperature_section(edited, _notify)),
        _register(
            "Metrics", _build_metrics_section(edited, _notify, run_metrics_checkbox)
        ),
        _register("Parametric", _build_parametric_section(edited, _notify)),
        _register("Output", output_panel),
        sizing_mode="stretch_width",
        styles=row_styles,
    )
    row3 = pn.Row(
        _register("Background", _build_background_section(state, edited, _notify)),
        _register("Responses", _build_responses_section(state, edited, _notify)),
        sizing_mode="stretch_width",
        styles=row_styles,
    )

    # Set initial status icons — a freshly loaded config may already
    # have gaps, and cards shouldn't wait for the first edit to show it.
    _refresh_statuses()

    return pn.Column(
        row1, row2, row3, sizing_mode="stretch_width", styles={"gap": "10px"}
    )


def panel(state):
    """Return the Config tab content.

    Args:
        state (AppState): Shared application state.

    Returns:
        pn.Column: Tab layout.
    """
    dir_picker = FilePicker(label="Select the working directory", directory=True)
    if state.working_dir:
        dir_picker.set_path(state.working_dir)

    def _on_picker_changed(event):
        state.working_dir = event.new

    def _on_state_changed(event):
        # Keep the picker in sync with working_dir changes driven from
        # elsewhere (e.g. auto-derived from a loaded config's location).
        if event.new != dir_picker.path:
            dir_picker.set_path(event.new)

    dir_picker.param.watch(_on_picker_changed, "path")
    state.param.watch(_on_state_changed, "working_dir")

    form_placeholder = pn.Column(sizing_mode="stretch_width")
    empty_msg = pn.pane.Markdown(
        "⚠️ Create a new configuration or load an existing one "
        "from the sidebar to get started."
    )

    def _rebuild(_event=None):
        if state.edited_config is None:
            form_placeholder.objects = [empty_msg]
        else:
            form_placeholder.objects = [_build_form(state)]

    state.param.watch(_rebuild, "config_generation")
    _rebuild()

    return pn.Column(
        pn.pane.Markdown(TITLE),
        dir_picker,
        form_placeholder,
        sizing_mode="stretch_width",
        styles={"gap": "10px", "margin-top": "15px"},
    )
