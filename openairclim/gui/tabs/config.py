"""Config tab: working directory and the full configuration form.

The form is rebuilt from scratch whenever ``state.config_generation`` changes
(i.e. a fresh config was loaded or created via the sidebar's Load/New buttons) -
individual field edits mutate the same ``state.edited_config`` dict in place and
only trigger it (via ``_notify``), so they don't tear down and rebuild
the widgets.
"""

from pathlib import Path
from typing import get_args, get_origin

import panel as pn

from .. import config_io
from ..components.file_picker import FilePicker
from ...addon._premium import OAC_PREMIUM_AVAILABLE, LOW_SOOT_CASES
from ...core.config_model import Config

TITLE = """
### Edit configuration
Use this tab to edit the OpenAirClim configuration. All variables are available
within the cards below. The card titles will show a warning if any required
variables are invalid or not set. The validity of the configuration can be
checked using the "validate" button in the sidebar.

If you prefer modifying the config file directly in text form, use the Config
(Expert) tab.
"""


def _submodel(path):
    """Resolve a dotted config path to its pydantic submodel class. Walks
    through `Config.model_fields` (e.g. "responses.CO2.rf") rather than
    importing the per-section classes directly, so that the data can be
    referenced in the actual toml shape.

    Args:
        path (str): Dotted field path, starting from Config.

    Returns:
        The pydantic model class at that path.
    """
    model = Config
    for part in path.split("."):
        model = model.model_fields[part].annotation
    return model


def _literal_choices(model, field):
    """Return the allowed values of a `Literal[...]` or `list[Literal[...]]`
    field, read off a config_model submodel (see `_submodel`).

    Args:
        model: A pydantic model class, e.g. `_submodel("species")`.
        field (str): Name of one of its fields.

    Returns:
        list: Allowed values for that field. A list, not a tuple —
            Panel's SelectBase.options only accepts dict/list.
    """
    annotation = model.model_fields[field].annotation
    if get_origin(annotation) is list:
        return list(get_args(get_args(annotation)[0]))
    return list(get_args(annotation))


def _field_description(model, field):
    """Return a config_model submodel field's `Field(description=...)`. This
    can be passed straight through to a panel widget's own `description`
    kwarg to provide a tooltip.

    Args:
        model: A pydantic model class, e.g. `_submodel("temperature")`.
        field (str): Name of one of its fields.

    Returns:
        str or None: The field's description, if one is set.
    """
    return model.model_fields[field].description


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
}
BACKGROUND_FILE_DEFAULTS = {
    "CO2": "co2_bg.nc",
    "CH4": "ch4_bg.nc",
    "N2O": "n2o_bg.nc",
}

# Width/wrap hint used for the sub-columns inside the wide Background and
# Responses cards, so their content forms roughly two columns instead of
# one long vertical stack.
_SUBCOL_STYLES = {"flex": "1 1 45%", "min-width": "260px"}


def _resolve_dir_or_none(working_dir, dir_str):
    """Resolve a directory string, or None if nothing has been chosen yet.

    Used instead of ``config_io.resolve_dir`` directly wherever a blank
    ``dir_str`` must stay blank (not silently fall back to
    ``working_dir``, or to "." if that's empty too) — otherwise a fresh
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
        return value[len(_MISSING_PREFIX):-len(_MISSING_SUFFIX)]
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
# Per-card required-field status
#
# Each card gets a small "status" check that returns None (nothing
# missing) or "⚠️" (required data missing/incomplete) — shown as an icon
# appended to the card's title.
#
# Most cards just need a flat list of dotted-key paths that must be
# non-empty (REQUIRED_FIELDS_*) — add or remove a path there to change
# what a card requires. A few cards have conditional or multi-level
# logic (e.g. Background/Responses warn on any unset dropdown, Emission
# inventories' base folder is only required when "Relative to base" is on,
# Simulation period warns on end <= start rather than a blank check) and
# get their own _check_* function instead, following the same pattern.
# ======================================================================

REQUIRED_FIELDS_SPECIES = ["species.inv", "species.out"]
REQUIRED_FIELDS_TIME_EVOLUTION = []
REQUIRED_FIELDS_TEMPERATURE = []
REQUIRED_FIELDS_PARAMETRIC = []
REQUIRED_FIELDS_OUTPUT = ["output.dir", "output.name"]


def _is_blank(value):
    """Return True if a config field counts as "not filled in".

    Args:
        value: Field value read from the edited config dict.

    Returns:
        bool: True for None, "", [], or {}.
    """
    return value in (None, "", [], {})


def _get_path(edited, path):
    """Look up a dotted key path in the config dict, e.g. "species.inv".

    Args:
        edited (dict): Working configuration dict.
        path (str): Dotted key path.

    Returns:
        The value at that path, or None if any part of the path is
        missing.
    """
    value = edited
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _required_fields_status(edited, paths):
    """Return "⚠️" if any dotted-path field in `paths` is blank.

    Args:
        edited (dict): Working configuration dict.
        paths (list): Dotted key paths that must be non-empty.

    Returns:
        str or None: "⚠️" if something's missing, else None.
    """
    for path in paths:
        if _is_blank(_get_path(edited, path)):
            return "⚠️"
    return None


def _check_species(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_SPECIES)


def _check_time(edited):
    """Warn if the simulation period is unset or empty (end <= start) —
    same condition _build_time_section checks live, so the card ⚠️ and
    the inline widget warning always agree."""
    start, end, _ = edited["time"]["range"]
    return "⚠️" if end <= start else None


def _check_time_evolution(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_TIME_EVOLUTION)


def _check_temperature(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_TEMPERATURE)


def _check_parametric(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_PARAMETRIC)


def _check_output(edited):
    return _required_fields_status(edited, REQUIRED_FIELDS_OUTPUT)


def _has_missing_files(dir_str, files):
    """Return True if any of `files` doesn't exist in `dir_str`.

    `dir_str` is expected to already be an absolute path — that's how
    it's stored once a folder is picked in the Config tab (see
    `_build_dir_files_widgets`) — so this is a plain existence check,
    no resolution against working_dir needed.

    Args:
        dir_str (str): Absolute folder path, or "" if unset.
        files (list): Filenames expected to live in that folder.

    Returns:
        bool: True if dir_str is set and at least one file is missing.
    """
    if not dir_str:
        return False
    base = Path(dir_str)
    return any(not (base / f).exists() for f in files)


def _check_inventories(edited):
    """Folder + files are always required; base folder + files only
    when "Relative to base" is enabled. Also warns if any selected
    file (main, or base when relevant) can't be found on disk — the
    same "not found here" files flagged in the multiselect itself."""
    inv = edited["inventories"]
    paths = ["inventories.dir", "inventories.files"]
    if inv.get("rel_to_base"):
        paths = paths + ["inventories.base.dir", "inventories.base.files"]

    status = _required_fields_status(edited, paths)
    if status:
        return status

    if _has_missing_files(inv.get("dir", ""), inv.get("files", [])):
        return "⚠️"
    if inv.get("rel_to_base"):
        base = inv.get("base", {})
        if _has_missing_files(base.get("dir", ""), base.get("files", [])):
            return "⚠️"
    return None


def _check_background(edited):
    """Folder required; warn if any species' file or scenario is unset."""
    bg = edited["background"]
    if _is_blank(bg.get("dir")):
        return "⚠️"
    for species in ("CO2", "CH4", "N2O"):
        sub = bg.get(species, {})
        if _is_blank(sub.get("file")) or _is_blank(sub.get("scenario")):
            return "⚠️"
    return None


def _check_responses(edited):
    """Folder required; warn if any response file is unset."""
    resp = edited["responses"]
    if _is_blank(resp.get("dir")):
        return "⚠️"
    file_fields = [
        resp.get("H2O", {}).get("rf", {}).get("file"),
        resp.get("O3", {}).get("rf", {}).get("file"),
        resp.get("CH4", {}).get("tau", {}).get("file"),
        resp.get("cont", {}).get("resp", {}).get("file"),
    ]
    if any(_is_blank(v) for v in file_fields):
        return "⚠️"
    return None


def _check_metrics(edited):
    """Warn if the metrics fields are missing/incomplete.

    OpenAirClim needs at least one value in each of types/H/t_0 to run
    metrics. Warns if:
    - "Calculate climate metrics" (Output) is on and any of the three
      is unset, or
    - only one or two of the three are filled in, regardless of
      whether metrics calculation is enabled (partially-filled card).
    """
    metrics = edited["metrics"]
    filled = sum(
        0 if _is_blank(metrics.get(key)) else 1 for key in ("types", "H", "t_0")
    )
    run_metrics = bool(edited.get("output", {}).get("run_metrics"))

    if run_metrics and filled < 3:
        return "⚠️"
    if 0 < filled < 3:
        return "⚠️"
    return None


# Card title -> check function, in card order
CARD_CHECKS = {
    "Species": _check_species,
    "Simulation period": _check_time,
    "Time evolution (optional)": _check_time_evolution,
    "Emission inventories": _check_inventories,
    "Temperature": _check_temperature,
    "Metrics": _check_metrics,
    "Parametric": _check_parametric,
    "Output": _check_output,
    "Background": _check_background,
    "Responses": _check_responses,
}


def check_required_fields(edited_config):
    """Run every card's required-field check against a config dict.

    Lets other modules (the sidebar's Validate button) run the exact
    same checks shown as ⚠️ icons on the Config tab's card titles,
    without needing the cards to actually be rendered.

    Args:
        edited_config (dict): Working configuration dict.

    Returns:
        list: (card_title, status_icon) tuples for cards with a
            non-None status, in card order. Empty if nothing's missing.
    """
    problems = []
    for title, check_fn in CARD_CHECKS.items():
        status = check_fn(edited_config)
        if status:
            problems.append((title, status))
    return problems


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
        name=label, options=list(current),
        value=list(current),
        sizing_mode="stretch_width"
    )
    add_input = pn.widgets.IntInput(name="Add value", value=0, width=110)
    add_btn = pn.widgets.Button(name="Add", width=50, margin=(25, 0, 0, 6))

    def _sync(event=None):
        parent[key] = [int(v) for v in select.value]
        notify()

    def _on_add(event=None):
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
    species_model = _submodel("species")

    inv_select = pn.widgets.MultiChoice(
        name="Input species (from inventories)",
        options=_literal_choices(species_model, "inv"),
        value=list(species["inv"]),
        description=_field_description(species_model, "inv")
    )
    out_select = pn.widgets.MultiChoice(
        name="Output species (responses)",
        options=_literal_choices(species_model, "out"),
        value=list(species["out"]),
        description=_field_description(species_model, "out")
    )
    nox_select = pn.widgets.Select(
        name="Assumed NOx species in inventories",
        options=_literal_choices(species_model, "nox"),
        value=species["nox"],
        description=_field_description(species_model, "nox")
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


def _build_time_section(state, edited, notify):
    """Build the Simulation period section.

    Args:
        state (AppState): Shared application state.
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
        "2051 is selected, the last year simulated will be 2050."
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

    def _on_time_changed(event=None):
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
        description=_field_description(_submodel("time"), "dir")
    )
    if time_cfg["dir"]:
        dir_resolved = config_io.resolve_dir(state.working_dir, time_cfg["dir"])
        time_cfg["dir"] = str(dir_resolved)
        dir_picker._text_input.value = str(dir_resolved)

    file_select = pn.widgets.Select(
        name="Time evolution file (optional)",
        options=[_NONE_OPTION],
        value=_NONE_OPTION,
        description=_field_description(_submodel("time"), "file")
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

    def _on_clear(event=None):
        file_select.value = _NONE_OPTION

    dir_picker.param.watch(_on_dir_changed, "path")
    clear_btn.on_click(_on_clear)

    # Populate before wiring the "value changed" watcher below, so
    # setting the initial selection doesn't itself count as an edit
    # (and incorrectly mark the config dirty right after loading it).
    _refresh_time_file()
    file_select.param.watch(_on_file_changed, "value")

    return pn.Column(
        dir_picker,
        pn.Row(file_select, clear_btn)
    )


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
        dir_picker._text_input.value = str(resolved)

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

    dir_picker = FilePicker(label="Folder", directory=True)
    existing_dir = bg.get("dir", "")
    if existing_dir:
        dir_resolved = config_io.resolve_dir(state.working_dir, existing_dir)
        bg["dir"] = str(dir_resolved)
        dir_picker._text_input.value = str(dir_resolved)

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
            resolved = _resolve_dir_or_none(state.working_dir, dir_picker.path)
            if file_select.value == _NONE_OPTION or resolved is None:
                scenario_select.options = [_NONE_OPTION]
                scenario_select.value = _NONE_OPTION
                return
            variables = config_io.list_nc_data_vars(resolved / file_select.value)
            scenario_select.options = [_NONE_OPTION] + variables
            current = sub.get("scenario", "")
            scenario_select.value = current if current in variables else _NONE_OPTION

        def _refresh_file():
            resolved = _resolve_dir_or_none(state.working_dir, dir_picker.path)
            files = config_io.list_nc_files(resolved) if resolved is not None else []
            file_select.options = [_NONE_OPTION] + files
            current = sub.get("file", "")
            default_filename = BACKGROUND_FILE_DEFAULTS.get(species_key, "")
            if current in files:
                file_select.value = current
            elif not current and default_filename in files:
                # Suggest the example config's filename if it's present
                # and nothing has been chosen yet.
                file_select.value = default_filename
            else:
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
            pn.pane.Markdown(f"**{label}**"), file_select, scenario_select,
            styles=_SUBCOL_STYLES,
        )

    species_columns.append(_make_species_widgets("CO2", "CO₂"))
    species_columns.append(_make_species_widgets("CH4", "CH₄"))
    species_columns.append(_make_species_widgets("N2O", "N₂O"))

    def _on_dir_changed(event):
        bg["dir"] = event.new
        for refresh in refresh_funcs:
            refresh()
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")

    return pn.Column(
        dir_picker,
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

    dir_picker = FilePicker(label="Folder", directory=True)
    existing_dir = resp.get("dir", "")
    if existing_dir:
        dir_resolved = config_io.resolve_dir(state.working_dir, existing_dir)
        resp["dir"] = str(dir_resolved)
        dir_picker._text_input.value = str(dir_resolved)

    refresh_funcs = []

    def _make_file_select(label, sub_dict, default_filename):
        """Build a single response-file dropdown bound to sub_dict["file"]."""
        select = pn.widgets.Select(name=label, options=[_NONE_OPTION], value=_NONE_OPTION)

        def _refresh():
            resolved = _resolve_dir_or_none(state.working_dir, dir_picker.path)
            files = config_io.list_nc_files(resolved) if resolved is not None else []
            select.options = [_NONE_OPTION] + files
            current = sub_dict.get("file", "")
            if current in files:
                select.value = current
            elif not current and default_filename in files:
                # Suggest the example config's filename if it's present
                # and nothing has been chosen yet.
                select.value = default_filename
            else:
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
        value=current_low_soot if current_low_soot in low_soot_options else _NONE_OPTION,
        description=_field_description(_submodel("responses.cont"), "low_soot_case")
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
        for refresh in refresh_funcs:
            refresh()
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")

    # ---- CO2 / CH4 method & attribution dropdowns ----------------------
    # response_grid isn't shown — it's filled in via DEFAULT_CONFIG and
    # isn't something the user needs to set directly.
    # CH4's attribution options are identical to CO2's (see config_model.py),
    # so both dropdowns share the same choice set, read once here.
    rf_attr_options = _literal_choices(_submodel("responses.CO2.rf"), "attr")
    co2_conc_method = pn.widgets.Select(
        name="CO₂ concentration method",
        options=_literal_choices(_submodel("responses.CO2.conc"), "method"),
        value=resp["CO2"]["conc"]["method"],
        description=_field_description(_submodel("responses.CO2.conc"), "method"),
    )
    co2_rf_method = pn.widgets.Select(
        name="CO₂ RF method",
        options=_literal_choices(_submodel("responses.CO2.rf"), "method"),
        value=resp["CO2"]["rf"]["method"],
        description=_field_description(_submodel("responses.CO2.rf"), "method"),
    )
    co2_rf_attr = pn.widgets.Select(
        name="CO₂ RF attribution",
        options=rf_attr_options,
        value=resp["CO2"]["rf"]["attr"],
        description=_field_description(_submodel("responses.CO2.rf"), "attr")
    )
    ch4_rf_attr = pn.widgets.Select(
        name="CH₄ RF attribution",
        options=rf_attr_options,
        value=resp["CH4"]["rf"]["attr"],
        description=_field_description(_submodel("responses.CH4.rf"), "attr")
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
        pn.FlexBox(
            pn.Column(
                pn.pane.Markdown("**CO₂**"), co2_conc_method, co2_rf_method, co2_rf_attr,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**H₂O**"), h2o_select,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**O₃**"), o3_select,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**CH₄**"), ch4_select, ch4_rf_attr,
                styles=_SUBCOL_STYLES,
            ),
            pn.Column(
                pn.pane.Markdown("**Contrails**"), cont_select, low_soot_select,
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
    temperature_model = _submodel("temperature")
    efficacy_species = [
        f for f in temperature_model.model_fields if f not in ("method", "CO2")
    ]

    method_select = pn.widgets.Select(
        name="Method", options=["Boucher&Reddy"], value=temp.get("method", "Boucher&Reddy")
    )
    lambda_input = pn.widgets.FloatInput(
        name="CO2 climate sensitivity (lambda)",
        value=float(temp["CO2"]["lambda"]),
        description=_field_description(_submodel("temperature.CO2"), "lambda_"),
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
            description=_field_description(temperature_model, species),
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
        options=_literal_choices(_submodel("metrics"), "types"),
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
    parametric_species = [f for f in _submodel("parametric").model_fields if f != "enabled"]

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
        dir_picker._text_input.value = str(dir_resolved)

    name_input = pn.widgets.TextInput(name="Output file name", value=out["name"])

    run_oac_cb = pn.widgets.Checkbox(
        name="Run OpenAirClim",
        value=bool(out["run_oac"])
    )
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

    panel = pn.Column(
        dir_picker, name_input, run_oac_cb, run_metrics_cb, run_plots_cb, overwrite_cb
    )
    return panel, run_metrics_cb


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
    # function from the shared CARD_CHECKS registry.
    cards = {}

    def _register(title, content):
        card = _card(title, content)
        cards[title] = card
        return card

    def _refresh_statuses():
        for title, card in cards.items():
            status = CARD_CHECKS[title](edited)
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

    return pn.Column(row1, row2, row3, sizing_mode="stretch_width", styles={"gap": "10px"})


def panel(state):
    """Return the Config tab content.

    Args:
        state (AppState): Shared application state.

    Returns:
        pn.Column: Tab layout.
    """
    dir_picker = FilePicker(
        label="Select the working directory", directory=True
    )
    if state.working_dir:
        dir_picker._text_input.value = state.working_dir

    def _on_picker_changed(event):
        state.working_dir = event.new

    def _on_state_changed(event):
        # Keep the picker in sync with working_dir changes driven from
        # elsewhere (e.g. auto-derived from a loaded config's location).
        if event.new != dir_picker.path:
            dir_picker._text_input.value = event.new

    dir_picker.param.watch(_on_picker_changed, "path")
    state.param.watch(_on_state_changed, "working_dir")

    form_placeholder = pn.Column(sizing_mode="stretch_width")
    empty_msg = pn.pane.Markdown(
        "⚠️ Create a new configuration or load an existing one "
        "from the sidebar to get started."
    )

    def _rebuild(event=None):
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
