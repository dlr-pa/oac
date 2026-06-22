"""Sidebar: working directory, load/new, config accordion, and Run."""

from pathlib import Path

import panel as pn

from . import config_io
from .components.file_picker import FilePicker

INPUT_SPECIES_OPTIONS = ["CO2", "H2O", "NOx", "distance"]
OUTPUT_SPECIES_OPTIONS = ["CO2", "H2O", "O3", "CH4", "PMO", "cont", "SWV"]
NOX_OPTIONS = ["NO", "NO2"]
RF_ATTR_OPTIONS = ["none", "residual", "marginal", "proportional", "differential"]
CO2_RF_METHOD_OPTIONS = ["Etminan_2016", "IPCC_2001_1", "IPCC_2001_2", "IPCC_2001_3"]
CO2_CONC_METHOD_OPTIONS = ["Sausen&Schumann"]
METRICS_TYPE_OPTIONS = ["AGWP", "ATR", "AGTP"]
PARAMETRIC_SPECIES = ["CO2", "H2O", "O3", "CH4", "cont"]
TEMPERATURE_EFFICACY_SPECIES = ["H2O", "O3", "PMO", "CH4", "cont", "SWV"]

# Sentinel option used by optional single-file Select widgets, since
# Select requires `value` to be one of `options` — there's no built-in
# "nothing selected" state.
_NONE_OPTION = "\u2014 none \u2014"

# Suggested response filenames, matching the example config — used to
# pre-select a sensible default once a folder is chosen, but only if
# that file actually exists there.
RESPONSE_FILE_DEFAULTS = {
    "H2O": "resp_RF.nc",
    "O3": "resp_RF_O3.nc",
    "CH4": "resp_ch4.nc",
    "cont": "resp_cont_lf.nc",
}


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
        pn.Row: The widget group.
    """
    current = [str(v) for v in parent.get(key, [])]
    select = pn.widgets.MultiChoice(name=label, options=list(current), value=list(current))
    add_input = pn.widgets.IntInput(name="Add value", value=0, width=110)
    add_btn = pn.widgets.Button(name="Add", width=60, margin=(18, 0, 0, 6))

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

    return pn.Row(select, pn.Column(add_input, add_btn))


# ======================================================================
# Accordion section builders
# ======================================================================


def _build_species_section(edited, notify):
    """Build the Species accordion section.

    Args:
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    species = edited["species"]
    species.setdefault("nox", "NO")

    inv_select = pn.widgets.MultiChoice(
        name="Input species (from inventories)",
        options=INPUT_SPECIES_OPTIONS,
        value=list(species["inv"]),
    )
    out_select = pn.widgets.MultiChoice(
        name="Output species (responses)",
        options=OUTPUT_SPECIES_OPTIONS,
        value=list(species["out"]),
    )
    nox_select = pn.widgets.Select(
        name="Assumed NOx species in inventories",
        options=NOX_OPTIONS,
        value=species["nox"],
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
    """Build the Simulation period accordion section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    time_cfg = edited["time"]
    time_cfg.setdefault("dir", "")

    t_start, t_end, t_step = time_cfg["range"]

    start_input = pn.widgets.IntInput(name="Start year", value=int(t_start))
    end_input = pn.widgets.IntInput(name="End year (exclusive)", value=int(t_end))
    step_input = pn.widgets.IntInput(name="Step", value=int(t_step), start=1)
    warning = pn.pane.Markdown("")

    def _on_time_changed(event=None):
        start, end, step = start_input.value, end_input.value, step_input.value
        if end <= start:
            warning.object = "\u26a0\ufe0f End year must be after the start year."
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
    """Build the time evolution accordion section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    time_cfg = edited["time"]
    time_cfg.setdefault("dir", "")

    dir_picker = FilePicker(label="Folder (for time evolution file)", directory=True)
    dir_resolved = config_io.resolve_dir(state.working_dir, time_cfg["dir"])
    time_cfg["dir"] = str(dir_resolved)
    dir_picker._text_input.value = str(dir_resolved)

    file_select = pn.widgets.Select(
        name="Time evolution file (optional)",
        options=[_NONE_OPTION],
        value=_NONE_OPTION,
    )
    clear_btn = pn.widgets.Button(name="Clear", width=70, margin=(24, 10, 0, 6))

    def _refresh_time_file():
        resolved = config_io.resolve_dir(state.working_dir, dir_picker.path)
        files = config_io.list_nc_files(resolved)
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
    file_select.param.watch(_on_file_changed, "value")
    clear_btn.on_click(_on_clear)

    _refresh_time_file()

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
    resolved = config_io.resolve_dir(state.working_dir, section.get("dir", ""))
    # Canonicalize to absolute now, so the stored path stays correct
    # even if state.working_dir changes (or is set) later on.
    section["dir"] = str(resolved)
    dir_picker._text_input.value = str(resolved)

    files_select = pn.widgets.MultiChoice(name="Files", options=[], value=[])
    status = pn.pane.Markdown("")

    def _refresh(initial_selection=None):
        resolved = config_io.resolve_dir(state.working_dir, dir_picker.path)
        files = config_io.list_nc_files(resolved)
        files_select.options = files

        if initial_selection is not None:
            keep = [f for f in initial_selection if f in files]
            missing = [f for f in initial_selection if f not in files]
            files_select.value = keep
            if missing:
                status.object = (
                    f"\u26a0\ufe0f Previously selected file(s) not found "
                    f"in this folder: {', '.join(missing)}"
                )
            else:
                status.object = ""
        else:
            keep = [f for f in files_select.value if f in files]
            files_select.value = keep
            status.object = "" if files else "\u26a0\ufe0f No .nc files found."

    def _on_dir_changed(event):
        # Keep the absolute path during editing — converted to relative
        # only at save time, in config_io.prepare_for_save.
        section["dir"] = event.new
        _refresh()
        notify()

    def _on_files_changed(event):
        section["files"] = list(event.new)
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")
    files_select.param.watch(_on_files_changed, "value")

    _refresh(initial_selection=list(initial_files or []))

    return pn.Column(dir_picker, files_select, status)


def _build_inventories_section(state, edited, notify):
    """Build the Emission inventories accordion section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    inv = edited["inventories"]
    # Older / hand-written config files may not have a [inventories.base]
    # section at all — only required when rel_to_base is True.
    inv.setdefault("base", {"dir": "", "files": []})

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
    """Build the Background accordion section.

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
    dir_resolved = config_io.resolve_dir(state.working_dir, bg.get("dir", ""))
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
            if file_select.value == _NONE_OPTION:
                scenario_select.options = [_NONE_OPTION]
                scenario_select.value = _NONE_OPTION
                return
            resolved = config_io.resolve_dir(state.working_dir, dir_picker.path)
            variables = config_io.list_nc_data_vars(resolved / file_select.value)
            scenario_select.options = [_NONE_OPTION] + variables
            current = sub.get("scenario", "")
            scenario_select.value = current if current in variables else _NONE_OPTION

        def _refresh_file():
            resolved = config_io.resolve_dir(state.working_dir, dir_picker.path)
            files = config_io.list_nc_files(resolved)
            file_select.options = [_NONE_OPTION] + files
            current = sub.get("file", "")
            file_select.value = current if current in files else _NONE_OPTION
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

        file_select.param.watch(_on_file_changed, "value")
        scenario_select.param.watch(_on_scenario_changed, "value")

        _refresh_file()
        refresh_funcs.append(_refresh_file)

        return pn.Column(pn.pane.Markdown(f"**{label}**"), file_select, scenario_select)

    species_columns.append(_make_species_widgets("CO2", "CO\u2082"))
    species_columns.append(_make_species_widgets("CH4", "CH\u2084"))
    species_columns.append(_make_species_widgets("N2O", "N\u2082O"))

    def _on_dir_changed(event):
        bg["dir"] = event.new
        for refresh in refresh_funcs:
            refresh()
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")

    return pn.Column(dir_picker, *species_columns)


def _build_responses_section(state, edited, notify):
    """Build the Responses accordion section.

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    resp = edited["responses"]

    # Defensive defaults — CONFIG_TEMPLATE/DEFAULT_CONFIG only guarantee
    # responses.dir and (via DEFAULT_CONFIG) the response_grid/method/attr
    # fields; the actual response *files* have no universal default, so
    # a brand-new config (or an old hand-edited one) may not have these
    # nested dicts at all yet.
    resp.setdefault("CO2", {})
    resp["CO2"].setdefault("conc", {})
    resp["CO2"]["conc"].setdefault("method", "Sausen&Schumann")
    resp["CO2"].setdefault("rf", {})
    resp["CO2"]["rf"].setdefault("method", "Etminan_2016")
    resp["CO2"]["rf"].setdefault("attr", "proportional")

    resp.setdefault("H2O", {})
    resp["H2O"].setdefault("rf", {})
    resp["H2O"]["rf"].setdefault("file", "")

    resp.setdefault("O3", {})
    resp["O3"].setdefault("rf", {})
    resp["O3"]["rf"].setdefault("file", "")

    resp.setdefault("CH4", {})
    resp["CH4"].setdefault("tau", {})
    resp["CH4"]["tau"].setdefault("file", "")
    resp["CH4"].setdefault("rf", {})
    resp["CH4"]["rf"].setdefault("attr", "proportional")

    resp.setdefault("cont", {})
    resp["cont"].setdefault("resp", {})
    resp["cont"]["resp"].setdefault("file", "")

    dir_picker = FilePicker(label="Folder", directory=True)
    dir_resolved = config_io.resolve_dir(state.working_dir, resp.get("dir", ""))
    resp["dir"] = str(dir_resolved)
    dir_picker._text_input.value = str(dir_resolved)

    refresh_funcs = []

    def _make_file_select(label, sub_dict, default_filename):
        """Build a single response-file dropdown bound to sub_dict["file"]."""
        select = pn.widgets.Select(name=label, options=[_NONE_OPTION], value=_NONE_OPTION)

        def _refresh():
            resolved = config_io.resolve_dir(state.working_dir, dir_picker.path)
            files = config_io.list_nc_files(resolved)
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

        select.param.watch(_on_change, "value")
        _refresh()
        refresh_funcs.append(_refresh)
        return select

    h2o_select = _make_file_select(
        "H\u2082O response file", resp["H2O"]["rf"], RESPONSE_FILE_DEFAULTS["H2O"]
    )
    o3_select = _make_file_select(
        "O\u2083 response file", resp["O3"]["rf"], RESPONSE_FILE_DEFAULTS["O3"]
    )
    ch4_select = _make_file_select(
        "CH\u2084 response file", resp["CH4"]["tau"], RESPONSE_FILE_DEFAULTS["CH4"]
    )
    cont_select = _make_file_select(
        "Contrail response file", resp["cont"]["resp"], RESPONSE_FILE_DEFAULTS["cont"]
    )

    def _on_dir_changed(event):
        resp["dir"] = event.new
        for refresh in refresh_funcs:
            refresh()
        notify()

    dir_picker.param.watch(_on_dir_changed, "path")

    # ---- CO2 / CH4 method & attribution dropdowns ----------------------
    # response_grid isn't shown — it's filled in via DEFAULT_CONFIG and
    # isn't something the user needs to set directly.
    co2_conc_method = pn.widgets.Select(
        name="CO\u2082 concentration method",
        options=CO2_CONC_METHOD_OPTIONS,
        value=resp["CO2"]["conc"]["method"],
    )
    co2_rf_method = pn.widgets.Select(
        name="CO\u2082 RF method",
        options=CO2_RF_METHOD_OPTIONS,
        value=resp["CO2"]["rf"]["method"],
    )
    co2_rf_attr = pn.widgets.Select(
        name="CO\u2082 RF attribution", options=RF_ATTR_OPTIONS, value=resp["CO2"]["rf"]["attr"]
    )
    ch4_rf_attr = pn.widgets.Select(
        name="CH\u2084 RF attribution", options=RF_ATTR_OPTIONS, value=resp["CH4"]["rf"]["attr"]
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
        pn.pane.Markdown("**CO\u2082**"),
        co2_conc_method,
        co2_rf_method,
        co2_rf_attr,
        pn.pane.Markdown("**H\u2082O**"),
        h2o_select,
        pn.pane.Markdown("**O\u2083**"),
        o3_select,
        pn.pane.Markdown("**CH\u2084**"),
        ch4_select,
        ch4_rf_attr,
        pn.pane.Markdown("**Contrails**"),
        cont_select,
    )


def _build_temperature_section(edited, notify):
    """Build the Temperature accordion section.

    Args:
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    temp = edited["temperature"]

    # Defensive defaults — only temperature.method and temperature.CO2.lambda
    # are guaranteed by CONFIG_TEMPLATE; efficacies have no universal
    # default, so seed them with the example config's values if missing.
    efficacy_defaults = {
        "H2O": 1.14, "O3": 1.37, "PMO": 1.37, "CH4": 1.14, "cont": 0.59, "SWV": 1.0,
    }
    for species, default in efficacy_defaults.items():
        temp.setdefault(species, {})
        temp[species].setdefault("efficacy", default)
    temp.setdefault("CO2", {})
    temp["CO2"].setdefault("lambda", 0.73)

    method_select = pn.widgets.Select(
        name="Method", options=["Boucher&Reddy"], value=temp.get("method", "Boucher&Reddy")
    )
    lambda_input = pn.widgets.FloatInput(
        name="CO2 climate sensitivity (lambda)", value=float(temp["CO2"]["lambda"])
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
    for species in TEMPERATURE_EFFICACY_SPECIES:
        fi = pn.widgets.FloatInput(
            name=f"{species} efficacy", value=float(temp[species]["efficacy"])
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
    """Build the Metrics accordion section.

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
        name="Metric types", options=METRICS_TYPE_OPTIONS, value=list(metrics.get("types", []))
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
        "\u2139\ufe0f Enable **Calculate climate metrics** in the Output section "
        "to configure metrics.",
        visible=not run_metrics_checkbox.value,
    )

    def _on_run_metrics_changed(event):
        metrics_content.visible = event.new
        info_pane.visible = not event.new

    run_metrics_checkbox.param.watch(_on_run_metrics_changed, "value")

    return pn.Column(info_pane, metrics_content)


def _build_parametric_section(edited, notify):
    """Build the Parametric accordion section.

    Args:
        edited (dict): Working configuration dict, mutated in place.
        notify (callable): Called after every edit.

    Returns:
        pn.Column: Section content.
    """
    param_cfg = edited["parametric"]

    ratio_defaults = {
        "CO2": 1.0019972, "H2O": 0.25401992, "O3": 0.7016167,
        "CH4": 1.246515, "cont": 0.22705537,
    }
    for species, default in ratio_defaults.items():
        param_cfg.setdefault(species, default)
    param_cfg.setdefault("enabled", False)

    enabled_cb = pn.widgets.Checkbox(name="Enabled", value=bool(param_cfg["enabled"]))

    ratio_widgets = []
    for species in PARAMETRIC_SPECIES:
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
    """Build the Output accordion section.

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
    dir_resolved = config_io.resolve_dir(state.working_dir, out["dir"])
    out["dir"] = str(dir_resolved)
    dir_picker._text_input.value = str(dir_resolved)

    name_input = pn.widgets.TextInput(name="Output file name", value=out["name"])

    run_oac_cb = pn.widgets.Checkbox(name="Run", value=bool(out["run_oac"]))
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


def _build_validate_save_section(state, edited, on_save=None):
    """Build the Validate and save section (lives below the accordion).

    Args:
        state (AppState): Shared application state.
        edited (dict): Working configuration dict (read at click-time
            via state.edited_config, which aliases the same object).
        on_save (callable, optional): Called after a successful save,
            used to clear the "unsaved edits" flag in the sidebar.

    Returns:
        pn.Column: Section content.
    """
    validate_btn = pn.widgets.Button(
        name="Validate configuration", button_type="primary"
    )
    status = pn.pane.Markdown("")
    save_btn = pn.widgets.Button(name="Save configuration\u2026", button_type="success")

    def _on_validate(event=None):
        if not state.working_dir:
            status.object = "\u26a0\ufe0f Select a working directory first."
            return
        if not state.edited_config:
            status.object = "\u26a0\ufe0f No configuration to validate yet."
            return
        status.object = "\u23f3 Validating\u2026"
        errors = config_io.check_files_exist(state.working_dir, state.edited_config)
        status.object = config_io.format_validation_result(state.edited_config, errors)

    def _on_save(event=None):
        if not state.edited_config:
            status.object = "\u26a0\ufe0f No configuration to save yet."
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
            status.object = f"\u274c Failed to save: {e}"
            return

        state.config_path = selected
        status.object = f"\u2705 Saved to `{selected}`"
        if on_save is not None:
            on_save()

    validate_btn.on_click(_on_validate)
    save_btn.on_click(_on_save)

    return pn.Column(
        validate_btn, save_btn, status
    )


def _build_accordion(state, config, on_change=None):
    """Build the full configuration accordion.

    Mutates ``state.edited_config`` in place as the user makes changes.

    Args:
        state (AppState): Shared application state.
        config (dict): Configuration to seed the form with — either a
            validated config loaded from file, or a blank skeleton.
        on_change (callable, optional): Called whenever any field is
            edited, used to track unsaved changes.

    Returns:
        pn.Accordion: The configuration accordion, collapsed by default.
    """
    from copy import deepcopy

    edited = deepcopy(config)
    state.edited_config = edited

    def _notify():
        if on_change is not None:
            on_change()
        # edited_config is mutated in place, so a plain reassignment
        # wouldn't fire watchers (same object, equal by identity).
        # Trigger explicitly so other tabs can react to live edits.
        state.param.trigger("edited_config")

    def _padded(content):
        """Wrap a section's content with bottom padding inside the accordion.

        Args:
            content: Any Panel object returned by a section builder.

        Returns:
            pn.Column: Content wrapped with 10px bottom padding.
        """
        return pn.Column(content, styles={"padding-bottom": "10px"})

    # Output is built first since Metrics needs to watch its
    # run_metrics checkbox to decide whether to show its own content.
    output_panel, run_metrics_checkbox = _build_output_section(state, edited, _notify)

    return pn.Accordion(
        ("Species", _padded(_build_species_section(edited, _notify))),
        ("Simulation period", _padded(_build_time_section(state, edited, _notify))),
        ("Time evolution (optional)", _padded(_build_time_evolution_section(state, edited, _notify))),
        ("Emission inventories", _padded(_build_inventories_section(state, edited, _notify))),
        ("Background", _padded(_build_background_section(state, edited, _notify))),
        ("Responses", _padded(_build_responses_section(state, edited, _notify))),
        ("Temperature", _padded(_build_temperature_section(edited, _notify))),
        ("Metrics", _padded(_build_metrics_section(edited, _notify, run_metrics_checkbox))),
        ("Parametric", _padded(_build_parametric_section(edited, _notify))),
        ("Output", _padded(output_panel)),
        active=[],
        margin=(0, 10, 0, 0),
    )


# ======================================================================
# Sidebar layout
# ======================================================================


def panel(state):
    """Return the sidebar content.

    Args:
        state (AppState): Shared application state.

    Returns:
        pn.Column: Sidebar layout.
    """
    dir_picker = FilePicker(label="Working directory", directory=True)
    if state.working_dir:
        dir_picker._text_input.value = state.working_dir

    def _on_working_dir_changed(event):
        state.working_dir = event.new

    dir_picker.param.watch(_on_working_dir_changed, "path")

    load_btn = pn.widgets.Button(name="Load", button_type="primary")
    new_btn = pn.widgets.Button(name="New", button_type="default")
    load_status = pn.pane.Markdown("")

    confirm_msg = pn.pane.Markdown(
        "\u26a0\ufe0f You have unsaved edits to the current configuration. "
        "Continuing will discard them."
    )
    confirm_yes = pn.widgets.Button(name="Discard and continue", button_type="danger")
    confirm_no = pn.widgets.Button(name="Cancel", button_type="default")
    confirm_row = pn.Column(
        confirm_msg, pn.Row(confirm_yes, confirm_no), visible=False
    )

    accordion_placeholder = pn.Column()
    validate_save_placeholder = pn.Column()

    run_target_label = pn.pane.Markdown("")
    run_btn = pn.widgets.Button(name="Run OpenAirClim", button_type="primary")
    run_status = pn.pane.Markdown("")

    dirty = {"flag": False}
    pending_action = {"type": None}

    def _mark_dirty():
        dirty["flag"] = True

    def _mark_clean():
        dirty["flag"] = False

    def _update_run_target_label(event=None):
        if state.config_path:
            run_target_label.object = f"Will run: `{Path(state.config_path).name}`"
        else:
            run_target_label.object = "*No configuration saved yet.*"

    state.param.watch(_update_run_target_label, "config_path")
    _update_run_target_label()

    def _do_load(config_path):
        """Validate the given file and build the form from it.

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
        dir_picker._text_input.value = config_dir

        load_status.object = "\u23f3 Loading\u2026"
        config, errors = config_io.parse_and_check_structure(
            state.working_dir, config_path
        )

        if errors:
            load_status.object = "\n".join(f"\u274c {e}" for e in errors)
            return

        state.config_path = config_path
        accordion_placeholder.objects = [
            _build_accordion(state, config, on_change=_mark_dirty)
        ]
        validate_save_placeholder.objects = [
            _build_validate_save_section(state, state.edited_config, on_save=_mark_clean)
        ]
        load_status.object = f"\u2139\ufe0f Loaded `{Path(config_path).name}`."
        dirty["flag"] = False
        try:
            main_content.visible = True
        except NameError:
            pass  # Called during startup before main_content is defined;
                  # visibility is handled by visible=state.edited_config is not None

    def _do_new():
        """Start a blank configuration and build the form from it."""
        blank = config_io.blank_config()
        accordion_placeholder.objects = [
            _build_accordion(state, blank, on_change=_mark_dirty)
        ]
        validate_save_placeholder.objects = [
            _build_validate_save_section(state, state.edited_config, on_save=_mark_clean)
        ]
        load_status.object = "\u2139\ufe0f Started a new blank configuration."
        dirty["flag"] = False
        main_content.visible = True

    def _request_load(event=None):
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

        if dirty["flag"]:
            pending_action["type"] = "load"
            pending_action["path_getter"] = _open_dialog
            confirm_row.visible = True
        else:
            selected = _open_dialog()
            if selected:
                _do_load(selected)

    def _request_new(event=None):
        if dirty["flag"]:
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
    # Run
    # ------------------------------------------------------------------

    def _on_run(event=None):
        if dirty["flag"]:
            run_status.object = (
                "\u26a0\ufe0f You have unsaved edits \u2014 "
                "save the configuration before running."
            )
            return
        if not state.config_path:
            run_status.object = "\u26a0\ufe0f Save the configuration before running."
            return
        if not state.working_dir:
            run_status.object = "\u26a0\ufe0f Select a working directory first."
            return

        run_btn.loading = True
        run_status.object = "\u23f3 Running OpenAirClim\u2026"
        try:
            config_io.run_config(state.working_dir, state.config_path)
            run_status.object = "\u2705 Run completed."
        except Exception as e:
            run_status.object = f"\u274c Run failed: {e}"
        finally:
            run_btn.loading = False

    run_btn.on_click(_on_run)

    # ------------------------------------------------------------------
    # Restore an in-progress accordion if one already exists in state
    # ------------------------------------------------------------------

    if state.edited_config is not None:
        accordion_placeholder.objects = [
            _build_accordion(state, state.edited_config, on_change=_mark_dirty)
        ]
        validate_save_placeholder.objects = [
            _build_validate_save_section(state, state.edited_config, on_save=_mark_clean)
        ]
    elif state.config_path:
        # A config path was supplied at launch (e.g. via --config on the
        # command line). Load it now, before main_content is constructed,
        # so that main_content.visible evaluates to True below.
        # _do_load normally sets main_content.visible = True, but
        # main_content doesn't exist yet here — the visibility is instead
        # handled by the `visible=state.edited_config is not None` argument
        # below, which evaluates after _do_load has populated edited_config.
        _do_load(state.config_path)

    # ------------------------------------------------------------------

    # Everything below the load/new buttons starts hidden and is revealed
    # the first time the user loads a file or creates a new configuration.
    main_content = pn.Column(
        dir_picker,
        pn.layout.Divider(),
        pn.pane.Markdown("### Settings"),
        accordion_placeholder,
        pn.layout.Divider(),
        pn.pane.Markdown("### Validate and save"),
        validate_save_placeholder,
        pn.layout.Divider(),
        pn.pane.Markdown("### Run OpenAirClim"),
        run_target_label,
        run_btn,
        run_status,
        visible=state.edited_config is not None,
    )

    return pn.Column(
        pn.pane.Markdown("## OpenAirClim Configuration"),
        pn.Row(load_btn, new_btn),
        confirm_row,
        load_status,
        pn.layout.Divider(),
        main_content,
    )
