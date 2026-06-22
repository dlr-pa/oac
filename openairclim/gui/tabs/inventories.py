"""Inventories tab: visualise the input emission inventories."""

import os
from pathlib import Path
import panel as pn

from ..components.utils import COLORS, MARKERS, auto_scale, load_inventory, get_numeric_vars


def _build_profile_figures(
    datasets,
    variable,
    p_bin_width=50,
    lat_bin_width=10,
    legend_loc_v="bottom_right",
    legend_loc_l="top_left",
):
    """Create two interactive Bokeh figures: vertical and latitudinal profiles.

    Args:
        datasets (dict): Mapping of label (str) to xarray.Dataset.
        variable (str): Data variable to plot.
        p_bin_width (int): Pressure bin width in hPa. Default 50.
        lat_bin_width (int): Latitude bin width in degrees. Default 10.
        legend_loc_v (str): Bokeh legend location for the vertical profile.
        legend_loc_l (str): Bokeh legend location for the latitudinal profile.

    Returns:
        tuple: Two bokeh.plotting.Figure objects (vertical, latitudinal).
    """
    import numpy as np
    from bokeh.models import Range1d
    from bokeh.plotting import figure

    # get unit (later, we could offer the option of converting between them)
    base_unit = next(iter(datasets.values()))[variable].attrs.get("units", "?")

    # bin edges derived from slider values
    p_bins = np.arange(50, 1050, p_bin_width)
    p_bw = float(p_bin_width)
    lat_bins = np.arange(-90, 91, lat_bin_width)
    lat_bw = float(lat_bin_width)

    # compute binned densities for every selected inventory
    plev_series = {}
    lat_series = {}

    for label, ds in datasets.items():
        data = ds[variable]

        p_binned = data.groupby_bins("plev", p_bins).sum().fillna(0)
        p_mids = p_binned.plev_bins.to_index().mid.values.astype(float)
        plev_series[label] = (p_mids, p_binned.values / p_bw)

        lat_binned = data.groupby_bins("lat", lat_bins).sum().fillna(0)
        lat_mids = lat_binned.lat_bins.to_index().mid.values.astype(float)
        lat_series[label] = (lat_mids, lat_binned.values / lat_bw)

    # global maxima and auto-scaling
    p_max = float(max((d.max() for _, d in plev_series.values()), default=1.0))
    l_max = float(max((d.max() for _, d in lat_series.values()), default=1.0))
    p_scale, p_prefix = auto_scale(p_max)
    l_scale, l_prefix = auto_scale(l_max)

    # vertical profile
    p_fig = figure(
        title="Vertical profile",
        x_axis_label=f"{variable} [{p_prefix}{base_unit} / hPa]",
        y_axis_label="Pressure level [hPa]",
        height=420,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=[("Inventory", "$name"), (variable, "$x"), ("plev", "$y hPa")],
    )
    p_fig.y_range.flipped = True
    p_x_max = p_max / p_scale * 1.1 if p_max > 0 else 1.0
    p_fig.x_range = Range1d(start=0, end=p_x_max)

    for i, (label, (mids, density)) in enumerate(plev_series.items()):
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        scaled = density / p_scale
        p_fig.line(
            scaled, mids, legend_label=label, color=c,
            line_width=2, name=label
        )
        p_fig.scatter(scaled, mids, marker=m, color=c, size=7, name=label)

    p_fig.legend.click_policy = "hide"
    p_fig.legend.location = legend_loc_v

    # latitudinal profile
    l_fig = figure(
        title="Latitudinal profile",
        x_axis_label="Latitude [\u00b0]",
        y_axis_label=f"{variable} [{l_prefix}{base_unit} / \u00b0]",
        height=420,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=[("Inventory", "$name"), ("lat", "$x\u00b0"), (variable, "$y")],
    )
    l_y_max = l_max / l_scale * 1.1 if l_max > 0 else 1.0
    l_fig.y_range = Range1d(start=0, end=l_y_max)

    for i, (label, (mids, density)) in enumerate(lat_series.items()):
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        scaled = density / l_scale
        l_fig.line(mids, scaled, legend_label=label, color=c,
                   line_width=2, name=label)
        l_fig.scatter(mids, scaled, marker=m, color=c, size=7, name=label)

    l_fig.legend.click_policy = "hide"
    l_fig.legend.location = legend_loc_l

    return p_fig, l_fig


# ======================================================================
# Tab layout
# ======================================================================


def panel(state):
    """Return the inventories tab content.

    Args:
        state (AppState): Shared application state.
    """

    # widgets used
    inventory_select = pn.widgets.CheckBoxGroup(
        name="Emission inventories",
        options=[],
        value=[],
    )
    variable_select = pn.widgets.Select(
        name="Variable",
        options=[],
        width=150,
    )
    p_bin_slider = pn.widgets.IntSlider(
        name="Pressure bin width [hPa]", start=10, end=200, step=10, value=50
    )
    lat_bin_slider = pn.widgets.IntSlider(
        name="Latitude bin width [\u00b0]", start=1, end=30, step=1, value=10
    )

    _LEGEND_LOCATIONS = [
        "top_left", "top_center", "top_right",
        "center_left", "center", "center_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]
    legend_v_select = pn.widgets.Select(
        name="Vertical profile legend",
        options=_LEGEND_LOCATIONS,
        value="bottom_right",
    )
    legend_l_select = pn.widgets.Select(
        name="Latitudinal profile legend",
        options=_LEGEND_LOCATIONS,
        value="top_left",
    )

    # Persistent panes — update .object rather than replacing the pane
    # to avoid "dropping a patch" warnings
    status_pane = pn.pane.Markdown("")
    plot_pane_v = pn.pane.Bokeh(None, sizing_mode="stretch_width")
    plot_pane_l = pn.pane.Bokeh(None, sizing_mode="stretch_width")

    # Dataset cache: option_str -> xarray.Dataset
    # _file_map:    option_str -> (inv_dir, filename)
    # Using the option string (e.g. "[base] rnd_inv_2020.nc") as the key
    # means main and base versions of the same filename are kept separate.
    _cache = {}
    _file_map = {}

    # ------------------------------------------------------------------
    # Redraw helper
    # ------------------------------------------------------------------

    def _update_plots():
        """Redraw both profile plots for the current selections."""
        selected = inventory_select.value
        variable = variable_select.value

        if not selected or not variable:
            plot_pane_v.object = None
            plot_pane_l.object = None
            return

        datasets = {}
        skipped = []
        for opt in selected:
            if opt in _cache:
                ds = _cache[opt]
                if variable not in ds.data_vars:
                    year = ds.attrs.get("Inventory_Year", "?")
                    is_base = opt.startswith("[base] ")
                    stem = Path(opt.replace("[base] ", "")).stem
                    skipped.append(f"{'[base] ' if is_base else ''}{stem} ({year})")
                    continue
                year = ds.attrs.get("Inventory_Year", "?")
                is_base = opt.startswith("[base] ")
                stem = Path(opt.replace("[base] ", "")).stem
                label = f"{'[base] ' if is_base else ''}{stem} ({year})"
                datasets[label] = ds

        if skipped:
            status_pane.object = (
                f"\u26a0\ufe0f **{variable}** not available in: "
                f"{', '.join(skipped)} \u2014 skipped."
            )

        if not datasets:
            plot_pane_v.object = None
            plot_pane_l.object = None
            return

        try:
            fig_v, fig_l = _build_profile_figures(
                datasets, variable,
                p_bin_width=p_bin_slider.value,
                lat_bin_width=lat_bin_slider.value,
                legend_loc_v=legend_v_select.value,
                legend_loc_l=legend_l_select.value,
            )
            plot_pane_v.object = fig_v
            plot_pane_l.object = fig_l
        except Exception as e:
            status_pane.object = f"\u274c Plot error: {e}"
            plot_pane_v.object = None
            plot_pane_l.object = None

    # ------------------------------------------------------------------
    # Live updates from the sidebar's edited configuration
    # ------------------------------------------------------------------

    def _on_edited_config_changed(event):
        """React to live edits in the sidebar configuration.

        Only refreshes the inventory list (and prunes the cache) when
        the set of inventory files actually changed — unrelated edits
        (species, time range, etc.) leave the current selection and
        plots untouched.

        Args:
            event: Param event carrying the current edited_config dict.
        """
        config = event.new

        if config is None:
            _cache.clear()
            _file_map.clear()
            inventory_select.options = []
            inventory_select.value = []
            variable_select.options = []
            plot_pane_v.object = None
            plot_pane_l.object = None
            status_pane.object = "\u26a0\ufe0f Create or load a configuration first."
            return

        inv_cfg = config.get("inventories", {})
        main_files = list(inv_cfg.get("files", []))
        rel_to_base = inv_cfg.get("rel_to_base", False)
        base_files = list(inv_cfg.get("base", {}).get("files", [])) if rel_to_base else []

        # Main files are shown as-is; base files are prefixed so
        # the same filename in both directories is distinguishable.
        inv_options = main_files + [f"[base] {f}" for f in base_files]

        if inv_options == inventory_select.options:
            return  # Nothing relevant changed — leave selection/plots as-is

        # Build the file map so the loader knows which dir each entry lives in.
        _file_map.clear()
        inv_dir = inv_cfg.get("dir", "")
        base_dir = inv_cfg.get("base", {}).get("dir", "")
        for f in main_files:
            _file_map[f] = (inv_dir, f)
        for f in base_files:
            _file_map[f"[base] {f}"] = (base_dir, f)

        # Drop cached datasets for options that are no longer available.
        for opt in list(_cache):
            if opt not in inv_options:
                del _cache[opt]

        prev_selected = inventory_select.value
        inventory_select.options = inv_options
        inventory_select.value = [f for f in prev_selected if f in inv_options]
        status_pane.object = ""

    state.param.watch(_on_edited_config_changed, "edited_config")

    # ------------------------------------------------------------------
    # Inventory selection changed → load & update variable list
    # ------------------------------------------------------------------

    def _on_inventory_changed(event):
        """Load newly selected inventories and refresh the variable list.

        Args:
            event: Param event carrying the list of selected filenames.
        """
        selected = event.new
        if not selected or not state.edited_config:
            plot_pane_v.object = None
            plot_pane_l.object = None
            return

        old_cwd = os.getcwd()
        try:
            os.chdir(state.working_dir)
            for opt in selected:
                if opt not in _cache and opt in _file_map:
                    inv_dir, filename = _file_map[opt]
                    _cache[opt] = load_inventory(state.working_dir, inv_dir, filename)
        except Exception as e:
            status_pane.object = f"\u274c Failed to load inventory: {e}"
            return
        finally:
            os.chdir(old_cwd)

        # Collect variables available across all selected inventories,
        # taking the union so the dropdown is as complete as possible.
        all_vars: set = set()
        for opt in selected:
            if opt in _cache:
                all_vars.update(get_numeric_vars(_cache[opt]))
        numeric_vars = sorted(all_vars)

        prev_var = variable_select.value
        variable_select.options = numeric_vars
        if prev_var in numeric_vars:
            variable_select.value = prev_var
        elif numeric_vars:
            variable_select.value = numeric_vars[0]

        labels = []
        missing_warnings = []
        for opt in selected:
            if opt not in _cache:
                continue
            ds = _cache[opt]
            year = ds.attrs.get("Inventory_Year", "?")
            is_base = opt.startswith("[base] ")
            stem = Path(opt.replace("[base] ", "")).stem
            label = f"{'[base] ' if is_base else ''}{stem} ({year})"
            labels.append(label)
            ds_vars = set(get_numeric_vars(ds))
            missing = all_vars - ds_vars
            if missing:
                missing_warnings.append(
                    f"**{label}** is missing: {', '.join(sorted(missing))}"
                )

        status_lines = [f"\u2705 Loaded: {', '.join(labels)}"]
        if missing_warnings:
            status_lines.append(
                "\u26a0\ufe0f Some variables are not available in all inventories:"
            )
            status_lines.extend(f"- {w}" for w in missing_warnings)
        status_pane.object = "\n\n".join(status_lines)

        _update_plots()

    inventory_select.param.watch(_on_inventory_changed, "value")

    # --------------------------------------------------------------------
    # catch changes in the variable choice, bin sizes and legend locations
    # --------------------------------------------------------------------

    variable_select.param.watch(lambda event: _update_plots(), "value")
    p_bin_slider.param.watch(lambda event: _update_plots(), "value")
    lat_bin_slider.param.watch(lambda event: _update_plots(), "value")
    legend_v_select.param.watch(lambda event: _update_plots(), "value")
    legend_l_select.param.watch(lambda event: _update_plots(), "value")

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    if state.edited_config is None:
        status_pane.object = "\u26a0\ufe0f Create or load a configuration first."

    # --------------------------------------------------------------------
    # Layout: three control cards side-by-side, full-width plot card below
    # --------------------------------------------------------------------

    card_inventories = pn.Card(
        inventory_select,
        status_pane,
        title="Emission inventories",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_display = pn.Card(
        variable_select,
        p_bin_slider,
        lat_bin_slider,
        legend_v_select,
        legend_l_select,
        title="Display options",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_extra = pn.Card(
        pn.pane.Markdown("*More options coming soon.*"),
        title="Additional options",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_plots = pn.Card(
        pn.Row(plot_pane_v, plot_pane_l, sizing_mode="stretch_width"),
        title="Profiles",
        collapsible=False,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.Row(
            card_inventories,
            card_display,
            card_extra,
            sizing_mode="stretch_width",
            styles={"gap": "10px", "align-items": "stretch"},
        ),
        card_plots,
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )
