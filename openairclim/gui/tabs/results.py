"""Results tab: explore OpenAirClim simulation output.

Loading is explicit (two buttons: "Load from config" / "Browse...") rather than
automatic on every config edit. Files are also loaded eagerly rather than
lazily, so the underlying file handle is closed immediately after reading and
never blocks a later run.
"""

from pathlib import Path

import panel as pn

from .. import config_io
from ..components.utils import COLORS, MARKERS, auto_scale

TITLE = """
### Results

Load a results NetCDF file to explore it below. **Load from config**
opens the file that the current configuration's output directory/name point
to, if it already exists. **Browse...** lets you pick a different results
file. Either way, the file is read once into memory rather than kept
open, so it won't block OpenAirClim from overwriting it when you next
click "Run".
"""


# ======================================================================
# Output structure helpers
# ======================================================================

# Known variable-name prefixes that group into physical categories.
# The order here determines the display order in the variable dropdown.
_CATEGORY_PREFIXES = [
    ("dT", "Temperature response"),
    ("RF", "Radiative forcing"),
    ("AGWP", "AGWP"),
    ("AGTP", "AGTP"),
    ("ATR", "ATR"),
    ("conc", "Concentration"),
]


def _load_results(filepath):
    """Read a results NetCDF file into memory and return an xarray Dataset.

    Uses the eager `xr.load_dataset` to ensure that the results file is not
    blocked, thereby preventing an OpenAirClim run. Promotes `ac` to a
    coordinate if stored as a data variable, mirroring the inventory loading
    pattern.

    Args:
        filepath (str or Path): Path to the NetCDF file.

    Returns:
        xarray.Dataset: Loaded dataset.

    Raises:
        Exception: Propagated from xarray if the file cannot be read.
    """
    import xarray as xr

    ds = xr.load_dataset(filepath)
    promote = [c for c in ("ac",) if c in ds.data_vars]
    if promote:
        ds = ds.set_coords(promote)
    return ds


def _candidate_results_path(state):
    """Return the results file path implied by the current config, or None.

    Args:
        state (AppState): Shared application state.

    Returns:
        Path or None: `output.dir/output.name.nc`, resolved against
            `state.working_dir`, or None if the config has no output
            dir/name set yet.
    """
    config = state.edited_config
    if not config:
        return None
    output_cfg = config.get("output", {})
    out_dir = output_cfg.get("dir", "")
    out_name = output_cfg.get("name", "")
    if not (out_dir and out_name):
        return None
    out_dir_path = (
        config_io.resolve_dir(state.working_dir, out_dir)
        if state.working_dir else Path(out_dir)
    )
    return out_dir_path / f"{out_name}.nc"


def _get_time_coord(ds):
    """Find the name of the time coordinate in a dataset.

    Args:
        ds (xarray.Dataset): Results dataset.

    Returns:
        str or None: Name of the time coordinate, or None if not found.
    """
    for name in ("time", "t", "year", "years"):
        if name in ds.coords or name in ds.dims:
            return name
    # Fall back to the first integer/float coordinate
    for name, coord in ds.coords.items():
        if coord.dtype.kind in ("i", "f"):
            return name
    return None


def _categorise_variables(ds):
    """Group data variables by physical category based on name prefixes.

    Variables that don't match any known prefix land in "Other".

    Args:
        ds (xarray.Dataset): Results dataset.

    Returns:
        dict: Mapping category_label -> list of variable names.
    """
    categories = {label: [] for _, label in _CATEGORY_PREFIXES}
    categories["Other"] = []

    for varname in ds.data_vars:
        matched = False
        for prefix, label in _CATEGORY_PREFIXES:
            if varname.startswith(prefix):
                categories[label].append(varname)
                matched = True
                break
        if not matched:
            categories["Other"].append(varname)

    return {k: sorted(v) for k, v in categories.items() if v}


def _has_ac_dim(ds, varname):
    """Return True if the variable has an aircraft dimension.

    Args:
        ds (xarray.Dataset): Results dataset.
        varname (str): Variable name to check.

    Returns:
        bool: True if ``ac`` is a dimension of the variable.
    """
    return "ac" in ds[varname].dims


def _build_figure(ds, time_coord, variables, selected_ac, legend_loc):
    """Create a Bokeh line plot of selected variables over time.

    If the variables have an `ac` dimension, one line is drawn per
    selected aircraft.  Otherwise one line per variable.

    Args:
        ds (xarray.Dataset): Results dataset.
        time_coord (str): Name of the time coordinate.
        variables (list): Variable names to plot.
        selected_ac (list): Aircraft identifiers to show.  Ignored if
            the variables have no `ac` dimension.
        legend_loc (str): Bokeh legend location string.

    Returns:
        bokeh.plotting.Figure: The assembled figure.
    """
    import numpy as np
    from bokeh.plotting import figure

    time_vals = ds[time_coord].values.tolist()

    # Build the list of (label, data_array) series to plot
    series = []
    for varname in variables:
        var = ds[varname]
        if "ac" in var.dims and selected_ac:
            for ac in selected_ac:
                try:
                    data = var.sel(ac=ac).values.tolist()
                    series.append((f"{varname} [{ac}]", data))
                except KeyError:
                    pass
        else:
            # Sum or squeeze out the ac dim if present but not selected
            if "ac" in var.dims:
                try:
                    data = var.sel(ac="TOTAL").values.tolist()
                except KeyError:
                    data = var.isel(ac=0).values.tolist()
            else:
                data = var.values.tolist()
            series.append((varname, data))

    if not series:
        return None

    # Determine y-axis label from variable units (use first variable)
    first_var = ds[variables[0]]
    unit = first_var.attrs.get("units", "")
    long_name = first_var.attrs.get("long_name", variables[0])
    y_label = f"{long_name} [{unit}]" if unit else long_name

    all_vals = [v for _, vals in series for v in vals
                if v is not None and np.isfinite(float(v))]
    y_max = max(all_vals) if all_vals else 1.0
    y_min = min(all_vals) if all_vals else 0.0
    scale, prefix = auto_scale(max(abs(y_max), abs(y_min)))

    if prefix:
        y_label = f"{long_name} [{prefix}{unit}]" if unit else f"{long_name} [{prefix}]"

    fig = figure(
        title=", ".join(variables),
        x_axis_label="Year",
        y_axis_label=y_label,
        height=420,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=[("Year", "$x{0}"), ("Value", "$y")],
    )

    for i, (label, vals) in enumerate(series):
        c = COLORS[i % len(COLORS)]
        m = MARKERS[i % len(MARKERS)]
        scaled = [v / scale if v is not None and np.isfinite(float(v)) else float("nan")
                  for v in vals]
        fig.line(time_vals, scaled, color=c, line_width=2,
                 legend_label=label, name=label)
        fig.scatter(time_vals, scaled, marker=m, color=c, size=5, name=label)

    fig.legend.click_policy = "hide"
    fig.legend.location = legend_loc

    return fig


# ======================================================================
# Tab layout
# ======================================================================


def panel(state):
    """Return the results tab content.

    Args:
        state (AppState): Shared application state.
    """
    # ── internal state ────────────────────────────────────────────────
    _ds = {"dataset": None}

    # ── widgets ───────────────────────────────────────────────────────
    load_from_config_btn = pn.widgets.Button(
        name="Load from config", button_type="primary"
    )
    browse_btn = pn.widgets.Button(name="Browse...", button_type="default")

    status_pane = pn.pane.Markdown(
        "⚠️ Load a results file first."
    )

    category_select = pn.widgets.Select(
        name="Category",
        options=[],
        width=200,
    )
    variable_select = pn.widgets.CheckBoxGroup(
        name="Variables",
        options=[],
        value=[],
    )
    ac_select = pn.widgets.CheckBoxGroup(
        name="Aircraft",
        options=[],
        value=[],
    )
    ac_card_title = pn.pane.Markdown("**Aircraft**")
    ac_section = pn.Column(ac_card_title, ac_select)

    _legend_locations = [
        "top_left", "top_center", "top_right",
        "center_left", "center", "center_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]
    legend_select = pn.widgets.Select(
        name="Legend location",
        options=_legend_locations,
        value="top_left",
    )

    # Persistent Bokeh pane
    plot_pane = pn.pane.Bokeh(None, sizing_mode="stretch_width")

    # ── helpers ───────────────────────────────────────────────────────

    def _update_plot():
        """Redraw the plot for the current widget selections."""
        ds = _ds["dataset"]
        if ds is None:
            plot_pane.object = None
            return

        variables = variable_select.value
        if not variables:
            plot_pane.object = None
            return

        time_coord = _get_time_coord(ds)
        if time_coord is None:
            status_pane.object = "⚠️ No time coordinate found in results."
            plot_pane.object = None
            return

        selected_ac = ac_select.value

        try:
            fig = _build_figure(ds, time_coord, variables, selected_ac, legend_select.value)
            plot_pane.object = fig
        except Exception as e:  # pylint: disable=broad-exception-caught
            status_pane.object = f"❌ Plot error: {e}"
            plot_pane.object = None

    def _load_from_path(path):
        """Load a results file and refresh all widgets.

        Args:
            path (str): Absolute path to the NetCDF file.
        """
        if not path:
            return

        try:
            ds = _load_results(path)
            _ds["dataset"] = ds
        except Exception as e:  # pylint: disable=broad-exception-caught
            status_pane.object = f"❌ Could not load results: {e}"
            return

        # set the variable list directly here rather than relying on
        # _on_category_changed firing
        cats = _categorise_variables(ds)
        cat_options = list(cats.keys())
        category_select.options = cat_options
        if cat_options:
            category_select.value = cat_options[0]
            variable_select.options = cats[cat_options[0]]
            variable_select.value = list(cats[cat_options[0]])
        else:
            variable_select.options = []
            variable_select.value = []

        # Populate aircraft selector if relevant
        if "ac" in ds.coords:
            ac_ids = [str(v) for v in ds["ac"].values]
            ac_select.options = ac_ids
            ac_select.value = ac_ids
            ac_section.visible = True
        else:
            ac_select.options = []
            ac_select.value = []
            ac_section.visible = False

        status_pane.object = (
            f"✅ Loaded `{Path(path).name}` — "
            f"{len(ds.data_vars)} variable(s), "
            f"{len(ds.coords)} coordinate(s)"
        )

        # Explicit redraw - don't rely solely on the widget watchers above,
        # for the same reason: they're a no-op when values didn't change.
        _update_plot()

    def _on_category_changed(event):
        """Update the variable checkboxes when the category changes.

        Args:
            event: Param event.
        """
        ds = _ds["dataset"]
        if ds is None:
            return
        cats = _categorise_variables(ds)
        new_cat = event.new
        options = cats.get(new_cat, [])
        variable_select.options = options
        # Pre-select all variables in the new category
        variable_select.value = list(options)

    def _on_load_from_config_click(_event=None):
        """Load the results file implied by the current config, if any."""
        candidate = _candidate_results_path(state)
        if candidate is None:
            status_pane.object = (
                "⚠️ Current configuration has no output directory/name set."
            )
            return
        if not candidate.exists():
            status_pane.object = f"⚠️ No results file found at `{candidate}`."
            return
        state.results_path = str(candidate)
        _load_from_path(str(candidate))

    def _on_browse_click(_event=None):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select results file",
            filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")],
            initialdir=state.working_dir or None,
        )
        root.destroy()

        if selected:
            path = str(Path(selected).resolve())
            state.results_path = path
            _load_from_path(path)

    load_from_config_btn.on_click(_on_load_from_config_click)
    browse_btn.on_click(_on_browse_click)

    category_select.param.watch(_on_category_changed, "value")
    variable_select.param.watch(lambda e: _update_plot(), "value")
    ac_select.param.watch(lambda e: _update_plot(), "value")
    legend_select.param.watch(lambda e: _update_plot(), "value")

    # ── initial state ─────────────────────────────────────────────────
    # Only loads if a results file was passed explicitly via --results on
    # the command line — no automatic loading from the config otherwise

    if state.results_path:
        _load_from_path(state.results_path)

    ac_section.visible = False

    # ── layout ────────────────────────────────────────────────────────

    card_variables = pn.Card(
        category_select,
        variable_select,
        title="Variables",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_aircraft = pn.Card(
        ac_section,
        title="Aircraft",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_display = pn.Card(
        legend_select,
        title="Display options",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_plot = pn.Card(
        plot_pane,
        title="Results",
        collapsible=False,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.pane.Markdown(TITLE),
        status_pane,
        pn.Row(load_from_config_btn, browse_btn),
        pn.Row(
            card_variables,
            card_aircraft,
            card_display,
            sizing_mode="stretch_width",
            styles={"gap": "10px", "align-items": "stretch"},
        ),
        card_plot,
        sizing_mode="stretch_width",
        styles={"gap": "10px", "margin-top": "15px"},
    )
