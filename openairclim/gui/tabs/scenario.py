"""Scenario tab: visualise the emission scenario over time."""

import os
from pathlib import Path

import panel as pn

from ..components.utils import COLORS, MARKERS, auto_scale, load_inventory, get_numeric_vars


# ======================================================================
# Plot builders
# ======================================================================


def _build_global_sum_figure(years, sums, variable, base_unit, t_start, t_end):
    """Create a Bokeh scatter + line of global variable sum vs inventory year.

    Args:
        years (list): Inventory years (x values).
        sums (list): Global sums of the variable (y values, raw units).
        variable (str): Variable name, used for axis label.
        base_unit (str): Unit string from dataset attributes.
        t_start (int): Simulation start year (for axis extent).
        t_end (int): Simulation end year (for axis extent).

    Returns:
        bokeh.plotting.Figure: The assembled figure.
    """
    from bokeh.models import Range1d
    from bokeh.plotting import figure

    scale, prefix = auto_scale(max(sums) if sums else 1.0)
    scaled = [s / scale for s in sums]
    margin = max(2, int((t_end - t_start) * 0.05))

    fig = figure(
        title=f"Global {variable} sum",
        x_axis_label="Year",
        y_axis_label=f"{variable} [{prefix}{base_unit}]",
        height=420,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=[("Year", "$x{0}"), (variable, "$y")],
    )
    fig.x_range = Range1d(start=t_start - margin, end=t_end + margin)
    fig.y_range = Range1d(start=0, end=max(scaled) * 1.1 if scaled else 1.0)
    fig.line(years, scaled, color=COLORS[0], line_width=2)
    fig.scatter(years, scaled, marker=MARKERS[0], color=COLORS[0], size=8)

    return fig


def _build_norm_figure(ds, norm_variable):
    """Create a Bokeh line plot of a norm time evolution variable vs time.

    Args:
        ds (xarray.Dataset): Loaded time evolution dataset (type "norm").
        norm_variable (str): Data variable to plot.

    Returns:
        bokeh.plotting.Figure: The assembled figure.
    """
    from bokeh.models import Range1d
    from bokeh.plotting import figure

    time_vals = ds["time"].values.tolist()
    data_vals = ds[norm_variable].values.tolist()

    base_unit = ds[norm_variable].attrs.get("units", "?")
    scale, prefix = auto_scale(max(abs(v) for v in data_vals) if data_vals else 1.0)
    scaled = [v / scale for v in data_vals]

    fig = figure(
        title=f"{norm_variable} — time evolution",
        x_axis_label="Year",
        y_axis_label=f"{norm_variable} [{prefix}{base_unit}]",
        height=420,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=[("Year", "$x{0}"), (norm_variable, "$y")],
    )
    fig.line(time_vals, scaled, color=COLORS[1], line_width=2)
    fig.scatter(time_vals, scaled, marker=MARKERS[1], color=COLORS[1], size=5)

    return fig


# ======================================================================
# Tab layout
# ======================================================================


def panel(state):
    """Return the scenario tab content.

    Args:
        state (AppState): Shared application state.
    """
    # ── widgets ──────────────────────────────────────────────────────
    variable_select = pn.widgets.Select(
        name="Emission variable",
        options=[],
    )
    norm_var_select = pn.widgets.Select(
        name="Time evolution variable",
        options=[],
        visible=False,
    )
    status_left = pn.pane.Markdown("")
    status_right = pn.pane.Markdown("")

    # Persistent panes to avoid "dropping a patch" warnings
    plot_pane_sum = pn.pane.Bokeh(None, sizing_mode="stretch_width")
    plot_pane_norm = pn.pane.Bokeh(None, sizing_mode="stretch_width")

    # ── internal state ────────────────────────────────────────────────
    # Inventory cache: filename -> xarray.Dataset (main inventories only)
    _cache = {}
    # Time evolution file state
    _evo = {"ds": None, "type": None}
    # What was last loaded (for change detection in _on_edited_config_changed)
    _loaded = {"inv_files": [], "evo_path": None}

    # ── helpers ───────────────────────────────────────────────────────

    def _evo_path_from_config():
        """Return the absolute path to the time evolution file, or None.

        Returns:
            str or None: Absolute path if a file is configured, else None.
        """
        config = state.edited_config
        if not config:
            return None
        time_cfg = config.get("time", {})
        evo_file = time_cfg.get("file")
        if not evo_file:
            return None
        evo_dir = time_cfg.get("dir", "")
        return str(Path(evo_dir) / evo_file)

    def _load_inventories():
        """Load main inventory files from the current config into _cache.

        Updates _cache in place and refreshes the variable dropdown.
        """
        config = state.edited_config
        if not config:
            return

        inv_cfg = config.get("inventories", {})
        inv_dir = inv_cfg.get("dir", "")
        inv_files = list(inv_cfg.get("files", []))

        old_cwd = os.getcwd()
        try:
            if state.working_dir:
                os.chdir(state.working_dir)
            for f in inv_files:
                if f not in _cache:
                    _cache[f] = load_inventory(state.working_dir, inv_dir, f)
        except Exception as e:
            status_left.object = f"\u26a0\ufe0f Failed to load inventory: {e}"
            return
        finally:
            os.chdir(old_cwd)

        # Prune stale entries
        for f in list(_cache):
            if f not in inv_files:
                del _cache[f]

        # Update variable dropdown from the union of all loaded inventories
        all_vars: set = set()
        for f in inv_files:
            if f in _cache:
                all_vars.update(get_numeric_vars(_cache[f]))
        numeric_vars = sorted(all_vars)

        prev = variable_select.value
        variable_select.options = numeric_vars
        if prev in numeric_vars:
            variable_select.value = prev
        elif numeric_vars:
            variable_select.value = numeric_vars[0]

        _loaded["inv_files"] = list(inv_files)
        status_left.object = ""

    def _load_evo():
        """Load the time evolution file (if configured) into _evo.

        Updates _evo in place, sets the norm variable dropdown, and
        writes a status message if the file is missing or invalid.
        """
        import xarray as xr

        evo_path = _evo_path_from_config()
        _loaded["evo_path"] = evo_path

        if not evo_path:
            _evo["ds"] = None
            _evo["type"] = None
            norm_var_select.visible = False
            norm_var_select.options = []
            status_right.object = ""
            return

        try:
            ds = xr.load_dataset(evo_path)
        except Exception as e:
            status_right.object = (
                f"\u26a0\ufe0f Could not load time evolution file: {e}"
            )
            _evo["ds"] = None
            _evo["type"] = None
            norm_var_select.visible = False
            return

        evo_type = ds.attrs.get("Type")
        _evo["ds"] = ds
        _evo["type"] = evo_type

        if evo_type is None:
            status_right.object = (
                "\u26a0\ufe0f Time evolution file has no **Type** attribute "
                "(expected `norm` or `scaling`)."
            )
            norm_var_select.visible = False
            norm_var_select.options = []
        elif evo_type == "norm":
            status_right.object = ""
            evo_vars = sorted(ds.data_vars)
            prev = norm_var_select.value
            norm_var_select.options = evo_vars
            norm_var_select.value = prev if prev in evo_vars else (evo_vars[0] if evo_vars else "")
            norm_var_select.visible = True
        elif evo_type == "scaling":
            status_right.object = (
                "\u2139\ufe0f Scaling time evolution \u2014 "
                "visualisation not yet supported."
            )
            norm_var_select.visible = False
            norm_var_select.options = []
        else:
            status_right.object = (
                f"\u26a0\ufe0f Unknown time evolution type: `{evo_type}`."
            )
            norm_var_select.visible = False
            norm_var_select.options = []

    # ── plot updaters ─────────────────────────────────────────────────

    def _update_sum_plot():
        """Redraw the global sum scatter from the loaded inventory cache."""
        config = state.edited_config
        variable = variable_select.value

        if not config or not variable:
            plot_pane_sum.object = None
            return

        inv_files = list(config.get("inventories", {}).get("files", []))
        t_cfg = config.get("time", {}).get("range", [None, None, 1])
        t_start, t_end = t_cfg[0], t_cfg[1]

        points = []
        base_unit = "?"
        for f in inv_files:
            if f not in _cache:
                continue
            ds = _cache[f]
            if variable not in ds.data_vars:
                continue
            year = ds.attrs.get("Inventory_Year")
            if year is None:
                continue
            base_unit = ds[variable].attrs.get("units", "?")
            points.append((int(year), float(ds[variable].sum().item())))

        if not points:
            plot_pane_sum.object = None
            return

        points.sort(key=lambda p: p[0])
        years = [p[0] for p in points]
        sums = [p[1] for p in points]

        # Fall back to inventory year range if time config is incomplete
        if t_start is None:
            t_start = years[0]
        if t_end is None:
            t_end = years[-1]

        try:
            plot_pane_sum.object = _build_global_sum_figure(
                years, sums, variable, base_unit, t_start, t_end
            )
        except Exception as e:
            status_left.object = f"\u274c Plot error: {e}"
            plot_pane_sum.object = None

    def _update_norm_plot():
        """Redraw the time evolution plot for norm-type files."""
        if _evo["type"] != "norm" or _evo["ds"] is None:
            plot_pane_norm.object = None
            return

        norm_variable = norm_var_select.value
        if not norm_variable or norm_variable not in _evo["ds"].data_vars:
            plot_pane_norm.object = None
            return

        try:
            plot_pane_norm.object = _build_norm_figure(_evo["ds"], norm_variable)
        except Exception as e:
            status_right.object = f"\u274c Plot error: {e}"
            plot_pane_norm.object = None

    def _update_plots():
        """Redraw both plots."""
        _update_sum_plot()
        _update_norm_plot()

    # ── config change watcher ─────────────────────────────────────────

    def _on_edited_config_changed(event):
        """React to live edits in the sidebar configuration.

        Reloads inventories if the file list changed, reloads the time
        evolution file if the path changed, then redraws all plots.

        Args:
            event: Param event carrying the current edited_config dict.
        """
        config = event.new

        if config is None:
            _cache.clear()
            _evo["ds"] = None
            _evo["type"] = None
            _loaded["inv_files"] = []
            _loaded["evo_path"] = None
            variable_select.options = []
            norm_var_select.options = []
            norm_var_select.visible = False
            plot_pane_sum.object = None
            plot_pane_norm.object = None
            status_left.object = "\u26a0\ufe0f Create or load a configuration first."
            status_right.object = ""
            return

        inv_files = list(config.get("inventories", {}).get("files", []))
        if inv_files != _loaded["inv_files"]:
            _load_inventories()

        evo_path = _evo_path_from_config()
        if evo_path != _loaded["evo_path"]:
            _load_evo()

        _update_plots()

    state.param.watch(_on_edited_config_changed, "edited_config")

    variable_select.param.watch(lambda e: _update_sum_plot(), "value")
    norm_var_select.param.watch(lambda e: _update_norm_plot(), "value")

    # ── initial state ─────────────────────────────────────────────────

    if state.edited_config is None:
        status_left.object = "\u26a0\ufe0f Create or load a configuration first."
    else:
        _load_inventories()
        _load_evo()
        _update_plots()

    # ── layout ────────────────────────────────────────────────────────

    card_variable = pn.Card(
        variable_select,
        status_left,
        title="Emission variable",
        collapsible=False,
        sizing_mode="stretch_width",
    )
    card_evo = pn.Card(
        norm_var_select,
        status_right,
        title="Time evolution",
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
        pn.Row(plot_pane_sum, plot_pane_norm, sizing_mode="stretch_width"),
        title="Scenario",
        collapsible=False,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.Row(
            card_variable,
            card_evo,
            card_extra,
            sizing_mode="stretch_width",
            styles={"gap": "10px", "align-items": "stretch"},
        ),
        card_plots,
        sizing_mode="stretch_width",
        styles={"gap": "10px", "margin-top": "15px"},
    )
