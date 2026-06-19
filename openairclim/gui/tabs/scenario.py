"""Scenario tab: visualise emission inventory profiles."""

import os
from pathlib import Path

import panel as pn

# define unicode superscripts, rather than using math mode
_SUPERSCRIPTS = str.maketrans(
    "0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b"
)

# visual style
COLORS = [
    "#2271B2", "#3DB7E9", "#F748A5", "#359B73",
    "#D55E00", "#E69F00", "#F0E442"
]
MARKERS = [
    "circle", "square", "triangle", "inverted_triangle",
    "plus", "x", "hex", "diamond",
]


def _superscript(n: str):
    """Convert an integer to Unicode superscript characters.

    Args:
        n (int): Number to convert.

    Returns:
        str: Unicode superscript representation.
    """
    return str(n).translate(_SUPERSCRIPTS)


def _load_inventory(working_dir: str, inv_dir: str, inv_file: str):
    """Load a single emission inventory, promoting spatial fields to coords.

    Args:
        working_dir (str): Project working directory.
        inv_dir (str): Inventory subdirectory from config.
        inv_file (str): Inventory filename.

    Returns:
        xarray.Dataset: Loaded inventory with plev, lat, lon as coordinates.
    """
    import xarray as xr

    filepath = Path(working_dir) / inv_dir / inv_file
    ds = xr.load_dataset(filepath)

    # make lat, lon and plev into coordinates for easier manipulation
    coord_names = [c for c in ("plev", "lat", "lon") if c in ds.data_vars]
    if coord_names:
        ds = ds.set_coords(coord_names)

    return ds


def _get_numeric_vars(ds):
    """Return names of plottable numeric data variables. Allows for all
    numeric data within the inventories to be visualised, even if it is not
    used by OpenAirClim.

    Args:
        ds (xarray.Dataset): Emission inventory.

    Returns:
        list: Names of numeric data variables, excluding spatial fields and ac.
    """
    skip = {"ac", "plev", "lat", "lon"}
    return [
        name for name, var in ds.data_vars.items()
        if name not in skip and var.dtype.kind == "f"
    ]


def _auto_scale(max_val):
    """Determine a scaling factor for clean axis labels.

    Values in the range [0.1, 1000) are left unscaled. Otherwise, the
    appropriate power of 10 is extracted and returned as a label prefix
    using Unicode superscript characters.

    Args:
        max_val (float): Maximum value on the axis.

    Returns:
        tuple: (divisor, label_prefix) where divisor is the power of 10
            to divide data by, and label_prefix is a string like
            ``"10\u2078 "`` or ``""`` if no scaling is needed.
    """
    import numpy as np

    if max_val == 0 or not np.isfinite(max_val):
        return 1.0, ""
    exponent = int(np.floor(np.log10(abs(max_val))))
    if -1 <= exponent <= 2:
        return 1.0, ""
    return 10.0 ** exponent, f"10{_superscript(exponent)} "


def _get_unit_params(datasets, variable, use_tg):
    """Determine unit conversion factor and display label. This could be
    updated in the future to allow for further units.

    Args:
        datasets (dict): Mapping of label (str) to xarray.Dataset.
        variable (str): Data variable name.
        use_tg (bool): If True and the variable unit is "kg", convert to Tg.

    Returns:
        tuple: (unit_factor, base_unit) where unit_factor is the
            multiplier to apply to raw data and base_unit is the string
            label for the converted unit.
    """
    first_ds = next(iter(datasets.values()))
    raw_unit = first_ds[variable].attrs.get("units", "?")

    if use_tg and raw_unit == "kg":
        return 1e-9, "Tg"
    return 1.0, raw_unit


def _build_profile_figures(datasets, variable, use_tg):
    """Create two interactive Bokeh figures: vertical and latitudinal profiles.

    Args:
        datasets (dict): Mapping of label (str) to xarray.Dataset.
        variable (str): Data variable to plot.
        use_tg (bool): If True and the variable unit is "kg", convert to Tg.

    Returns:
        tuple: Two bokeh.plotting.Figure objects (vertical, latitudinal).
    """
    import numpy as np
    from bokeh.models import Range1d
    from bokeh.plotting import figure

    unit_factor, base_unit = _get_unit_params(datasets, variable, use_tg)

    # bin edges
    p_bins = np.arange(50, 1050, 50)
    p_bw = float(p_bins[1] - p_bins[0])
    lat_bins = np.arange(-90, 100, 10)
    lat_bw = float(lat_bins[1] - lat_bins[0])

    # compute binned densities for every selected inventory
    plev_series = {}
    lat_series = {}

    for label, ds in datasets.items():
        data = ds[variable] * unit_factor

        p_binned = data.groupby_bins("plev", p_bins).sum().fillna(0)
        p_mids = p_binned.plev_bins.to_index().mid.values.astype(float)
        plev_series[label] = (p_mids, p_binned.values / p_bw)

        lat_binned = data.groupby_bins("lat", lat_bins).sum().fillna(0)
        lat_mids = lat_binned.lat_bins.to_index().mid.values.astype(float)
        lat_series[label] = (lat_mids, lat_binned.values / lat_bw)

    # global maxima and auto-scaling
    p_max = float(max((d.max() for _, d in plev_series.values()), default=1.0))
    l_max = float(max((d.max() for _, d in lat_series.values()), default=1.0))
    p_scale, p_prefix = _auto_scale(p_max)
    l_scale, l_prefix = _auto_scale(l_max)

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
        p_fig.line(scaled, mids, legend_label=label, color=c,
                   line_width=2, name=label)
        p_fig.scatter(scaled, mids, marker=m, color=c, size=7, name=label)

    p_fig.legend.click_policy = "hide"
    p_fig.legend.location = "bottom_right"

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
    l_fig.legend.location = "top_left"

    return p_fig, l_fig


# ======================================================================
# Tab layout
# ======================================================================


def panel(state):
    """Return the scenario tab content.

    Args:
        state (AppState): Shared application state.
    """
    inventory_select = pn.widgets.CheckBoxGroup(
        name="Emission inventories",
        options=[],
        value=[],
    )
    variable_select = pn.widgets.Select(
        name="Variable",
        options=[],
        width=200,
    )
    unit_select = pn.widgets.RadioButtonGroup(
        name="Unit",
        options=["kg", "Tg"],
        value="kg",
        width=150,
    )
    status_pane = pn.pane.Markdown("")

    # Persistent panes — update .object rather than replacing the pane
    # to avoid "dropping a patch" warnings
    plot_pane_v = pn.pane.Bokeh(None, sizing_mode="stretch_width")
    plot_pane_l = pn.pane.Bokeh(None, sizing_mode="stretch_width")
    profile_row = pn.Row(plot_pane_v, plot_pane_l, sizing_mode="stretch_width")

    # Dataset cache: filename -> xarray.Dataset
    _cache = {}

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
        for f in selected:
            if f in _cache:
                ds = _cache[f]
                year = ds.attrs.get("Inventory_Year", "?")
                datasets[f"{Path(f).stem} ({year})"] = ds

        if not datasets:
            plot_pane_v.object = None
            plot_pane_l.object = None
            return

        use_tg = unit_select.value == "Tg"

        try:
            fig_v, fig_l = _build_profile_figures(datasets, variable, use_tg)
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
            inventory_select.options = []
            inventory_select.value = []
            variable_select.options = []
            plot_pane_v.object = None
            plot_pane_l.object = None
            status_pane.object = "\u26a0\ufe0f Create or load a configuration first."
            return

        inv_files = list(config.get("inventories", {}).get("files", []))
        if inv_files == inventory_select.options:
            return  # Nothing relevant changed — leave selection/plots as-is

        for f in list(_cache):
            if f not in inv_files:
                del _cache[f]

        prev_selected = inventory_select.value
        inventory_select.options = inv_files
        inventory_select.value = [f for f in prev_selected if f in inv_files]
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

        config = state.edited_config
        inv_dir = config["inventories"].get("dir", "")

        old_cwd = os.getcwd()
        try:
            os.chdir(state.working_dir)
            for f in selected:
                if f not in _cache:
                    _cache[f] = _load_inventory(state.working_dir, inv_dir, f)
        except Exception as e:
            status_pane.object = f"\u274c Failed to load inventory: {e}"
            return
        finally:
            os.chdir(old_cwd)

        # Update variable dropdown from the first selected inventory
        first_ds = _cache[selected[0]]
        numeric_vars = _get_numeric_vars(first_ds)
        prev_var = variable_select.value
        variable_select.options = numeric_vars
        if prev_var in numeric_vars:
            variable_select.value = prev_var
        elif numeric_vars:
            variable_select.value = numeric_vars[0]

        labels = []
        for f in selected:
            ds = _cache[f]
            year = ds.attrs.get("Inventory_Year", "?")
            labels.append(f"{Path(f).stem} ({year})")
        status_pane.object = f"\u2705 Loaded: {', '.join(labels)}"

        _update_plots()

    inventory_select.param.watch(_on_inventory_changed, "value")

    # ------------------------------------------------------------------
    # Variable changed → toggle unit widget visibility, redraw
    # ------------------------------------------------------------------

    def _on_variable_changed(event):
        """Update unit toggle visibility and redraw plots.

        Args:
            event: Param event carrying the new variable name.
        """
        variable = event.new
        if not variable:
            return

        selected = inventory_select.value
        if selected and selected[0] in _cache:
            raw_unit = _cache[selected[0]][variable].attrs.get("units", "?")
            unit_select.visible = raw_unit != "km"

        _update_plots()

    variable_select.param.watch(_on_variable_changed, "value")

    # ------------------------------------------------------------------
    # Unit changed → redraw
    # ------------------------------------------------------------------

    unit_select.param.watch(lambda event: _update_plots(), "value")

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    if state.edited_config is None:
        status_pane.object = "\u26a0\ufe0f Create or load a configuration first."

    # ------------------------------------------------------------------

    controls = pn.Column(
        pn.pane.Markdown("**Select inventories:**"),
        inventory_select,
        pn.Row(variable_select, pn.Column(pn.pane.Markdown("**Unit:**"), unit_select)),
        status_pane,
    )

    return pn.Column(
        pn.Card(
            controls,
            profile_row,
            title="Emission inventory profiles",
            collapsible=False,
            sizing_mode="stretch_width",
        ),
    )
