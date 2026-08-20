"""Scenario tab: visualise the emission scenario over time.

Two plots, each driven by its own variable dropdown:

- Left ("Inventories") — any numeric variable found in the loaded
  emission inventories, plotted as a raw sum per inventory year. Can
  be split by aircraft type and shown as a relative to yearly total.
- Right ("Time evolution") — whatever variables are actually present
  in the loaded time evolution file, each overlaid with the matching
  inventory-derived quantities.

Unit reconciliation (needed only when overlaying two sources on one
axis) uses `cf_units`. Time evolution's "fuel" is a rate (e.g. "Tg yr-1")
that represents a total accumulated over exactly one year — handled by
multiplying its unit by "yr" before converting (unit algebra, not a
hardcoded numeric factor), which cancels the rate exactly.
"""

import os
from pathlib import Path

import panel as pn

from ..components.utils import COLORS, MARKERS, auto_scale, load_inventory, get_numeric_vars
from ...core.interpolate_time import KEY_TABLE

# Canonical unit each combined (evolution-side) variable is converted to
# before plotting, regardless of what a given file happens to declare —
# keeps axis labels predictable across different loaded files.
CANONICAL_UNITS = {
    "fuel": "Tg",
    "EI_CO2": "1",
    "EI_H2O": "1",
    "EI_NOx": "1",
    "dis_per_fuel": "km kg-1",
}


# ======================================================================
# Unit helpers
# ======================================================================


def _unit_str(raw):
    """Return a cf_units-parseable unit string.

    Args:
        raw (str or None): Unit string from a NetCDF attribute — may be
            missing or "" for a dimensionless quantity (e.g. an
            emission index), which cf_units doesn't treat as
            dimensionless unless given "1" explicitly.

    Returns:
        str: "1" if raw is blank, else raw unchanged.
    """
    return raw if raw else "1"


def _display_unit(unit_str):
    """Return a unit string suitable for an axis label.

    Args:
        unit_str (str): Unit string (possibly "1"/"" for dimensionless).

    Returns:
        str: "-" for dimensionless, else unit_str unchanged.
    """
    return "-" if unit_str in ("1", "", None) else unit_str


def _convert_value(value, src_units, target_units, per_year=False):
    """Convert a single value between unit strings using cf_units.

    Args:
        value (float): Value to convert.
        src_units (str): Source unit string, as declared in the file.
        target_units (str): Target unit string.
        per_year (bool): If True, src_units is a rate per year (e.g.
            "Tg yr-1") representing a total accumulated over exactly
            one year — multiplied by a "yr" unit to cancel the rate
            (exact, via unit algebra) before converting, rather than
            converting the rate itself.

    Returns:
        float: Converted value.

    Raises:
        ValueError: If the units aren't convertible.
    """
    from cf_units import Unit  # type: ignore[import-untyped]

    src = Unit(_unit_str(src_units))
    if per_year:
        src = src * Unit("yr")
    return src.convert(value, Unit(_unit_str(target_units)))


def _convert_ratio(value, numerator_units, denominator_units, target_units):
    """Convert a numerator/denominator ratio to a target unit.

    Args:
        value (float): Ratio value (e.g. sum(CO2) / sum(fuel)).
        numerator_units (str): Numerator's unit string.
        denominator_units (str): Denominator's unit string.
        target_units (str): Target unit string.

    Returns:
        float: Converted value.

    Raises:
        ValueError: If the units aren't convertible.
    """
    from cf_units import Unit

    src = Unit(_unit_str(numerator_units)) / Unit(_unit_str(denominator_units))
    return src.convert(value, Unit(_unit_str(target_units)))


# ======================================================================
# Plot builders
# ======================================================================


def _build_inventory_bar_figure(
    variable_name, unit, categories, years, values_by_category,
    t_start, t_end, relative=False, show_legend=True,
    legend_location="top_left", show_period=True, period_color="#808080",
):
    """Create a stacked-bar Bokeh figure of an inventory variable per year.

    One bar per inventory year, its segments the variable's sum for
    each aircraft-type ("ac") category — or a single "Total" segment
    if not split by category.

    Args:
        variable_name (str): Variable name, for title/axis label.
        unit (str): Unit string, for the y-axis label (ignored if
            relative=True, since relative values are percentages).
        categories (list): Category names (stack order), e.g. the
            sorted "ac" values, or ["Total"] if not split.
        years (list): Inventory years (x positions), sorted.
        values_by_category (dict): category -> list of values, aligned
            with `years`.
        t_start (int or None): Simulation start year, for the shaded
            period annotation — skipped if either is None.
        t_end (int or None): Simulation end year (exclusive) — shading
            drawn up to t_end - 1, the last year actually simulated.
        relative (bool): If True, values are already percentages of
            each year's total (0-100), and the axis is labelled
            accordingly rather than with `unit`.
        show_legend (bool): Whether the legend is visible.
        legend_location (str): Bokeh legend location string.
        show_period (bool): Whether the simulation period is shaded.
        period_color (str): Fill color for the simulation period shading.

    Returns:
        bokeh.plotting.Figure or None: The assembled figure, or None
            if there's no data to plot.
    """
    from bokeh.models import BoxAnnotation, ColumnDataSource, HoverTool, Range1d
    from bokeh.plotting import figure

    if not years or not categories:
        return None

    y_label = f"{variable_name} [% of yearly total]" if relative else f"{variable_name} [{unit}]"

    if relative:
        scale, prefix = 1.0, ""
        max_total = 100.0
    else:
        totals = [
            sum(values_by_category[cat][i] for cat in categories) for i in range(len(years))
        ]
        scale, prefix = auto_scale(max(totals) if totals else 1.0)
        max_total = max(totals) / scale if totals else 1.0
        y_label = f"{variable_name} [{prefix}{unit}]"

    fig = figure(
        title=f"Global {variable_name} sum" + (" by aircraft type" if len(categories) > 1 else ""),
        x_axis_label="Year",
        y_axis_label=y_label,
        height=420,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    fig.y_range = Range1d(start=0, end=max_total * 1.1)

    has_period = show_period and t_start is not None and t_end is not None
    if has_period:
        fig.add_layout(BoxAnnotation(
            left=t_start, right=t_end - 1, fill_alpha=0.08, fill_color=period_color,
            line_color=None, level="underlay",
        ))

    source = ColumnDataSource({
        "x": years,
        **{cat: [v / scale for v in values_by_category[cat]] for cat in categories},
    })
    colors = [COLORS[i % len(COLORS)] for i in range(len(categories))]
    bars = fig.vbar_stack(
        categories, x="x", width=0.8, color=colors, source=source,
        legend_label=categories,
    )
    fig.add_tools(HoverTool(
        tooltips=[("Aircraft type", "$name"), ("Year", "@x"), ("Value", "@$name{0.00}")],
        renderers=bars,
    ))

    if has_period:
        # BoxAnnotation isn't a glyph renderer, so it doesn't get a
        # legend entry on its own — a zero-area dummy glyph matching
        # its fill stands in for it, purely for the legend swatch.
        fig.quad(
            top=0, bottom=0, left=t_start, right=t_end - 1,
            fill_color=period_color, fill_alpha=0.3, line_color=None,
            legend_label="Simulation period",
        )

    if fig.legend:
        fig.legend.click_policy = "hide"
        fig.legend.location = legend_location
        fig.legend.visible = show_legend and (len(categories) > 1 or has_period)

    return fig


def _build_figure(
    title, variable_name, unit, t_start, t_end, evo_points=None, inv_points=None,
    show_legend=True, legend_location="top_left", show_period=True, period_color="#808080",
):
    """Create a Bokeh figure showing an evolution line and/or inventory scatter.

    Args:
        title (str): Figure title.
        variable_name (str): Variable name, for the y-axis label.
        unit (str): Unit string, for the y-axis label (e.g. "Tg").
        t_start (int or None): Simulation start year, for the shaded
            period annotation — skipped if either is None.
        t_end (int or None): Simulation end year (exclusive) — the
            shaded region is drawn up to t_end - 1, the last year
            actually simulated.
        evo_points (tuple, optional): (years, values) for the time
            evolution line+markers.
        inv_points (tuple, optional): (years, values) for the
            inventory scatter.
        show_legend (bool): Whether the legend is visible.
        legend_location (str): Bokeh legend location string.
        show_period (bool): Whether the simulation period is shaded.
        period_color (str): Fill color for the simulation period shading.

    Returns:
        bokeh.plotting.Figure or None: The assembled figure, or None
            if neither evo_points nor inv_points has any data.
    """
    from bokeh.models import BoxAnnotation, HoverTool, Range1d
    from bokeh.plotting import figure

    all_vals = []
    if evo_points:
        all_vals.extend(evo_points[1])
    if inv_points:
        all_vals.extend(inv_points[1])
    if not all_vals:
        return None

    scale, prefix = auto_scale(max(abs(v) for v in all_vals))

    fig = figure(
        title=title,
        x_axis_label="Year",
        y_axis_label=f"{variable_name} [{prefix}{unit}]",
        height=420,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    fig.y_range = Range1d(start=0, end=max(v / scale for v in all_vals) * 1.1)

    has_period = show_period and t_start is not None and t_end is not None
    if has_period:
        fig.add_layout(BoxAnnotation(
            left=t_start, right=t_end - 1, fill_alpha=0.08, fill_color=period_color,
            line_color=None, level="underlay",
        ))

    # Hover is restricted to the marker (scatter) renderers below, not the
    # connecting line — hovering along a line otherwise reports whatever
    # interpolated point sits under the cursor (e.g. "2051" between two
    # yearly markers) rather than only the actual discrete data points.
    marker_renderers = []

    if evo_points:
        years, vals = evo_points
        scaled = [v / scale for v in vals]
        fig.line(years, scaled, color=COLORS[0], line_width=2, legend_label="Time evolution")
        marker_renderers.append(fig.scatter(
            years, scaled, marker=MARKERS[0], color=COLORS[0], size=7,
            legend_label="Time evolution", name="Time evolution",
        ))
    if inv_points:
        years, vals = inv_points
        scaled = [v / scale for v in vals]
        marker_renderers.append(fig.scatter(
            years, scaled, marker=MARKERS[1], color=COLORS[1], size=10,
            legend_label="Inventories", name="Inventories",
        ))

    fig.add_tools(HoverTool(
        tooltips=[("Series", "$name"), ("Year", "@x{0}"), ("Value", "@y")],
        renderers=marker_renderers,
    ))

    if has_period:
        # BoxAnnotation isn't a glyph renderer, so it doesn't get a
        # legend entry on its own — a zero-area dummy glyph matching
        # its fill stands in for it, purely for the legend swatch.
        fig.quad(
            top=0, bottom=0, left=t_start, right=t_end - 1,
            fill_color=period_color, fill_alpha=0.3, line_color=None,
            legend_label="Simulation period",
        )

    if fig.legend:
        fig.legend.click_policy = "hide"
        fig.legend.location = legend_location
        fig.legend.visible = show_legend

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
        name="Inventory variable",
        options=[],
    )
    split_ac_cb = pn.widgets.Checkbox(name="Split by aircraft type (ac)", value=False)
    relative_cb = pn.widgets.Checkbox(
        name="Show relative to yearly total", value=False, visible=False,
    )
    norm_var_select = pn.widgets.Select(
        name="Time evolution variable",
        options=[],
        visible=False,
    )
    status_left = pn.pane.Markdown("")
    status_right = pn.pane.Markdown("")

    # Display options (applied to both plots)
    _legend_locations = [
        "top_left", "top_center", "top_right",
        "center_left", "center", "center_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]
    show_legend_cb = pn.widgets.Checkbox(name="Show legend", value=True)
    legend_loc_select = pn.widgets.Select(
        name="Legend location", options=_legend_locations, value="top_left",
    )
    show_period_cb = pn.widgets.Checkbox(name="Show simulation period", value=True)
    period_color_picker = pn.widgets.ColorPicker(
        name="Simulation period colour", value="#808080",
    )

    # Persistent panes to avoid "dropping a patch" warnings
    plot_pane_sum = pn.pane.Bokeh(None, sizing_mode="stretch_width")
    plot_pane_norm = pn.pane.Bokeh(None, sizing_mode="stretch_width")

    # ── internal state ────────────────────────────────────────────────
    # Inventory cache: filename -> xarray.Dataset (main inventories only)
    _cache = {}
    # Time evolution file state
    _evo = {"ds": None, "type": None}
    # What was last loaded/plotted (for change detection in
    # _on_edited_config_changed, so unrelated config edits elsewhere —
    # e.g. the aircraft tab — don't trigger a full plot rebuild).
    _loaded = {"inv_files": [], "evo_path": None, "sim_range": (None, None)}

    # ── helpers ───────────────────────────────────────────────────────

    def _sim_range():
        """Return (t_start, t_end) from the config, or (None, None).

        Returns:
            tuple: (int or None, int or None).
        """
        config = state.edited_config
        if not config:
            return None, None
        t_cfg = config.get("time", {}).get("range", [None, None, 1])
        return t_cfg[0], t_cfg[1]

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
        # time.dir is normally already absolute (canonicalized by the
        # Config tab as soon as a folder's picked) — but right after a
        # Load, edited_config briefly holds the file's own (possibly
        # relative) dir until the Config tab rebuilds. Path's "/" with
        # an absolute right-hand side ignores the left, so this is
        # correct either way, matching how load_inventory() resolves
        # inventories.dir.
        return str(Path(state.working_dir) / evo_dir / evo_file)

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
        except Exception as e:  # pylint: disable=broad-exception-caught
            status_left.object = f"⚠️ Failed to load inventory: {e}"
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

        # Only offer splitting by aircraft type if at least one loaded
        # inventory actually has an "ac" variable to split by.
        any_has_ac = any(
            "ac" in _cache[f].data_vars for f in inv_files if f in _cache
        )
        split_ac_cb.visible = any_has_ac
        if not any_has_ac:
            split_ac_cb.value = False

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
        except Exception as e:  # pylint: disable=broad-exception-caught
            status_right.object = (
                f"⚠️ Could not load time evolution file: {e}"
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
                "⚠️ Time evolution file has no **Type** attribute "
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
                "ℹ️ Scaling time evolution — "
                "visualisation not yet supported."
            )
            norm_var_select.visible = False
            norm_var_select.options = []
        else:
            status_right.object = (
                f"⚠️ Unknown time evolution type: `{evo_type}`."
            )
            norm_var_select.visible = False
            norm_var_select.options = []

    # ── plot updaters ─────────────────────────────────────────────────

    def _update_sum_plot():
        """Redraw the inventories-only stacked-bar plot for the selected variable."""
        config = state.edited_config
        variable = variable_select.value

        if not config or not variable:
            plot_pane_sum.object = None
            return

        inv_files = list(config.get("inventories", {}).get("files", []))
        split = split_ac_cb.value

        # One (year, {category: value}) entry per inventory year.
        year_data = []
        raw_unit = "?"
        for f in inv_files:
            if f not in _cache:
                continue
            ds = _cache[f]
            if variable not in ds.data_vars:
                continue
            year = ds.attrs.get("Inventory_Year")
            if year is None:
                continue
            raw_unit = ds[variable].attrs.get("units", "?")

            if split and "ac" in ds.data_vars:
                grouped = ds[variable].groupby(ds["ac"]).sum()
                cat_values = {
                    str(cat): float(v)
                    for cat, v in zip(grouped["ac"].values, grouped.values)
                }
            elif split:
                # No "ac" variable in this inventory — matches
                # read_netcdf.py's fallback: every row is treated as
                # aircraft type "DEFAULT".
                cat_values = {"DEFAULT": float(ds[variable].sum().item())}
            else:
                cat_values = {"Total": float(ds[variable].sum().item())}

            year_data.append((int(year), cat_values))

        if not year_data:
            plot_pane_sum.object = None
            return

        year_data.sort(key=lambda p: p[0])
        years = [p[0] for p in year_data]
        categories = sorted({cat for _, cat_values in year_data for cat in cat_values})
        values_by_category = {
            cat: [cat_values.get(cat, 0.0) for _, cat_values in year_data]
            for cat in categories
        }

        relative = relative_cb.value and len(categories) > 1
        if relative:
            totals = [sum(cat_values.values()) for _, cat_values in year_data]
            for cat in categories:
                values_by_category[cat] = [
                    (v / t * 100 if t else 0.0)
                    for v, t in zip(values_by_category[cat], totals)
                ]

        t_start, t_end = _sim_range()

        try:
            plot_pane_sum.object = _build_inventory_bar_figure(
                variable, raw_unit, categories, years, values_by_category,
                t_start, t_end, relative=relative,
                show_legend=show_legend_cb.value,
                legend_location=legend_loc_select.value,
                show_period=show_period_cb.value,
                period_color=period_color_picker.value,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            status_left.object = f"❌ Plot error: {e}"
            plot_pane_sum.object = None

    def _inventory_ratio_points(evo_variable):
        """Compute per-inventory-year points matching an evolution variable.

        For "fuel", this is the raw inventory fuel sum. For every other
        KEY_TABLE entry (EI_CO2, EI_H2O, EI_NOx, dis_per_fuel), it's the
        inventory species-sum divided by the inventory fuel-sum for
        that year — the same emission-index calculation core itself
        performs (see calc_inv_quantities in core/interpolate_time.py)
        — converted into CANONICAL_UNITS[evo_variable].

        A given inventory year is skipped (not an error) if it's
        missing the species and/or fuel needed for this variable.

        Args:
            evo_variable (str): Time evolution data variable name.

        Returns:
            tuple: (years, values) — possibly empty if nothing could
                be computed.
        """
        target_unit = CANONICAL_UNITS.get(evo_variable)
        if target_unit is None:
            return [], []

        inv_species = KEY_TABLE.get(evo_variable)
        config = state.edited_config
        inv_files = list(config.get("inventories", {}).get("files", [])) if config else []

        points = []
        for f in inv_files:
            ds = _cache.get(f)
            if ds is None or "fuel" not in ds.data_vars:
                continue
            year = ds.attrs.get("Inventory_Year")
            if year is None:
                continue
            fuel_sum = float(ds["fuel"].sum().item())
            fuel_units = ds["fuel"].attrs.get("units", "?")

            try:
                if evo_variable == "fuel":
                    value = _convert_value(fuel_sum, fuel_units, target_unit)
                else:
                    if inv_species is None or inv_species not in ds.data_vars:
                        continue
                    if fuel_sum == 0:
                        continue
                    spec_sum = float(ds[inv_species].sum().item())
                    spec_units = ds[inv_species].attrs.get("units", "?")
                    ratio = spec_sum / fuel_sum
                    value = _convert_ratio(ratio, spec_units, fuel_units, target_unit)
            except ValueError:
                # Incompatible units for this file — skip this year
                # rather than failing the whole plot.
                continue

            points.append((int(year), value))

        points.sort(key=lambda p: p[0])
        years = [p[0] for p in points]
        values = [p[1] for p in points]
        return years, values

    def _update_norm_plot():
        """Redraw the time evolution plot, overlaid with inventory data."""
        import xarray as xr

        if _evo["type"] != "norm" or _evo["ds"] is None:
            plot_pane_norm.object = None
            return

        norm_variable = norm_var_select.value
        ds: xr.Dataset = _evo["ds"]
        if not norm_variable or norm_variable not in ds.data_vars:
            plot_pane_norm.object = None
            return

        evo_unit = ds[norm_variable].attrs.get("units", "")
        target_unit = CANONICAL_UNITS.get(norm_variable, evo_unit)

        try:
            evo_raw = ds[norm_variable].values.tolist()
            per_year = norm_variable == "fuel"
            evo_vals = [
                _convert_value(v, evo_unit, target_unit, per_year=per_year)
                for v in evo_raw
            ]
        except ValueError as e:
            status_right.object = f"❌ Could not convert units: {e}"
            plot_pane_norm.object = None
            return

        evo_years = ds["time"].values.tolist()
        inv_years, inv_vals = _inventory_ratio_points(norm_variable)

        t_start, t_end = _sim_range()

        try:
            plot_pane_norm.object = _build_figure(
                f"{norm_variable} — time evolution",
                norm_variable, _display_unit(target_unit), t_start, t_end,
                evo_points=(evo_years, evo_vals),
                inv_points=(inv_years, inv_vals) if inv_years else None,
                show_legend=show_legend_cb.value,
                legend_location=legend_loc_select.value,
                show_period=show_period_cb.value,
                period_color=period_color_picker.value,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            status_right.object = f"❌ Plot error: {e}"
            plot_pane_norm.object = None

    def _update_plots():
        """Redraw both plots."""
        _update_sum_plot()
        _update_norm_plot()

    # ── config change watcher ─────────────────────────────────────────

    def _on_edited_config_changed(event):
        """React to live edits in the sidebar configuration.

        Reloads inventories if the file list changed, reloads the time
        evolution file if the path changed, then redraws all plots — but
        only if something this tab actually cares about changed. This
        watcher fires on *every* edited_config trigger app-wide (e.g.
        every single aircraft-tab table edit), so redrawing unconditionally
        would rebuild both plots on edits that have nothing to do with
        them.

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
            _loaded["sim_range"] = (None, None)
            variable_select.options = []
            norm_var_select.options = []
            norm_var_select.visible = False
            plot_pane_sum.object = None
            plot_pane_norm.object = None
            status_left.object = "⚠️ Create or load a configuration first."
            status_right.object = ""
            return

        inv_files = list(config.get("inventories", {}).get("files", []))
        inv_changed = inv_files != _loaded["inv_files"]
        if inv_changed:
            _load_inventories()

        evo_path = _evo_path_from_config()
        evo_changed = evo_path != _loaded["evo_path"]
        if evo_changed:
            _load_evo()

        sim_range = _sim_range()
        range_changed = sim_range != _loaded["sim_range"]
        _loaded["sim_range"] = sim_range

        if inv_changed or evo_changed or range_changed:
            _update_plots()

    state.param.watch(_on_edited_config_changed, "edited_config")

    variable_select.param.watch(lambda e: _update_sum_plot(), "value")
    norm_var_select.param.watch(lambda e: _update_norm_plot(), "value")
    show_legend_cb.param.watch(lambda e: _update_plots(), "value")
    legend_loc_select.param.watch(lambda e: _update_plots(), "value")
    show_period_cb.param.watch(lambda e: _update_plots(), "value")
    period_color_picker.param.watch(lambda e: _update_plots(), "value")
    relative_cb.param.watch(lambda e: _update_sum_plot(), "value")

    def _on_split_ac_changed(event):
        relative_cb.visible = event.new
        _update_sum_plot()

    split_ac_cb.param.watch(_on_split_ac_changed, "value")

    # ── initial state ─────────────────────────────────────────────────

    if state.edited_config is None:
        status_left.object = "⚠️ Create or load a configuration first."
    else:
        _load_inventories()
        _load_evo()
        _loaded["sim_range"] = _sim_range()
        _update_plots()

    # ── layout ────────────────────────────────────────────────────────

    card_variable = pn.Card(
        variable_select,
        split_ac_cb,
        relative_cb,
        status_left,
        title="Inventories",
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
        show_legend_cb,
        legend_loc_select,
        show_period_cb,
        period_color_picker,
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
