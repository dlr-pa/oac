"""Plot routines for the OpenAirClim framework."""

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import xarray as xr

# %config InlineBackend.figure_format='retina'
BINS = 50


def plot_inventory_vertical_profiles(inv_dict: dict, output_dir: str | Path) -> None:
    """Plots vertical emission profiles of dictionary of inventories.

    Args:
        inv_dict (dict): Dictionary of xarray Datasets,
            keys are years of emission inventories
        output_dir (str or Path): Directory to save the plot to
    """
    n_inv = len(inv_dict.keys())
    fig, axs = plt.subplots(ncols=n_inv, sharex=True, sharey=True, num="Inventories")
    if n_inv == 1:
        year, inv = next(iter(inv_dict.items()))
        axs.hist(
            inv.plev.values,
            bins=BINS,
            weights=inv.fuel.values,
            histtype="step",
            orientation="horizontal",
        )
        axs.set_title(year)
        axs.grid(True)
        # axs.set_xlabel("fuel (kg)")
        # axs.set_ylabel("plev (hPa)")
    else:
        # axs[0].set_xlabel("fuel (kg)")
        # axs[0].set_ylabel("plev (hPa)")
        i = 0
        for year, inv in inv_dict.items():
            axs[i].hist(
                inv.plev.values,
                bins=BINS,
                weights=inv.fuel.values,
                histtype="step",
                orientation="horizontal",
            )
            axs[i].set_title(year)
            axs[i].grid(True)
            i = i + 1
    fig.supxlabel("fuel (kg)")
    fig.supylabel("plev (hPa)")
    plt.gca().invert_yaxis()
    fig.savefig(Path(output_dir) / "inventory_vertical_profiles.png")


def _group_vars_by_species(result: xr.Dataset) -> dict[str, list[str]]:
    """Groups the "type_species" data variable names in result by species.

    Args:
        result (xarray.Dataset): Dataset whose data variable names follow
            the "type_species" naming pattern (e.g. "RF_CO2").

    Returns:
        dict[str, list[str]]: Mapping of species name to the list of metric
            (var_type) prefixes found for that species.
    """
    fig_dic: dict[str, list[str]] = {}
    pattern = "(.+)_(.+)"
    for var_name in result:
        match = re.search(pattern, str(var_name))
        if match is None:
            raise ValueError(
                f"Data variable '{var_name}' does not match the expected "
                "'type_species' naming pattern."
            )
        var_type = match.group(1)
        var_spec = match.group(2)
        fig_dic.setdefault(var_spec, []).append(var_type)
    return fig_dic


def _subplot_layout(num_plots: int) -> tuple[int, int]:
    """Determines the number of subplot rows/columns for a given plot count.

    Args:
        num_plots (int): Number of subplots required for one species.

    Returns:
        tuple[int, int]: Number of rows and columns for the subplot grid.

    Raises:
        ValueError: If num_plots exceeds the supported maximum of 9.
    """
    if num_plots == 1:
        return 1, 1
    if num_plots == 2:  # noqa: PLR2004
        return 1, 2
    if num_plots in (3, 4):
        return 2, 2
    if num_plots in range(5, 10):
        return 3, 3
    raise ValueError("Number of plots per species is limited to 9.")


def plot_results(
    config: dict, result_dic: dict, ac: str = "TOTAL", **kwargs: Any
) -> None:
    """Plots results from dictionary of :class:`xarray.Dataset`s.

    Args:
        config (dict): Configuration dictionary from config file
        result_dic (dict): Dictionary of xarrays
        ac (str, optional): Aircraft identifier, defaults to TOTAL
        **kwargs (Line2D properties, optional): kwargs are parsed to matplotlib
            plot command to specify properties like a line label, linewidth,
            antialiasing, marker face color

    Raises:
        ValueError: If more than 9 subplots per species are parsed, or if
            the requested aircraft identifier is not found
    """
    title = config["output"]["name"]
    output_dir = config["output"]["dir"]
    for result_name, result in result_dic.items():
        # handle multi-aircraft results
        if "ac" in result.dims:
            if ac not in result.coords["ac"].values:
                raise ValueError(
                    f"'ac' coordinate exists in {result_name}, but no '{ac}' "
                    "entry found."
                )
            result_ac = result.sel(ac=ac)
        else:
            result_ac = result
        # Get prefixes (metric) and suffixes (species)
        fig_dic = _group_vars_by_species(result_ac)
        #  Iterate over species and metrics
        for spec, var_type_arr in fig_dic.items():
            # Get number of required rows and columns for suplots
            num_rows, num_cols = _subplot_layout(len(var_type_arr))
            # Generate figure and subplots
            fig = plt.figure(title + ": " + spec)
            # fig.tight_layout()
            plt_i = 1
            for var_type in var_type_arr:
                axis = fig.add_subplot(num_rows, num_cols, plt_i)
                result_ac[var_type + "_" + spec].plot(**kwargs)
                axis.ticklabel_format(axis="y", scilimits=(-3, 3))
                axis.grid(True)
                plt_i = plt_i + 1
            fig.savefig(Path(output_dir) / f"{result_name}_{spec}.png")


def plot_concentrations(config: dict, spec: str, conc_dict: dict) -> None:
    """Plot species concentration change colormaps, one colormap for each year.

    Args:
        config (dict): Configuration dictionary from config file
        spec (str): Species name
        conc_dict (dict): Dictionary of time series numpy arrays (time, lat, plev),
            keys are species
    """
    output_dir = config["output"]["dir"]
    conc = conc_dict[spec][("conc_" + spec)]
    plot_object = conc.plot(x="lat", y="plev", col="time", col_wrap=4)
    axs = plt.gca()
    axs.invert_yaxis()
    fig = plot_object.fig
    fig.canvas.manager.set_window_title(spec)
    fig.savefig(Path(output_dir) / f"conc_{spec}.png")
