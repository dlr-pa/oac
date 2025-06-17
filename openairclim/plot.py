"""
Plot routines for the OpenAirClim framework
"""

import re
import matplotlib.pyplot as plt


# %config InlineBackend.figure_format='retina'
BINS = 50


def plot_inventory_vertical_profiles(inv_dict):
    """Plots vertical emission profiles of dictionary of inventories

    Args:
        inv_dict (dict): Dictionary of xarray Datasets,
            keys are years of emission inventories
    """
    n_inv = len(inv_dict.keys())
    fig, axs = plt.subplots(
        ncols=n_inv, sharex=True, sharey=True, num="Inventories"
    )
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
    plt.show()


def plot_results(config, result_dic, ac="TOTAL", **kwargs):
    """Plots results from dictionary of xarrays

    Args:
        config (dic): Configuration dictionary from config file
        result_dic (dic): Dictionary of xarrays
        ac (str, optional): Aircraft identifier, defaults to TOTAL
        **kwargs (Line2D properties, optional): kwargs are parsed to matplotlib
            plot command to specify properties like a line label, linewidth,
            antialiasing, marker face color

    Raises:
        IndexError: If more than 9 subplots per species are parsed
    """
    title = config["output"]["name"]
    output_dir = config["output"]["dir"]
    for result_name, result in result_dic.items():
        # handle multi-aircraft results
        if "ac" in result.dims:
            if ac in result.coords["ac"].values:
                result = result.sel(ac=ac)
            else:
                raise ValueError(
                    f"'ac' coordinate exists in {result_name}, but no '{ac}'"
                    "entry found."
                )
        fig_dic = {}
        pattern = "(.+)_(.+)"
        # Get prefixes (metric) and suffixes (species)
        for var_name in result.keys():
            match = re.search(pattern, var_name)
            var_type = match.group(1)
            var_spec = match.group(2)
            # Get the names of different species
            if var_spec not in fig_dic:
                fig_dic.update({var_spec: []})
            else:
                pass
            fig_dic[var_spec].append(var_type)
        #  Iterate over species and metrics
        for spec, var_type_arr in fig_dic.items():
            # Get number of required rows and columns for suplots
            num_plots = len(var_type_arr)
            if num_plots == 1:
                num_rows = 1
                num_cols = 1
            elif num_plots == 2:
                num_rows = 1
                num_cols = 2
            elif num_plots in (3, 4):
                num_rows = 2
                num_cols = 2
            elif num_plots in range(5, 10):
                num_rows = 3
                num_cols = 3
            else:
                raise ValueError(
                    "Number of plots per species is limited to 9."
                )
            # Generate figure and subplots
            fig = plt.figure((title + ": " + spec))
            # fig.tight_layout()
            plt_i = 1
            for var_type in var_type_arr:
                axis = fig.add_subplot(num_rows, num_cols, plt_i)
                result[var_type + "_" + spec].plot(**kwargs)
                axis.ticklabel_format(axis="y", scilimits=(-3, 3))
                axis.grid(True)
                plt_i = plt_i + 1
            fig.savefig(output_dir + result_name + "_" + spec + ".png")
        plt.show()


def plot_concentrations(config, spec, conc_dict):
    """Plot species concentration change colormaps, one colormap for each year

    Args:
        config (dic): Configuration dictionary from config file
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
    fig.savefig(output_dir + "conc_" + spec + ".png")
    plt.show()
