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
        ncols=n_inv, sharex=True, sharey=True, figsize=(12, 6), num="Inventories"
    )
    if n_inv == 1:
        year, inv = next(iter(inv_dict.items()))
        axs.hist(
            inv.plev.values,
            bins=BINS,
            weights=inv.fuel.values,
            histtype="stepfilled",
            orientation="horizontal",
            color="lightblue",
            edgecolor="blue",
        )
        axs.set_title(year, fontsize=12)
        axs.grid(True)
    else:
        i = 0
        for year, inv in inv_dict.items():
            axs[i].hist(
                inv.plev.values,
                bins=BINS,
                weights=inv.fuel.values,
                histtype="stepfilled",
                orientation="horizontal",
                color="lightblue",
                edgecolor="blue",
            )
            axs[i].set_title(year, fontsize=12)
            axs[i].grid(True)
            i = i + 1
    fig.supxlabel("fuel (kg)")
    fig.supylabel("plev (hPa)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_results(config, result_dic, **kwargs):
    """Plots results from dictionary of xarrays

    Args:
        config (dic): Configuration dictionary from config file
        result_dic (dic): Dictionary of xarrays
        **kwargs (Line2D properties, optional): kwargs are parsed to matplotlib
            plot command to specify properties like a line label, linewidth,
            antialiasing, marker face color

    Raises:
        IndexError: If more than 9 subplots per species are parsed
    """
    title = config["output"]["name"]
    output_dir = config["output"]["dir"]
    # result_name = key, result = xarray Dataset
    for result_name, result in result_dic.items():
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
            plt_i = 1
            for var_type in var_type_arr:
                axis = fig.add_subplot(num_rows, num_cols, plt_i)
                result[var_type + "_" + spec].plot(**kwargs)
                axis.ticklabel_format(axis="y", scilimits=(-3, 3))
                axis.grid(True, linestyle="--", alpha=0.7)
                axis.set_facecolor("#f9f9f9")
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


def plot_cross_check(data1, data2, labels, title="Cross-Check Plot"):
    """
    Plots a side-by-side comparison of two datasets for cross-checks.

    Args:
        data1 (array-like): First dataset (e.g., input data).
        data2 (array-like): Second dataset (e.g., reference data).
        labels (list): List containing labels for the datasets [label1, label2].
        title (str, optional): Title of the plot. Defaults to 'Cross-Check Plot'.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, num="Cross-Check")
    axs[0].hist(data1, bins=BINS, color="skyblue", edgecolor="black")
    axs[0].set_title(labels[0])
    axs[0].grid(True)
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(data2, bins=BINS, color="salmon", edgecolor="black")
    axs[1].set_title(labels[1])
    axs[1].grid(True)
    axs[1].set_xlabel("Value")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
