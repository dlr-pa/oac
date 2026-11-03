from openairclim.run_repeated_sampling import load_uncertainty_distributions
import openairclim as oac
from openairclim import get_config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import json

"""
Uncertainty and Variance-Based Sensitivity Analysis.
Global Sensitivity Analysis follows methodology by Saltelli et al. (2010)

Implementation J.S. Dunselman (2026) MSc Thesis.
"""

def Get_Sobol_Indices(file_name, N, Distributions, N_bootstraps):
    """
    Calculates the first-order and total-order Sobol' indices S_i and S_Ti, using the estimators 
    by Saltelli et al. (2010) and Janssen (1999) respectively. Includes bootstrapping, and creates a boxplot.

    Args:
        file_name (string): The .csv file directory of the file containing the climate metric outputs.
        N (integer): The number of samples for which the Sobol' indices are calculated. Must be a power of 2.
        Distributions (list): The list of distributions for each uncertain input as defined in uncertainties.py.

    Returns:
        S_i_list (list of floats): 
            List containing the first-order Sobol' indices for the uncertain input parameters.
        S_Ti_list (list of floats): 
            List containing the total-order Sobol' indices for the uncertain input parameters.
        Index_names (list of strings): 
            List containing the names of the Sobol' indices.
        S_i_Bootstrap_df (Pandas dataframe):
            Dataframe containing bootstrap first-order Sobol' indices. 
        S_Ti_Bootstrap_df (Pandas dataframe):
            Dataframe containing bootstrap total-order Sobol' indices. 
    """
    Metric_name = file_name.split("_")[0]
    # Create the correct directory to locate the files containing the climate metrics.

    # Open the contents of the csv file for the desired number of rows.
    Model_output_raw = pd.read_csv(file_name, nrows=N, dtype=object, keep_default_na=False)

    if (Model_output_raw == "").any().any():
        raise ValueError(f"{file_name} has empty cells in the first {N} rows. "
                     "Was it generated with GSA_sampling = true?")

    # Check that the desired number of samples is present within the file.
    assert len(Model_output_raw) == N, f"Expected {N} rows, got {len(Model_output_raw)}"

    # Output dataframe created
    Model_output = pd.DataFrame(index=range(N))
    for col in Model_output_raw.columns:
        # Sums each entry to obtain total Climate metric.
        Model_output[col] = Model_output_raw[col].apply(cell_total)
    
    # Number of indices (-2 to account for A and B)
    N_indices = Model_output.shape[1]-2

    # Creates list of index names 
    Index_names = [""] * N_indices
    for i in range(len(Distributions)):
        idx = Distributions[i][4]   # Index representing to which Sobol' index it corresponds.
        name = Distributions[i][0]
        # Merge names if multiple uncertainties contribute to single Sobol' index!
        Index_names[idx] = Index_names[idx] + ("/" if Index_names[idx] else "") + name 

    # Calculate the Sobol' indices for the actual model evaluations.
    S_i_list, S_Ti_list = Calculate_Sobol_Indices(Model_output, N_indices, N)

    # Create dataframes to hold bootstrapped Sobol' indices. Each column will represent one index.
    S_i_Bootstrap_df = pd.DataFrame(np.zeros(shape=(N_bootstraps, N_indices)), columns = Index_names)
    S_Ti_Bootstrap_df = pd.DataFrame(np.zeros(shape=(N_bootstraps, N_indices)), columns = Index_names)

    # Create N_Bootstrap_Samples Sobol' indices.
    for bootstrap in range(N_bootstraps):
        # Get bootstrap output, sampled (with resampling) from the original model output.
        Bootstrap_output = Get_Bootstrap_Sample(Model_output, N) 
        # Calculate the bootstrap Sobol' indices outputs.
        S_i_Bootstrap, S_Ti_Bootstrap = Calculate_Sobol_Indices(Bootstrap_output, N_indices, N)
        # Write bootstrap indices to dataframes.
        S_i_Bootstrap_df.iloc[bootstrap] = S_i_Bootstrap
        S_Ti_Bootstrap_df.iloc[bootstrap] = S_Ti_Bootstrap
        print("Performing Bootstrap iterations: "+ str(bootstrap) + " of "+ str(N_bootstraps), end='\r')

    # Print statement
    print("\nSobol' Indices for " + Metric_name)
    print("=" * 80)
    print(f"{'Parameter':<45} {'S_i':>12} {'S_Ti':>12}")
    print("-" * 80)

    for i in range(len(S_i_list)):
        print(f"{Index_names[i]:<45} {S_i_list[i]:12.3f} {S_Ti_list[i]:12.3f}")

    print("=" * 80)
    print(f"{'Sum':<45} {sum(S_i_list):12.3f} {'N.A.':>12}")

    # Return the Sobol' indices and names.
    return S_i_list, S_Ti_list, Index_names, S_i_Bootstrap_df, S_Ti_Bootstrap_df

def Calculate_Sobol_Indices(Output_Matrix, N_indices, N_samples):
    """
    Calculates the first-order and total-order Sobol' indices S_i and S_Ti, using the estimators 
    by Saltelli et al. (2010) and Janssen (1999) respectively. 

    Args:
        Output_Matrix (pd Dataframe): Dataframe containing model evaluations. Columns represent A, B, A_B^i.
        N_indices (integer): The number of Sobol' indices expected.
        N_samples (list): The number of samples for which the Sobol' indices are calculated. Must be a power of 2.

    Returns:
        S_i_list (list of floats): 
            List containing the first-order Sobol' indices for the uncertain input parameters.
        S_Ti_list (list of floats): 
            List containing the total-order Sobol' indices for the uncertain input parameters.
    """
    # Create list of model outputs for matrix A and B, once.
    yA = Output_Matrix["A"].values
    yB = Output_Matrix["B"].values

    # Calculate the variance, required to normalize for S_i and S_Ti
    VarY = np.var(yA, ddof=1)
    
    S_i_list = []       # Empty list to append first-order Sobol' index.
    S_Ti_list = []      # Empty list to append total-order Sobol' index.

    for i in range(N_indices):
        yABi = Output_Matrix[str(i+1)]

        S_i_estimator = 1/N_samples * sum((yB)*(yABi-yA))    # Saltelli et al. (2010)
        S_i_list.append(S_i_estimator/VarY)         

        S_Ti_estimator = 1/(2*N_samples) * sum((yA-yABi)**2) # Janssen (1999)
        S_Ti_list.append(S_Ti_estimator/VarY)

    return S_i_list, S_Ti_list

def Get_Bootstrap_Sample(Output_Matrix, N_Samples):
    """
    Calculates the bootstrap sample by randomly sampling from rows of the original output matrix.
    
    Args:
        Output_Matrix (pd Dataframe): Dataframe containing model evaluations. Columns represent A, B, A_B^i.
        N_samples (integer): The number of samples for which the Sobol' indices are calculated. Must be a power of 2.

    Returns:
        Bootstrap_Output (pd Dataframe): 
            Dataframe containing the resampled rows using for bootstrapping.
    """
    # Obtain random samples for bootstrapping.
    indices = np.random.randint(0, N_Samples, size=N_Samples)
    Bootstrap_Output = Output_Matrix.iloc[indices].reset_index(drop=True)
    # Return the Output matrix with resampled rows.
    return Bootstrap_Output

def parse_cell(cell):
    v = json.loads(cell)
    return v if isinstance(v, dict) else v 

def cell_total(cell):
    v = json.loads(cell)
    return sum(v.values()) if isinstance(v, dict) else sum(v)

def Calculate_distribution(file_name, N, Species):
    """
    Calculates the bootstrap sample by randomly sampling from rows of the original output matrix. Presents
    boxplot containing the total and species contribution to the climate metrics.
    
    Args:
        file_name (string): File name of results CSV file for desired metric.
        N (integer): The number of samples for which distributions are calculated. Must be a power of 2.
        Species (list of strings): Contains the species evaluated. Must be unchanged thoughout model runs.

    Returns:
        Bootstrap_Output (pd Dataframe): 
            Dataframe containing the resampled rows using for bootstrapping.
    """
    Metric_name = file_name.split("_")[0]
    Model_output_raw = pd.read_csv(file_name, nrows=N, dtype=object, keep_default_na=False)
    assert len(Model_output_raw) == N, f"Expected {N} rows, got {N}"

    Metric_Outputs = pd.DataFrame(np.zeros(shape=(N, len(Species))), columns=Species)
    
    def process_species(cell):
        values = json.loads(cell)

        return values
    processed = Model_output_raw["A"].apply(json.loads)

    Metric_Outputs = pd.DataFrame(index=range(N))
    Metric_Outputs['Total'] = processed.apply(lambda d: sum(d.values()))
    for name in Species:
        Metric_Outputs[name] = processed.apply(lambda d, n=name: d[n])

    # NOx-driven contribution: row-wise sum, preserves per-sample correlation.
    nox_species = ['SWV', 'O3', 'PMO', 'CH4']
    Metric_Outputs['Net NOx'] = Metric_Outputs[nox_species].sum(axis=1)

    # Explicit display order: Total, CO2, H2O, NOx, [SWV, O3, PMO, CH4], cont
    if 'CO2' in Species:
        display_order = ['Total', 'CO2', 'Net NOx'] + nox_species + ['H2O', 'cont']
    else: 
        display_order = ['Total', 'Net NOx'] + nox_species + ['H2O', 'cont']
    Metric_Outputs = Metric_Outputs[display_order]

    return Metric_name, Metric_Outputs

def Sobol_Box_Plot(S_i_Bootstrap, S_Ti_Bootstrap, Metric_name, N_Samples):
    # Sobol' indices box plot
    flierprops = dict(
    marker='x',
    markerfacecolor='none',      # hollow marker
    markeredgecolor='black',
    markersize=4,
    alpha=0.5,
    linestyle='none'
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    S_i_Bootstrap.boxplot(ax=ax1, rot=45, grid=False, whis=[2.5, 97.5], flierprops=flierprops)
    ax1.set_ylabel(r"$S_i$", fontsize=16)
    ax1.set_title("First-Order Sobol' Indices", fontsize=14)
    ax1.set_axisbelow(True)
    ax1.grid(axis='both')

    S_Ti_Bootstrap.boxplot(ax=ax2, rot=45, grid=False, whis=[2.5, 97.5], flierprops=flierprops)
    for label in ax2.get_xticklabels():
        label.set_ha('right')
    ax2.set_ylabel(r"$S_{Ti}$", fontsize=16)
    ax2.set_title("Total-Order Sobol' Indices", fontsize=14)
    ax2.set_axisbelow(True)
    ax2.grid(axis='both')
    ax2.tick_params(axis='both', labelsize=14)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.suptitle("Bootstrap Distribution of Sobol' Indices for " + Metric_name + " (N="+ str(N_Samples) + ").", fontsize=16)
    plt.tight_layout()
    plt.show()

def Metric_Raincloud_Plot(Metric_name, Metric_outputs):

    columns = Metric_outputs.columns.tolist()
    n = len(columns)
 
    # ── colour by sign of median (positive RF → blue, negative → orange) ──────
    POS_COLOUR = "#3A7DC9"   # blue  – warming contributions (CO2, contrails…)
    POS_COLOUR_EMPH = "#20446E"   # blue  – warming contributions (CO2, contrails…)
    NEG_COLOUR = "#EC8722"   # orange – cooling contributions
    NEG_COLOUR_EMPH = "#AA5500"   # orange – cooling contributions
 
    def _colour(data):
        return NEG_COLOUR if np.nanmedian(data) <= 0 else POS_COLOUR
 
    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 6))
 
    for i, col in enumerate(columns):
        colour = _colour(Metric_outputs[col])
        data = Metric_outputs[col].dropna().values
        if len(data) < 2:
            continue
 
        x_center = i                                # integer x-position
 
        # ── 1. half-violin (KDE, drawn to the RIGHT of x_center) ────────────
        kde = gaussian_kde(data, bw_method="scott")
        y_grid = np.linspace(data.min(), data.max(), 300)
        density = kde(y_grid)
        density_scaled = density / density.max() * 0.38   # max half-width
 
        ax.fill_betweenx(
            y_grid,
            x_center,
            x_center + density_scaled,
            alpha=0.55,
            color=colour,
            linewidth=0,
        )
        ax.plot(
            x_center + density_scaled,
            y_grid,
            color=colour,
            linewidth=1.2,
            alpha=0.8,
        )
 
        # ── 2. jittered raw data points (to the LEFT of x_center) ───────────
        rng = np.random.default_rng(seed=42 + i)
        jitter = rng.uniform(-0.18, 0, size=len(data))   # negative → left
 
        ax.scatter(
            x_center + jitter,
            data,
            color=colour,
            alpha=0.45,
            s=14,
            linewidths=0,
            zorder=3,
        )
 
        # ── 3. boxplot summary (thin, centred on x_center) ──────────────────
        if colour == POS_COLOUR:
            emph_color = POS_COLOUR_EMPH
        else:
            emph_color = NEG_COLOUR_EMPH
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisker_lo, whisker_hi = np.percentile(data, [2.5, 97.5])
        box_w = 0.06
 
        # IQR box
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x_center - box_w, q1),
                2 * box_w,
                iqr,
                boxstyle="square,pad=0",
                facecolor="white",
                edgecolor=emph_color,
                linewidth=1.6,
                zorder=4,
            )
        )
        # median line
        ax.plot(
            [x_center - box_w, x_center + box_w],
            [median, median],
            color=emph_color,
            linewidth=2.2,
            zorder=5,
        )
        # whiskers
        ax.plot(
            [x_center, x_center],
            [whisker_lo, q1],
            color=emph_color,
            linewidth=1.2,
            zorder=4,
        )
        ax.plot(
            [x_center, x_center],
            [q3, whisker_hi],
            color=emph_color,
            linewidth=1.2,
            zorder=4,
        )
        # whisker caps
        for y_cap in (whisker_lo, whisker_hi):
            ax.plot(
                [x_center - box_w * 0.6, x_center + box_w * 0.6],
                [y_cap, y_cap],
                color=emph_color,
                linewidth=1.2,
                zorder=4,
            )
    # ── NOx composition bracket ─────────────────────────────────────────────
    if 'Net NOx' in columns and all(s in columns for s in ['SWV', 'O3', 'PMO', 'CH4']):
        nox_idx = columns.index('Net NOx')
        group_idx = [columns.index(s) for s in ['SWV', 'O3', 'PMO', 'CH4']]
        left, right = min(group_idx), max(group_idx)

        y_top = ax.get_ylim()[1] * 0.94   # bracket height near top of plot
        bracket_color = "#666666"

        # horizontal bar spanning the four component columns
        ax.plot([left, right+0.2], [y_top, y_top], color=bracket_color, linewidth=1.2, zorder=6)
        # small downward ticks at each end
        for x in (left, right+0.2):
            ax.plot([x, x], [y_top, y_top - 0.01 * ax.get_ylim()[1]], color=bracket_color, linewidth=1.2, zorder=6)

        # connector from bracket midpoint down toward NOx
        mid = (left + right) / 2

        # label above the bracket
        ax.text(mid, y_top + 0.015 * ax.get_ylim()[1], "NOx cycle emissions",
                 ha="center", va="bottom", fontsize=13, color=bracket_color)

        # light shaded background behind the four component columns for grouping
        ax.axvspan(left - 0.4, right + 0.6, color=bracket_color, alpha=0.1, zorder=0)

    # ── axes cosmetics ────────────────────────────────────────────────────────
    ax.set_xticks(range(n))
    ax.set_xticklabels(columns, rotation=40, ha="right", fontsize=15)
    ax.set_xlim(-0.6, n - 0.4)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='-', zorder=1)

    ax.set_ylabel(f"{Metric_name} [K]", fontsize=16)
    ax.set_title(f"{Metric_name} , with contributions per species", fontsize=20, pad=12)
    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=POS_COLOUR, label="Positive "+ Metric_name+" (warming)"),
        mpatches.Patch(color=NEG_COLOUR, label="Negative "+ Metric_name+" (cooling)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=13, loc="upper right")
 
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
 
    plt.tight_layout()
    plt.show()

def Print_Distribution_Summary(Metric_name, Metric_outputs, lower_pct=2.5, upper_pct=97.5):
    """
    Prints the median and the central confidence interval per species, using the
    same percentiles as the whiskers in the raincloud plot. Returns the table
    as a dataframe so it can be saved or reused.
    """
    lo_label = f"{lower_pct:g}th pct"
    hi_label = f"{upper_pct:g}th pct"

    rows = []
    for col in Metric_outputs.columns:
        data = Metric_outputs[col].dropna().values
        lo, med, hi = np.percentile(data, [lower_pct, 50, upper_pct])
        rows.append({"Species": col, "Median": med, lo_label: lo,
                     hi_label: hi, "Mean": np.mean(data)})
    summary = pd.DataFrame(rows).set_index("Species")

    width = 88
    print("\nDistribution summary for " + Metric_name)
    print("=" * width)
    print(f"{'Species':<20} {'Median':>16} {lo_label:>16} {hi_label:>16} {'Mean':>16}")
    print("-" * width)
    for name, r in summary.iterrows():
        print(f"{name:<20} {r['Median']:16.3e} {r[lo_label]:16.3e} {r[hi_label]:16.3e} {r['Mean']:16.3e}")
    print("=" * width)

def Sobol_Convergence_Plot(File_dir, Max_Samples, Parametric_Uncertainty_Distributions):
    Name = File_dir.split("_")[0]
    N_intervals = int(np.log2(Max_Samples))
    Samples = np.logspace(1, N_intervals, N_intervals, base=2)
    S_i_convergence = []
    S_Ti_convergence = []
    for i in Samples:
        S_i, S_Ti, Sobol_Names, S_i_Bootstrap, S_Ti_Bootstrap = Get_Sobol_Indices(
            File_dir,
            int(i),
            Parametric_Uncertainty_Distributions,
            N_bootstraps=0
        )

        S_i_convergence.append(S_i)
        S_Ti_convergence.append(S_Ti)
        
    fig, ax = plt.subplots()
    base_colors = plt.cm.viridis(np.linspace(0, 1, 9))
    colors = np.vstack([base_colors, base_colors])   # actually duplicate: shape (16, 4)
    from itertools import cycle
    linestyles = cycle(['-'] * 9 + ['--'] * 9)
    for i, (values, name) in enumerate(zip(zip(*S_i_convergence), Sobol_Names)):
        ax.plot(Samples, values, marker="x", label=name, color=colors[i], linestyle=next(linestyles))
    # for values, name in zip(zip(*S_i_convergence), Sobol_Names):
    #     ax.plot(Samples, values, marker="o", label=name, color="tab20")
    ax.set_xscale("log", base=2)
    ax.set_title("Convergence of First-order Sobol' indices for " + Name + ".")
    ax.set_xlabel("Number of model evaluations")
    ax.axhline(0, color='black', linewidth=2.0, linestyle='-', zorder=1)
    ax.set_ylabel(r"$S_i$")
    ax.legend()
    ax.grid('major')
    plt.show()

def Sobol_Rate_Convergence_Plot(File_dir, Max_Samples, Parametric_Uncertainty_Distributions, N_bootstraps):
    Name = File_dir.split("_")[0]
    N_intervals = int(np.log2(Max_Samples))
    Samples = np.logspace(1, N_intervals, N_intervals, base=2)
    S_i_convergence = []
    S_i_95 = []
    S_Ti_convergence = []
    S_Ti_95 = []
    S_i_95_rel = []
    for i in Samples:
        S_i, S_Ti, Sobol_Names, S_i_Bootstrap, S_Ti_Bootstrap = Get_Sobol_Indices(
            File_dir,
            int(i),
            Parametric_Uncertainty_Distributions,
            N_bootstraps
        )
        S_i_convergence.append(S_i)
        S_Ti_convergence.append(S_Ti)
        
        S_i_CI_Width = np.array(S_i_Bootstrap.apply(ci_width))
        S_i_95.append(S_i_CI_Width)
        S_Ti_CI_Width = np.array(S_Ti_Bootstrap.apply(ci_width))
        S_Ti_95.append(S_Ti_CI_Width)
        
        S_i_95_rel.append(S_i_CI_Width/np.array(S_i))

    fig, ax = plt.subplots()
    base_colors = plt.cm.viridis(np.linspace(0, 1, 9))
    colors = np.vstack([base_colors, base_colors])   # actually duplicate: shape (16, 4)
    from itertools import cycle
    linestyles = cycle(['-'] * 9 + ['--'] * 9)
    for i, (values, name) in enumerate(zip(zip(*S_i_95), Sobol_Names)):
        ax.plot(Samples, values, marker="x", label=name, color=colors[i], linestyle=next(linestyles))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title("Confidence Interval Width of First-order Sobol' indices for " + Name+ ".")
    ax.set_xlabel("Number of Samples N")
    ax.axhline(0, color='black', linewidth=2.0, linestyle='-', zorder=1)
    ax.set_ylabel(r"$S_i$")
    ax.legend()
    ax.grid('major')
    plt.show()

def ci_width(col, lower_pct=2.5, upper_pct=97.5):
    """
    Calculates the confidence interval width. Standard is 95% CI.

    Args:
        col: Column of which the confidence interval is calculated.
        lower_pct (float): Lower percentile.
        upper_pct (float): Upper percentile.

    Returns:
        CI width (float): Expressed in terms of climate metric (e.g. ATR100).
    """
    lower, upper = np.percentile(col, [lower_pct, upper_pct])
    return upper - lower

def uncertainty(file_name):   
    """
    Main function for uncertainty quantification. Considers only column "A" and
    is therefore usable without running GSA sampling.
    """
    config = oac.get_config(file_name)
    N_Samples = config["uncertainty_sensitivity_analysis"]["N_samples"]
    N_Bootstrap = config["uncertainty_sensitivity_analysis"]["N_bootstrap"]
    Generate_Plots = config["uncertainty_sensitivity_analysis"]["Generate_plots"]
    File_dir = config["uncertainty_sensitivity_analysis"]["File_dir"]
    Species = config["species"]["out"]

    # Calculate uncertainty distributions.
    Name, outputs = Calculate_distribution(
        File_dir,
        N_Samples,
        Species
    )

    # Print uncertainty distributions
    Print_Distribution_Summary(Name, outputs)

    # Generate raincloud plot.
    if Generate_Plots == True:
        Metric_Raincloud_Plot(Name, outputs)

def sensitivity(file_name):
    """
    Main function for sensitivity analysis following Saltelli et al. (2010).
    Considers all columns and is therefore not usable without running GSA sampling.
    """
    config = oac.get_config(file_name)
    N_Samples = config["uncertainty_sensitivity_analysis"]["N_samples"]
    Species = config["species"]["out"]
    N_Bootstrap = config["uncertainty_sensitivity_analysis"]["N_bootstrap"]
    Generate_Plots = config["uncertainty_sensitivity_analysis"]["Generate_plots"]
    File_dir = config["uncertainty_sensitivity_analysis"]["File_dir"]
    uncertainties_file_path = config["uncertainty_sensitivity_analysis"]["PDF_dir"]

    # Load uncertainty distributions.
    Parametric_Uncertainty_Distributions = load_uncertainty_distributions(uncertainties_file_path)

    # Calculate Sobol' indices
    S_i, S_Ti, Sobol_Names, S_i_Bootstrap, S_Ti_Bootstrap = Get_Sobol_Indices(
        File_dir, 
        N_Samples, 
        Parametric_Uncertainty_Distributions,
        N_Bootstrap
        )

    # Calculate distributions
    Name, outputs = Calculate_distribution(
        File_dir,
        N_Samples,
        Species
    )

    # Generate Sobol' convergence plots.
    if Generate_Plots == True:
        Sobol_Box_Plot(S_i_Bootstrap, S_Ti_Bootstrap, Name, N_Samples)
        Sobol_Convergence_Plot(File_dir, N_Samples, Parametric_Uncertainty_Distributions)
        Sobol_Rate_Convergence_Plot(File_dir, N_Samples, Parametric_Uncertainty_Distributions, N_Bootstrap)
