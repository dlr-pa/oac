from openairclim.run_sobol_sampling import Parametric_Uncertainty_Distributions, Species
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
Script_dir = os.path.dirname(os.path.abspath(__file__))
Base_dir = os.path.abspath(os.path.join(Script_dir, "../Sensitivity_Analysis"))

N_Samples = 1024          # Must be a power of 2!
N_Bootstrap_Samples = 5000 # Number of bootstrap samples. Recommended > 5000
File_dir = "ATR100_Sobol_GSA_1.csv"
Generate_Plots = True

def Get_Sobol_Indices(file_name, N, Distributions, N_bootstraps = N_Bootstrap_Samples):
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
    import json
    # Create the correct directory to locate the files containing the climate metrics.
    file_name = os.path.join(Base_dir, file_name)

    # Open the contents of the csv file for the desired number of rows.
    Model_output_raw = pd.read_csv(file_name, nrows=N, dtype=object, keep_default_na=False)

    # Check that the desired number of samples is present within the file.
    assert len(Model_output_raw) == N, f"Expected {N} rows, got {len(Model_output)}"

    # Output dataframe created
    Model_output = pd.DataFrame(index=range(N))
    for col in Model_output_raw.columns:
        # Sums each entry to obtain total Climate metric.
        Model_output[col] = Model_output_raw[col].apply(lambda cell: sum(json.loads(cell)))

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
        Bootstrap_output = Get_Bootstrap_Sample(Model_output, N_Samples) 
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
    # Obtain metric name from file.
    Metric_name = file_name.split("_")[0]
    import json
    # Create the correct directory to locate the files containing the climate metrics.
    file_name = os.path.join(Base_dir, file_name)
    # Open the contents of the csv file for the desired number of rows.
    Model_output_raw = pd.read_csv(file_name, nrows=N, dtype=object, keep_default_na=False)

    # Check that the desired number of samples is present within the file.
    assert len(Model_output_raw) == N, f"Expected {N} rows, got {len(Model_output_raw)}"

    # Add "Total" as initial column
    Species.insert(0, 'Total')

    # Create dataframe containing the metric outputs.
    Metric_Outputs = pd.DataFrame(np.zeros(shape=(N_Samples, (len(Species)))), columns = Species)

    counter = 0
    # Iterate over indices to obtain model outputs.
    for i in Species:
        # If species = Total, sum all indices.
        if i == "Total":
            Metric_Outputs[i] = Model_output_raw["A"].apply(lambda cell: sum(json.loads(cell)))
        # Else: assess individual species
        else:
            Metric_Outputs[i] = Model_output_raw["A"].apply(lambda cell: json.loads(cell)[counter])
            counter += 1

    return Metric_name, Metric_Outputs

def Sobol_Box_Plot(S_i_Bootstrap, S_Ti_Bootstrap, Metric_name):
    # Sobol' indices box plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    S_i_Bootstrap.boxplot(ax=ax1, rot=45, grid=False)
    ax1.set_ylabel("S_i")
    ax1.set_title("First-Order Sobol' Indices")
    ax1.set_axisbelow(True)
    ax1.grid(axis='both')

    S_Ti_Bootstrap.boxplot(ax=ax2, rot=45, grid=False)
    for label in ax2.get_xticklabels():
        label.set_ha('right')
    ax2.set_ylabel("S_Ti")
    ax2.set_title("Total-Order Sobol' Indices")
    ax2.set_axisbelow(True)
    ax2.grid(axis='both')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.suptitle("Bootstrap Distribution of Sobol' Indices for " + Metric_name, fontsize=14)
    plt.tight_layout()
    plt.show()

def Sobol_Raincloud_Plot(S_i_Bootstrap, S_Ti_Bootstrap, Metric_name, Columns):    
    COLOUR   = "#5A7FA8"   # muted steel-blue; no sign encoding needed
    n        = len(Columns)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 1.4), 10), sharex=True)
 
    for ax, df, ylabel, title in [
        (ax1, S_i_Bootstrap,  r"$S_i$",      "First-Order Sobol' Indices"),
        (ax2, S_Ti_Bootstrap, r"$S_{T_i}$",  "Total-Order Sobol' Indices"),
    ]:
        for i, col in enumerate(Columns):
            data = df[col].dropna().values
            if len(data) < 2:
                continue
 
            x_center = i
 
            # ── 1. half-violin (KDE, right of centre) ────────────────────────
            if data.std() > 0:                          # guard against singular KDE
                kde            = gaussian_kde(data, bw_method="scott")
                y_grid         = np.linspace(data.min(), data.max(), 300)
                density        = kde(y_grid)
                density_scaled = density / density.max() * 0.38

                ax.fill_betweenx(y_grid, x_center, x_center + density_scaled,
                                alpha=0.50, color=COLOUR, linewidth=0)
                ax.plot(x_center + density_scaled, y_grid,
                        color=COLOUR, linewidth=1.2, alpha=0.85)
 
            # ── 3. mini boxplot (centred) ─────────────────────────────────────
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            iqr        = q3 - q1
            whisker_lo = max(data.min(), q1 - 1.5 * iqr)
            whisker_hi = min(data.max(), q3 + 1.5 * iqr)
            box_w      = 0.06
 
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_center - box_w, q1), 2 * box_w, iqr,
                boxstyle="square,pad=0", facecolor="white",
                edgecolor=COLOUR, linewidth=1.6, zorder=4,
            ))
            ax.plot([x_center - box_w, x_center + box_w], [median, median],
                    color=COLOUR, linewidth=2.2, zorder=5)
            ax.plot([x_center, x_center], [whisker_lo, q1],
                    color=COLOUR, linewidth=1.2, zorder=4)
            ax.plot([x_center, x_center], [q3, whisker_hi],
                    color=COLOUR, linewidth=1.2, zorder=4)
            for y_cap in (whisker_lo, whisker_hi):
                ax.plot([x_center - box_w * 0.6, x_center + box_w * 0.6],
                        [y_cap, y_cap], color=COLOUR, linewidth=1.2, zorder=4)
 
        # ── panel cosmetics ───────────────────────────────────────────────────
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
 
    # x-tick labels only on the bottom panel
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(Columns, rotation=40, ha="right", fontsize=10)
 
    fig.suptitle(
        f"Bootstrap Distribution of Sobol' Indices — {Metric_name}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()

def Metric_Violin_Plot(Metric_name, Metric_outputs):
    POS_COLOUR = "#3A7DC9"
    NEG_COLOUR = "#E8821A"
 
    columns = Metric_outputs.columns.tolist()
    n = len(columns)
 
    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 6))
 
    parts = ax.violinplot(
        [Metric_outputs[col].dropna() for col in columns],
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
 
    # Colour each violin body and its lines by the sign of the column median
    for i, (col, body) in enumerate(zip(columns, parts["bodies"])):
        colour = POS_COLOUR if np.nanmedian(Metric_outputs[col]) >= 0 else NEG_COLOUR
        body.set_facecolor(colour)
        body.set_edgecolor(colour)
        body.set_alpha(0.55)
 
    # Colour the shared line collections (cmins, cmaxes, cmedians, cbars)
    # by iterating per-segment so each violin gets its own colour
    for line_key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if line_key not in parts:
            continue
        collection = parts[line_key]
        colours = []
        for i, col in enumerate(columns):
            c = POS_COLOUR if np.nanmedian(Metric_outputs[col]) >= 0 else NEG_COLOUR
            colours.append(c)
        collection.set_colors(colours)
        if line_key == "cmedians":
            collection.set_linewidth(2.0)
 
    # Zero reference line
    ax.axhline(0, color="black", linewidth=1.5, linestyle="-", zorder=1)
 
    # Legend
    legend_handles = [
        mpatches.Patch(color=POS_COLOUR, label="Positive RF (warming)"),
        mpatches.Patch(color=NEG_COLOUR, label="Negative RF (cooling)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="upper right")
 
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(f"{Metric_name} value", fontsize=11)
    ax.set_title(f"{Metric_name} decomposition, per species", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
 
    plt.tight_layout()
    plt.show()

def Metric_Raincloud_Plot(Metric_name, Metric_outputs):

    columns = Metric_outputs.columns.tolist()
    n = len(columns)
 
    # ── colour by sign of median (positive RF → blue, negative → orange) ──────
    POS_COLOUR = "#3A7DC9"   # blue  – warming contributions (CO2, contrails…)
    NEG_COLOUR = "#E8821A"   # orange – cooling contributions
 
    def _colour(data):
        return POS_COLOUR if np.nanmedian(data) >= 0 else NEG_COLOUR
 
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
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisker_lo = max(data.min(), q1 - 1.5 * iqr)
        whisker_hi = min(data.max(), q3 + 1.5 * iqr)
        box_w = 0.06
 
        # IQR box
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x_center - box_w, q1),
                2 * box_w,
                iqr,
                boxstyle="square,pad=0",
                facecolor="white",
                edgecolor=colour,
                linewidth=1.6,
                zorder=4,
            )
        )
        # median line
        ax.plot(
            [x_center - box_w, x_center + box_w],
            [median, median],
            color=colour,
            linewidth=2.2,
            zorder=5,
        )
        # whiskers
        ax.plot(
            [x_center, x_center],
            [whisker_lo, q1],
            color=colour,
            linewidth=1.2,
            zorder=4,
        )
        ax.plot(
            [x_center, x_center],
            [q3, whisker_hi],
            color=colour,
            linewidth=1.2,
            zorder=4,
        )
        # whisker caps
        for y_cap in (whisker_lo, whisker_hi):
            ax.plot(
                [x_center - box_w * 0.6, x_center + box_w * 0.6],
                [y_cap, y_cap],
                color=colour,
                linewidth=1.2,
                zorder=4,
            )
 
    # ── axes cosmetics ────────────────────────────────────────────────────────
    ax.set_xticks(range(n))
    ax.set_xticklabels(columns, rotation=40, ha="right", fontsize=10)
    ax.set_xlim(-0.6, n - 0.4)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='-', zorder=1)

    ax.set_ylabel(f"{Metric_name} value", fontsize=11)
    ax.set_title(f"{Metric_name} decomposition, per species", fontsize=13, pad=12)
 
    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=POS_COLOUR, label="Positive RF (warming)"),
        mpatches.Patch(color=NEG_COLOUR, label="Negative RF (cooling)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="upper right")
 
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
 
    plt.tight_layout()
    plt.show()

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

    for values, name in zip(zip(*S_i_convergence), Sobol_Names):
        ax.plot(Samples, values, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_title("Convergence of First-order Sobol' indices for " + Name)
    ax.set_xlabel("Number of model evaluations")
    ax.axhline(0, color='black', linewidth=2.0, linestyle='-', zorder=1)
    ax.set_ylabel("S_i")
    ax.legend()
    ax.grid('major')
    plt.show()

S_i, S_Ti, Sobol_Names, S_i_Bootstrap, S_Ti_Bootstrap = Get_Sobol_Indices(
    File_dir, 
    N_Samples, 
    Parametric_Uncertainty_Distributions
    )

Name, outputs = Calculate_distribution(
    File_dir,
    N_Samples,
    Species
)

if Generate_Plots == True:
    Sobol_Box_Plot(S_i_Bootstrap, S_Ti_Bootstrap, Name)
    Sobol_Raincloud_Plot(S_i_Bootstrap, S_Ti_Bootstrap, Name, Sobol_Names)
    Metric_Violin_Plot(Name, outputs)
    Metric_Raincloud_Plot(Name, outputs)
    Sobol_Convergence_Plot(File_dir, N_Samples, Parametric_Uncertainty_Distributions)
