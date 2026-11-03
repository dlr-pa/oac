from scipy.stats import qmc
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import openairclim as oac
import netCDF4 as nc
import numpy as np
import logging
import openairclim.shared_uncertainties

"""
Variance-Based Sensitivity Analysis.
Globally follows methodology by Saltelli et al. (2010)

Implementation J.S. Dunselman (2026) MSc Thesis.
"""

Currently_Implemented_Distributions = ["Normal", "Lognormal", "Uniform"]

def load_uncertainty_distributions(path):
    """
    Reads the parametric uncertainties from an Excel sheet and returns
    [name, distribution, param1, param2, sobol_index] per row. Uses these to define PDF's of uncertainties.
    Accepts headers Param1/Param2 or Mean/SD.

    Args:
        path (string): path to the Excel sheet containing uncertainties.

    Returns:
        distributions
    """

    # Creates dataframe of uncertainty distributions in Excel.
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Mean": "Param1", "SD": "Param2"})

    # Requires the following parameters
    required = ["Variable", "Distribution", "Param1", "Param2", "SI"]
    missing = [c for c in required if c not in df.columns]
    # Raise error if parameter is missing!
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=required)

    distributions = []
    for _, r in df.iterrows():
        name = str(r["Variable"]).strip()
        dist = str(r["Distribution"]).strip().capitalize()
        if dist not in Currently_Implemented_Distributions:
            raise ValueError(f"'{name}': unknown distribution '{r['Distribution']}'. "
                             f"Use one of {sorted(Currently_Implemented_Distributions)}.")
        p1, p2 = float(r["Param1"]), float(r["Param2"])
        # Check if parameter smaller than 0, which is not physically possible for implemented PDF's.
        if p2 <= 0:
            raise ValueError(f"'{name}': Param2 must be > 0 (SD, sigma or width), got {p2}.")
        distributions.append([name, dist, p1, p2, int(r["SI"])])

    names = [d[0] for d in distributions]
    dupes = {n for n in names if names.count(n) > 1}
    # Check for duplicates in input parametric uncertainties.
    if dupes:
        raise ValueError(f"Duplicate variable names in {path}: {dupes}")

    indices = sorted({d[4] for d in distributions})
    if indices != list(range(len(indices))):
        raise ValueError(f"Sobol' indices must be 0..k-1 without gaps, got {indices}")

    return distributions

def Create_Sobol_Sequence(k, N_max):
    """
    Create a Sobol' Sequence used for the sensitivity analysis.

    Args:
        k (integer): The total number of parametric uncertainties implemented.
        N_max (integer): The number of samples per uncertainty. Must be a power of 2.

    Returns:
        (N_max, 2k) matrix containing Sobol' Sequences
    """

    # Check if N_max is a power of 2, else Sobol' sequence loses integrity.
    if (N_max & (N_max - 1)) != 0:
        msg = "WARNING: N_max must be a power of 2."
        raise ValueError(msg)
    
    Sobol_Sequence_Dimensions = 2*k

    #Scramble set to TRUE to avoid row of zeros.
    Sobol_sampler = qmc.Sobol(d=Sobol_Sequence_Dimensions, scramble=True, seed=29052026)


    power_2 = int(np.log2(N_max))
    Sobol_Sequence = Sobol_sampler.random_base2(m=power_2)
    print(np.shape(Sobol_Sequence))
    return Sobol_Sequence

def Write_Sobol_CSV(sequence_file_name, sequence, output_file_name, k, N_max, Metrics):
    """
    Writes a CSV file containing the Sobol' sequences, and an empty file in the 
    same directory for future model outputs.

    Args:
        sequence_file_name (string): The file directory of the Sobol' Sequence CSV file.
        sequence (matrix): Matrix containing the Sobol' Sequence.
        output_file_name (string): The file directory of the output CSV file.
        k (integer): The total number of parametric uncertainties implemented.
        N_max (integer): The number of samples per uncertainty. Must be a power of 2.
        Metrics (list of strings): Contains the names of metrics to be saved.

    Returns:
        None
    """
    # Create input header showing which matrices the Sobol' sequence samples belong to.
    input_header = ",".join(f"A{i}" for i in range(1, k + 1)) + "," + \
         ",".join(f"B{i}" for i in range(1, k + 1))
    
    # Writes Sobol' Sequence to CSV file.
    np.savetxt(sequence_file_name, sequence, delimiter=",", header=input_header, comments="")

    col_names = ["A", "B"] + [str(i) for i in range(1, k + 1)]
    # Write header and empty data to CSV file.
    for metric in Metrics:
        empty_df = pd.DataFrame("", index=range(N_max), columns=col_names, dtype=object)
        empty_df.to_csv(metric + "_" + output_file_name, index=False)

def Write_Model_Output(Sample_N, Row_List, output_file_name, Metric, GSA):
    """
    Writes list of model outputs to the appropriate CSV file.
    Each entry in Row_List is a list of species values, JSON-encoded per cell.

    Args:
        Sample_N (integer): The index of the samples to which the results correspond
        Row_List (List of Lists): Each element is a list of per-species values for one matrix column.
        output_file_name (string): The file directory of the output CSV file.
        Metric (string): The metric name (ATR100, AGWP100, AGTP100).
        GSA (Bool): Toggles sampling methodology. If False, only column "A" sampled.

    Returns:
        None
    """
    import json

    # Write to corresponding CSV, named with Metric name.
    df = pd.read_csv(Metric + "_" + output_file_name, dtype=object, keep_default_na=False)

    # JSON-encode each cell so a list of species values fits in one CSV cell
    if GSA == True:
        df.iloc[Sample_N] = [json.dumps(entry) for entry in Row_List]
    else:
        df.iloc[Sample_N, 0] = json.dumps(Row_List[0])
    df.to_csv((Metric + "_" + output_file_name), index=False)

def Model_Samples(Sample_N, file_location, GSA):
    """
    Calculates the different matrix rows A, B, and A_b^i.

    Args:
        Sample_N (integer): The current Sobol' sample being analysed.
        file_location (string): The file directory of the Sobol' Sequence CSV file.
        GSA (Bool): Toggles sampling methodology. If False, only column "A" sampled.
        
    Returns:
        Matrix containing model inputs for each matrix (Saltelli et al. (2010))
    """

    df = pd.read_csv(file_location)
    # Reads row of current sample from Sobol' Sequence CSV file
    Current_Sample = df.iloc[Sample_N].to_numpy() 

    # Calculates k for any CSV input.
    k = int(len(Current_Sample)/2)

    # Creates matrix representing row N of matrices A, B and A_B^i
    Sample_N_Matrix = np.zeros((k+2, k))
    
    Sample_N_Matrix[0] = Current_Sample[:k] # Row 1: A only

    if GSA == True:
        Sample_N_Matrix[1] = Current_Sample[k:] # Row 2: B only

        for i in range(k):
            # Row i+2: A only, with i sampled from B
            Sample_N_Matrix[i+2] = Current_Sample[:k]
            Sample_N_Matrix[i+2, i] = Current_Sample[(i+k)]
    
    return Sample_N_Matrix, k

def Calculate_Multiplication_Factors(row, Distributions):
    """
    Calculates the multiplication factors following defined distributions, following
    the Probability Integral Transform (universality of the uniform).

    Args:
        row (list): The row taken from Matrix A or B or A_B^i, containing Sobol' Sequence samples
        Distributions (2D array): Array containing information on uncertainties of different species, 
            including correlations.

    Returns:
        List containing the multiplication factors for uncertain parameters.
    """    
    # Create empty list to store multiplication factors.
    Multiplication_Factors = np.zeros(len(Distributions))

    # Iterates over number of distributions.
    for i in range(len(Distributions)):

        Sobol_index = Distributions[i][4]
        Sobol_value = row[Sobol_index] # Value of the sample of Sobol' Sequence

        # For normal distributions, calculate percentile!
        if Distributions[i][1] == "Normal":
            percentile = Sobol_value
            mean = Distributions[i][2]
            SD = Distributions[i][3]
            Multiplication_Factors[i] = stats.norm.ppf(percentile, mean, SD)

        # For uniform distributions, calculate percentile!
        elif Distributions[i][1] == "Uniform":
            percentile = Sobol_value
            lower_bound = Distributions[i][2]
            width = Distributions[i][3]
            Multiplication_Factors[i] = stats.uniform.ppf(percentile, loc=lower_bound, scale=width)

        # For lognormal distributions, calculate percentile!
        elif Distributions[i][1] == "Lognormal":
            percentile = Sobol_value
            sigma = Distributions[i][3]
            scale = Distributions[i][2]
            Multiplication_Factors[i] = stats.lognorm.ppf(percentile, s=sigma, scale=scale)

        # If distribution is not defined:
        else:
            raise ValueError("WARNING: Unknown distribution. Only Normal and Lognormal implemented.")

    return Multiplication_Factors

def get_metric_specs(config):
    """
    Builds the metric definitions from the [metrics] section of the .toml.

    Returns:
        specs (list of dicts): label (used in CSV file names, e.g. 'ATR100')
    and var (name of the variable in the metrics netCDF file).
    """
    types = config["metrics"]["types"]
    Hs = config["metrics"]["H"]
    t0s = config["metrics"]["t_0"]
    specs = []
    for metric_type in types:
        for H, t0 in zip(Hs, t0s):
            specs.append({
                "label": f"{metric_type}{H}",
                "var": f"{metric_type}_{H}_{t0}",
            })
    return specs

def repeated_sampling(file_name):
    """
    Main function for uncertainty or sensitivity sampling. Iterates OpenAirClim based
    a Sobol' sequence.
    """
    config = oac.get_config(file_name)
    N_Samples = config["uncertainty_sensitivity_analysis"]["N_sampling_points"]
    Model_Evaluation_Range = config["uncertainty_sensitivity_analysis"]["Evaluation_range"]
    Sobol_Sequence_CSV_File = config["uncertainty_sensitivity_analysis"]["Sobol_sequence_file"]
    Model_Outputs_File = config["uncertainty_sensitivity_analysis"]["Model_outputs_file"]
    Create_new_CSV = config["uncertainty_sensitivity_analysis"]["Create_new_CSV"]
    Global_Sensitivity_Analysis = config["uncertainty_sensitivity_analysis"]["GSA_sampling"]
    Sobol_Sequence_CSV_File = config["uncertainty_sensitivity_analysis"]["Sobol_sequence_file"]
    Species = config["species"]["out"]
    uncertainty_file_path = config["uncertainty_sensitivity_analysis"]["PDF_dir"]

    Parametric_Uncertainty_Distributions = load_uncertainty_distributions(uncertainty_file_path)
    specs = get_metric_specs(config)
    labels = [s["label"] for s in specs]
    metrics_file = os.path.join(config["output"]["dir"], config["output"]["name"] + "_metrics.nc")

    N_Independent_Uncertainties = len({d[4] for d in Parametric_Uncertainty_Distributions})    # os.chdir(Base_dir)
    Sobol_Sequence_Dir = Sobol_Sequence_CSV_File
    Output_Dir = Model_Outputs_File

    # Creates new CSV files with the Sobol' Sequence and corresponding output.
    if Create_new_CSV == True:
        Sobol_Sequence = Create_Sobol_Sequence(N_Independent_Uncertainties, N_Samples)
        Write_Sobol_CSV(Sobol_Sequence_Dir, Sobol_Sequence, Output_Dir,
                        N_Independent_Uncertainties, N_Samples, labels)

    # Check that number of independent input parameters equals number of Sobol' samples!
    Sobol_Sample_Indices = [row[-1] for row in Parametric_Uncertainty_Distributions]

    # Check that number of independent input parameters equals number of Sobol' samples!
    Sobol_Sample_Indices = [row[-1] for row in Parametric_Uncertainty_Distributions]
    if len(set(Sobol_Sample_Indices)) != N_Independent_Uncertainties:
        raise ValueError("Number of columns of Sobol' Sequence is incorrect! "
                         "Please check the input parameters.")

    # Counter to check computational expense, which should follow N(k+2)
    counter = 0

    # Loops over all samples if no range is set.
    if Model_Evaluation_Range == "none":
        loop = range(N_Samples)
    else:
        Model_Evaluation_Range = Model_Evaluation_Range[0], Model_Evaluation_Range[1]
        loop = range(*Model_Evaluation_Range)

    Multiplication_Matrix = []
    for N in loop:
        Inputs, k = Model_Samples(N, Sobol_Sequence_Dir, Global_Sensitivity_Analysis)
        Iteration_results = {label: [] for label in labels}

        iterate_over = k + 2 if Global_Sensitivity_Analysis == True else 1

        for row in range(iterate_over):
            Model_Inputs_Factors = Calculate_Multiplication_Factors(
                Inputs[row], Parametric_Uncertainty_Distributions)

            Uncertainty_Factor_Dict = {
                dist[0]: Model_Inputs_Factors[i]
                for i, dist in enumerate(Parametric_Uncertainty_Distributions)
            }

            print("\nUncertainty Factors:")
            print("\nSample", N + 1, ", Simulation ", row + 1, "/", iterate_over)
            print("-" * 45)
            for key, value in Uncertainty_Factor_Dict.items():
                print(f"  {key:<30} {value:.4f}")
            print("-" * 45)

            openairclim.shared_uncertainties.Uncertainty_Factor = Uncertainty_Factor_Dict

            logging.disable(logging.CRITICAL)
            oac.run(file_name)
            logging.disable(logging.NOTSET)

            ds = nc.Dataset(metrics_file)
            species = list(ds.variables['species'][:])
            for s in specs:
                if s["var"] not in ds.variables:
                    ds.close()
                    raise KeyError(f"{s['var']} not found in {metrics_file}. "
                                   f"Available: {list(ds.variables)}")
                values = ds.variables[s["var"]][:]
                Iteration_results[s["label"]].append(
                    {sp: float(values[species.index(sp)]) for sp in Species}
                )
            ds.close()
            counter += 1

        # Once the sample is complete, write each metric to its own CSV file.
        for label in labels:
            Write_Model_Output(N, Iteration_results[label], Output_Dir, label,
                               Global_Sensitivity_Analysis)

    # Check computational expense, which should follow N(k+2) for GSA, N otherwise.
    expected = len(loop) * (k + 2 if Global_Sensitivity_Analysis else 1)
    if counter == expected:
        print("Sampling concluded successfully!")
    

        