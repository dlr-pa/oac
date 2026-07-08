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
Script_dir = os.path.dirname(os.path.abspath(__file__))
Base_dir = os.path.abspath(os.path.join(Script_dir, "../Sensitivity_Analysis"))

Create_new_CSV = False
Sobol_Sequence_CSV_File = "Sobol_Sequence_GSA_1.csv"
Model_Outputs_File = "Sobol_GSA_1.csv"
# CSV_dir = "Sensitivity_Analysis/" 
Metrics = ["ATR100", "AGWP100", "AGTP100"]
Model_Evaluation_Range = 981, 1200   # Enter range of Sobol' samples which should be evaluated!
N_Parametric_Uncertainties = 15 # Note that coupled inputs are NOT included!
N_Samples = 512*4                  # Must be a power of 2!

Show_Sample_Distribution = False
Species = ["CO2", "H2O", "SWV", "O3", "PMO", "CH4", "cont"]
Parametric_Uncertainty_Distributions = [
#   [ Parameter,              Distribution, Mean, SD, Sobol' Index]
    ["CO2 Climate Sensitivity", "Normal", 1, 0.2551, 0],
    ["CO2 Radiative Forcing",   "Normal", 1, 0.0608, 1],
    ["H2O Efficacy",            "Normal", 1, 0.0918, 2],
    ["H2O Radiative Forcing",   "Normal", 1, 0.2551, 3],
    ["SWV Efficacy",            "Normal", 1, 0.0918, 4],
    ["SWV Radiative Forcing",   "Normal", 1, 0.1939, 5],
    ["O3 Efficacy",             "Normal", 1, 0.0765, 6],
    ["O3 Radiative Forcing",    "Normal", 1, 0.1480, 7],
    ["PMO Efficacy",            "Normal", 1, 0.0765, 8],
    ["PMO Radiative Forcing",   "Normal", 1, 0.1888, 9],
    ["CH4 Efficacy",            "Normal", 1, 0.0306, 10],
    ["CH4 Radiative Forcing",   "Normal", 1, 0.0851, 7],    #Coupled with index 7: O3 RF
    ["CH4 Trop. Perturbation",  "Normal", 1, 0.0918, 11],
    ["CH4 Strat. Perturbation", "Normal", 1, 0.2041, 12],
#   [ Parameter,               Distribution, scale, sigma] FOR LOGNORMAL ONLY
    ["cont Efficacy",          "Lognormal", 1, 0.3690, 13],
    ["cont Radiative Forcing",   "Normal", 1, 0.3571, 14],
    ]

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
    # Sobol_sampler = qmc.Sobol(d=Sobol_Sequence_Dimensions, scramble=True, seed="Sensitivity_Testing")

    power_2 = int(np.log2(N_max))
    Sobol_Sequence = Sobol_sampler.random_base2(m=power_2)
    print(np.shape(Sobol_Sequence))
    return Sobol_Sequence

def Write_Sobol_CSV(sequence_file_name, sequence, output_file_name, k, N_max):
    """
    Writes a CSV file containing the Sobol' sequences, and an empty file in the 
    same directory for future model outputs.

    Args:
        sequence_file_name (string): The file directory of the Sobol' Sequence CSV file.
        Sequence (matrix): Matrix containing the Sobol' Sequence.
        output_file_name (string): The file directory of the output CSV file.
        k (integer): The total number of parametric uncertainties implemented.
        N_max (integer): The number of samples per uncertainty. Must be a power of 2.

    Returns:
        None
    """
    # Create file paths

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

def Write_Model_Output(Sample_N, Row_List, output_file_name, Metric):
    """
    Writes list of model outputs to the appropriate CSV file.
    Each entry in Row_List is a list of species values, JSON-encoded per cell.

    Args:
        Sample_N (integer): The index of the samples to which the results correspond
        Row_List (List of Lists): Each element is a list of per-species values for one matrix column.
        output_file_name (string): The file directory of the output CSV file.
        Metric (string): The metric name (ATR100, AGWP100, AGTP100).

    Returns:
        None
    """
    import json

    df = pd.read_csv(Metric + "_" + output_file_name, dtype=object, keep_default_na=False)

    # JSON-encode each cell so a list of species values fits in one CSV cell
    df.iloc[Sample_N] = [json.dumps(entry) for entry in Row_List]
    df.to_csv((Metric + "_" + output_file_name), index=False)

def Model_Samples(Sample_N, file_location):
    """
    Calculates the different matrix rows A, B, and A_b^i.

    Args:
        Sample_N (integer): The current Sobol' sample being analysed.
        file_location (string): The file directory of the Sobol' Sequence CSV file.
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

if __name__ == "__main__":
    # Create file directories  
    os.chdir(Base_dir)
    Sobol_Sequence_Dir = Sobol_Sequence_CSV_File
    Output_Dir = Model_Outputs_File

    # Creates new CSV files with the Sobol' Sequence and corresponding output.
    if Create_new_CSV == True:
        Sobol_Sequence = Create_Sobol_Sequence(N_Parametric_Uncertainties, N_Samples)
        Write_Sobol_CSV(Sobol_Sequence_Dir, Sobol_Sequence, Output_Dir, N_Parametric_Uncertainties, N_Samples)

    # Check that number of independent input parameters equals number of Sobol' samples!
    Sobol_Sample_Indices = [row[-1] for row in Parametric_Uncertainty_Distributions]

    if len(set(Sobol_Sample_Indices)) != N_Parametric_Uncertainties:
        raise ValueError("Number of columns of Sobol' Sequence is incorrect! " \
        "Please check the input parameters.")

    # Counter to check computational expense, which should follow N(k+2)
    counter = 0

    # Loops over all samples if no range is set.
    if Model_Evaluation_Range:
        loop = range(*Model_Evaluation_Range)
    else:
        loop = range(N_Samples)

    Multiplication_Matrix = []
    for N in loop:
        Inputs, k = Model_Samples(N, Sobol_Sequence_Dir)
        ATR100_Iteration_results = []
        AGTP100_Iteration_results = []
        AGWP100_Iteration_results = []
        for row in range(k+2):
            
            # Determine Model multiplication factors for each row.
            Model_Inputs_Factors = Calculate_Multiplication_Factors(Inputs[row], Parametric_Uncertainty_Distributions)

            Uncertainty_Factor_Dict = {}
            for i in range(len(Parametric_Uncertainty_Distributions)):
                Uncertainty_Factor_Dict[Parametric_Uncertainty_Distributions[i][0]] = Model_Inputs_Factors[i]

            
            print("\nUncertainty Factors:")
            print("\nSample",N+1, ", Simulation ", row+1, "/", len(Inputs))
            print("-" * 45)
            for key, value in Uncertainty_Factor_Dict.items():
                print(f"  {key:<30} {value:.4f}")
            print("-" * 45)

            openairclim.shared_uncertainties.Uncertainty_Factor = Uncertainty_Factor_Dict

            logging.disable(logging.CRITICAL)  # Suppresses everything below CRITICAL level
            # change directory to match current file
            
            oac.run("Sensitivity_Analysis.toml")

            logging.disable(logging.NOTSET) 

            ds = nc.Dataset('results/example_metrics.nc')
            species = list(ds.variables['species'][:])

            ATR100 = ds.variables['ATR_100_1940'][:]
            AGWP100 = ds.variables['AGWP_100_1940'][:]
            AGTP100 = ds.variables['AGTP_100_1940'][:]    

            ds.close()
            ATR100_List = []
            AGWP100_List = []
            AGTP100_List = []
            for spec in Species:
                ATR100_List.append(float(ATR100[species.index(spec)]))
                AGTP100_List.append(float(AGTP100[species.index(spec)]))
                AGWP100_List.append(float(AGWP100[species.index(spec)]))
            
            # Append result of row to Iteration_results
            ATR100_Iteration_results.append(ATR100_List)
            AGTP100_Iteration_results.append(AGTP100_List)
            AGWP100_Iteration_results.append(AGWP100_List)
            counter += 1

        # Once row is complete, add it to Output CSV file.
        Write_Model_Output(N, ATR100_Iteration_results, Output_Dir, "ATR100")
        Write_Model_Output(N, AGWP100_Iteration_results, Output_Dir, "AGWP100")
        Write_Model_Output(N, AGTP100_Iteration_results, Output_Dir, "AGTP100")

    # Check computational expense, which should follow N(k+2)
    if counter == (2+k)*N_Samples:
        print("Succesful!")

    if Show_Sample_Distribution:
        Multiplication_Matrix = np.array(Multiplication_Matrix)  # Convert once at the end

        for i in range(16):
            col = Multiplication_Matrix[:, i]  # All samples for parameter i
            plt.hist(col, 25)
            plt.title(Parametric_Uncertainty_Distributions[i][0])
            plt.show()

        