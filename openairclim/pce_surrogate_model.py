import numpy as np
import os
import pandas as pd
import warnings
from UQpy.distributions import Normal, Lognormal, Uniform, JointIndependent
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import TotalDegreeBasis
from UQpy.surrogates.polynomial_chaos.regressions.LeastAngleRegression import LeastAngleRegression
from UQpy.surrogates.polynomial_chaos.regressions.LeastSquareRegression import LeastSquareRegression
import openairclim as oac
from openairclim.run_repeated_sampling import load_uncertainty_distributions
import json

"""
Polynomial Chaos Expansion surrogate model.
Intended for Variance-based Global Sensitivity Analysis.
Returns the sensitivity indices for a specific scenario.

Implementation J.S. Dunselman (2026) MSc Thesis.
"""

def build_marginals(Distributions):
    """
    One UQpy marginal per Sobol' index, ordered by index. Coupled inputs
    (several rows sharing an index) become Uniform(0, 1).
    Returns (marginals, index_names).
    """
    groups = {}
    for name, dist, p1, p2, idx in Distributions:
        groups.setdefault(idx, []).append((name, dist, p1, p2))

    marginals, index_names = [], []
    for idx in sorted(groups):
        members = groups[idx]
        index_names.append("/".join(m[0] for m in members))

        if len(members) > 1:
            marginals.append(Uniform(loc=0, scale=1))
            continue

        name, dist, p1, p2 = members[0]
        if dist == "Normal":
            marginals.append(Normal(loc=p1, scale=p2))
        elif dist == "Uniform":
            marginals.append(Uniform(loc=p1, scale=p2))
        elif dist == "Lognormal":
            warnings.warn(f"'{name}': Lognormal approximated by Normal(loc={p1}, scale={p2}) in the PCE.")
            marginals.append(Normal(loc=p1, scale=p2))
        else:
            raise ValueError(f"'{name}': unsupported distribution '{dist}'.")
    return marginals, index_names

def cell_total(cell):
    v = json.loads(cell)
    return sum(v.values()) if isinstance(v, dict) else sum(v)

def Data_Prepping(input_file_dir, output_file_dir, marginals, N_total, N_validation):
    """
    Prepares data for training and validation of PCE surrogate.

    Args:
        input_file_dir (string): file directory of OAC input data, in this case the Sobol' sequence.
        output_file_dir (string): file directory of OAC output data, in this case a climate metric.
        marginals (list): the uncertainty distributions used to train PCE.
        N_total (int): Number of total simulations usable for PCE training and validation.
        N_validation (int): Number of simulations usable for PCE validation, included in N_total.

    Returns:
        X_complete (array): Numpy array containing Sobol' sequence OAC inputs for N_total.
        y_complete (array): Numpy array containing total climate metric results for N_total.
        X_val (array): Numpy array containing Sobol' sequence OAC inputs for N_validation.
        y_val (array): Numpy array containing total climate metric results for N_validation.
    
    """

    # read csv file as a pandas dataframe.
    Model_output_raw = pd.read_csv(output_file_dir, nrows=N_total, dtype=object, keep_default_na=False)

    # Check length and availability of model output.
    if len(Model_output_raw) < N_total:
        raise ValueError(f"{output_file_dir} has {len(Model_output_raw)} rows, expected {N_total}.")
    if (Model_output_raw["A"] == "").any():
        raise ValueError(f"{output_file_dir}: empty cells in column A within the first {N_total} rows.")

    # Total climate metric = sum over species.
    y_complete = np.asarray(Model_output_raw["A"].apply(cell_total)).ravel()

    # First k columns = matrix A, one column per independent parametric uncertainty.
    Model_input_raw = pd.read_csv(input_file_dir, nrows=N_total).iloc[:, :len(marginals)]
    if Model_input_raw.shape[1] != len(marginals):
        raise ValueError(f"{input_file_dir} has {Model_input_raw.shape[1]} input columns, "
                         f"but {len(marginals)} marginals were defined.")

    # Transform uniform Sobol' samples to the marginal distributions.
    eps = np.finfo(float).eps
    Model_input_transformed = pd.DataFrame({
        col: dist.icdf(Model_input_raw[col].clip(eps, 1 - eps))
        for col, dist in zip(Model_input_raw.columns, marginals)
    })

    X_complete = Model_input_transformed.to_numpy()
    X_val, y_val = X_complete[-N_validation:], y_complete[-N_validation:]

    return X_complete, y_complete, X_val, y_val

def PCE_fit_and_evaluate(N_training, p, X_complete, y_complete, X_val, y_val,
                         distributions, Target_error):
    """
    Fit and evaluate the PCE surrogate model.

    Args:
        N_training (int): Number of samples used for training, taken from X_complete and y_complete.
        p (int): Polynomial degree p.
        X_complete (array): Numpy array containing Sobol' sequence OAC inputs for N_total.
        y_complete (array): Numpy array containing total climate metric results for N_total.
        X_val (array): Numpy array containing Sobol' sequence OAC inputs for N_validation.
        y_val (array): Numpy array containing total climate metric results for N_validation.
        distributions (list): the uncertainty distributions used to train PCE. 
        Target_error (float): Target error, terminates training once reached.
    
    Returns:
        dictionary containing PCE accuracy and fit data.
    """
    # Select only first samples for training.
    X_training = X_complete[:N_training]
    y_training = y_complete[:N_training]

    # Set up basis of PCE based on number of distributions and degree
    basis = TotalDegreeBasis(distributions, max_degree=p)

    # Define the dense PCE
    dense_PCE = PolynomialChaosExpansion(
        polynomial_basis=basis,
        # Uses Least Squares for initial fitting
        regression_method=LeastSquareRegression()
    )

    # Fit the PCE to the transformed model inputs, using Training_output as reference.
    # In addition, transforms dataframes to numpy array formats.
    dense_PCE.fit(X_training, y_training)

    # Create sparse PCE using LAR.
    sparse_PCE = LeastAngleRegression.model_selection(
        pce_object=dense_PCE,
        target_error=Target_error,
        check_overfitting=True
    )

    # Calculate first-order and total-order Sobol' indices
    LOO_error = PolynomialChaosExpansion.leaveoneout_error(sparse_PCE)

    # Validation-set error (independent of training data)
    y_val_predicted = sparse_PCE.predict(X_val).ravel()
    val_rmse = np.sqrt(np.mean((y_val_predicted - y_val) ** 2))
    val_rel_rmse = val_rmse / np.std(y_val)
    val_q2 = 1 - np.sum((y_val_predicted - y_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

    # Index of Sparsity calculation:
    # How many terms are actually retained by Sparse PCE?
    n_active = len(sparse_PCE.polynomial_basis.polynomials)
    n_candidate = len(basis.polynomials)
    index_of_sparsity = n_active / n_candidate

    return {
        "N": N_training,
        "p": p,
        "LOO_error": LOO_error,
        "val_rel_rmse": val_rel_rmse,
        "val_Q2": val_q2,
        "n_active": n_active,
        "n_candidate": n_candidate,
        "index_of_sparsity": index_of_sparsity,
        "sparse_PCE": sparse_PCE,
    }

def get_sobol_indices(sparse_PCE):
    """ 
    Obtains first- and total-order Sobol' indices from fitted PCE.

    Args:
        sparse_PCE (object): Fitted sparse PCE.

    Returns:
        S_i (array of floats): First-order Sobol' indices.
        S_Ti (array of floats): Total-order Sobol' indices.
    """
    from UQpy.sensitivity.PceSensitivity import PceSensitivity
    
    pce_sens = PceSensitivity(sparse_PCE)
    pce_sens.run()
    S_i = np.asarray(pce_sens.first_order_indices).flatten()
    S_Ti = np.asarray(pce_sens.total_order_indices).flatten()

    return S_i, S_Ti

def print_sobol_table(S_i_list, S_Ti_list, Index_names, metric_name):
    """
    Prints table containing Sobol' indices.
    """
    print("\nSobol' Indices for " + metric_name)
    print("=" * 80)
    print(f"{'Parameter':<45} {'S_i':>12} {'S_Ti':>12}")
    print("-" * 80)
    for i in range(len(S_i_list)):
        print(f"{Index_names[i]:<45} {S_i_list[i]:12.6f} {S_Ti_list[i]:12.6f}")
    print("=" * 80)
    print(f"{'Sum':<45} {sum(S_i_list):12.4f} {'N.A.':>12}")


def Hyperparameter_Tuning(X_complete, y_complete, X_val, y_val, distributions,
                          Index_names, metric_name, Set_N, Set_p, Target_error):
    """
    Function performing Hyperparameter tuning, aiming to obtain best-performing hyperparameters.

    Args:
        X_complete (array): Numpy array containing Sobol' sequence OAC inputs for N_total.
        y_complete (array): Numpy array containing total climate metric results for N_total.
        X_val (array): Numpy array containing Sobol' sequence OAC inputs for N_validation.
        y_val (array): Numpy array containing total climate metric results for N_validation.
        distributions (list): the uncertainty distributions used to train PCE. 
        metric_name (string): Name of the climate metric for which PCE is fit.
        Set_N (list): List containing the number of training samples N, used for hyperparameter tuning.
        Set_p (list): List containing the polynomial degrees p used for hyperparameter tuning.
        Target_error (float): Target error, terminates training once reached.

    Prints pivot tables used for hyperparameter selection.
    
    """
    results_list = []
    fitted_models = {}

    # Iterate over N and p
    for N in Set_N:
        for p in Set_p:
            # Fit and evaluate PCE
            results = PCE_fit_and_evaluate(N, p, X_complete, y_complete, X_val, y_val, distributions, Target_error)
            fitted_models[(N, p)] = results.pop("sparse_PCE")
            results_list.append(results)
            # Print results of PCE for specific N and p.
            print(
                f"N={N:5d}, p={p}: LOO={results['LOO_error']:.5f}  "
                f"Val_Q2={results['val_Q2']:.5f}  Val_relRMSE={results['val_rel_rmse']:.5f}  "
                f"IS={results['index_of_sparsity']:.3f} ({results['n_active']}/{results['n_candidate']} terms)"
            )

    results_df = pd.DataFrame(results_list)

    # Selecting the best hyperparameters for Q^2 maximizing
    best_row = results_df.loc[results_df["val_Q2"].idxmax()]
    best_N, best_p = int(best_row["N"]), int(best_row["p"])
    print(f"\nSelected hyperparameters: N={best_N}, p={best_p} "
          f"(val_Q2={best_row['val_Q2']:.5f}, LOO={best_row['LOO_error']:.5f})")

    best_pce = fitted_models[(best_N, best_p)]
    S_i, S_Ti = get_sobol_indices(best_pce)
    print_sobol_table(S_i, S_Ti, Index_names, metric_name)

    # pivot tables for the thesis (LOO error and validation Q^2 heatmaps)
    loo_pivot = results_df.pivot(index="N", columns="p", values="LOO_error")
    val_q2_pivot = results_df.pivot(index="N", columns="p", values="val_Q2")
    print("\nLOO error (N x p):\n", loo_pivot)
    print("\nValidation Q^2 (N x p):\n", val_q2_pivot)


def surrogate_gsa(file_name):
    """
    Main function for generating PCE-based Sobol' index estimates. Can be run using oac.surrogate_gsa.
    """
    config = oac.get_config(file_name)
    output_file_name = config["uncertainty_sensitivity_analysis"]["File_dir"]
    input_file_name = config["uncertainty_sensitivity_analysis"]["Sobol_sequence_file"]

    # Number of samples used to train the PCE
    N_total = config["uncertainty_sensitivity_analysis"]["Total_samples"]
    N_validation = config["uncertainty_sensitivity_analysis"]["Validation_samples"]

    # Path to the Excel sheet containing the distributions.
    uncertainty_file_path = config["uncertainty_sensitivity_analysis"]["PDF_dir"]

    # Specified target error.
    Target_error = config["uncertainty_sensitivity_analysis"]["Target_error"]

    # Set of PCE truncation numbers for Hyperparam fitting
    Set_p = config["uncertainty_sensitivity_analysis"]["Tuning_p"]
    # Set of N_training samples for Hyperparam fitting
    Set_N = config["uncertainty_sensitivity_analysis"]["Tuning_N"]

    # Training rows are the first N, validation rows the last N_validation: they must not overlap.
    if max(Set_N) + N_validation > N_total:
        raise ValueError(f"max(Tuning_N) + Validation_samples = {max(Set_N) + N_validation} "
                         f"exceeds Total_samples = {N_total}; training and validation would overlap.")
    
    Distributions = load_uncertainty_distributions(uncertainty_file_path)
    marginals, Index_names = build_marginals(Distributions)
    distributions = JointIndependent(marginals)
    
    X_complete, y_complete, X_val, y_val = Data_Prepping(
        input_file_name, output_file_name, marginals, N_total, N_validation)
    
    metric_name = os.path.basename(output_file_name).split("_")[0]

    return Hyperparameter_Tuning(
        X_complete, y_complete, X_val, y_val, distributions,
        Index_names, metric_name, Set_N, Set_p, Target_error)