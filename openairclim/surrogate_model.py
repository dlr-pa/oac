import numpy as np
import os
import pandas as pd
from UQpy.distributions import Normal, Lognormal, Uniform, JointIndependent
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import TotalDegreeBasis
from UQpy.surrogates.polynomial_chaos.regressions.LeastAngleRegression import LeastAngleRegression
from UQpy.surrogates.polynomial_chaos.regressions.LeastSquareRegression import LeastSquareRegression
from openairclim.run_sobol_sampling import Parametric_Uncertainty_Distributions as Distributions

Script_dir = os.path.dirname(os.path.abspath(__file__))
Base_dir = os.path.abspath(os.path.join(Script_dir, "../Sensitivity_Analysis"))
output_file_name = "ATR100_Sobol_GSA_1.csv"
input_file_name = "Sobol_Sequence_GSA_1.csv"
# Number of samples used to train the PCE
N_training = 512
# Number of terms of the PCE
p = 3

# Definition of the distributions assumed for the PCE.
marginals = [
    Normal(loc=1, scale=0.2551), #A1. CO2 Climate Sensitivity
    Normal(loc=1, scale=0.0608), #A2. CO2 Radiative Forcing
    Normal(loc=1, scale=0.0918), #A3. H2O Efficacy
    Normal(loc=1, scale=0.2551), #A4. H2O Radiative Forcing
    Normal(loc=1, scale=0.0918), #A5. SWV Efficacy
    Normal(loc=1, scale=0.1939), #A6. SWV Radiative Forcing
    Normal(loc=1, scale=0.0765), #A7. O3 Efficacy
    # Assumed uniform for O3 and CH4 forcing, due to underlying distributions
    Uniform(loc=0, scale=1),     #A8. O3 and CH4 Radiative Forcing
    Normal(loc=1, scale=0.0765), #A9. PMO Efficacy
    Normal(loc=1, scale=0.1888), #A10. PMO Radiative Forcing
    Normal(loc=1, scale=0.0306), #A11. CH4 Efficacy
    Normal(loc=1, scale=0.0851), #A12. CH4 Trop. Perturbation
    Normal(loc=1, scale=0.0918), #A13. CH4 Strat. Perturbation
    # Assumed Normal for contrail efficacy, due to inability of UQpy to assess Lognormal.
    Normal(loc=1, scale=0.3690), #A14. cont Efficacy
    Normal(loc=1, scale=0.3571)  #A15. cont Radiative Forcing
]

distributions = JointIndependent(marginals)

# Create file directory for file containing outputs.
output_file_dir = os.path.join(Base_dir, output_file_name)
Model_output_raw = pd.read_csv(output_file_dir, nrows=N_training, dtype=object, keep_default_na=False)
import json 
# Get sum of output values yielding total climate metric.
Training_output = Model_output_raw["A"].apply(lambda cell: sum(json.loads(cell)))
print(Training_output)

# Create file directory for file containing inputs.
input_file_dir = os.path.join(Base_dir, input_file_name)

# Create dataframe containing first N columns, N being equivalent to number of parametric uncertainties.
Model_input_raw = pd.read_csv(input_file_dir, nrows=N_training).iloc[:, :len(marginals)]

print(Model_input_raw)

# Transform the uniformly distributed values from Model_input_raw to their corresponding PDF                                                                                                                                                                                                                                                        
eps = np.finfo(float).eps
Model_input_transformed = pd.DataFrame({
    col: dist.icdf(Model_input_raw[col].clip(eps, 1 - eps))
    for col, dist in zip(Model_input_raw.columns, marginals)
})

print(Model_input_transformed)

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
dense_PCE.fit(Model_input_transformed.to_numpy(), np.asarray(Training_output).ravel())

# Create sparse PCE using LAR.
sparse_PCE = LeastAngleRegression.model_selection(
    pce_object=dense_PCE,
    target_error=0.999,
    check_overfitting=True
)

# y_pred = sparse_PCE.predict(x_test)

# Calculate first-order and total-order Sobol' indices
from UQpy.sensitivity.PceSensitivity import PceSensitivity
pce_sens = PceSensitivity(sparse_PCE)
pce_sens.run()
S_i_list = np.asarray(pce_sens.first_order_indices).flatten()
S_Ti_list = np.asarray(pce_sens.total_order_indices).flatten()

# Get index names
Index_names = [""] * len(marginals)
for i in range(len(Distributions)):
    idx = Distributions[i][4]   # Index representing to which Sobol' index it corresponds.
    name = Distributions[i][0]
    # Merge names if multiple uncertainties contribute to single Sobol' index!
    Index_names[idx] = Index_names[idx] + ("/" if Index_names[idx] else "") + name 

# Calculate metric name from model output file name
Metric_name = output_file_name.split("_")[0]

# Print Sobol' indices 
print("\nSobol' Indices for " + Metric_name)
print("=" * 80)
print(f"{'Parameter':<45} {'S_i':>12} {'S_Ti':>12}")
print("-" * 80)

for i in range(len(S_i_list)):
    print(f"{Index_names[i]:<45} {S_i_list[i]:12.3f} {S_Ti_list[i]:12.3f}")

print("=" * 80)
# Calculate sum of first-order Sobol' indices.
print(f"{'Sum':<45} {sum(S_i_list):12.3f} {'N.A.':>12}")