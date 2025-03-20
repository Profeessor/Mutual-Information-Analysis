# Import libraries
import mutual_inf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Run reliability analysis without drift rate parameters
print("Running Reliability Analysis Without Drift Rates")
print("=" * 70)

# Run the standard reliability analysis but with empty drift_vars list
reliability_results_no_drift = mutual_inf.run_reliability_analysis(
    df_merged, 
    alpha_vars=['alpha_s1', 'alpha_s2'],
    threshold_vars=['a_mean_s1', 'a_mean_s2'], 
    ndt_vars=['ndt_mean_s1', 'ndt_mean_s2'],
    drift_vars=[]  # Exclude drift rate parameters
)

# Visualize the results
mutual_inf.plot_reliability_results(reliability_results_no_drift)

# Additionally, run a direct comparison between with/without drift rates
print("\nComparing Analyses With vs. Without Drift Rate Parameters")
print("=" * 70)

# Run the standard reliability analysis with all parameters
reliability_results_full = mutual_inf.run_reliability_analysis(
    df_merged, 
    alpha_vars=['alpha_s1', 'alpha_s2'],
    threshold_vars=['a_mean_s1', 'a_mean_s2'], 
    ndt_vars=['ndt_mean_s1', 'ndt_mean_s2'],
    drift_vars=['v1_mean_s1', 'v1_mean_s2', 'v2_mean_s1', 'v2_mean_s2']
)

# Calculate how much controlling for threshold and NDT alone affects alpha reliability
control_vars = ['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']

# Function to calculate regression-based variance explained
def calc_variance_explained(df, target='alpha_s2', predictor='alpha_s1', controls=None):
    """Calculate variance in target explained by predictor with/without controls"""
    # Variance explained by predictor alone
    X1 = df[[predictor]]
    y = df[target]
    model1 = LinearRegression().fit(X1, y)
    r2_simple = model1.score(X1, y)
    
    if controls:
        # Variance explained by controls alone
        X2 = df[controls]
        model2 = LinearRegression().fit(X2, y)
        r2_controls = model2.score(X2, y)
        
        # Variance explained by predictor + controls
        X3 = df[[predictor] + controls]
        model3 = LinearRegression().fit(X3, y)
        r2_full = model3.score(X3, y)
        
        # Unique variance explained by predictor
        unique_var = r2_full - r2_controls
        
        # Shared variance
        shared_var = r2_simple - unique_var
        
        return {
            'r2_simple': r2_simple,
            'r2_controls': r2_controls,
            'r2_full': r2_full,
            'unique_variance': unique_var,
            'shared_variance': shared_var,
            'unique_percent': unique_var / r2_simple * 100 if r2_simple > 0 else 0
        }
    else:
        return {'r2_simple': r2_simple}

# Calculate variance components
variance_results = calc_variance_explained(
    df_merged, target='alpha_s2', predictor='alpha_s1', controls=control_vars
)

print("\nVariance Decomposition:")
print(f"  Total variance in alpha_s2 explained by alpha_s1: {variance_results['r2_simple']:.4f}")
print(f"  Variance in alpha_s2 explained by control parameters: {variance_results['r2_controls']:.4f}")
print(f"  Variance in alpha_s2 explained by alpha_s1 + controls: {variance_results['r2_full']:.4f}")
print(f"  Unique variance contribution of alpha_s1: {variance_results['unique_variance']:.4f}")
print(f"  Shared variance (alpha_s1 & controls): {variance_results['shared_variance']:.4f}")
print(f"  Percent of alpha_s1's effect that is unique: {variance_results['unique_percent']:.1f}%")

# Create a visualization of the variance decomposition
labels = ['Total alpha_s1\nvariance', 'Unique alpha_s1\nvariance', 'Shared variance\nwith controls']
values = [
    variance_results['r2_simple'], 
    variance_results['unique_variance'], 
    variance_results['shared_variance']
]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color=['#1f77b4', '#2ca02c', '#ff7f0e'])

# Add value annotations on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=12)

plt.ylabel('R² (Variance Explained)', fontsize=14)
plt.title('Decomposition of Alpha Reliability', fontsize=16)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
