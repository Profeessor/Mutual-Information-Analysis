import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.linear_model import LinearRegression
import seaborn as sns

def estimate_mutual_information(X, y, n_neighbors=3, random_state=42):
    """
    Estimate mutual information between features X and target y.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    mi : numpy array
        Mutual information between each feature and the target.
    """
    return mutual_info_regression(
        X, y, 
        discrete_features=False, 
        n_neighbors=n_neighbors, 
        random_state=random_state
    )

def estimate_conditional_mutual_information(X, y, condition_indices, n_neighbors=3, random_state=42):
    """
    Estimate conditional mutual information I(X[0]; y | X[condition_indices]).
    
    This measures how much information feature X[0] provides about y
    beyond what's already provided by X[condition_indices].
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix. X[0] is the target feature (alpha_s1).
    y : array-like of shape (n_samples,)
        Target vector (alpha_s2).
    condition_indices : list of int
        Indices of conditioning variables (threshold, ndt, etc.)
    n_neighbors : int, default=3
        Number of neighbors for k-NN regression.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    cmi : float
        Conditional mutual information estimate.
    """
    # Standardize X and y for better estimation
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Extract conditioning variables
    if len(condition_indices) > 0:
        Z = X_scaled[:, condition_indices]
        
        # 1. Estimate residuals of X[0] after predicting from Z
        knn_x = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X) - 1))
        knn_x.fit(Z, X_scaled[:, 0])
        x_given_z = X_scaled[:, 0] - knn_x.predict(Z)
        
        # 2. Estimate residuals of y after predicting from Z
        knn_y = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X) - 1))
        knn_y.fit(Z, y_scaled)
        y_given_z = y_scaled - knn_y.predict(Z)
        
        # 3. Calculate mutual information between residuals
        return max(0, mutual_info_regression(
            x_given_z.reshape(-1, 1), 
            y_given_z, 
            discrete_features=False,
            n_neighbors=min(n_neighbors, len(X) - 1),
            random_state=random_state
        )[0])
    else:
        # If no conditioning variables, return regular mutual information
        return mutual_info_regression(
            X_scaled[:, 0].reshape(-1, 1),
            y_scaled,
            discrete_features=False,
            n_neighbors=min(n_neighbors, len(X) - 1),
            random_state=random_state
        )[0]

def permutation_test_cmi(X, y, condition_indices, n_permutations=1000, n_neighbors=3, random_state=42):
    """
    Perform permutation test to assess significance of conditional mutual information.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    condition_indices : list of int
        Indices of conditioning variables.
    n_permutations : int, default=1000
        Number of permutations for the test.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    observed_cmi : float
        Observed conditional mutual information.
    p_value : float
        P-value from permutation test.
    null_distribution : numpy array
        Distribution of CMI values under the null hypothesis.
    """
    # Calculate observed CMI
    observed_cmi = estimate_conditional_mutual_information(
        X, y, condition_indices, n_neighbors, random_state
    )
    
    # Generate null distribution by permuting alpha_s1
    null_distribution = []
    rng = np.random.RandomState(random_state)
    
    for i in range(n_permutations):
        # Create copy of X with permuted alpha_s1
        X_perm = X.copy()
        X_perm[:, 0] = rng.permutation(X[:, 0])
        
        # Calculate CMI with permuted data
        null_cmi = estimate_conditional_mutual_information(
            X_perm, y, condition_indices, n_neighbors, random_state=random_state + i
        )
        null_distribution.append(null_cmi)
    
    # Calculate p-value
    null_distribution = np.array(null_distribution)
    
    # Handle the case where all values in the null distribution are very close to zero
    if np.std(null_distribution) < 1e-6:
        # If observed_cmi is also close to zero, consider it non-significant
        if observed_cmi < 1e-6:
            p_value = 1.0
        else:
            p_value = 0.0
    else:
        # Regular p-value calculation
        p_value = (np.sum(null_distribution >= observed_cmi) + 1) / (n_permutations + 1)
    
    return observed_cmi, p_value, np.array(null_distribution)

def run_mutual_information_analysis(df_merged, feature_cols=['alpha_s1', 'a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2','v1_mean_s1','v1_mean_s2','v2_mean_s1','v2_mean_s2'], 
                                   target_col='alpha_s2', n_neighbors=3, n_permutations=1000, random_state=42):
    """
    Run complete mutual information analysis for alpha reliability.
    
    Parameters:
    -----------
    df_merged : pandas DataFrame
        DataFrame containing session 1 and session 2 data.
    feature_cols : list of str, default=['alpha_s1', 'a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
        Column names for features.
    target_col : str, default='alpha_s2'
        Column name for target variable.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
    n_permutations : int, default=1000
        Number of permutations for significance testing.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    results : dict
        Dictionary containing all results of the analysis.
    """
    print("Running Mutual Information Analysis for Alpha Reliability")
    print("=" * 70)
    
    # 1. Prepare data
    X = df_merged[feature_cols].values
    y = df_merged[target_col].values
    
    # 2. Calculate raw mutual information between each feature and alpha_s2
    raw_mi = estimate_mutual_information(X, y, n_neighbors, random_state)
    
    print("Raw Mutual Information with alpha_s2:")
    for i, feature in enumerate(feature_cols):
        print(f"  {feature}: {raw_mi[i]:.4f} bits")
    
    # 3. Calculate conditional MI for alpha_s1, controlling for other parameters
    condition_indices = list(range(1, X.shape[1]))  # all except alpha_s1
    observed_cmi, p_value, null_distribution = permutation_test_cmi(
        X, y, condition_indices, n_permutations, n_neighbors, random_state
    )
    
    print("\nConditional Mutual Information:")
    print(f"  I(alpha_s1; alpha_s2 | other params): {observed_cmi:.4f} bits")
    print(f"  Permutation test p-value: {p_value:.4f}")
    
    # 4. Calculate percentage of information retained after conditioning
    # Handle the case where raw_mi[0] is very small or zero
    if raw_mi[0] > 0.001:  # Using a small threshold to avoid division by very small numbers
        info_retained = min((observed_cmi / raw_mi[0]) * 100, 100.0)  # Cap at 100%
        print(f"  Percentage of information retained: {info_retained:.1f}%")
    else:
        # If raw MI is too small, we can't reliably calculate percentage
        if observed_cmi > raw_mi[0]:
            print(f"  Note: Conditional MI ({observed_cmi:.4f}) is larger than raw MI ({raw_mi[0]:.4f})")
            print("  This suggests estimation uncertainty or that conditioning variables")
            print("  are suppressing shared information between alpha_s1 and alpha_s2.")
        else:
            print("  Raw MI is too small to reliably calculate percentage of information retained.")
    
    # 5. Calculate Pearson correlation for comparison
    pearson_r, pearson_p = pearsonr(X[:, 0], y)
    print("\nPearson correlation (for comparison):")
    print(f"  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    
    # 6. Create visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of mutual information
    axs[0].bar(feature_cols, raw_mi, color='skyblue')
    axs[0].set_title('Mutual Information with alpha_s2')
    axs[0].set_ylabel('Mutual Information (bits)')
    axs[0].set_xlabel('Features')
    axs[0].tick_params(axis='x', rotation=45)
    
    # Histogram of null distribution with observed value marked
    axs[1].hist(null_distribution, bins=30, alpha=0.7, color='gray', density=True)
    axs[1].axvline(x=observed_cmi, color='red', linestyle='--', linewidth=2)
    axs[1].set_title('Permutation Test for Conditional MI')
    axs[1].set_xlabel('Conditional Mutual Information (bits)')
    axs[1].set_ylabel('Density')
    axs[1].text(0.7, 0.8, f'p = {p_value:.4f}', transform=axs[1].transAxes, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # 7. Return all results
    results = {
        'raw_mutual_information': {feature_cols[i]: raw_mi[i] for i in range(len(feature_cols))},
        'conditional_mutual_information': observed_cmi,
        'p_value': p_value,
        'null_distribution': null_distribution,
        'pearson_correlation': (pearson_r, pearson_p),
        'feature_cols': feature_cols,
    }
    
    return results

def load_and_merge(session1_path, session2_path):
    """
    Loads two CSV files (for Session 1 and Session 2), renames specific columns to include session suffixes,
    merges them on the 'ID', 'participant', 'Experiment', or 'participant_ID' column, drops any missing values,
    and renames the transformed alpha columns for clarity.
    
    Parameters:
      session1_path (str): File path to Session 1 CSV.
      session2_path (str): File path to Session 2 CSV.
      
    Returns:
      pd.DataFrame: Merged DataFrame containing both sessions.
    """
    df_s1 = pd.read_csv(session1_path)
    df_s2 = pd.read_csv(session2_path)
    
    # Determine the common identifier column
    id_col_s1 = next((col for col in ['ID', 'participant', 'participant_ID'] if col in df_s1.columns), None)
    id_col_s2 = next((col for col in ['ID', 'participant', 'participant_ID'] if col in df_s2.columns), None)
    
    if not id_col_s1 or not id_col_s2:
        raise ValueError("No common identifier column found in one or both datasets.")
    
    # Rename only the alpha_boxcox_after_arcsin column to include session suffixes before merging
    if 'alpha_boxcox_after_arcsin' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'alpha_boxcox_after_arcsin': 'alpha_boxcox_after_arcsin_s1'})
    if 'alpha_boxcox_after_arcsin' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'alpha_boxcox_after_arcsin': 'alpha_boxcox_after_arcsin_s2'})


    if 'alpha_mean_boxcox_after_arcsin' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'alpha_mean_boxcox_after_arcsin': 'alpha_boxcox_after_arcsin_s1'})
    if 'alpha_mean_boxcox_after_arcsin' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'alpha_mean_boxcox_after_arcsin': 'alpha_boxcox_after_arcsin_s2'})


        

        # Rename 'a' and 'ndt' columns to include session suffixes
    if 'a' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'a': 'a_mean_s1'})
    if 'a' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'a': 'a_mean_s2'})
    if 'ndt' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'ndt': 'ndt_mean_s1'})
    if 'ndt' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'ndt': 'ndt_mean_s2'})

    if 'v1' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'v1': 'v1_mean_s1'})
    if 'v1' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'v1': 'v1_mean_s2'})
    if 'v2' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'v2': 'v2_mean_s1'})
    if 'v2' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'v2': 'v2_mean_s2'})
        
        # Rename 'a_mean' and 'ndt_mean' columns to include session suffixes
    if 'a_mean' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'a_mean': 'a_mean_s1'})
    if 'a_mean' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'a_mean': 'a_mean_s2'})
    if 'ndt_mean' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'ndt_mean': 'ndt_mean_s1'})
    if 'ndt_mean' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'ndt_mean': 'ndt_mean_s2'})
        
    if 'v1_mean' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'v1_mean': 'v1_mean_s1'})
    if 'v1_mean' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'v1_mean': 'v1_mean_s2'})
    if 'v2_mean' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'v2_mean': 'v2_mean_s1'})
    if 'v2_mean' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'v2_mean': 'v2_mean_s2'})
    
    
    # Merge on the common identifier column
    df_merged = pd.merge(df_s1, df_s2, on=id_col_s1)
    df_merged.dropna(inplace=True)
    
    # Determine the correct alpha column names
    alpha_col_s1 = 'alpha_mean_boxcox_after_arcsin_s1' if 'alpha_mean_boxcox_after_arcsin_s1' in df_merged.columns else 'alpha_boxcox_after_arcsin_s1'
    alpha_col_s2 = 'alpha_mean_boxcox_after_arcsin_s2' if 'alpha_mean_boxcox_after_arcsin_s2' in df_merged.columns else 'alpha_boxcox_after_arcsin_s2'
    
    # Rename the transformed alpha columns for convenience:
    df_merged["alpha_s1"] = df_merged[alpha_col_s1]
    df_merged["alpha_s2"] = df_merged[alpha_col_s2]
    
    return df_merged

def compute_normalized_mi(X, y, n_neighbors=3, random_state=42):
    """
    Compute normalized mutual information estimates to help with interpretation.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    results : dict
        Dictionary containing various normalized MI metrics.
    """
    # Calculate entropy of X[0] (alpha_s1)
    entropy_x = mutual_info_regression(
        X[:, 0].reshape(-1, 1),
        X[:, 0],
        discrete_features=False,
        n_neighbors=min(n_neighbors, len(X) - 1),
        random_state=random_state
    )[0]
    
    # Calculate entropy of y (alpha_s2)
    entropy_y = mutual_info_regression(
        y.reshape(-1, 1),
        y,
        discrete_features=False,
        n_neighbors=min(n_neighbors, len(X) - 1),
        random_state=random_state
    )[0]
    
    # Calculate mutual information between X[0] and y
    mi = mutual_info_regression(
        X[:, 0].reshape(-1, 1),
        y,
        discrete_features=False,
        n_neighbors=min(n_neighbors, len(X) - 1),
        random_state=random_state
    )[0]
    
    # Calculate uncertainty coefficient (asymmetric normalized MI)
    u_xy = 0 if entropy_y == 0 else mi / entropy_y  # How much X helps predict y
    u_yx = 0 if entropy_x == 0 else mi / entropy_x  # How much y helps predict X
    
    # Calculate symmetric normalized MI
    norm_mi = 0 if (entropy_x + entropy_y) == 0 else 2 * mi / (entropy_x + entropy_y)
    
    return {
        'entropy_x': entropy_x,
        'entropy_y': entropy_y,
        'mutual_information': mi,
        'uncertainty_coefficient_xy': u_xy,
        'uncertainty_coefficient_yx': u_yx,
        'normalized_mutual_information': norm_mi
    }

def run_detailed_analysis(df_merged, feature_cols=['alpha_s1', 'a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2','v1_mean_s1','v1_mean_s2','v2_mean_s1','v2_mean_s2'], 
                         target_col='alpha_s2', n_neighbors=3, n_permutations=1000, random_state=42):
    """
    Run an extended analysis with additional metrics for more comprehensive interpretation.
    
    This is particularly useful when the standard mutual information values are very small.
    
    Parameters:
    -----------
    df_merged : pandas DataFrame
        DataFrame containing session 1 and session 2 data.
    feature_cols : list of str, default=['alpha_s1', 'a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2','v1_mean_s1','v1_mean_s2','v2_mean_s1','v2_mean_s2']
        Column names for features.
    target_col : str, default='alpha_s2'
        Column name for target variable.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
    n_permutations : int, default=1000
        Number of permutations for significance testing.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    results : dict
        Dictionary containing all results of the analysis.
    """
    # First run the standard analysis
    standard_results = run_mutual_information_analysis(
        df_merged, feature_cols, target_col, n_neighbors, n_permutations, random_state
    )
    
    print("\n" + "=" * 70)
    print("Running Extended Analysis with Additional Metrics")
    print("=" * 70)
    
    # Prepare data
    X = df_merged[feature_cols].values
    y = df_merged[target_col].values
    
    # Compute normalized metrics
    norm_metrics = compute_normalized_mi(X, y, n_neighbors, random_state)
    
    print("\nAdditional Metrics for alpha_s1 and alpha_s2:")
    print(f"  Entropy of alpha_s1: {norm_metrics['entropy_x']:.4f} bits")
    print(f"  Entropy of alpha_s2: {norm_metrics['entropy_y']:.4f} bits")
    print(f"  Uncertainty coefficient (alpha_s1 -> alpha_s2): {norm_metrics['uncertainty_coefficient_xy']:.4f}")
    print(f"  Uncertainty coefficient (alpha_s2 -> alpha_s1): {norm_metrics['uncertainty_coefficient_yx']:.4f}")
    print(f"  Normalized mutual information: {norm_metrics['normalized_mutual_information']:.4f}")
    
    # Combine the results
    combined_results = {**standard_results, 'normalized_metrics': norm_metrics}
    
    return combined_results

def compute_partial_correlation(df, target_var='alpha_s2', predictor_var='alpha_s1', 
                              control_vars=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']):
    """
    Compute partial correlation between two variables while controlling for other variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing all variables.
    target_var : str, default='alpha_s2'
        The target variable (y).
    predictor_var : str, default='alpha_s1'
        The predictor variable (x).
    control_vars : list of str, default=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
        Variables to control for.
        
    Returns:
    --------
    partial_r : float
        Partial correlation coefficient.
    p_value : float
        P-value for the partial correlation.
    """
    # Step 1: Regress target_var on control_vars
    if len(control_vars) > 0:
        y = df[target_var].values
        X_control = df[control_vars].values
        model_y = LinearRegression().fit(X_control, y)
        residuals_y = y - model_y.predict(X_control)
        
        # Step 2: Regress predictor_var on control_vars
        x = df[predictor_var].values
        model_x = LinearRegression().fit(X_control, x)
        residuals_x = x - model_x.predict(X_control)
        
        # Step 3: Calculate correlation between residuals
        partial_r, p_value = pearsonr(residuals_x, residuals_y)
    else:
        # If no control variables, just compute regular correlation
        partial_r, p_value = pearsonr(df[predictor_var], df[target_var])
    
    return partial_r, p_value

def run_reliability_analysis(df_merged, alpha_vars=['alpha_s1', 'alpha_s2'], 
                          threshold_vars=['a_mean_s1', 'a_mean_s2'], 
                          ndt_vars=['ndt_mean_s1', 'ndt_mean_s2'],
                          drift_vars=['v1_mean_s1', 'v1_mean_s2', 'v2_mean_s1', 'v2_mean_s2']):
    """
    Run a comprehensive reliability analysis for alpha, controlling for other parameters.
    
    Parameters:
    -----------
    df_merged : pandas DataFrame
        DataFrame containing session 1 and session 2 data.
    alpha_vars : list of str, default=['alpha_s1', 'alpha_s2']
        Column names for alpha from session 1 and 2.
    threshold_vars : list of str, default=['a_mean_s1', 'a_mean_s2']
        Column names for threshold from session 1 and 2.
    ndt_vars : list of str, default=['ndt_mean_s1', 'ndt_mean_s2']
        Column names for non-decision time from session 1 and 2.
    drift_vars : list of str, default=['v1_mean_s1', 'v1_mean_s2', 'v2_mean_s1', 'v2_mean_s2']
        Column names for drift rates from session 1 and 2.
        
    Returns:
    --------
    results : dict
        Dictionary containing all results of the analysis.
    """
    print("Running Comprehensive Reliability Analysis")
    print("=" * 70)
    
    # Combine all control variables
    all_controls = threshold_vars + ndt_vars + drift_vars
    
    # 1. Calculate raw correlation for alpha
    raw_r, raw_p = pearsonr(df_merged[alpha_vars[0]], df_merged[alpha_vars[1]])
    print(f"Raw correlation between {alpha_vars[0]} and {alpha_vars[1]}:")
    print(f"  r = {raw_r:.4f}, p = {raw_p:.4f}")
    
    # 2. Calculate partial correlation controlling for all other parameters
    partial_r_all, partial_p_all = compute_partial_correlation(
        df_merged, alpha_vars[1], alpha_vars[0], all_controls
    )
    print(f"\nPartial correlation controlling for all parameters:")
    print(f"  r = {partial_r_all:.4f}, p = {partial_p_all:.4f}")
    
    # 3. Calculate partial correlations controlling for different parameter types
    partial_r_threshold, partial_p_threshold = compute_partial_correlation(
        df_merged, alpha_vars[1], alpha_vars[0], threshold_vars
    )
    print(f"\nPartial correlation controlling for threshold parameters:")
    print(f"  r = {partial_r_threshold:.4f}, p = {partial_p_threshold:.4f}")
    
    partial_r_ndt, partial_p_ndt = compute_partial_correlation(
        df_merged, alpha_vars[1], alpha_vars[0], ndt_vars
    )
    print(f"\nPartial correlation controlling for non-decision time parameters:")
    print(f"  r = {partial_r_ndt:.4f}, p = {partial_p_ndt:.4f}")
    
    partial_r_drift, partial_p_drift = compute_partial_correlation(
        df_merged, alpha_vars[1], alpha_vars[0], drift_vars
    )
    print(f"\nPartial correlation controlling for drift rate parameters:")
    print(f"  r = {partial_r_drift:.4f}, p = {partial_p_drift:.4f}")
    
    # 4. Calculate correlations between alpha and other parameters
    correlations = {}
    for var in all_controls:
        for alpha_var in alpha_vars:
            r, p = pearsonr(df_merged[alpha_var], df_merged[var])
            correlations[f"{alpha_var}_{var}"] = (r, p)
    
    print("\nCorrelations between alpha and other parameters:")
    for key, (r, p) in correlations.items():
        print(f"  {key}: r = {r:.4f}, p = {p:.4f}")
    
    # 5. Compare the results
    print("\nReliability comparison:")
    print(f"  Raw correlation:                               r = {raw_r:.4f}")
    print(f"  Controlling for all parameters:                r = {partial_r_all:.4f}")
    print(f"  Controlling for threshold:                     r = {partial_r_threshold:.4f}")
    print(f"  Controlling for non-decision time:             r = {partial_r_ndt:.4f}")
    print(f"  Controlling for drift rates:                   r = {partial_r_drift:.4f}")
    
    # 6. Percentage change in correlation
    pct_change_all = (partial_r_all - raw_r) / raw_r * 100
    pct_change_threshold = (partial_r_threshold - raw_r) / raw_r * 100
    pct_change_ndt = (partial_r_ndt - raw_r) / raw_r * 100
    pct_change_drift = (partial_r_drift - raw_r) / raw_r * 100
    
    print("\nPercentage change in correlation coefficient:")
    print(f"  When controlling for all parameters:           {pct_change_all:.1f}%")
    print(f"  When controlling for threshold:                {pct_change_threshold:.1f}%")
    print(f"  When controlling for non-decision time:        {pct_change_ndt:.1f}%")
    print(f"  When controlling for drift rates:              {pct_change_drift:.1f}%")
    
    # 7. Return all results
    results = {
        'raw_correlation': (raw_r, raw_p),
        'partial_correlation_all': (partial_r_all, partial_p_all),
        'partial_correlation_threshold': (partial_r_threshold, partial_p_threshold),
        'partial_correlation_ndt': (partial_r_ndt, partial_p_ndt),
        'partial_correlation_drift': (partial_r_drift, partial_p_drift),
        'parameter_correlations': correlations,
        'percent_changes': {
            'all': pct_change_all,
            'threshold': pct_change_threshold,
            'ndt': pct_change_ndt,
            'drift': pct_change_drift
        }
    }
    
    return results

def plot_reliability_results(results):
    """
    Create visualizations for the reliability analysis results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_reliability_analysis.
    """
    # 1. Create a bar plot comparing raw and partial correlations
    corr_labels = [
        'Raw correlation', 
        'Controlling for\nall parameters',
        'Controlling for\nthreshold',
        'Controlling for\nnon-decision time',
        'Controlling for\ndrift rates'
    ]
    
    corr_values = [
        results['raw_correlation'][0],
        results['partial_correlation_all'][0],
        results['partial_correlation_threshold'][0],
        results['partial_correlation_ndt'][0],
        results['partial_correlation_drift'][0]
    ]
    
    p_values = [
        results['raw_correlation'][1],
        results['partial_correlation_all'][1],
        results['partial_correlation_threshold'][1],
        results['partial_correlation_ndt'][1],
        results['partial_correlation_drift'][1]
    ]
    
    # Create a color map based on significance
    colors = ['darkgreen' if p < 0.05 else 'lightgreen' if p < 0.1 else 'gray' for p in p_values]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(corr_labels, corr_values, color=colors)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value annotations on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + (0.01 if height >= 0 else -0.04),
                f'r = {corr_values[i]:.3f}\np = {p_values[i]:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)
    
    plt.ylabel('Correlation Coefficient (r)', fontsize=12)
    plt.title('Alpha Reliability: Raw vs. Partial Correlations', fontsize=14)
    plt.ylim(min(min(corr_values) - 0.1, -0.1), max(max(corr_values) + 0.1, 0.5))
    plt.grid(axis='y', alpha=0.3)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', label='p < 0.05'),
        Patch(facecolor='lightgreen', label='p < 0.10'),
        Patch(facecolor='gray', label='p ≥ 0.10')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Create a heatmap of correlations between parameters
    param_corrs = results['parameter_correlations']
    
    # Extract unique parameter names (excluding alpha)
    alpha_vars = ['alpha_s1', 'alpha_s2']
    unique_params = list(set([key.split('_', 1)[1] for key in param_corrs.keys()]))
    
    # Create matrices for correlation values and p-values
    corr_matrix = np.zeros((len(alpha_vars), len(unique_params)))
    p_matrix = np.zeros((len(alpha_vars), len(unique_params)))
    
    for i, alpha in enumerate(alpha_vars):
        for j, param in enumerate(unique_params):
            key = f"{alpha}_{param}"
            if key in param_corrs:
                corr_matrix[i, j] = param_corrs[key][0]
                p_matrix[i, j] = param_corrs[key][1]
    
    # Create a mask for non-significant correlations
    mask = p_matrix >= 0.05
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
               xticklabels=unique_params, yticklabels=alpha_vars, mask=mask, 
               cbar_kws={'label': 'Correlation Coefficient (r)'})
    plt.title('Significant Correlations Between Alpha and Other Parameters', fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_alpha_threshold_ndt_relationship(df_merged, alpha_vars=['alpha_s1', 'alpha_s2'], 
                                         threshold_vars=['a_mean_s1', 'a_mean_s2'], 
                                         ndt_vars=['ndt_mean_s1', 'ndt_mean_s2']):
    """
    Analyze in detail how alpha reliability relates to threshold and non-decision time parameters.
    This function excludes drift rates from the analysis to focus specifically on the relationship
    between alpha, threshold, and non-decision time.
    
    Parameters:
    -----------
    df_merged : pandas DataFrame
        DataFrame containing session 1 and session 2 data.
    alpha_vars : list of str, default=['alpha_s1', 'alpha_s2']
        Column names for alpha from session 1 and 2.
    threshold_vars : list of str, default=['a_mean_s1', 'a_mean_s2']
        Column names for threshold from session 1 and 2.
    ndt_vars : list of str, default=['ndt_mean_s1', 'ndt_mean_s2']
        Column names for non-decision time from session 1 and 2.
        
    Returns:
    --------
    results : dict
        Dictionary containing all results of the analysis.
    """
    print("Analyzing Alpha, Threshold, and Non-Decision Time Relationships")
    print("=" * 70)
    
    # 1. Calculate raw correlations between all parameters
    all_vars = alpha_vars + threshold_vars + ndt_vars
    corr_matrix = np.zeros((len(all_vars), len(all_vars)))
    p_matrix = np.zeros((len(all_vars), len(all_vars)))
    
    for i, var1 in enumerate(all_vars):
        for j, var2 in enumerate(all_vars):
            if i != j:  # Skip self-correlations
                r, p = pearsonr(df_merged[var1], df_merged[var2])
                corr_matrix[i, j] = r
                p_matrix[i, j] = p
            else:
                corr_matrix[i, j] = 1.0  # Self-correlation is 1
                p_matrix[i, j] = 0.0     # p-value for self-correlation is 0
    
    # 2. Calculate partial correlations
    # Alpha reliability controlling for threshold
    partial_r_threshold, partial_p_threshold = compute_partial_correlation(
        df_merged, alpha_vars[1], alpha_vars[0], threshold_vars
    )
    
    # Alpha reliability controlling for non-decision time
    partial_r_ndt, partial_p_ndt = compute_partial_correlation(
        df_merged, alpha_vars[1], alpha_vars[0], ndt_vars
    )
    
    # Alpha reliability controlling for both threshold and non-decision time
    partial_r_both, partial_p_both = compute_partial_correlation(
        df_merged, alpha_vars[1], alpha_vars[0], threshold_vars + ndt_vars
    )
    
    # 3. Calculate variance components using regression
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
    
    # Calculate variance components for different control variable sets
    variance_threshold = calc_variance_explained(
        df_merged, target=alpha_vars[1], predictor=alpha_vars[0], controls=threshold_vars
    )
    
    variance_ndt = calc_variance_explained(
        df_merged, target=alpha_vars[1], predictor=alpha_vars[0], controls=ndt_vars
    )
    
    variance_both = calc_variance_explained(
        df_merged, target=alpha_vars[1], predictor=alpha_vars[0], controls=threshold_vars + ndt_vars
    )
    
    # 4. Print summary results
    print("\nAlpha Reliability:")
    raw_r, raw_p = pearsonr(df_merged[alpha_vars[0]], df_merged[alpha_vars[1]])
    print(f"  Raw correlation: r = {raw_r:.4f}, p = {raw_p:.4f}")
    print(f"  Controlling for threshold: r = {partial_r_threshold:.4f}, p = {partial_p_threshold:.4f}")
    print(f"  Controlling for non-decision time: r = {partial_r_ndt:.4f}, p = {partial_p_ndt:.4f}")
    print(f"  Controlling for both: r = {partial_r_both:.4f}, p = {partial_p_both:.4f}")
    
    # Print variance decomposition
    print("\nVariance Decomposition:")
    print("  When controlling for threshold parameters:")
    print(f"    Unique variance contribution of alpha_s1: {variance_threshold['unique_variance']:.4f}")
    print(f"    Shared variance with threshold: {variance_threshold['shared_variance']:.4f}")
    print(f"    Percent of alpha_s1's effect that is unique: {variance_threshold['unique_percent']:.1f}%")
    
    print("\n  When controlling for non-decision time parameters:")
    print(f"    Unique variance contribution of alpha_s1: {variance_ndt['unique_variance']:.4f}")
    print(f"    Shared variance with non-decision time: {variance_ndt['shared_variance']:.4f}")
    print(f"    Percent of alpha_s1's effect that is unique: {variance_ndt['unique_percent']:.1f}%")
    
    print("\n  When controlling for both threshold and non-decision time:")
    print(f"    Unique variance contribution of alpha_s1: {variance_both['unique_variance']:.4f}")
    print(f"    Shared variance with both: {variance_both['shared_variance']:.4f}")
    print(f"    Percent of alpha_s1's effect that is unique: {variance_both['unique_percent']:.1f}%")
    
    # 5. Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    mask = p_matrix >= 0.05  # Mask non-significant correlations
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
               xticklabels=all_vars, yticklabels=all_vars, mask=mask, 
               cbar_kws={'label': 'Correlation Coefficient (r)'})
    plt.title('Significant Correlations Between Parameters', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 6. Plot the variance decomposition
    # Create a comparative bar chart of unique vs. shared variance
    labels = ['Controlling for\nThreshold', 'Controlling for\nNDT', 'Controlling for\nBoth']
    unique_values = [
        variance_threshold['unique_variance'],
        variance_ndt['unique_variance'],
        variance_both['unique_variance']
    ]
    shared_values = [
        variance_threshold['shared_variance'],
        variance_ndt['shared_variance'],
        variance_both['shared_variance']
    ]
    
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    ax.bar(np.arange(len(labels)) - width/2, unique_values, width, label='Unique Variance', color='#2ca02c')
    ax.bar(np.arange(len(labels)) + width/2, shared_values, width, label='Shared Variance', color='#ff7f0e')
    
    # Add some text for labels, title and axes ticks
    ax.set_ylabel('R² (Variance Explained)', fontsize=12)
    ax.set_title('Variance Decomposition of Alpha Reliability', fontsize=14)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value annotations
    for i, v in enumerate(unique_values):
        ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(shared_values):
        ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 7. Return results
    results = {
        'raw_correlation': (raw_r, raw_p),
        'partial_correlations': {
            'threshold': (partial_r_threshold, partial_p_threshold),
            'ndt': (partial_r_ndt, partial_p_ndt),
            'both': (partial_r_both, partial_p_both)
        },
        'variance_components': {
            'threshold': variance_threshold,
            'ndt': variance_ndt,
            'both': variance_both
        },
        'correlation_matrix': corr_matrix,
        'p_value_matrix': p_matrix,
        'parameter_labels': all_vars
    }
    
    return results

def analyze_mutual_info_with_controls(df_merged, feature='alpha_s1', target='alpha_s2', 
                                   control_vars=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2'],
                                   n_neighbors=5, random_state=42):
    """
    Analyze the mutual information between two variables while accounting for control variables.
    This approach is more appropriate for transformed variables since it can detect nonlinear relationships.
    
    Parameters:
    -----------
    df_merged : pandas DataFrame
        DataFrame containing all variables.
    feature : str, default='alpha_s1'
        The feature variable (X).
    target : str, default='alpha_s2'
        The target variable (y).
    control_vars : list of str, default=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
        Variables to control for.
    n_neighbors : int, default=5
        Number of neighbors for MI estimation.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    results : dict
        Dictionary containing all results of the analysis.
    """
    print("Analyzing Mutual Information Between Alpha Parameters")
    print("=" * 70)
    
    # 1. Calculate raw mutual information
    X = df_merged[[feature]].values
    y = df_merged[target].values
    
    raw_mi = mutual_info_regression(X, y, discrete_features=False, 
                                 n_neighbors=n_neighbors, random_state=random_state)[0]
    
    print(f"Raw mutual information between {feature} and {target}: {raw_mi:.4f} bits")
    
    # 2. Calculate conditional mutual information using each control variable separately
    cond_mi_individual = {}
    
    for var in control_vars:
        # Create feature matrix X with the variable to condition on
        X_cond = df_merged[[feature, var]].values
        
        # Standardize X for better estimation
        scaler_X = StandardScaler()
        X_cond_scaled = scaler_X.fit_transform(X_cond)
        
        # Standardize y
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Extract the conditioning variable
        Z = X_cond_scaled[:, 1].reshape(-1, 1)
        
        # 1. Residualize X[0] with respect to Z
        knn_x = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X) - 1))
        knn_x.fit(Z, X_cond_scaled[:, 0])
        residuals_x = X_cond_scaled[:, 0] - knn_x.predict(Z)
        
        # 2. Residualize y with respect to Z
        knn_y = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X) - 1))
        knn_y.fit(Z, y_scaled)
        residuals_y = y_scaled - knn_y.predict(Z)
        
        # 3. Calculate mutual information between residuals
        cmi = mutual_info_regression(
            residuals_x.reshape(-1, 1), residuals_y, discrete_features=False,
            n_neighbors=min(n_neighbors, len(X) - 1), random_state=random_state)[0]
        
        cond_mi_individual[var] = cmi
        
        # Calculate how much mutual information is retained
        if raw_mi > 0:
            retention = (cmi / raw_mi) * 100
            if retention > 100:
                print(f"  Conditioning on {var}: CMI = {cmi:.4f} bits ({retention:.1f}% retained) *")
                print(f"    * Note: Value >100% indicates {var} may be suppressing or masking the relationship")
            else:
                print(f"  Conditioning on {var}: CMI = {cmi:.4f} bits ({retention:.1f}% retained)")
        else:
            print(f"  Conditioning on {var}: CMI = {cmi:.4f} bits")
    
    # 3. Calculate conditional mutual information with all control variables
    if len(control_vars) > 0:
        # Create feature matrix with all variables
        X_all = df_merged[[feature] + control_vars].values
        
        # Standardize X for better estimation
        scaler_X = StandardScaler()
        X_all_scaled = scaler_X.fit_transform(X_all)
        
        # Standardize y
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Extract all conditioning variables
        Z = X_all_scaled[:, 1:]
        
        # 1. Residualize X[0] with respect to Z
        knn_x = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X) - 1))
        knn_x.fit(Z, X_all_scaled[:, 0])
        residuals_x = X_all_scaled[:, 0] - knn_x.predict(Z)
        
        # 2. Residualize y with respect to Z
        knn_y = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X) - 1))
        knn_y.fit(Z, y_scaled)
        residuals_y = y_scaled - knn_y.predict(Z)
        
        # 3. Calculate mutual information between residuals
        cmi_all = mutual_info_regression(
            residuals_x.reshape(-1, 1), residuals_y, discrete_features=False,
            n_neighbors=min(n_neighbors, len(X) - 1), random_state=random_state)[0]
        
        # Calculate how much mutual information is retained
        if raw_mi > 0:
            retention_all = (cmi_all / raw_mi) * 100
            if retention_all > 100:
                print(f"\nConditional MI with all control variables: {cmi_all:.4f} bits ({retention_all:.1f}% retained) *")
                print("  * Note: Value >100% indicates some control variables may be suppressing the relationship")
                print("    between alpha parameters, revealing stronger connections after conditioning.")
            else:
                print(f"\nConditional MI with all control variables: {cmi_all:.4f} bits ({retention_all:.1f}% retained)")
        else:
            print(f"\nConditional MI with all control variables: {cmi_all:.4f} bits")
    else:
        cmi_all = raw_mi
        retention_all = 100.0
    
    # Add note about information retention
    print("\nNote about information retention percentages:")
    print("  - Values less than 100% indicate the control variable(s) explain some of the relationship")
    print("  - Values equal to 100% indicate the control variable(s) have no effect on the relationship")
    print("  - Values greater than 100% indicate the control variable(s) were suppressing or masking")
    print("    the relationship, making it appear weaker than it actually is when properly conditioned")
    
    # 4. Perform permutation test for significance
    # Calculate observed CMI with all control variables
    observed_cmi = cmi_all
    
    # Generate null distribution by permuting feature
    null_distribution = []
    rng = np.random.RandomState(random_state)
    n_permutations = 1000
    
    for i in range(n_permutations):
        # Create copy of X with permuted feature
        X_all_perm = X_all.copy()
        X_all_perm[:, 0] = rng.permutation(X_all[:, 0])
        
        # Standardize X for better estimation
        X_all_perm_scaled = scaler_X.fit_transform(X_all_perm)
        
        # Extract feature and conditioning variables
        X_perm = X_all_perm_scaled[:, 0]
        Z_perm = X_all_perm_scaled[:, 1:]
        
        # Residualize X with respect to Z
        knn_x = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(X) - 1))
        knn_x.fit(Z_perm, X_perm)
        residuals_x_perm = X_perm - knn_x.predict(Z_perm)
        
        # Calculate MI between permuted residuals and original target residuals
        null_cmi = mutual_info_regression(
            residuals_x_perm.reshape(-1, 1), residuals_y, discrete_features=False,
            n_neighbors=min(n_neighbors, len(X) - 1), random_state=random_state+i)[0]
        
        null_distribution.append(null_cmi)
    
    # Calculate p-value
    null_distribution = np.array(null_distribution)
    
    # Handle the case where all values in the null distribution are very close to zero
    if np.std(null_distribution) < 1e-6:
        # If observed_cmi is also close to zero, consider it non-significant
        if observed_cmi < 1e-6:
            p_value = 1.0
        else:
            p_value = 0.0
    else:
        # Regular p-value calculation
        p_value = (np.sum(null_distribution >= observed_cmi) + 1) / (n_permutations + 1)
    
    print(f"Permutation test p-value: {p_value:.4f}")
    
    # 5. Calculate correlation for comparison
    pearson_r, pearson_p = pearsonr(df_merged[feature], df_merged[target])
    spearman_r, spearman_p = spearmanr(df_merged[feature], df_merged[target])
    print("\nCorrelations for comparison:")
    print(f"  Pearson: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    print(f"  Spearman: rho = {spearman_r:.4f}, p = {spearman_p:.4f}")
    
    # 6. Visualize the results
    # Bar plot of raw MI and conditional MI values
    plt.figure(figsize=(10, 6))
    all_labels = ['Raw MI'] + [f'Conditioning\non {var}' for var in control_vars] + ['Conditioning\non all']
    all_values = [raw_mi] + [cond_mi_individual[var] for var in control_vars] + [cmi_all]
    
    colors = ['blue'] + ['lightblue'] * len(control_vars) + ['darkblue']
    
    plt.bar(all_labels, all_values, color=colors)
    plt.ylabel('Mutual Information (bits)', fontsize=12)
    plt.title('Mutual Information Between Alpha Parameters', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot correlation between alpha_s1 and alpha_s2 along with MI
    plt.figure(figsize=(8, 8))
    plt.scatter(df_merged[feature], df_merged[target], alpha=0.6)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title(f"Relationship Between {feature} and {target}\n" +
            f"MI: {raw_mi:.4f} bits, Pearson r: {pearson_r:.4f}, Spearman rho: {spearman_r:.4f}", 
            fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 7. Return all results
    results = {
        'raw_mi': raw_mi,
        'cond_mi_individual': cond_mi_individual,
        'cond_mi_all': cmi_all,
        'retention_all': retention_all if raw_mi > 0 else None,
        'p_value': p_value,
        'null_distribution': null_distribution,
        'pearson': (pearson_r, pearson_p),
        'spearman': (spearman_r, spearman_p)
    }
    
    return results

def compare_mutual_info_across_k(df_merged, feature='alpha_s1', target='alpha_s2', 
                              control_vars=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2'],
                              k_values=[3, 5, 7, 10, 15]):
    """
    Compare mutual information calculations across different values of k (number of neighbors).
    This helps assess the robustness of the mutual information estimates.
    
    Parameters:
    -----------
    df_merged : pandas DataFrame
        DataFrame containing all variables.
    feature : str, default='alpha_s1'
        The feature variable (X).
    target : str, default='alpha_s2'
        The target variable (y).
    control_vars : list of str, default=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
        Variables to control for.
    k_values : list of int, default=[3, 5, 7, 10, 15]
        Numbers of neighbors to try.
        
    Returns:
    --------
    results : dict
        Dictionary containing all results of the analysis.
    """
    print("Comparing Mutual Information Estimates Across Different k Values")
    print("=" * 70)
    
    # Store results for each k
    all_results = {}
    raw_mi_values = []
    cond_mi_values = []
    retention_values = []
    
    # Calculate MI for each k
    for k in k_values:
        print(f"\nAnalysis with k={k} neighbors:")
        result_k = analyze_mutual_info_with_controls(
            df_merged, feature, target, control_vars, n_neighbors=k
        )
        
        all_results[k] = result_k
        raw_mi_values.append(result_k['raw_mi'])
        cond_mi_values.append(result_k['cond_mi_all'])
        
        if result_k['raw_mi'] > 0:
            retention_values.append(result_k['retention_all'])
        else:
            retention_values.append(0)
    
    # Plot results as a function of k
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, raw_mi_values, 'o-', label='Raw MI', color='blue', linewidth=2)
    plt.plot(k_values, cond_mi_values, 's-', label='Conditional MI', color='green', linewidth=2)
    plt.xlabel('Number of neighbors (k)', fontsize=12)
    plt.ylabel('Mutual Information (bits)', fontsize=12)
    plt.title('MI vs. Number of Neighbors', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, retention_values, 'o-', color='red', linewidth=2)
    plt.xlabel('Number of neighbors (k)', fontsize=12)
    plt.ylabel('% of MI retained after conditioning', fontsize=12)
    plt.title('Information Retention vs. Number of Neighbors', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation between transformed variables
    plt.figure(figsize=(10, 6))
    
    # Scatterplot of alpha_s1 vs alpha_s2 with color based on a_mean values
    sc = plt.scatter(df_merged[feature], df_merged[target], 
                   c=df_merged['a_mean_s1'], cmap='viridis', 
                   alpha=0.7, s=50)
    
    plt.colorbar(sc, label='a_mean_s1 (threshold)')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title(f'Relationship between {feature} and {target}\ncolored by threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return all results
    return {
        'all_results': all_results,
        'raw_mi_values': raw_mi_values,
        'cond_mi_values': cond_mi_values,
        'retention_values': retention_values,
        'k_values': k_values
    }

def comprehensive_alpha_analysis(df_merged, feature='alpha_s1', target='alpha_s2', 
                               control_vars=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2'],
                               n_neighbors=5, k_values=[3, 5, 7, 10, 15]):
    """
    Run a comprehensive analysis of mutual information between alpha parameters across sessions.
    
    This function runs all analyses in one call and generates informative visualizations and
    interpretations of the results.
    
    Parameters:
    -----------
    df_merged : pandas DataFrame
        DataFrame containing session 1 and session 2 data.
    feature : str, default='alpha_s1'
        Column name for alpha from session 1.
    target : str, default='alpha_s2'
        Column name for alpha from session 2.
    control_vars : list of str, default=['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
        Variables to control for in the analysis.
    n_neighbors : int, default=5
        Number of neighbors for primary MI estimation.
    k_values : list of int, default=[3, 5, 7, 10, 15]
        Numbers of neighbors to try for robustness analysis.
        
    Returns:
    --------
    results : dict
        Dictionary containing all results of the analyses.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set styling for plots
    sns.set_theme(style="whitegrid")
    
    # Run the mutual information analysis with control variables
    print("Analyzing mutual information between alpha parameters across sessions")
    print("=" * 70)
    
    # Run individual analysis with specified neighbors
    mi_result = analyze_mutual_info_with_controls(
        df_merged, 
        feature=feature,
        target=target,
        control_vars=control_vars,
        n_neighbors=n_neighbors
    )
    
    # Check robustness across different k values
    print("\nAssessing robustness across different k values:")
    k_results = compare_mutual_info_across_k(
        df_merged, 
        feature=feature,
        target=target,
        control_vars=control_vars,
        k_values=k_values
    )
    
    # Create additional visualization showing the relationship between alpha parameters
    # and threshold/NDT parameters
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Alpha_s1 vs Alpha_s2 colored by threshold
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(df_merged[feature], df_merged[target], 
                   c=df_merged[control_vars[0]], cmap='viridis', 
                   alpha=0.7, s=50)
    plt.colorbar(sc1, label=f'{control_vars[0]} (threshold)')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title('Alpha Relationship\nColored by Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Alpha_s1 vs Alpha_s2 colored by non-decision time
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(df_merged[feature], df_merged[target], 
                   c=df_merged[control_vars[2]], cmap='plasma', 
                   alpha=0.7, s=50)
    plt.colorbar(sc2, label=f'{control_vars[2]} (non-decision time)')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title('Alpha Relationship\nColored by Non-Decision Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary of findings
    print("\nSummary of Findings:")
    print("-" * 60)
    
    # Get the raw mutual information and retention percentage
    raw_mi = mi_result['raw_mi']
    retention = mi_result['retention_all'] if mi_result['retention_all'] is not None else 0
    
    print(f"Raw mutual information between {feature} and {target}: {raw_mi:.4f} bits")
    print(f"After controlling for threshold and NDT parameters: {mi_result['cond_mi_all']:.4f} bits")
    print(f"Percentage of information retained: {retention:.1f}%")
    print(f"Permutation test p-value: {mi_result['p_value']:.4f}")
    
    print("\nPearson correlation for comparison: r = {:.4f}, p = {:.4f}".format(
        mi_result['pearson'][0], mi_result['pearson'][1]
    ))
    
    print("\nInterpretation:")
    if retention < 25:
        print("The alpha reliability appears to be largely shared with threshold and non-decision time parameters.")
        print("This suggests that alpha's stability across sessions may be primarily due to its relationship with these parameters.")
    elif retention < 50:
        print("A substantial portion of alpha reliability appears to be shared with threshold and non-decision time parameters.")
        print("However, some unique reliability remains after accounting for these relationships.")
    else:
        print("A significant portion of alpha reliability appears to be independent of threshold and non-decision time parameters.")
        print("This suggests alpha has unique variance that is stable across sessions.")
    
    # Create a pie chart showing the breakdown of information
    plt.figure(figsize=(8, 8))
    labels = ['Unique to Alpha', 'Shared with\nThreshold & NDT']
    sizes = [retention, 100-retention]
    colors = ['#2ca02c', '#ff7f0e']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
    plt.axis('equal')
    plt.title('Breakdown of Alpha Reliability\n(Based on Mutual Information)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Return combined results
    return {
        'mi_result': mi_result,
        'k_results': k_results,
        'retention': retention,
        'raw_mi': raw_mi
    }

# Example usage:
# results = run_mutual_information_analysis(df_merged)