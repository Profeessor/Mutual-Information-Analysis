#!/usr/bin/env python
"""
GBR.py

This module contains functions to load and merge two-session datasets,
train a Gradient Boosting Regressor on the following features:
    - alpha_mean_boxcox_after_arcsin_s1 (renamed to "alpha_s1")
    - a_mean_s1, a_mean_s2 (threshold measures)
    - ndt_mean_s1, ndt_mean_s2 (non-decision time measures)
to predict alpha_mean_boxcox_after_arcsin_s2 (renamed to "alpha_s2").

It also prints feature importances and plots a partial dependence curve
for alpha_s1 (Session 1) as a predictor of alpha_s2 (Session 2).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

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

def train_gradient_boosting(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets, trains a Gradient Boosting Regressor,
    and returns the trained model along with the split data and R² scores.
    
    Parameters:
      X (pd.DataFrame): Predictor features.
      y (pd.Series): Target variable.
      test_size (float): Proportion of data to be used for testing.
      random_state (int): Seed for random splitting.
      
    Returns:
      model: Trained GradientBoostingRegressor.
      X_train, X_test, y_train, y_test: Split data.
      train_r2, test_r2: R² scores for training and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=random_state
    )
    gbr.fit(X_train, y_train)
    train_r2 = gbr.score(X_train, y_train)
    test_r2  = gbr.score(X_test, y_test)
    
    return gbr, X_train, X_test, y_train, y_test, train_r2, test_r2

def display_feature_importances(gbr, X_test, y_test, feature_cols):
    """
    Displays Gini-based feature importances and permutation importances for each predictor.
    
    Parameters:
      gbr: Trained GradientBoostingRegressor.
      X_test (pd.DataFrame): Test data.
      y_test (pd.Series): Target values for the test data.
      feature_cols (list): List of feature column names.
      
    Returns:
      perm_imp: The permutation importance result.
    """
    # Gini-based importances:
    importances = gbr.feature_importances_
    print("\nGini-based Feature Importances:")
    for col, imp in zip(feature_cols, importances):
        print(f"  {col}: {imp:.3f}")
    
    # Permutation importances:
    perm_imp = permutation_importance(gbr, X_test, y_test, n_repeats=5, random_state=42)
    perm_sorted_idx = perm_imp.importances_mean.argsort()[::-1]
    print("\nPermutation Importances (Test set):")
    for idx in perm_sorted_idx:
        print(f"  {feature_cols[idx]}: Mean={perm_imp.importances_mean[idx]:.4f}, std={perm_imp.importances_std[idx]:.4f}")
    
    return perm_imp

def plot_partial_dependence_curve(gbr, X_test, feature_cols, feature_index=0):
    """
    Plots the partial dependence curve for the specified feature (by index).
    Default: feature_index=0 corresponds to 'alpha_s1'.
    
    Parameters:
      gbr: Trained GradientBoostingRegressor.
      X_test (pd.DataFrame): Test data.
      feature_cols (list): List of feature column names.
      feature_index (int): Index of the feature to plot.
    """
    disp = PartialDependenceDisplay.from_estimator(
        estimator=gbr,
        X=X_test,
        features=[feature_index],
        kind='average',
        feature_names=feature_cols
    )
    plt.title(f"Partial Dependence: Predicted αₛ₂ vs. {feature_cols[feature_index]}")
    plt.xlabel(feature_cols[feature_index])
    plt.ylabel("Predicted αₛ₂")
    plt.show()

def analyze_dataset(session1_path, session2_path, test_size=0.2, random_state=42):
    """
    Main analysis function.
    
    Steps:
      1. Load and merge Session 1 and Session 2 data.
      2. Define predictor features and the target variable.
      3. Train a Gradient Boosting Regressor.
      4. Print training and test R² scores.
      5. Display Gini-based and permutation feature importances.
      6. Plot the partial dependence curve for 'alpha_s1'.
    
    Parameters:
      session1_path (str): File path for Session 1 CSV.
      session2_path (str): File path for Session 2 CSV.
      test_size (float): Fraction of data to use as the test set.
      random_state (int): Random seed.
    """
    # Load and merge the datasets
    df_merged = load_and_merge(session1_path, session2_path)
    
    # Define predictors and target variable
    feature_cols = [
        "alpha_s1",     # Transformed alpha from Session 1
        "a_mean_s1",    # Threshold from Session 1
        "a_mean_s2",    # Threshold from Session 2
        "ndt_mean_s1",  # Non-decision time from Session 1
        "ndt_mean_s2",  # Non-decision time from Session 2
        "v1_mean_s1",   # drift rate from Session 1
        "v1_mean_s2",   # drift rate from Session 2
        "v2_mean_s1",   # drift rate from Session 1
        "v2_mean_s2"    # drift rate from Session 2
    ]
    target_col = "alpha_s2"  # Transformed alpha from Session 2
    
    X = df_merged[feature_cols].copy()
    y = df_merged[target_col].copy()
    
    # Train the model
    gbr, X_train, X_test, y_train, y_test, train_r2, test_r2 = train_gradient_boosting(
        X, y, test_size, random_state
    )
    
    print(f"GradientBoostingRegressor R^2 on TRAIN: {train_r2:.3f}")
    print(f"GradientBoostingRegressor R^2 on TEST:  {test_r2:.3f}")
    
    # Display feature importances (now passing y_test)
    _ = display_feature_importances(gbr, X_test, y_test, feature_cols)
    
    # Plot the partial dependence for 'alpha_s1' (index 0)
    plot_partial_dependence_curve(gbr, X_test, feature_cols, feature_index=0)

if __name__ == "__main__":
    # Example file paths (modify as necessary)
    session1_path = '/mnt/data/ldt_session_1_data_transformed.csv'
    session2_path = '/mnt/data/ldt_session_2_data_transformed.csv'
    analyze_dataset(session1_path, session2_path)
