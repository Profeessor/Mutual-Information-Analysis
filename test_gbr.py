#!/usr/bin/env python

# Import required libraries
import pandas as pd
import numpy as np
import time
import os
import importlib

# Re-import the fixed module
import GBR_APT_optimized
importlib.reload(GBR_APT_optimized)
from GBR_APT_optimized import load_and_merge_datasets_apt

# Scikit-learn imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error

# --- Configuration ---
session1_path = 'Model1 APT/study2_associative_session_1_data_transformed.csv'
session2_path = 'Model1 APT/study2_associative_session_2_data_transformed.csv'
manual_dataset_name = "APT Model"
target_col = 'alpha_s2'
random_seed = 42
n_cv_folds = 5

print(f"Dataset Name: {manual_dataset_name}")

# Define just one feature set for testing
all_possible_features_apt = [
    'alpha_s1',
    'a_s1', 'a_s2',
    'ndt1_s1', 'ndt1_s2', 'ndt2_s1', 'ndt2_s2', 'ndt3_s1', 'ndt3_s2', 'ndt4_s1', 'ndt4_s2',
]

feature_sets = {
    "All_Controls_APT": all_possible_features_apt
}

# Define simplified hyperparameter grid for GBR
param_grid_gbr = {
    'n_estimators': [100],  # Reduced for faster testing
    'max_depth': [2],
    'learning_rate': [0.1],
    'min_samples_leaf': [5]
}

# Define scoring metrics for cross-validation
scoring_metrics = {
    'r2': make_scorer(r2_score),
    'neg_MAE': make_scorer(mean_absolute_error, greater_is_better=False),
    'neg_RMSE': make_scorer(mean_squared_error, squared=False, greater_is_better=False)
}

# --- Load APT Data ---
try:
    print("Loading and merging datasets...")
    df_merged = load_and_merge_datasets_apt(session1_path, session2_path)
    if df_merged is not None and not df_merged.empty:
        print(f"\nAPT Data loaded. Final N = {len(df_merged)}")
        # Verify target column exists
        if target_col not in df_merged.columns:
            raise ValueError(f"Target column '{target_col}' not found in loaded APT data.")
        
        # Get target variable
        y = df_merged[target_col]
        print(f"Target variable shape: {y.shape}, type: {type(y)}")
        
        # Run analysis for first condition as a test
        condition_name = "All_Controls_APT"
        current_features = feature_sets[condition_name]
        print("\n" + "="*80)
        print(f"--- Running CV Analysis for Condition: {condition_name} ---")
        print(f"Features: {current_features}")
        print("="*80)
        
        # Verify all features exist in the loaded data
        missing_features = [feat for feat in current_features if feat not in df_merged.columns]
        if missing_features:
            print(f"Skipping condition '{condition_name}' due to missing feature columns: {missing_features}")
        else:
            X = df_merged[current_features]
            print(f"Feature matrix shape: {X.shape}")
            
            # Cross-Validation for Performance Estimation
            print(f"Performing {n_cv_folds}-fold Cross-Validation with GridSearchCV...")
            gbr = GradientBoostingRegressor(random_state=random_seed)
            print("Running GridSearchCV on full data to find best params...")
            gbr_grid_search = GridSearchCV(gbr, param_grid=param_grid_gbr, cv=n_cv_folds, scoring='r2', n_jobs=-1)
            gbr_grid_search.fit(X, y)
            best_gbr_model = gbr_grid_search.best_estimator_
            print(f"Best GBR Params found: {gbr_grid_search.best_params_}")
            print("SUCCESS! The GridSearchCV completed without errors.")
            
            # Do a simple cross-validation to verify everything works
            print(f"\nRunning {n_cv_folds}-fold cross-validation using best estimator...")
            cv_results = cross_validate(best_gbr_model, X, y,
                                       cv=KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_seed),
                                       scoring=scoring_metrics,
                                       n_jobs=-1,
                                       return_train_score=True)
            
            print("\n--- Cross-Validation Performance ---")
            mean_test_r2 = np.mean(cv_results['test_r2'])
            std_test_r2 = np.std(cv_results['test_r2'])
            print(f"Mean Test R2: {mean_test_r2:.4f} (+/- {std_test_r2:.4f})")
            
            print("\nAll tests completed successfully!")
            
    else:
        print("Loaded APT data is empty or None.")
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc() 