#!/usr/bin/env python
"""
GBR_optimized.py

Provides functions for training and evaluating Gradient Boosting Regressor (GBR) models
to analyze the relationship between alpha_s2 and alpha_s1, potentially controlling for other variables.
Includes data loading, model training with hyperparameter tuning, feature importance, and partial dependence plotting.
"""

import pandas as pd
import numpy as np
# Removed imports no longer used directly here (plt, os, ensemble, inspection, model_selection, metrics)

def load_and_merge_datasets(session1_path, session2_path):
    """Loads, merges, renames columns, and handles NaNs for essential columns."""
    print(f"Loading S1: {session1_path}")
    print(f"Loading S2: {session2_path}")
    df_s1 = pd.read_csv(session1_path)
    df_s2 = pd.read_csv(session2_path)

    # Find a common participant identifier
    id_col = next((col for col in ['ID', 'participant', 'participant_ID', 'Experiment'] if col in df_s1.columns and col in df_s2.columns), None)
    if not id_col:
        raise ValueError("Could not find common participant ID column ('ID', 'participant', 'participant_ID', 'Experiment') in both files.")
    print(f"Using participant ID column: {id_col}")

    # Define potential column names and their standard names
    rename_map_s1 = {
        'alpha_mean_boxcox_after_arcsin': 'alpha_s1',
        'alpha_boxcox_after_arcsin': 'alpha_s1', # Handle alternative name
        'a_mean': 'a_mean_s1',
        'a': 'a_mean_s1', # Handle alternative name
        'ndt_mean': 'ndt_mean_s1',
        'ndt': 'ndt_mean_s1' # Handle alternative name
    }
    rename_map_s2 = {
        'alpha_mean_boxcox_after_arcsin': 'alpha_s2',
        'alpha_boxcox_after_arcsin': 'alpha_s2', # Handle alternative name
        'a_mean': 'a_mean_s2',
        'a': 'a_mean_s2', # Handle alternative name
        'ndt_mean': 'ndt_mean_s2',
        'ndt': 'ndt_mean_s2' # Handle alternative name
    }

    # Apply renaming, ignoring errors if columns don't exist
    df_s1.rename(columns=rename_map_s1, inplace=True, errors='ignore')
    df_s2.rename(columns=rename_map_s2, inplace=True, errors='ignore')

    # Merge datasets
    print(f"Merging on '{id_col}'. Initial S1 rows: {len(df_s1)}, S2 rows: {len(df_s2)}")
    df_merged = pd.merge(df_s1, df_s2, on=id_col, how='inner', suffixes=('_delme1', '_delme2')) # Use suffixes for non-renamed conflicts
    print(f"Rows after inner merge: {len(df_merged)}")

    # Define essential columns needed (target + all potential features)
    essential_cols = [
        'alpha_s1', 'alpha_s2', 'a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2'
    ]

    # Check if essential columns exist after rename and merge
    missing_essential = [col for col in essential_cols if col not in df_merged.columns]
    if missing_essential:
        # Add more debug info
        print("Columns available after merge:", df_merged.columns.tolist())
        raise ValueError(f"Essential columns missing from merged data after renaming: {missing_essential}. Check original CSV column names and renaming logic.")

    # Drop rows only if NaNs exist in ESSENTIAL columns needed for *any* analysis
    print(f"Shape before dropping NaNs in essential columns: {df_merged.shape}")
    df_merged.dropna(subset=essential_cols, inplace=True)
    print(f"Shape after dropping NaNs in essential columns: {df_merged.shape}")

    # Clean up potentially duplicated columns from merge if suffixes were used
    cols_to_drop = [col for col in df_merged.columns if col.endswith('_delme1') or col.endswith('_delme2')]
    if cols_to_drop:
        df_merged.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped merge conflict columns: {cols_to_drop}")

    return df_merged


# Removed train_gbr_model function

# Removed plot_partial_dependence_curve function
