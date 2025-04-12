#!/usr/bin/env python
"""
GBR_APT_optimized.py

Provides data loading and preparation functions specifically for the APT 
(Associative Priming Task) dataset, handling its unique parameter names 
for subsequent GBR analysis in a Jupyter Notebook.
"""

import pandas as pd
import numpy as np

def load_and_merge_datasets_apt(session1_path, session2_path):
    """
    Loads, merges, renames columns specifically for APT data, 
    and handles NaNs for essential columns.
    """
    print(f"Loading APT Session 1: {session1_path}")
    print(f"Loading APT Session 2: {session2_path}")
    try:
        df_s1 = pd.read_csv(session1_path)
        df_s2 = pd.read_csv(session2_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        raise
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise

    # Find a common participant identifier
    id_col = next((col for col in ['ID', 'participant', 'participant_ID', 'Experiment'] 
                 if col in df_s1.columns and col in df_s2.columns), None)
    if not id_col:
        raise ValueError("Could not find common participant ID column ('ID', 'participant', 'participant_ID', 'Experiment') in both APT files.")
    print(f"Using participant ID column: {id_col}")

    # --- Define Renaming Maps specifically for APT parameters ---
    # Session 1 Renaming
    rename_map_s1 = {
        'alpha_mean_boxcox_after_arcsin': 'alpha_s1',
        'alpha_mean': 'alpha_s1',  # Fallback if transformed not present
        'v1_mean': 'v1_s1',
        'v2_mean': 'v2_s1',
        'v3_mean': 'v3_s1',
        'v4_mean': 'v4_s1',
        'zr_mean': 'zr_s1',
        'a_mean': 'a_s1',
        'ndt1_mean': 'ndt1_s1',
        'ndt2_mean': 'ndt2_s1',
        'ndt3_mean': 'ndt3_s1',
        'ndt4_mean': 'ndt4_s1',
        'sndt_mean': 'sndt_s1'
    }
    # Session 2 Renaming
    rename_map_s2 = {
        'alpha_mean_boxcox_after_arcsin': 'alpha_s2',
        'alpha_mean': 'alpha_s2',  # Fallback
        'v1_mean': 'v1_s2',
        'v2_mean': 'v2_s2',
        'v3_mean': 'v3_s2',
        'v4_mean': 'v4_s2',
        'zr_mean': 'zr_s2',
        'a_mean': 'a_s2',
        'ndt1_mean': 'ndt1_s2',
        'ndt2_mean': 'ndt2_s2',
        'ndt3_mean': 'ndt3_s2',
        'ndt4_mean': 'ndt4_s2',
        'sndt_mean': 'sndt_s2'
    }

    # Apply renaming, ignoring errors if columns don't exist in source files
    df_s1.rename(columns=rename_map_s1, inplace=True, errors='ignore')
    df_s2.rename(columns=rename_map_s2, inplace=True, errors='ignore')

    # --- Merge datasets ---
    print(f"Merging on '{id_col}'. Initial S1 rows: {len(df_s1)}, S2 rows: {len(df_s2)}")
    df_merged = pd.merge(df_s1, df_s2, on=id_col, how='inner', suffixes=('_delme1', '_delme2'))
    print(f"Rows after inner merge: {len(df_merged)}")
    
    if len(df_merged) == 0:
        print("Warning: Merge resulted in an empty DataFrame. Check participant IDs match.")
        return df_merged  # Return empty DataFrame

    # --- Define essential columns needed for any potential APT GBR analysis ---
    # Includes target (alpha_s2) and all potential features after renaming
    essential_cols = [
        'alpha_s1', 'alpha_s2', 
        'a_s1', 'a_s2', 
        'ndt1_s1', 'ndt1_s2', 'ndt2_s1', 'ndt2_s2', 'ndt3_s1', 'ndt3_s2', 'ndt4_s1', 'ndt4_s2'
    ]

    # Check if essential columns exist after rename and merge
    available_columns = df_merged.columns.tolist()
    # Check only for the target and alpha_s1 initially, specific features checked later
    base_essential = ['alpha_s1', 'alpha_s2']
    missing_base = [col for col in base_essential if col not in available_columns]
    if missing_base:
         print("Columns available after merge:", available_columns)
         raise ValueError(f"Essential base APT columns missing from merged data after renaming: {missing_base}. Check original CSV column names and renaming logic.")
    
    # Identify which of the potential essential columns actually exist in the merged frame
    present_essential_cols = [col for col in essential_cols if col in available_columns]

    # Keep only essential columns that are present plus the ID column before dropping NaNs
    cols_to_keep = [id_col] + present_essential_cols
    df_merged = df_merged[cols_to_keep]
    print(f"Keeping columns: {cols_to_keep}")

    # --- Drop rows only if NaNs exist in the PRESENT ESSENTIAL columns ---
    print(f"Shape before dropping NaNs in essential APT columns: {df_merged.shape}")
    df_merged.dropna(subset=present_essential_cols, inplace=True)
    print(f"Shape after dropping NaNs in essential APT columns: {df_merged.shape}")
    
    if len(df_merged) == 0:
        print("Warning: DataFrame is empty after dropping NaNs from essential columns.")

    # --- Handle issue with alpha_s2 being a DataFrame instead of Series ---
    if 'alpha_s2' in df_merged.columns:
        print(f"DEBUG - alpha_s2 column type: {type(df_merged['alpha_s2'])}")
        
        # If it's a DataFrame, we need to extract just one column
        if isinstance(df_merged['alpha_s2'], pd.DataFrame):
            print(f"WARNING: alpha_s2 is a DataFrame with shape {df_merged['alpha_s2'].shape}. Converting to Series.")
            # Create a new column with the proper Series data type
            column_name = 'alpha_s2_fixed'
            df_merged[column_name] = df_merged['alpha_s2'].iloc[:, 0].values
            
            # Delete the old DataFrame column and rename the new one
            df_merged = df_merged.drop('alpha_s2', axis=1)
            df_merged = df_merged.rename(columns={column_name: 'alpha_s2'})
            
            print(f"After fix: alpha_s2 is now {type(df_merged['alpha_s2'])} with shape {df_merged['alpha_s2'].shape}")
        elif isinstance(df_merged['alpha_s2'], np.ndarray) and len(df_merged['alpha_s2'].shape) > 1:
            # If it's a 2D numpy array, take the first column
            print(f"WARNING: alpha_s2 is a 2D array with shape {df_merged['alpha_s2'].shape}. Converting to 1D array.")
            df_merged['alpha_s2'] = df_merged['alpha_s2'][:, 0]
            print(f"After fix: alpha_s2 is now {type(df_merged['alpha_s2'])} with shape {df_merged['alpha_s2'].shape}")

    print("--- APT Data Loading and Merging Complete ---")
    return df_merged


# Removed train_gbr_model function

# Removed plot_partial_dependence_curve function
