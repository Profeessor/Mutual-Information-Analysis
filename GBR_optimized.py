#!/usr/bin/env python
"""
GBR_optimized.py

This module provides a structured workflow for analyzing reliability of alpha across sessions using
Gradient Boosting Regressor (GBR). It includes data merging, preprocessing, model training with hyperparameter
tuning, feature importance evaluation, and partial dependence plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import train_test_split, GridSearchCV


def load_and_merge_datasets(session1_path, session2_path):
    df_s1 = pd.read_csv(session1_path)
    df_s2 = pd.read_csv(session2_path)

    # Find a common participant identifier
    id_col = next(col for col in ['ID', 'participant', 'participant_ID'] if col in df_s1.columns)

    # Rename relevant columns for consistency and clarity
    df_s1.rename(columns={
        'alpha_mean_boxcox_after_arcsin': 'alpha_s1',
        'alpha_mean_boxcox_after_arcsin': 'alpha_s1',
        'a_mean': 'a_mean_s1',
        'ndt_mean': 'ndt_mean_s1'
    }, inplace=True)

    df_s2.rename(columns={
        'alpha_mean_boxcox_after_arcsin': 'alpha_s2',
        'a_mean': 'a_mean_s2',
        'ndt_mean': 'ndt_mean_s2'
    }, inplace=True)

    # Merge datasets
    df_merged = pd.merge(df_s1, df_s2, on=id_col)
    df_merged = df_merged.dropna()

    # Ensure alpha_s1 column exists correctly
    if 'alpha_mean_boxcox_after_arcsin_s1' in df_merged:
        df_merged.rename(columns={'alpha_mean_boxcox_after_arcsin_s1': 'alpha_s1'}, inplace=True)
    elif 'alpha_boxcox_after_arcsin_s1' in df_merged:
        df_merged.rename(columns={'alpha_boxcox_after_arcsin_s1': 'alpha_s1'}, inplace=True)

    return df_merged


def train_gbr_model(X, y, test_size=0.3, random_state=42, hyper_tune=False):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.inspection import permutation_importance

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Hyperparameter optimization through GridSearchCV (optional but recommended)
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_leaf': [1, 3, 5]
    }

    from sklearn.model_selection import GridSearchCV
    gbr_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    gbr_grid.fit(X_train, y_train)

    best_model = gbr_grid.best_estimator_

    train_r2 = best_model.score(X_train, y_train)
    test_r2 = best_model.score(X_test, y_test)

    # Feature importance (Permutation-based)
    perm_imp = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=random_state)

    return best_model, X_train, X_test, y_train, y_test, train_r2, test_r2, perm_imp


def plot_partial_dependence_curve(model, X, feature, feature_names):
    from sklearn.inspection import PartialDependenceDisplay

    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        model, X, features=[feature], feature_names=feature_names, ax=ax
    )
    ax.set_title(f"Partial Dependence of αₛ₂ on {feature_names[feature]}")
    ax.set_xlabel(feature_names[feature])
    ax.set_ylabel("Predicted αₛ₂")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_dataset(session1_path, session2_path, test_size=0.3, random_state=42):
    df_merged = load_and_merge_datasets(session1_path, session2_path)

    feature_cols = ["alpha_s1", "a_mean_s1", "a_mean_s2", "ndt_mean_s1", "ndt_mean_s2"]
    X = df_merged[feature_cols]
    y = df_merged["alpha_s2"]

    model, X_train, X_test, y_train, y_test, train_r2, test_r2, perm_imp = train_gbr_model(
        X, y, test_size=0.3, random_state=42
    )

    print("Model Performance")
    print(f"R² Training: {train_r2:.3f}")
    print(f"R² Testing: {test_r2:.3f}")

    # Show Permutation Importance
    print("\nPermutation Importance (sorted):")
    sorted_idx = perm_imp.importances_mean.argsort()[::-1]
    for i in sorted_idx:
        print(f"{feature_cols[i]}: Mean={perm_imp.importances_mean[i]:.4f}, std={perm_imp.importances_std[i]:.4f}")

    # Plot partial dependence for alpha_s1
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_partial_dependence_curve(model, X_test, feature_cols, feature_index=0, ax=ax)


# Plot function standalone
def plot_partial_dependence_curve(model, X, feature_cols, feature_index=0, ax=None):
    from sklearn.inspection import PartialDependenceDisplay

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    disp = PartialDependenceDisplay.from_estimator(
        model, X, [feature_index], ax=ax, feature_names=feature_cols
    )

    ax.set_title(f"Partial Dependence: Predicted αₛ₂ vs. {feature_cols[feature_index]}")
    ax.set_xlabel(feature_cols[feature_index])
    ax.set_ylabel("Predicted αₛ₂")
    plt.tight_layout()
    plt.show()


# Main user-friendly function for Jupyter Notebook
def analyze(session1_path, session2_path, test_size=0.3, random_state=42):
    df_merged = load_and_merge_datasets(session1_path, session2_path)

    X = df_merged[["alpha_s1", "a_mean_s1", "a_mean_s2", "ndt_mean_s1", "ndt_mean_s2"]]
    y = df_merged["alpha_s2"]

    model, X_train, X_test, y_train, y_test, train_r2, test_r2, perm_imp = train_gbr_model(X, y)

    print(f"Training R²: {train_r2:.3f}")
    print(f"Testing R²: {test_r2:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_partial_dependence_curve(model, X_test, X.columns.tolist(), feature_index=0, ax=ax)

    return model
