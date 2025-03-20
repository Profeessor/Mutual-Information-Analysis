from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import BayesianRidge
from pygam import LinearGAM, s, f
from pyearth import Earth

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

        # Rename 'a_mean' and 'ndt_mean' columns to include session suffixes
    if 'a_mean' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'a_mean': 'a_mean_s1'})
    if 'a_mean' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'a_mean': 'a_mean_s2'})
    if 'ndt_mean' in df_s1.columns:
        df_s1 = df_s1.rename(columns={'ndt_mean': 'ndt_mean_s1'})
    if 'ndt_mean' in df_s2.columns:
        df_s2 = df_s2.rename(columns={'ndt_mean': 'ndt_mean_s2'})
    
    
    
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




def train_gradient_boosting_improved(X, y, test_size=0.2, random_state=42, n_cv_folds=5):
    """
    Improved version of the gradient boosting training function that works better with small datasets.
    Uses cross-validation, regularization, and simpler models when appropriate.
    
    Parameters:
      X (pd.DataFrame): Predictor features.
      y (pd.Series): Target variable.
      test_size (float): Proportion of data to be used for testing.
      random_state (int): Seed for random splitting.
      n_cv_folds (int): Number of cross-validation folds.
      
    Returns:
      best_model: Best trained model.
      X_train, X_test, y_train, y_test: Split data.
      cv_r2, test_r2: Cross-validation and test R² scores.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Check if dataset is small (fewer than 100 samples)
    small_dataset = len(X_train) < 100
    
    # Define models to try
    models = {
        'gbr': GradientBoostingRegressor(
            random_state=random_state,
            # More conservative parameters for small datasets
            n_estimators=100 if small_dataset else 300,
            max_depth=2 if small_dataset else 3,
            learning_rate=0.01 if small_dataset else 0.05,
            subsample=0.8,  # Use subsampling to reduce overfitting
            min_samples_split=5 if small_dataset else 2,
            min_samples_leaf=2 if small_dataset else 1
        ),
        'rf': RandomForestRegressor(
            random_state=random_state,
            n_estimators=100,
            max_depth=5 if small_dataset else 10,
            min_samples_split=5 if small_dataset else 2
        ),
        'ridge': Ridge(alpha=1.0, random_state=random_state),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
        'bayesian_ridge': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
        'svr_rbf': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
        'svr_linear': SVR(kernel='linear', C=0.1),
        'gpr': GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0), alpha=0.1)
    }
    
    # Create cross-validation object
    cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    
    # Evaluate each model with cross-validation
    best_model_name = None
    best_cv_score = -np.inf
    cv_results = {}
    
    for name, model in models.items():
        # Create a pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
        mean_cv_score = np.mean(cv_scores)
        cv_results[name] = {
            'mean_cv_score': mean_cv_score,
            'std_cv_score': np.std(cv_scores)
        }
        
        # Update best model if this one is better
        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_model_name = name
    
    # Print cross-validation results
    print("Cross-validation R² scores:")
    for name, result in cv_results.items():
        print(f"  {name}: {result['mean_cv_score']:.3f} ± {result['std_cv_score']:.3f}")
    
    # Train the best model on the full training set
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', models[best_model_name])
    ])
    best_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    test_r2 = best_pipeline.score(X_test, y_test)
    
    print(f"\nBest model: {best_model_name}")
    print(f"Cross-validation R²: {best_cv_score:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    return best_pipeline, X_train, X_test, y_train, y_test, best_cv_score, test_r2

def perform_feature_selection(X, y, random_state=42):
    """
    Performs feature selection to identify the most important features.
    
    Parameters:
      X (pd.DataFrame): Predictor features.
      y (pd.Series): Target variable.
      random_state (int): Random seed.
      
    Returns:
      selected_features (list): List of selected feature names.
      X_selected (pd.DataFrame): DataFrame with only selected features.
    """
    # Initialize a GradientBoostingRegressor with conservative parameters
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.01,
        random_state=random_state
    )
    
    # Fit the model
    gbr.fit(X, y)
    
    # Use SelectFromModel to select features
    sfm = SelectFromModel(gbr, threshold='mean')
    sfm.fit(X, y)
    
    # Get selected feature indices
    selected_indices = sfm.get_support(indices=True)
    
    # Get selected feature names
    selected_features = [X.columns[i] for i in selected_indices]
    
    # Create DataFrame with only selected features
    X_selected = X[selected_features]
    
    print(f"Selected {len(selected_features)} out of {X.shape[1]} features:")
    for feature in selected_features:
        print(f"  {feature}")
    
    return selected_features, X_selected

def tune_hyperparameters(X, y, model_type='gbr', random_state=42, n_cv_folds=5):
    """
    Performs hyperparameter tuning for the specified model type.
    
    Parameters:
      X (pd.DataFrame): Predictor features.
      y (pd.Series): Target variable.
      model_type (str): Type of model to tune ('gbr', 'rf', 'ridge', or 'elasticnet').
      random_state (int): Random seed.
      n_cv_folds (int): Number of cross-validation folds.
      
    Returns:
      best_params (dict): Best hyperparameters.
      best_score (float): Best cross-validation score.
    """
    # Define parameter grids for different model types
    param_grids = {
        'gbr': {
            'n_estimators': [50, 100, 200],
            'max_depth': [1, 2, 3],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_split': [2, 5, 10]
        },
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        },
        'ridge': {
            'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
        },
        'elasticnet': {
            'alpha': [0.01, 0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }
    
    # Define model based on model_type
    models = {
        'gbr': GradientBoostingRegressor(random_state=random_state),
        'rf': RandomForestRegressor(random_state=random_state),
        'ridge': Ridge(random_state=random_state),
        'elasticnet': ElasticNet(random_state=random_state)
    }
    
    # Create cross-validation object
    cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', models[model_type])
    ])
    
    # Set up parameter grid for pipeline
    pipeline_param_grid = {f'model__{param}': values for param, values in param_grids[model_type].items()}
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=pipeline_param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Get best parameters and score
    best_params = {param.replace('model__', ''): value for param, value in grid_search.best_params_.items()}
    best_score = grid_search.best_score_
    
    print(f"Best parameters for {model_type}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best cross-validation R²: {best_score:.3f}")
    
    return best_params, best_score

def analyze_dataset_improved(session1_path, session2_path, test_size=0.2, random_state=42):
    """
    Improved main analysis function with better handling of small datasets.
    
    Steps:
      1. Load and merge Session 1 and Session 2 data.
      2. Define predictor features and the target variable.
      3. Perform feature selection.
      4. Train multiple models and select the best one.
      5. Print cross-validation and test R² scores.
      6. Display feature importances.
      7. Plot the partial dependence curve for 'alpha_s1'.
    
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
        "ndt_mean_s2"   # Non-decision time from Session 2
    ]
    target_col = "alpha_s2"  # Transformed alpha from Session 2
    
    X = df_merged[feature_cols].copy()
    y = df_merged[target_col].copy()
    
    print(f"Dataset size: {len(X)} samples")
    
    # Perform feature selection if dataset is small
    if len(X) < 100:
        print("\nPerforming feature selection...")
        selected_features, X_selected = perform_feature_selection(X, y, random_state)
        X = X_selected
        feature_cols = selected_features
    
    # Train the improved model
    best_model, X_train, X_test, y_train, y_test, cv_r2, test_r2 = train_gradient_boosting_improved(
        X, y, test_size, random_state
    )
    
    # If the model is a pipeline, extract the actual model
    if hasattr(best_model, 'named_steps') and 'model' in best_model.named_steps:
        model = best_model.named_steps['model']
        
        # Display feature importances if the model supports it
        if hasattr(model, 'feature_importances_'):
            print("\nFeature Importances:")
            importances = model.feature_importances_
            for col, imp in zip(feature_cols, importances):
                print(f"  {col}: {imp:.3f}")
            
            # Plot the partial dependence for 'alpha_s1' if it's in the selected features
            if 'alpha_s1' in feature_cols:
                feature_index = feature_cols.index('alpha_s1')
                plot_partial_dependence_curve(model, X_test, feature_cols, feature_index)
    
    return best_model, X, y, cv_r2, test_r2

def plot_partial_dependence_curve(gbr, X_test, feature_cols, feature_index=0):
    """
    Plots the partial dependence curve for the specified feature (by index).
    Default: feature_index=0 corresponds to 'alpha_s1'.
    
    Parameters:
      gbr: Trained model.
      X_test (pd.DataFrame): Test data.
      feature_cols (list): List of feature column names.
      feature_index (int): Index of the feature to plot.
    """
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    
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

# Fit GAM with automatic smoothing
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(X, y)

# MARS model
mars_model = Earth(max_degree=2)  # Limit interactions to avoid overfitting
mars_model.fit(X_train, y_train)