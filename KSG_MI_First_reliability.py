import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings
from npeet import entropy_estimators as ee
import argparse
import os

class KSGMutualInformationAnalyzer:
    """
    A simplified mutual information analyzer implementing the 
    Kraskov-Stögbauer-Grassberger (KSG) estimator, specifically using the KSG-1 variant
    as implemented in the npeet library. Used as an alternative to correlation analysis.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing parameters from different sessions
        """
        self.data = data
        self.n_samples = len(data)
        
    def standardize_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize the input data by subtracting mean and dividing by standard deviation.
        This preprocessing step is recommended by Kraskov et al. for enhanced numerical stability.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
            
        Returns:
        --------
        x_std : np.ndarray
            Standardized first variable
        y_std : np.ndarray
            Standardized second variable
        """
        x_std = (x - np.mean(x)) / np.std(x)
        y_std = (y - np.mean(y)) / np.std(y)
        return x_std, y_std
    
    def _ksg_mi_core(self, x_std: np.ndarray, y_std: np.ndarray, k: int = 3, 
                   n_permutations: int = 5000) -> Tuple[float, float]:
        """
        Core function for KSG MI estimation - expects already standardized inputs.
        
        Parameters:
        -----------
        x_std : np.ndarray
            First variable (standardized)
        y_std : np.ndarray
            Second variable (standardized)
        k : int
            Number of nearest neighbors for KSG estimation
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        mi : float
            Estimated mutual information (in bits)
        p_value : float
            P-value from permutation test
        """
        # Reshape arrays for KSG estimator
        x_std = x_std.reshape(-1, 1)
        y_std = y_std.reshape(-1, 1)
        
        # Calculate MI using KSG-1 estimator (in nats)
        mi_nats = ee.mi(x_std, y_std, k=k)
        
        # Convert from nats to bits
        mi = mi_nats / np.log(2)
        
        # If MI is negative, it's likely due to numerical issues
        if mi < 0:
            warnings.warn(
                f"Negative MI value ({mi:.4f}) detected. This is likely due to "
                "numerical issues in the KSG estimator. Consider:\n"
                "1. Increasing the sample size\n"
                "2. Adjusting the k parameter\n"
                "3. Using a different MI estimation method"
            )
            # For very small negative values, we can assume they're numerical artifacts
            if abs(mi) < 0.01:  # threshold for considering it a numerical artifact
                mi = 0
        
        # Permutation test
        mi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y_std)
            mi_null[i] = ee.mi(x_std, y_perm.reshape(-1, 1), k=k) / np.log(2)
        
        # Calculate p-value
        p_value = (np.sum(mi_null >= mi) + 1) / (n_permutations + 1)
        
        return mi, p_value

    def ksg_mi(self, x: np.ndarray, y: np.ndarray, k: int = 3, 
              n_permutations: int = 5000) -> Tuple[float, float]:
        """
        Estimate mutual information using the KSG-1 estimator.
        The data is automatically standardized before estimation for enhanced numerical stability.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable (e.g., alpha_s1)
        y : np.ndarray
            Second variable (e.g., alpha_s2)
        k : int
            Number of nearest neighbors for KSG estimation
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        mi : float
            Estimated mutual information (in bits)
        p_value : float
            P-value from permutation test
        """
        # Standardize the data
        x_std, y_std = self.standardize_data(x, y)
        
        # Call the core KSG MI estimator function with standardized data
        return self._ksg_mi_core(x_std, y_std, k=k, n_permutations=n_permutations)

    
    
    def analyze_all_parameters(self, session1_params: List[str], session2_params: List[str],
                              k: int = 3, n_permutations: int = 5000) -> Dict:
        """
        Analyze multiple parameter pairs from session 1 and session 2 using KSG MI estimator.
        
        Parameters:
        -----------
        session1_params : List[str]
            Column names for parameters from session 1
        session2_params : List[str]
            Column names for parameters from session 2
        k : int
            Number of nearest neighbors for KSG estimation
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        results : Dict
            Dictionary containing MI and correlation results for each parameter pair
        """
        results = {}
        
        for param1, param2 in zip(session1_params, session2_params):
            if param1 in self.data.columns and param2 in self.data.columns:
                x = self.data[param1].values
                y = self.data[param2].values
                
                # Calculate MI
                mi, p_value = self.ksg_mi(x, y, k=k, n_permutations=n_permutations)
                
                # Calculate correlation for comparison
                r, p_corr = stats.pearsonr(x, y)
                
                results[f"{param1}_{param2}"] = {
                    'mi': mi,
                    'mi_p_value': p_value,
                    'correlation': r,
                    'correlation_p_value': p_corr
                }
                
                print(f"Parameters: {param1} - {param2}")
                print(f"MI: {mi:.4f} bits (p={p_value:.4f})")
                print(f"Correlation: {r:.4f} (p={p_corr:.4f})")
                print("-" * 50)
        
        return results
    
    def analyze_across_k_values(self, param1: str, param2: str, 
                           k_values: List[int] = [3, 5, 7, 10, 15, 20], 
                           n_permutations: int = 5000) -> Dict:
        """
        Analyze a parameter pair across multiple k values.
        
        Parameters:
        -----------
        param1 : str
            Column name for parameter from session 1
        param2 : str
            Column name for parameter from session 2
        k_values : List[int]
            List of k values to analyze
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        results : Dict
            Dictionary containing MI results for each k value
        """
        results = {}
    
        if param1 in self.data.columns and param2 in self.data.columns:
            x = self.data[param1].values
            y = self.data[param2].values
            
            for k in k_values:
                # Calculate MI with current k
                mi, p_value = self.ksg_mi(x, y, k=k, n_permutations=n_permutations)
                
                results[k] = {
                    'mi': mi,
                    'p_value': p_value
                }
                
                print(f"Parameters: {param1} - {param2}, k={k}")
                print(f"MI: {mi:.4f} bits (p={p_value:.4f})")
                print("-" * 40)
        else:
            if param1 not in self.data.columns:
                print(f"Warning: {param1} not found in data")
            if param2 not in self.data.columns:
                print(f"Warning: {param2} not found in data")
        
        return results



