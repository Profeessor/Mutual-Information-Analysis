import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union
import warnings
from npeet import entropy_estimators as ee
from scipy.spatial import cKDTree
import itertools
import argparse
import os

class KSGMutualInformationAnalyzer:
    """
    A comprehensive mutual information analyzer implementing the 
    Kraskov-Stögbauer-Grassberger (KSG) estimator, specifically using the KSG-1 variant
    as implemented in the npeet library. KSG-1 is the more commonly used variant
    that provides robust estimates for continuous variables.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing alpha parameters and other variables
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
    
    def validate_on_gaussian(self, n_samples: int = 1000, correlation: float = 0.5) -> Dict:
        """
        Validate the KSG estimator on correlated Gaussian distributions.
        This provides a theoretical check of the implementation.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        correlation : float
            Correlation coefficient between the variables
            
        Returns:
        --------
        results : Dict
            Dictionary containing theoretical and estimated MI values
        """
        # Generate correlated Gaussian data
        mean = np.array([0, 0])
        cov = np.array([[1, correlation], [correlation, 1]])
        data = np.random.multivariate_normal(mean, cov, n_samples)
        x, y = data[:, 0], data[:, 1]
        
        # Theoretical MI for bivariate Gaussian
        theoretical_mi = -0.5 * np.log(1 - correlation**2) / np.log(2)  # in bits
        
        # Standardize the data
        x_std, y_std = self.standardize_data(x, y)
        
        # Estimate MI using KSG
        estimated_mi, p_value = self.ksg_mi(x_std, y_std, k=3)
        
        # Calculate relative error
        relative_error = abs(estimated_mi - theoretical_mi) / theoretical_mi
        
        return {
            'theoretical_mi': theoretical_mi,
            'estimated_mi': estimated_mi,
            'relative_error': relative_error,
            'p_value': p_value,
            'correlation': correlation,
            'n_samples': n_samples
        }
    
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
        # We should check if it's significantly negative
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
            else:
                # For larger negative values, we should investigate further
                print(f"Warning: Large negative MI value ({mi:.4f}) detected.")
                print("This might indicate issues with the data or estimation method.")
        
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
    
    def ksg_sensitivity_analysis(self, x: np.ndarray, y: np.ndarray, 
                               k_values: List[int] = [3, 5, 7, 10],
                               n_subsamples: int = 100) -> Dict:
        """
        Perform sensitivity analysis for KSG estimator.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
        k_values : List[int]
            Different k values to test
        n_subsamples : int
            Number of subsamples for each k value
            
        Returns:
        --------
        sensitivity_results : Dict
            Dictionary containing sensitivity analysis results
        """
        results = {}
        
        # Analyze different k values
        for k in k_values:
            mi_values = []
            for _ in range(n_subsamples):
                # Random subsample
                idx = np.random.choice(len(x), size=int(0.8 * len(x)), replace=False)
                x_sub = x[idx]
                y_sub = y[idx]
                
                # Calculate MI
                mi = ee.mi(x_sub.reshape(-1, 1), y_sub.reshape(-1, 1), k=k)
                mi_values.append(mi)
            
            results[f'k_{k}'] = {
                'mean': np.mean(mi_values),
                'std': np.std(mi_values),
                'min': np.min(mi_values),
                'max': np.max(mi_values)
            }
        
        return results
    
    def analyze_alpha_reliability(self, 
                                feature: str = 'alpha_s1',
                                target: str = 'alpha_s2',
                                control_vars: List[str] = None,
                                k: int = 3,
                                n_permutations: int = 5000) -> Dict:
        """
        Comprehensive analysis of alpha parameter reliability using KSG estimator.
        Uses a stepwise approach for handling multiple control variables.
        
        Parameters:
        -----------
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        control_vars : List[str]
            Variables to control for in the analysis
        k : int
            Number of nearest neighbors for KSG estimation
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        results : Dict
            Dictionary containing results from KSG MI estimation
        """
        results = {}
        
        # Extract variables
        x_orig = self.data[feature].values
        y_orig = self.data[target].values
        
        # Standardize x and y upfront
        x_std, y_std = self.standardize_data(x_orig, y_orig)
        
        # Calculate raw MI using KSG estimator on standardized data
        raw_mi, raw_p = self._ksg_mi_core(x_std, y_std, k=k, n_permutations=n_permutations)
        
        # Store raw MI results in the dictionary
        results['ksg_mi'] = [raw_mi, raw_p]
        
        # Perform sensitivity analysis (this should also be updated to use standardized inputs)
        results['sensitivity'] = self.ksg_sensitivity_analysis(x_std, y_std)
        
        # Calculate conditional MI if control variables are provided
        if control_vars:
            results['conditional_mi'] = {}
            results['stepwise_retention'] = {}
            
            # Create standardized dictionary of all control variables
            control_std_dict = {}
            for var in control_vars:
                z_orig = self.data[var].values
                z_std, _ = self.standardize_data(z_orig, z_orig)  # Standardize each Z variable
                control_std_dict[var] = z_std
            
            # Separate threshold and non-decision time variables
            threshold_vars = [var for var in control_vars if 'a_' in var]
            ndt_vars = [var for var in control_vars if 'ndt_' in var]
            
            # Create dictionaries and arrays for different control variable subsets
            threshold_dict = {var: control_std_dict[var] for var in threshold_vars}
            ndt_dict = {var: control_std_dict[var] for var in ndt_vars}
            
            # Stepwise analysis: add one control variable at a time
            current_controls = []
            for var in control_vars:
                current_controls.append(var)
                z_values = np.column_stack([control_std_dict[v] for v in current_controls])
                
                # Calculate conditional MI using KSG
                # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
                x_reshaped = x_std.reshape(-1, 1)
                yz = np.column_stack([y_std, z_values])
                
                # Calculate MI with current set of controls
                mi_xyz = ee.mi(x_reshaped, yz, k=k)
                mi_xz = ee.mi(x_reshaped, z_values, k=k)
                
                # Conditional MI = I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
                cmi = mi_xyz - mi_xz
                
                # Permutation test for conditional MI
                cmi_null = np.zeros(n_permutations)
                for i in range(n_permutations):
                    y_perm = np.random.permutation(y_std)
                    yz_perm = np.column_stack([y_perm, z_values])
                    mi_xyz_perm = ee.mi(x_reshaped, yz_perm, k=k)
                    cmi_null[i] = mi_xyz_perm - mi_xz
                
                p_value = (np.sum(cmi_null >= cmi) + 1) / (n_permutations + 1)
                results['conditional_mi'][var] = (cmi, p_value)
                
                # Calculate retention percentage for this step
                raw_mi = results['ksg_mi'][0]
                retention = (cmi / raw_mi) * 100
                results['stepwise_retention'][var] = retention
            
            # Calculate retention percentage for individual variables
            raw_mi = results['ksg_mi'][0]
            cmi_values = [v[0] for v in results['conditional_mi'].values()]
            results['retention'] = {
                var: (cmi / raw_mi) * 100 
                for var, cmi in zip(control_vars, cmi_values)
            }
            
            # Calculate final conditional MI with all variables
            z_all_values = np.column_stack([control_std_dict[var] for var in control_vars])
            yz_all = np.column_stack([y_std, z_all_values])
            
            mi_xyz_all = ee.mi(x_std.reshape(-1, 1), yz_all, k=k)
            mi_xz_all = ee.mi(x_std.reshape(-1, 1), z_all_values, k=k)
            
            cmi_all = mi_xyz_all - mi_xz_all
            
            # Permutation test for joint conditional MI
            cmi_all_null = np.zeros(n_permutations)
            for i in range(n_permutations):
                y_perm = np.random.permutation(y_std)
                yz_all_perm = np.column_stack([y_perm, z_all_values])
                mi_xyz_all_perm = ee.mi(x_std.reshape(-1, 1), yz_all_perm, k=k)
                cmi_all_null[i] = mi_xyz_all_perm - mi_xz_all
            
            p_value_all = (np.sum(cmi_all_null >= cmi_all) + 1) / (n_permutations + 1)
            results['cond_mi_all'] = cmi_all
            results['p_value_all'] = p_value_all
            
            # Calculate overall retention percentage
            results['retention_all'] = (cmi_all / raw_mi) * 100 if raw_mi > 0 else 0
            
            # --------- NEW: Calculate threshold-only conditional MI -----------
            if threshold_vars:
                # Combine standardized threshold variables
                z_thresh_values = np.column_stack([threshold_dict[var] for var in threshold_vars])
                yz_thresh = np.column_stack([y_std, z_thresh_values])
                
                # Calculate MI components
                mi_xyz_thresh = ee.mi(x_std.reshape(-1, 1), yz_thresh, k=k)
                mi_xz_thresh = ee.mi(x_std.reshape(-1, 1), z_thresh_values, k=k)
                
                # Calculate threshold-only CMI
                cmi_thresh = mi_xyz_thresh - mi_xz_thresh
                
                # Permutation test
                cmi_thresh_null = np.zeros(n_permutations)
                for i in range(n_permutations):
                    y_perm = np.random.permutation(y_std)
                    yz_thresh_perm = np.column_stack([y_perm, z_thresh_values])
                    mi_xyz_thresh_perm = ee.mi(x_std.reshape(-1, 1), yz_thresh_perm, k=k)
                    cmi_thresh_null[i] = mi_xyz_thresh_perm - mi_xz_thresh
                
                p_value_thresh = (np.sum(cmi_thresh_null >= cmi_thresh) + 1) / (n_permutations + 1)
                
                # Store results
                results['cond_mi_thresh'] = cmi_thresh
                results['p_value_thresh'] = p_value_thresh
                results['retention_thresh'] = (cmi_thresh / raw_mi) * 100 if raw_mi > 0 else 0
            
            # --------- NEW: Calculate NDT-only conditional MI -----------
            if ndt_vars:
                # Combine standardized NDT variables
                z_ndt_values = np.column_stack([ndt_dict[var] for var in ndt_vars])
                yz_ndt = np.column_stack([y_std, z_ndt_values])
                
                # Calculate MI components
                mi_xyz_ndt = ee.mi(x_std.reshape(-1, 1), yz_ndt, k=k)
                mi_xz_ndt = ee.mi(x_std.reshape(-1, 1), z_ndt_values, k=k)
                
                # Calculate NDT-only CMI
                cmi_ndt = mi_xyz_ndt - mi_xz_ndt
                
                # Permutation test
                cmi_ndt_null = np.zeros(n_permutations)
                for i in range(n_permutations):
                    y_perm = np.random.permutation(y_std)
                    yz_ndt_perm = np.column_stack([y_perm, z_ndt_values])
                    mi_xyz_ndt_perm = ee.mi(x_std.reshape(-1, 1), yz_ndt_perm, k=k)
                    cmi_ndt_null[i] = mi_xyz_ndt_perm - mi_xz_ndt
                
                p_value_ndt = (np.sum(cmi_ndt_null >= cmi_ndt) + 1) / (n_permutations + 1)
                
                # Store results
                results['cond_mi_ndt'] = cmi_ndt
                results['p_value_ndt'] = p_value_ndt
                results['retention_ndt'] = (cmi_ndt / raw_mi) * 100 if raw_mi > 0 else 0
            
            # Add dimensionality information
            results['dimensionality'] = {
                'n_samples': self.n_samples,
                'n_controls': len(control_vars),
                'total_dimensions': 2 + len(control_vars)  # X, Y, and all Z variables
            }
            
            # Add warning if dimensionality might be too high
            if self.n_samples < (2 + len(control_vars)) * 10:
                warnings.warn(
                    f"Sample size ({self.n_samples}) might be too small for the "
                    f"current dimensionality ({2 + len(control_vars)}). "
                    "Consider reducing the number of control variables or increasing "
                    "the sample size."
                )
        
        return results
    
    def get_dataset_name(self, session1_path: str) -> str:
        """
        Get the dataset name based on the session1_path.
        
        Parameters:
        -----------
        session1_path : str
            Path to the first session data file
            
        Returns:
        --------
        str
            Dataset name for plotting
        """
        mapping = {
            'Model2 LDT RMT/EZ_recognition_memory_part1_transformed.csv': 'RMT Model 2',
            'Model1 LDT RMT/rmt_session_1_data_transformed.csv': 'RMT Model 1',
            'Model1 LDT RMT/ldt_session_1_data_transformed.csv': 'LDT Model 1',
            'Model2 LDT RMT/EZ_lexical_decision_part1_transformed.csv': 'LDT Model 2',
            'Model1 ELP/Transformed_recoverd_param_ELP_S1_second_100_2000_200_epoch_rt_acc_logAlphaSample.csv': 'ELP Model 1',
            'Model2 ELP/Transformed_recoverd_param_ELP_S1_second_100_2000_200_epoch_rt_acc_logAlphaSample.csv': 'ELP Model 2'
        }
        return mapping.get(session1_path, 'Unknown Dataset')

    def print_summary_table(self, all_k_results: Dict, session1_path: str = None, 
                           output_csv_path: str = None) -> None:
        """
        Print a comprehensive summary table of all results and optionally save to CSV.
        
        Parameters:
        -----------
        all_k_results : Dict
            Results from analyze_alpha_reliability for all k values
        session1_path : str
            Path to the first session data file (for dataset naming)
        output_csv_path : str, optional
            Path to save the results table as a CSV file
        """
        dataset_name = self.get_dataset_name(session1_path) if session1_path else ''
        
        # Print header
        print("\n" + "="*120)
        print(f"Summary Results for {dataset_name}")
        print(f"Number of Participants: {self.n_samples}")
        
        # Get sensitivity analysis info from first k value
        first_k = sorted(all_k_results.keys())[0]
        if 'sensitivity' in all_k_results[first_k]:
            n_subsamples = len(all_k_results[first_k]['sensitivity'])
            subsample_size = int(0.8 * self.n_samples)  # From ksg_sensitivity_analysis method
            print(f"Sensitivity Analysis: {n_subsamples} subsamples (each using {subsample_size} participants)")
        print("="*120)
        
        # Create headers
        headers = [
            "k Value",
            "Raw MI (bits)",
            "Raw MI p-value",
            "Cond MI All (bits)",
            "Cond MI All p-value",
            "Retention All (%)",
            "Cond MI Thresh (bits)",
            "Cond MI Thresh p-value",
            "Retention Thresh (%)",
            "Cond MI NDT (bits)",
            "Cond MI NDT p-value",
            "Retention NDT (%)",
            "Sensitivity Mean±SD"
        ]
        
        # Build a DataFrame to store results
        results_data = []
        
        # Fill the DataFrame with results for each k value
        for k in sorted(all_k_results.keys()):
            results = all_k_results[k]
            
            # Get all required values with appropriate defaults
            row_data = {
                'k Value': k,
                'Raw MI (bits)': results['ksg_mi'][0],
                'Raw MI p-value': results['ksg_mi'][1],
                'Cond MI All (bits)': results.get('cond_mi_all', 0),
                'Cond MI All p-value': results.get('p_value_all', 1),
                'Retention All (%)': results.get('retention_all', 0),
                'Cond MI Thresh (bits)': results.get('cond_mi_thresh', 0),
                'Cond MI Thresh p-value': results.get('p_value_thresh', 1),
                'Retention Thresh (%)': results.get('retention_thresh', 0),
                'Cond MI NDT (bits)': results.get('cond_mi_ndt', 0),
                'Cond MI NDT p-value': results.get('p_value_ndt', 1),
                'Retention NDT (%)': results.get('retention_ndt', 0)
            }
            
            # Get sensitivity analysis results
            sensitivity_str = ""
            sensitivity_mean = 0
            sensitivity_std = 0
            if 'sensitivity' in results:
                sens_key = f'k_{k}'
                if sens_key in results['sensitivity']:
                    sensitivity_mean = results['sensitivity'][sens_key]['mean']
                    sensitivity_std = results['sensitivity'][sens_key]['std']
                    sensitivity_str = f"{sensitivity_mean:.3f}±{sensitivity_std:.3f}"
                
            row_data['Sensitivity Mean'] = sensitivity_mean
            row_data['Sensitivity SD'] = sensitivity_std
            row_data['Sensitivity Mean±SD'] = sensitivity_str
            
            results_data.append(row_data)
        
        # Create DataFrame
        df_results = pd.DataFrame(results_data)
        
        # Print header row
        print(f"{headers[0]:<10} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} "
              f"{headers[4]:<15} {headers[5]:<15} {headers[6]:<15} {headers[7]:<15} "
              f"{headers[8]:<15} {headers[9]:<15} {headers[10]:<15} {headers[11]:<15} "
              f"{headers[12]}")
        print("-"*180)
        
        # Print results for each k value
        for _, row in df_results.iterrows():
            # Add asterisks for significance
            raw_sig = "*" if row['Raw MI p-value'] < 0.05 else " "
            all_sig = "*" if row['Cond MI All p-value'] < 0.05 else " "
            thresh_sig = "*" if row['Cond MI Thresh p-value'] < 0.05 else " "
            ndt_sig = "*" if row['Cond MI NDT p-value'] < 0.05 else " "
            
            print(f"{row['k Value']:<10} "
                  f"{row['Raw MI (bits)']:>6.3f}{raw_sig:<8} {row['Raw MI p-value']:<15.4f} "
                  f"{row['Cond MI All (bits)']:>6.3f}{all_sig:<8} {row['Cond MI All p-value']:<15.4f} {row['Retention All (%)']:<15.1f} "
                  f"{row['Cond MI Thresh (bits)']:>6.3f}{thresh_sig:<8} {row['Cond MI Thresh p-value']:<15.4f} {row['Retention Thresh (%)']:<15.1f} "
                  f"{row['Cond MI NDT (bits)']:>6.3f}{ndt_sig:<8} {row['Cond MI NDT p-value']:<15.4f} {row['Retention NDT (%)']:<15.1f} "
                  f"{row['Sensitivity Mean±SD']}")
        
        # Print footer with legend
        print("-"*180)
        print("* indicates significance at p < 0.05")
        print("Sensitivity Mean±SD shows the mean and standard deviation of MI estimates across subsamples")
        print("="*180)
        
        # Save to CSV if requested
        if output_csv_path:
            # Add metadata to the CSV
            metadata = pd.DataFrame({
                'Dataset': [dataset_name],
                'Number of Participants': [self.n_samples],
                'Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            
            # Write both metadata and results to CSV
            with open(output_csv_path, 'w') as f:
                f.write("# KSG Mutual Information Analysis Results\n")
                metadata.to_csv(f, index=False)
                f.write("\n# Detailed Results\n")
                df_results.to_csv(f, index=False)
            
            print(f"\nResults saved to: {output_csv_path}")

    def plot_results(self, results: Dict, feature: str = 'alpha_s1', 
                    target: str = 'alpha_s2', all_k_results: Dict = None,
                    session1_path: str = None, dataset_name: str = None) -> None:
        """
        Create visualizations of the KSG MI analysis results, focusing on how values change with k.
        Saves plots to a new folder named after the dataset.
        """
        # Determine dataset name
        if dataset_name is None:
            dataset_name = self.get_dataset_name(session1_path) if session1_path else 'Unknown Dataset'
        
        # Create directory for saving plots
        plot_dir = f"plots_{dataset_name.replace(' ', '_')}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Only create plots if we have results for multiple k values
        if not all_k_results or len(all_k_results) < 2:
            print("Warning: Need results for at least 2 k values to create plots.")
            return
        
        # Extract data for plots
        k_values = sorted(all_k_results.keys())
        
        # Prepare data dictionaries
        plot_data = {
            'k': k_values,
            'raw_mi': [],
            'raw_p': [],
            'cmi_thresh': [],
            'p_thresh': [],
            'retention_thresh': [],
            'cmi_ndt': [],
            'p_ndt': [],
            'retention_ndt': []
        }
        
        # Fill data for plots
        for k in k_values:
            res = all_k_results[k]
            
            # Raw MI data
            plot_data['raw_mi'].append(res['ksg_mi'][0])
            plot_data['raw_p'].append(res['ksg_mi'][1])
            
            # Threshold-only data
            plot_data['cmi_thresh'].append(res.get('cond_mi_thresh', 0))
            plot_data['p_thresh'].append(res.get('p_value_thresh', 1))
            plot_data['retention_thresh'].append(res.get('retention_thresh', 0))
            
            # NDT-only data
            plot_data['cmi_ndt'].append(res.get('cond_mi_ndt', 0))
            plot_data['p_ndt'].append(res.get('p_value_ndt', 1))
            plot_data['retention_ndt'].append(res.get('retention_ndt', 0))
        
        # Define colors and markers
        colors = {
            'raw': '#1f77b4',  # blue
            'thresh': '#ff7f0e',  # orange
            'ndt': '#d62728'   # red
        }
        
        markers = {
            'raw': 'o',
            'thresh': '^',
            'ndt': 'D'
        }
        
        # Plot 1: MI vs k
        plt.figure(figsize=(12, 8))
        
        plt.plot(plot_data['k'], plot_data['raw_mi'], 
                marker=markers['raw'], color=colors['raw'], 
                linewidth=2, markersize=10, label='Raw MI')
        
        plt.plot(plot_data['k'], plot_data['cmi_thresh'], 
                marker=markers['thresh'], color=colors['thresh'], 
                linewidth=2, markersize=10, label='CMI (Threshold Only)')
        
        plt.plot(plot_data['k'], plot_data['cmi_ndt'], 
                marker=markers['ndt'], color=colors['ndt'], 
                linewidth=2, markersize=10, label='CMI (NDT Only)')
        
        plt.title(f'Mutual Information vs k\n{dataset_name} (n={self.n_samples})', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('Mutual Information (bits)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # Add data-driven k if it's in the results
        k_dd = int(np.sqrt(self.n_samples))
        if k_dd in plot_data['k']:
            plt.axvline(x=k_dd, color='gray', linestyle='--', alpha=0.7)
            plt.text(k_dd, plt.ylim()[1]*0.95, f'k_dd = {k_dd}', 
                     ha='center', va='top', 
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'MI_vs_k.png'))
        plt.show()
        plt.close()
        
        # Plot 2: Retention vs k
        plt.figure(figsize=(12, 8))
        
        plt.plot(plot_data['k'], plot_data['retention_thresh'], 
                marker=markers['thresh'], color=colors['thresh'], 
                linewidth=2, markersize=10, label='Retention (Threshold Only)')
        
        plt.plot(plot_data['k'], plot_data['retention_ndt'], 
                marker=markers['ndt'], color=colors['ndt'], 
                linewidth=2, markersize=10, label='Retention (NDT Only)')
        
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='100% Retention')
        
        plt.title(f'Information Retention vs k\n{dataset_name} (n={self.n_samples})', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('Information Retention (%)', fontsize=14)
        plt.ylim(0, max(
            max(plot_data['retention_thresh'] or [0]), 
            max(plot_data['retention_ndt'] or [0]), 
            100) * 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # Add data-driven k if it's in the results
        if k_dd in plot_data['k']:
            plt.axvline(x=k_dd, color='gray', linestyle='--', alpha=0.7)
            plt.text(k_dd, plt.ylim()[1]*0.95, f'k_dd = {k_dd}', 
                     ha='center', va='top', 
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Retention_vs_k.png'))
        plt.show()
        plt.close()
        
        # Plot 3: P-value vs k
        plt.figure(figsize=(12, 8))
        
        plt.plot(plot_data['k'], plot_data['raw_p'], 
                marker=markers['raw'], color=colors['raw'], 
                linewidth=2, markersize=10, label='Raw MI p-value')
        
        plt.plot(plot_data['k'], plot_data['p_thresh'], 
                marker=markers['thresh'], color=colors['thresh'], 
                linewidth=2, markersize=10, label='CMI (Threshold Only) p-value')
        
        plt.plot(plot_data['k'], plot_data['p_ndt'], 
                marker=markers['ndt'], color=colors['ndt'], 
                linewidth=2, markersize=10, label='CMI (NDT Only) p-value')
        
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        
        plt.title(f'P-values vs k\n{dataset_name} (n={self.n_samples})', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('P-value', fontsize=14)
        plt.ylim(0, min(max(
            max(plot_data['raw_p'] or [0]), 
            max(plot_data['p_thresh'] or [0]), 
            max(plot_data['p_ndt'] or [0]), 
            0.1) * 1.1, 1.0))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # Add data-driven k if it's in the results
        if k_dd in plot_data['k']:
            plt.axvline(x=k_dd, color='gray', linestyle='--', alpha=0.7)
            plt.text(k_dd, plt.ylim()[1]*0.95, f'k_dd = {k_dd}', 
                     ha='center', va='top', 
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Pvalue_vs_k.png'))
        plt.show()
        plt.close()

    def analyze_individual_parameters(self, feature: str = 'alpha_s1',
                                   target: str = 'alpha_s2',
                                   control_vars: List[str] = None,
                                   k: int = 3,
                                   n_permutations: int = 5000) -> Dict:
        """
        Analyze the effect of each individual control parameter on MI.
        
        Parameters:
        -----------
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        control_vars : List[str]
            Variables to control for in the analysis
        k : int
            Number of nearest neighbors for KSG estimation
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        results : Dict
            Dictionary containing results for each individual parameter
        """
        results = {}
        
        # Extract variables
        x_orig = self.data[feature].values
        y_orig = self.data[target].values
        
        # Standardize x and y upfront
        x_std, y_std = self.standardize_data(x_orig, y_orig)
        
        # Calculate raw MI using KSG estimator on standardized data
        raw_mi, raw_p = self._ksg_mi_core(x_std, y_std, k=k, n_permutations=n_permutations)
        
        # Analyze each control variable individually
        for var in control_vars:
            z_orig = self.data[var].values
            z_std, _ = self.standardize_data(z_orig, z_orig)
            
            # Calculate conditional MI
            x_reshaped = x_std.reshape(-1, 1)
            yz = np.column_stack([y_std, z_std])
            
            # Calculate MI components
            mi_xyz = ee.mi(x_reshaped, yz, k=k)
            mi_xz = ee.mi(x_reshaped, z_std.reshape(-1, 1), k=k)
            
            # Conditional MI = I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
            cmi = mi_xyz - mi_xz
            
            # Permutation test for conditional MI
            cmi_null = np.zeros(n_permutations)
            for i in range(n_permutations):
                y_perm = np.random.permutation(y_std)
                yz_perm = np.column_stack([y_perm, z_std])
                mi_xyz_perm = ee.mi(x_reshaped, yz_perm, k=k)
                cmi_null[i] = mi_xyz_perm - mi_xz
            
            p_value = (np.sum(cmi_null >= cmi) + 1) / (n_permutations + 1)
            
            # Calculate retention percentage
            retention = (cmi / raw_mi) * 100 if raw_mi > 0 else 0
            
            results[var] = {
                'cmi': cmi,
                'p_value': p_value,
                'retention': retention
            }
        
        return results

    def plot_individual_parameter_effects(self, all_k_results: Dict,
                                       feature: str = 'alpha_s1',
                                       target: str = 'alpha_s2',
                                       control_vars: List[str] = None,
                                       session1_path: str = None,
                                       dataset_name: str = None) -> None:
        """
        Create plots showing the effect of each individual control parameter.
        """
        if dataset_name is None:
            dataset_name = self.get_dataset_name(session1_path) if session1_path else 'Unknown Dataset'
        
        # Create directory for saving plots
        plot_dir = f"plots_{dataset_name.replace(' ', '_')}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Extract data for plots
        k_values = sorted(all_k_results.keys())
        
        # Prepare data dictionaries for each parameter
        plot_data = {}
        for var in control_vars:
            plot_data[var] = {
                'k': k_values,
                'cmi': [],
                'p_value': [],
                'retention': []
            }
        
        # Fill data for plots
        for k in k_values:
            res = all_k_results[k]
            for var in control_vars:
                if var in res:
                    plot_data[var]['cmi'].append(res[var]['cmi'])
                    plot_data[var]['p_value'].append(res[var]['p_value'])
                    plot_data[var]['retention'].append(res[var]['retention'])
        
        # Define colors for different parameters
        colors = {
            'a_mean_s1': '#1f77b4',  # blue
            'a_mean_s2': '#ff7f0e',  # orange
            'ndt_mean_s1': '#2ca02c',  # green
            'ndt_mean_s2': '#d62728'   # red
        }
        
        # Plot 1: Individual CMI vs k
        plt.figure(figsize=(12, 8))
        for var in control_vars:
            plt.plot(plot_data[var]['k'], plot_data[var]['cmi'],
                    marker='o', color=colors[var],
                    linewidth=2, markersize=10,
                    label=f'CMI ({var})')
        
        plt.title(f'Individual Parameter CMI vs k\n{dataset_name} (n={self.n_samples})', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('Conditional Mutual Information (bits)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # Add data-driven k
        k_dd = int(np.sqrt(self.n_samples))
        if k_dd in k_values:
            plt.axvline(x=k_dd, color='gray', linestyle='--', alpha=0.7)
            plt.text(k_dd, plt.ylim()[1]*0.95, f'k_dd = {k_dd}',
                     ha='center', va='top',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Individual_CMI_vs_k.png'))
        plt.show()
        plt.close()
        
        # Plot 2: Individual Retention vs k
        plt.figure(figsize=(12, 8))
        for var in control_vars:
            plt.plot(plot_data[var]['k'], plot_data[var]['retention'],
                    marker='o', color=colors[var],
                    linewidth=2, markersize=10,
                    label=f'Retention ({var})')
        
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='100% Retention')
        
        plt.title(f'Individual Parameter Retention vs k\n{dataset_name} (n={self.n_samples})', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('Information Retention (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        if k_dd in k_values:
            plt.axvline(x=k_dd, color='gray', linestyle='--', alpha=0.7)
            plt.text(k_dd, plt.ylim()[1]*0.95, f'k_dd = {k_dd}',
                     ha='center', va='top',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Individual_Retention_vs_k.png'))
        plt.show()
        plt.close()
        
        # Plot 3: Individual P-values vs k
        plt.figure(figsize=(12, 8))
        for var in control_vars:
            plt.plot(plot_data[var]['k'], plot_data[var]['p_value'],
                    marker='o', color=colors[var],
                    linewidth=2, markersize=10,
                    label=f'p-value ({var})')
        
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        
        plt.title(f'Individual Parameter P-values vs k\n{dataset_name} (n={self.n_samples})', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('P-value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        if k_dd in k_values:
            plt.axvline(x=k_dd, color='gray', linestyle='--', alpha=0.7)
            plt.text(k_dd, plt.ylim()[1]*0.95, f'k_dd = {k_dd}',
                     ha='center', va='top',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Individual_Pvalue_vs_k.png'))
        plt.show()
        plt.close()

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
    id_col_s1 = next((col for col in ['ID', 'participant', 'participant_ID', 'Experiment'] if col in df_s1.columns), None)
    id_col_s2 = next((col for col in ['ID', 'participant', 'participant_ID', 'Experiment'] if col in df_s2.columns), None)
    
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
    
    # Print available columns for debugging
    print("Available columns in session 1:", df_s1.columns.tolist())
    print("Available columns in session 2:", df_s2.columns.tolist())
    print("Using identifier column:", id_col_s1)
    
    # Merge on the common identifier column
    df_merged = pd.merge(df_s1, df_s2, on=id_col_s1, how='inner')
    df_merged.dropna(inplace=True)
    
    # Determine the correct alpha column names
    alpha_col_s1 = 'alpha_mean_boxcox_after_arcsin_s1' if 'alpha_mean_boxcox_after_arcsin_s1' in df_merged.columns else 'alpha_boxcox_after_arcsin_s1'
    alpha_col_s2 = 'alpha_mean_boxcox_after_arcsin_s2' if 'alpha_mean_boxcox_after_arcsin_s2' in df_merged.columns else 'alpha_boxcox_after_arcsin_s2'
    
    # Rename the transformed alpha columns for convenience:
    df_merged["alpha_s1"] = df_merged[alpha_col_s1]
    df_merged["alpha_s2"] = df_merged[alpha_col_s2]
    
    return df_merged

# Example usage:
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='KSG Mutual Information Analysis')
    parser.add_argument('--session1', type=str, required=True, help='Path to Session 1 CSV file')
    parser.add_argument('--session2', type=str, required=True, help='Path to Session 2 CSV file')
    parser.add_argument('--k_values', type=str, help='Comma-separated list of k values (e.g., "3,5,7,10")')
    parser.add_argument('--output_csv', type=str, help='Path to save results CSV file')
    parser.add_argument('--dataset_name', type=str, help='Custom name for the dataset (overrides automatic name)')
    
    args = parser.parse_args()
    
    # Load data
    session1_path = args.session1
    session2_path = args.session2
    df_merged = load_and_merge(session1_path, session2_path)
    
    # Initialize analyzer
    analyzer = KSGMutualInformationAnalyzer(df_merged)
    
    # Determine dataset name
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = analyzer.get_dataset_name(session1_path)
    
    # Define control variables
    control_vars = ['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
    
    # Run analysis with different k values including the data-driven one
    results = {}
    individual_results = {}
    
    # Calculate data-driven k value
    k_data_driven = int(np.sqrt(len(df_merged)))
    print(f"\nData-driven k value (sqrt(N)): {k_data_driven}")
    
    # Define k values to test (including data-driven k)
    k_values = [3, 5, 7, 10, k_data_driven]
    k_values = sorted(list(set(k_values)))  # Remove duplicates and sort
    
    # Run analysis with different k values including the data-driven one
    for k in k_values:
        print(f"\nAnalyzing with k={k}")
        results[k] = analyzer.analyze_alpha_reliability(
            feature='alpha_s1',
            target='alpha_s2',
            control_vars=control_vars,
            k=k,
            n_permutations=5000
        )
        
        # Analyze individual parameters
        individual_results[k] = analyzer.analyze_individual_parameters(
            feature='alpha_s1',
            target='alpha_s2',
            control_vars=control_vars,
            k=k,
            n_permutations=5000
        )
    
    # Plot results
    analyzer.plot_results(
        results[k_values[-1]], 
        all_k_results=results, 
        session1_path=session1_path,
        dataset_name=dataset_name
    )
    
    # Plot individual parameter effects
    analyzer.plot_individual_parameter_effects(
        individual_results,
        feature='alpha_s1',
        target='alpha_s2',
        control_vars=control_vars,
        session1_path=session1_path,
        dataset_name=dataset_name
    )
    
    # Print and save summary table
    analyzer.print_summary_table(
        all_k_results=results, 
        session1_path=session1_path,
        output_csv_path=args.output_csv
    )
    
    # Print comprehensive results
    print("\nComprehensive Results:")
    for k in k_values:
        print(f"\nk={k}:")
        if k == k_data_driven:
            print(f"(Data-driven k value = sqrt(N) where N={len(df_merged)})")
        print(f"Raw MI: {results[k]['ksg_mi'][0]:.4f} bits (p={results[k]['ksg_mi'][1]:.4f})")
        
        # Print conditional MI results if available
        if 'cond_mi_all' in results[k]:
            print(f"Conditional MI (all parameters): {results[k]['cond_mi_all']:.4f} bits (p={results[k]['p_value_all']:.4f})")
            print(f"Overall Retention: {results[k]['retention_all']:.1f}%")
        
        if 'cond_mi_thresh' in results[k]:
            print(f"Conditional MI (threshold only): {results[k]['cond_mi_thresh']:.4f} bits (p={results[k]['p_value_thresh']:.4f})")
            print(f"Threshold Retention: {results[k]['retention_thresh']:.1f}%")
            
        if 'cond_mi_ndt' in results[k]:
            print(f"Conditional MI (NDT only): {results[k]['cond_mi_ndt']:.4f} bits (p={results[k]['p_value_ndt']:.4f})")
            print(f"NDT Retention: {results[k]['retention_ndt']:.1f}%") 