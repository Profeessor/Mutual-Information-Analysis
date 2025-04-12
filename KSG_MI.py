import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union
import warnings
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial import cKDTree
import itertools
from sklearn.decomposition import PCA

class KSGMutualInformationAnalyzer:
    """
    A comprehensive mutual information analyzer implementing the 
    Kraskov-Stögbauer-Grassberger (KSG) estimator using scikit-learn's
    mutual_info_regression implementation.
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
        This preprocessing step is recommended for enhanced numerical stability.
        
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
    
    def ksg_mi(self, x: np.ndarray, y: np.ndarray, k: int = 3, 
                n_permutations: int = 5000) -> Tuple[float, float]:
        """
        Estimate mutual information using scikit-learn's mutual_info_regression.
        The data is automatically standardized before estimation for enhanced numerical stability.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable (e.g., alpha_s1)
        y : np.ndarray
            Second variable (e.g., alpha_s2)
        k : int
            Number of nearest neighbors for estimation
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
        
        # Reshape arrays for scikit-learn
        x_std = x_std.reshape(-1, 1)
        y_std = y_std.reshape(-1, 1)
        
        # Calculate MI using scikit-learn's mutual_info_regression
        mi = mutual_info_regression(x_std, y_std.ravel(), n_neighbors=k)[0]
        
        # If MI is negative, it's likely due to numerical issues
        if mi < 0:
            warnings.warn(
                f"Negative MI value ({mi:.4f}) detected. This is likely due to "
                "numerical issues in the estimation. Consider:\n"
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
            y_perm = np.random.permutation(y_std.ravel())
            mi_null[i] = mutual_info_regression(x_std, y_perm, n_neighbors=k)[0]
        
        # Calculate p-value
        p_value = (np.sum(mi_null >= mi) + 1) / (n_permutations + 1)
        
        return mi, p_value
    
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
                mi = mutual_info_regression(x_sub.reshape(-1, 1), y_sub.reshape(-1, 1), n_neighbors=k)[0]
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
        Comprehensive analysis of alpha parameter reliability using scikit-learn's
        mutual information regression.
        
        Parameters:
        -----------
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        control_vars : List[str]
            Variables to control for in the analysis
        k : int
            Number of nearest neighbors for estimation
        n_permutations : int
            Number of permutations for significance testing
            
        Returns:
        --------
        results : Dict
            Dictionary containing results from MI estimation
        """
        results = {}
        
        # Extract variables
        x = self.data[feature].values
        y = self.data[target].values
        
        # Calculate raw MI
        results['ksg_mi'] = self.ksg_mi(x, y, k=k, n_permutations=n_permutations)
        
        # Calculate conditional MI if control variables are provided
        if control_vars:
            results['conditional_mi'] = {}
            results['stepwise_retention'] = {}
            
            # Stepwise analysis: add one control variable at a time
            current_controls = []
            for var in control_vars:
                current_controls.append(var)
                z = np.column_stack([self.data[v].values for v in current_controls])
                
                # Calculate conditional MI
                # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
                x_reshaped = x.reshape(-1, 1)
                
                # For joint variables (Y,Z), we need to create a single target variable
                # We'll use the first principal component of Y and Z
                yz = np.column_stack([y, z])
                pca = PCA(n_components=1)
                yz_pca = pca.fit_transform(yz).ravel()
                
                # Calculate MI with current set of controls
                mi_xyz = mutual_info_regression(x_reshaped, yz_pca, n_neighbors=k)[0]
                
                # For I(X;Z), we'll use the first principal component of Z
                z_pca = pca.fit_transform(z).ravel()
                mi_xz = mutual_info_regression(x_reshaped, z_pca, n_neighbors=k)[0]
                
                # Conditional MI = I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
                cmi = mi_xyz - mi_xz
                
                # Permutation test for conditional MI
                cmi_null = np.zeros(n_permutations)
                for i in range(n_permutations):
                    y_perm = np.random.permutation(y)
                    yz_perm = np.column_stack([y_perm, z])
                    yz_perm_pca = pca.fit_transform(yz_perm).ravel()
                    mi_xyz_perm = mutual_info_regression(x_reshaped, yz_perm_pca, n_neighbors=k)[0]
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
            z_values = np.column_stack([self.data[var].values for var in control_vars])
            yz = np.column_stack([y, z_values])
            
            # Use PCA for joint variables
            yz_pca = pca.fit_transform(yz).ravel()
            z_pca = pca.fit_transform(z_values).ravel()
            
            mi_xyz = mutual_info_regression(x_reshaped, yz_pca, n_neighbors=k)[0]
            mi_xz = mutual_info_regression(x_reshaped, z_pca, n_neighbors=k)[0]
            
            cmi_all = mi_xyz - mi_xz
            
            # Permutation test for joint conditional MI
            cmi_null = np.zeros(n_permutations)
            for i in range(n_permutations):
                y_perm = np.random.permutation(y)
                yz_perm = np.column_stack([y_perm, z_values])
                yz_perm_pca = pca.fit_transform(yz_perm).ravel()
                mi_xyz_perm = mutual_info_regression(x_reshaped, yz_perm_pca, n_neighbors=k)[0]
                cmi_null[i] = mi_xyz_perm - mi_xz
            
            p_value_all = (np.sum(cmi_null >= cmi_all) + 1) / (n_permutations + 1)
            results['cond_mi_all'] = cmi_all
            results['p_value_all'] = p_value_all
            
            # Calculate overall retention percentage
            results['retention_all'] = (cmi_all / raw_mi) * 100
            
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

    def print_summary_table(self, all_k_results: Dict, session1_path: str = None) -> None:
        """
        Print a comprehensive summary table of all results.
        
        Parameters:
        -----------
        all_k_results : Dict
            Results from analyze_alpha_reliability for all k values
        session1_path : str
            Path to the first session data file (for dataset naming)
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
            "Cond MI (bits)",
            "Cond MI p-value",
            "Retention (%)",
            "Sensitivity Mean±SD",
            "Stepwise Retention (%)"
        ]
        
        # Print header row
        print(f"{headers[0]:<10} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} "
              f"{headers[4]:<15} {headers[5]:<15} {headers[6]:<20} {headers[7]}")
        print("-"*120)
        
        # Print results for each k value
        for k in sorted(all_k_results.keys()):
            results = all_k_results[k]
            
            # Get stepwise retention values if available
            stepwise_str = ""
            if 'stepwise_retention' in results:
                stepwise_values = []
                for var in results['stepwise_retention']:
                    if 'a_mean_s1' in var:
                        name = 'a_s1'
                    elif 'a_mean_s2' in var:
                        name = 'a_s2'
                    elif 'ndt_mean_s1' in var:
                        name = 'ndt_s1'
                    elif 'ndt_mean_s2' in var:
                        name = 'ndt_s2'
                    else:
                        name = var
                    val = results['stepwise_retention'][var]
                    stepwise_values.append(f"{name}:{val:.1f}")
                stepwise_str = ", ".join(stepwise_values)
            
            # Format the row
            raw_mi = results['ksg_mi'][0]
            raw_p = results['ksg_mi'][1]
            cond_mi = results.get('cond_mi_all', 0)
            cond_p = results.get('p_value_all', 1)
            retention = results.get('retention_all', 0)
            
            # Get sensitivity analysis results
            sensitivity_str = ""
            if 'sensitivity' in results:
                sens_key = f'k_{k}'
                if sens_key in results['sensitivity']:
                    mean = results['sensitivity'][sens_key]['mean']
                    std = results['sensitivity'][sens_key]['std']
                    sensitivity_str = f"{mean:.3f}±{std:.3f}"
            
            # Add asterisks for significance
            raw_sig = "*" if raw_p < 0.05 else " "
            cond_sig = "*" if cond_p < 0.05 else " "
            
            print(f"{k:<10} {raw_mi:>6.3f}{raw_sig:<8} {raw_p:<15.4f} {cond_mi:>6.3f}{cond_sig:<8} "
                  f"{cond_p:<15.4f} {retention:<15.1f} {sensitivity_str:<20} {stepwise_str}")
        
        # Print footer with legend
        print("-"*120)
        print("* indicates significance at p < 0.05")
        print("Sensitivity Mean±SD shows the mean and standard deviation of MI estimates across subsamples")
        print("Stepwise Retention shows retention percentage after adding each control variable")
        print("="*120)

    def plot_results(self, results: Dict, feature: str = 'alpha_s1', 
                    target: str = 'alpha_s2', all_k_results: Dict = None,
                    session1_path: str = None) -> None:
        """
        Create visualizations of the KSG MI analysis results.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze_alpha_reliability for a single k value
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        all_k_results : Dict
            Results from analyze_alpha_reliability for all k values
        session1_path : str
            Path to the first session data file (for plot naming)
        """
        dataset_name = self.get_dataset_name(session1_path) if session1_path else ''
        
        # New plots - only create these once for the last k value
        if all_k_results:
            # Print summary table
            self.print_summary_table(all_k_results, session1_path)
            
            # Plot 1: Combined Stepwise Information Retention for all k values
            if any('stepwise_retention' in k_results for k_results in all_k_results.values()):
                plt.figure(figsize=(15, 8))
                
                # Set up colors and markers for different k values
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                markers = ['o', 's', '^', 'D', 'v']
                
                # Plot for each k value
                for idx, (k, k_results) in enumerate(all_k_results.items()):
                    if 'stepwise_retention' in k_results:
                        vars_list = list(k_results['stepwise_retention'].keys())
                        retention_values = [max(0, k_results['stepwise_retention'][var]) for var in vars_list]
                        
                        # Simplify variable names
                        display_names = []
                        for var in vars_list:
                            if 'a_mean_s1' in var:
                                display_names.append('a_s1')
                            elif 'a_mean_s2' in var:
                                display_names.append('a_s2')
                            elif 'ndt_mean_s1' in var:
                                display_names.append('ndt_s1')
                            elif 'ndt_mean_s2' in var:
                                display_names.append('ndt_s2')
                            else:
                                display_names.append(var)
                        
                        plt.plot(range(len(display_names)), retention_values, 
                                marker=markers[idx % len(markers)],
                                color=colors[idx % len(colors)],
                                linewidth=2, markersize=10,
                                label=f'k={k} (Raw MI: {k_results["ksg_mi"][0]:.3f}, p={k_results["ksg_mi"][1]:.4f})')
                
                plt.title(f'Stepwise Information Retention\n{dataset_name} (n={self.n_samples})', 
                         fontsize=16)
                plt.xlabel('Control Variables Added', fontsize=14)
                plt.ylabel('Information Retention (%)', fontsize=14)
                plt.ylim(0, 100)  # Set y-axis from 0 to 100%
                plt.xticks(range(len(display_names)), display_names, rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                plt.tight_layout()
                plt.show()
                
                # Plot 2: Pie charts for each k value
                k_values = list(all_k_results.keys())
                
                # Calculate grid dimensions based on number of k values
                n_plots = len(k_values)
                n_cols = min(3, n_plots)  # Maximum 3 columns
                n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
                
                fig = plt.figure(figsize=(15, 5*n_rows))  # Adjust height based on number of rows
                
                # Add main title with dataset name
                plt.suptitle(f'Information Retention Analysis\n{dataset_name}',
                            fontsize=16, y=0.95)
                
                # Create a grid for pie charts with appropriate dimensions
                gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0.8, wspace=0.3)
                
                # Create pie charts in the grid
                for idx, k in enumerate(k_values):
                    row = idx // n_cols
                    col = idx % n_cols
                    
                    # Create subplot for pie chart
                    ax = fig.add_subplot(gs[row, col])
                    
                    # Get values for the pie chart
                    retention = all_k_results[k].get('retention_all', 0)
                    retention = max(0, min(retention, 100))  # Ensure between 0 and 100
                    sizes = [retention, 100-retention]
                    
                    # Create pie chart
                    colors = ['#2ca02c', '#ff7f0e']  # Green and orange
                    wedges, texts, autotexts = plt.pie(sizes, colors=colors, autopct='%1.1f%%', 
                                                      startangle=90, textprops={'fontsize': 12})
                    
                    # Add text box with MI values and k value below the pie chart
                    raw_mi = all_k_results[k]['ksg_mi'][0]
                    p_val = all_k_results[k]['ksg_mi'][1]
                    cond_mi = all_k_results[k].get('cond_mi_all', 0)
                    
                    textstr = f'k = {k}\nRaw MI: {raw_mi:.3f} bits\np-value: {p_val:.4f}\nCond MI: {cond_mi:.3f} bits'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    
                    # Position the text box closer to the pie chart
                    y_pos = -0.2 if row == 0 else -0.25
                    plt.text(0.5, y_pos, textstr, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', horizontalalignment='center',
                            bbox=props)
                
                # Create a separate legend
                legend_labels = ['Unique to Alpha', 'Shared with\nThreshold & NDT']
                legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label)
                                 for color, label in zip(colors, legend_labels)]
                
                # Add the legend in the top right corner
                fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95),
                          fontsize=12, frameon=True)
                
                # Adjust layout
                plt.tight_layout(rect=[0, 0, 0.95, 0.92])  # Make room for suptitle and legend
                plt.show()

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
    # Load data
    session1_path = "path_to_session1_data.csv"
    session2_path = "path_to_session2_data.csv"
    df_merged = load_and_merge(session1_path, session2_path)
    
    # Initialize analyzer
    analyzer = KSGMutualInformationAnalyzer(df_merged)
    
    # Define control variables
    control_vars = ['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
    
    # Run analysis with different k values including the data-driven one
    results = {}
    
    # Calculate data-driven k value
    k_data_driven = int(np.sqrt(len(df_merged)))
    print(f"\nData-driven k value (sqrt(N)): {k_data_driven}")
    
    # For large datasets (N > 500), use larger k values
    if len(df_merged) > 500:
        k_values = [10, 15, 20, k_data_driven, 35]
    else:
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
    
    # Plot results for the last k value (will include all results in the plots)
    last_k = k_values[-1]
    analyzer.plot_results(results[last_k], all_k_results=results, session1_path=session1_path)
    
    # Print comprehensive results
    print("\nComprehensive Results:")
    for k in k_values:
        print(f"\nk={k}:")
        if k == k_data_driven:
            print(f"(Data-driven k value = sqrt(N) where N={len(df_merged)})")
        print(f"Raw MI: {results[k]['ksg_mi'][0]:.4f} bits (p={results[k]['ksg_mi'][1]:.4f})")
        if 'cond_mi_all' in results[k]:
            print(f"Conditional MI (all parameters): {results[k]['cond_mi_all']:.4f} bits (p={results[k]['p_value_all']:.4f})")
            print(f"Overall Retention: {results[k]['retention_all']:.1f}%") 