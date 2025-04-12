import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union
import warnings
from scipy.integrate import dblquad

class MutualInformationAnalyzer:
    """
    Mutual Information analyzer using kernel density estimation with optimal bandwidth.
    """
    
    def __init__(self, data):
        """
        Initialize the analyzer with data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing alpha parameters and other variables
        """
        self.data = data
        self.n_samples = len(data)
        
        # Check sample size
        if self.n_samples < 100:
            warnings.warn(
                f"Sample size ({self.n_samples}) is small. "
                "For reliable mutual information estimation, "
                "it's recommended to have at least 100 samples."
            )
    
    def shrinkage_mi(self, x: np.ndarray, y: np.ndarray, 
                    alpha: float = 0.5,
                    bandwidth: float = 0.2) -> Tuple[float, float]:
        """
        Estimate mutual information using shrinkage estimation.
        Combines empirical estimates with structured target estimators.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
        alpha : float
            Shrinkage parameter (0 to 1)
        bandwidth : float
            Bandwidth for kernel density estimation
            
        Returns:
        --------
        mi : float
            Estimated mutual information
        p_value : float
            P-value from permutation test
        """
        # Calculate empirical MI using k-NN
        k = max(3, int(np.sqrt(self.n_samples)))
        kd_x = KernelDensity(bandwidth=bandwidth).fit(x.reshape(-1, 1))
        kd_y = KernelDensity(bandwidth=bandwidth).fit(y.reshape(-1, 1))
        kd_xy = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([x, y]))
        
        log_px = kd_x.score_samples(x.reshape(-1, 1))
        log_py = kd_y.score_samples(y.reshape(-1, 1))
        log_pxy = kd_xy.score_samples(np.column_stack([x, y]))
        
        mi_empirical = np.mean(log_pxy - log_px - log_py)
        
        # Calculate Gaussian MI
        r = np.corrcoef(x, y)[0, 1]
        mi_gaussian = -0.5 * np.log(1 - r**2)
        
        # Combine estimates
        mi = alpha * mi_empirical + (1 - alpha) * mi_gaussian
        
        # Permutation test using Gaussian MI only (to avoid recursion)
        n_permutations = 5000
        mi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            r_perm = np.corrcoef(x, y_perm)[0, 1]
            mi_null[i] = -0.5 * np.log(1 - r_perm**2)
        
        p_value = np.mean(mi_null >= mi)
        return mi, p_value
    
    def analyze_alpha_reliability(self, 
                                feature='alpha_s1',
                                target='alpha_s2',
                                control_vars=None,
                                n_permutations=5000,
                                alpha=0.5,
                                bandwidth=0.2):
        """
        Analyze alpha parameter reliability using mutual information.
        Calculates conditional MI separately for threshold and non-decision time parameters.
        
        Parameters:
        -----------
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        control_vars : list
            Variables to control for in the analysis
        n_permutations : int
            Number of permutations for significance testing
        alpha : float
            Shrinkage parameter (0 to 1)
        bandwidth : float
            Bandwidth for kernel density estimation
            
        Returns:
        --------
        dict
            Dictionary containing results from MI estimation
        """
        results = {}
        
        # Extract variables
        x = self.data[feature].values
        y = self.data[target].values
        
        # Calculate raw MI
        mi, p_value = self.shrinkage_mi(x, y, alpha=alpha, bandwidth=bandwidth)
        results['mi'] = (mi, p_value)
        
        # Calculate conditional MI if control variables are provided
        if control_vars:
            results['conditional_mi'] = {}
            results['stepwise_retention'] = {}
            
            # Separate threshold and non-decision time parameters
            threshold_vars = [var for var in control_vars if 'a_' in var]
            ndt_vars = [var for var in control_vars if 'ndt_' in var]
            
            # Calculate conditional MI for threshold parameters
            if threshold_vars:
                # Extract threshold control variables
                z_threshold = np.column_stack([self.data[v].values for v in threshold_vars])
                
                # Conditional MI approach: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
                # First, calculate MI between X and Z
                mi_xz_threshold, _ = self.shrinkage_mi(x, z_threshold[:,0], alpha=alpha, bandwidth=bandwidth)
                
                # For multi-dimensional Z, add each dimension's contribution
                for i in range(1, z_threshold.shape[1]):
                    mi_xz_i, _ = self.shrinkage_mi(x, z_threshold[:,i], alpha=alpha, bandwidth=bandwidth)
                    mi_xz_threshold += mi_xz_i
                
                # Calculate a combined Gaussian MI for X and Y given Z
                combined_mi = 0
                for i in range(len(x)):
                    # For each sample, calculate MI between X and Y conditioned on Z
                    x_i = np.array([x[i]])
                    y_i = np.array([y[i]])
                    z_i = z_threshold[i,:]
                    mi_xy_i, _ = self.shrinkage_mi(x_i, y_i, alpha=alpha, bandwidth=bandwidth)
                    combined_mi += mi_xy_i
                
                # Average over all samples
                mi_xyz_threshold = combined_mi / len(x)
                
                # Conditional MI is the difference
                cmi_threshold = mi - mi_xz_threshold
                
                # Permutation test for threshold conditional MI
                cmi_null = np.zeros(n_permutations)
                for i in range(n_permutations):
                    y_perm = np.random.permutation(y)
                    # Calculate MI between X and permuted Y
                    mi_xy_perm, _ = self.shrinkage_mi(x, y_perm, alpha=alpha, bandwidth=bandwidth)
                    # Calculate conditional MI with permuted Y
                    cmi_null[i] = mi_xy_perm - mi_xz_threshold
                
                p_value_threshold = (np.sum(cmi_null >= cmi_threshold) + 1) / (n_permutations + 1)
                results['conditional_mi']['threshold'] = (cmi_threshold, p_value_threshold)
                results['stepwise_retention']['threshold'] = (cmi_threshold / mi) * 100
            
            # Calculate conditional MI for non-decision time parameters
            if ndt_vars:
                # Extract non-decision time control variables
                z_ndt = np.column_stack([self.data[v].values for v in ndt_vars])
                
                # Conditional MI approach: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
                # First, calculate MI between X and Z
                mi_xz_ndt, _ = self.shrinkage_mi(x, z_ndt[:,0], alpha=alpha, bandwidth=bandwidth)
                
                # For multi-dimensional Z, add each dimension's contribution
                for i in range(1, z_ndt.shape[1]):
                    mi_xz_i, _ = self.shrinkage_mi(x, z_ndt[:,i], alpha=alpha, bandwidth=bandwidth)
                    mi_xz_ndt += mi_xz_i
                
                # Calculate a combined Gaussian MI for X and Y given Z
                combined_mi = 0
                for i in range(len(x)):
                    # For each sample, calculate MI between X and Y conditioned on Z
                    x_i = np.array([x[i]])
                    y_i = np.array([y[i]])
                    z_i = z_ndt[i,:]
                    mi_xy_i, _ = self.shrinkage_mi(x_i, y_i, alpha=alpha, bandwidth=bandwidth)
                    combined_mi += mi_xy_i
                
                # Average over all samples
                mi_xyz_ndt = combined_mi / len(x)
                
                # Conditional MI is the difference
                cmi_ndt = mi - mi_xz_ndt
                
                # Permutation test for NDT conditional MI
                cmi_null = np.zeros(n_permutations)
                for i in range(n_permutations):
                    y_perm = np.random.permutation(y)
                    # Calculate MI between X and permuted Y
                    mi_xy_perm, _ = self.shrinkage_mi(x, y_perm, alpha=alpha, bandwidth=bandwidth)
                    # Calculate conditional MI with permuted Y
                    cmi_null[i] = mi_xy_perm - mi_xz_ndt
                
                p_value_ndt = (np.sum(cmi_null >= cmi_ndt) + 1) / (n_permutations + 1)
                results['conditional_mi']['ndt'] = (cmi_ndt, p_value_ndt)
                results['stepwise_retention']['ndt'] = (cmi_ndt / mi) * 100
            
            # Calculate retention percentages
            results['retention'] = {
                'threshold': results['stepwise_retention'].get('threshold', 0),
                'ndt': results['stepwise_retention'].get('ndt', 0)
            }
            
            # Add dimensionality information
            results['dimensionality'] = {
                'n_samples': self.n_samples,
                'n_threshold_vars': len(threshold_vars),
                'n_ndt_vars': len(ndt_vars),
                'total_dimensions': {
                    'threshold': 2 + len(threshold_vars),
                    'ndt': 2 + len(ndt_vars)
                }
            }
            
            # Add warnings for small sample sizes
            if self.n_samples < 100:
                warnings.warn(
                    f"Sample size ({self.n_samples}) is small. "
                    "Results should be interpreted with caution."
                )
            
            # Add warning if dimensionality might be too high
            for var_type, dim in results['dimensionality']['total_dimensions'].items():
                if self.n_samples < dim * 10:
                    warnings.warn(
                        f"Sample size ({self.n_samples}) might be too small for "
                        f"{var_type} analysis with {dim} dimensions. "
                        "Consider reducing the number of control variables."
                    )
        
        return results
    
    def bandwidth_sensitivity_analysis(self, x: np.ndarray, y: np.ndarray,
                                    bandwidths: List[float] = None,
                                    alpha: float = 0.5) -> Dict:
        """
        Perform sensitivity analysis for different bandwidths.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
        bandwidths : List[float], optional
            List of bandwidths to test. If None, uses default range.
        alpha : float
            Shrinkage parameter (0 to 1)
            
        Returns:
        --------
        Dict
            Dictionary containing results for each bandwidth
        """
        if bandwidths is None:
            # Default bandwidth range based on Silverman's rule
            base_bandwidth = np.std(x) * (4/(3*len(x)))**(1/5)
            bandwidths = [
                base_bandwidth * 0.5,
                base_bandwidth * 0.75,
                base_bandwidth,
                base_bandwidth * 1.25,
                base_bandwidth * 1.5,
                base_bandwidth * 2,
                base_bandwidth * 3,
              
            ]
        
        results = {
            'bandwidths': bandwidths,
            'mi_values': [],
            'p_values': [],
            'empirical_mi': [],
            'gaussian_mi': []
        }
        
        for h in bandwidths:
            # Calculate MI with current bandwidth
            mi, p_value = self.shrinkage_mi(x, y, alpha=alpha, bandwidth=h)
            
            # Calculate individual components for analysis
            kd_x = KernelDensity(bandwidth=h).fit(x.reshape(-1, 1))
            kd_y = KernelDensity(bandwidth=h).fit(y.reshape(-1, 1))
            kd_xy = KernelDensity(bandwidth=h).fit(np.column_stack([x, y]))
            
            log_px = kd_x.score_samples(x.reshape(-1, 1))
            log_py = kd_y.score_samples(y.reshape(-1, 1))
            log_pxy = kd_xy.score_samples(np.column_stack([x, y]))
            
            mi_empirical = np.mean(log_pxy - log_px - log_py)
            r = np.corrcoef(x, y)[0, 1]
            mi_gaussian = -0.5 * np.log(1 - r**2)
            
            results['mi_values'].append(mi)
            results['p_values'].append(p_value)
            results['empirical_mi'].append(mi_empirical)
            results['gaussian_mi'].append(mi_gaussian)
        
        # Calculate stability metrics
        results['stability'] = {
            'mi_std': np.std(results['mi_values']),
            'mi_range': np.max(results['mi_values']) - np.min(results['mi_values']),
            'mi_cv': np.std(results['mi_values']) / np.mean(results['mi_values'])
        }
        
        return results
    
    def plot_bandwidth_sensitivity(self, sensitivity_results: Dict):
        """
        Plot results from bandwidth sensitivity analysis.
        
        Parameters:
        -----------
        sensitivity_results : Dict
            Results from bandwidth_sensitivity_analysis
        """
        plt.figure(figsize=(12, 6))
        
        # Plot MI values
        plt.subplot(1, 2, 1)
        plt.plot(sensitivity_results['bandwidths'], sensitivity_results['mi_values'], 
                'o-', label='Combined MI')
        plt.plot(sensitivity_results['bandwidths'], sensitivity_results['empirical_mi'], 
                '--', label='Empirical MI')
        plt.plot(sensitivity_results['bandwidths'], sensitivity_results['gaussian_mi'], 
                '--', label='Gaussian MI')
        plt.xlabel('Bandwidth')
        plt.ylabel('Mutual Information (bits)')
        plt.title('MI vs Bandwidth')
        plt.legend()
        
        # Plot p-values
        plt.subplot(1, 2, 2)
        plt.plot(sensitivity_results['bandwidths'], sensitivity_results['p_values'], 'o-')
        plt.xlabel('Bandwidth')
        plt.ylabel('P-value')
        plt.title('P-value vs Bandwidth')
        
        plt.tight_layout()
        plt.show()
        
        # Print stability metrics
        print("\nStability Metrics:")
        print(f"Standard Deviation: {sensitivity_results['stability']['mi_std']:.4f}")
        print(f"Range: {sensitivity_results['stability']['mi_range']:.4f}")
        print(f"Coefficient of Variation: {sensitivity_results['stability']['mi_cv']:.4f}")
    
    def plot_results(self, results: Dict, dataset_name: str = None):
        """
        Plot results from mutual information analysis.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze_alpha_reliability
        dataset_name : str, optional
            Name of the dataset for plot title
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Add title with dataset name if provided
        if dataset_name:
            plt.suptitle(f"Mutual Information Analysis: {dataset_name}", fontsize=16)
        
        # Plot raw MI and conditional MI for threshold and NDT
        mi_values = [results['mi'][0]]
        mi_labels = ['Raw MI']
        p_values = [results['mi'][1]]
        
        if 'conditional_mi' in results:
            for label, (cmi, p_value) in results['conditional_mi'].items():
                mi_values.append(cmi)
                mi_labels.append(f"CMI ({label})")
                p_values.append(p_value)
        
        # Add significance asterisks
        for i, p in enumerate(p_values):
            if p < 0.001:
                mi_labels[i] += ' ***'
            elif p < 0.01:
                mi_labels[i] += ' **'
            elif p < 0.05:
                mi_labels[i] += ' *'
        
        # Plot MI values
        plt.subplot(2, 1, 1)
        bars = plt.bar(mi_labels, mi_values, color=['blue', 'green', 'orange'])
        plt.ylabel('Mutual Information (bits)')
        plt.title('Mutual Information Estimates')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot retention percentages if available
        if 'retention' in results:
            retention_values = list(results['retention'].values())
            retention_labels = [f"{label}" for label in results['retention'].keys()]
            
            plt.subplot(2, 1, 2)
            bars = plt.bar(retention_labels, retention_values, color=['green', 'orange'])
            plt.ylabel('Retention (%)')
            plt.title('Information Retention After Controlling for Parameters')
            plt.ylim(0, 110)  # Cap at 110% to leave room for text
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        if dataset_name:
            plt.subplots_adjust(top=0.9)  # Make room for suptitle
        plt.show()
        
        # Print summary table
        self.print_summary_table(results, dataset_name)
    
    def print_summary_table(self, results: Dict, dataset_name: str = None):
        """
        Print a summary table of mutual information results.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze_alpha_reliability
        dataset_name : str, optional
            Name of the dataset for header
        """
        # Print header
        print("\n" + "="*65)
        if dataset_name:
            print(f"MUTUAL INFORMATION ANALYSIS SUMMARY: {dataset_name}")
        else:
            print("MUTUAL INFORMATION ANALYSIS SUMMARY")
        print("="*65)
        
        # Print dataset information
        print(f"Number of participants: {self.n_samples}")
        
        if 'dimensionality' in results:
            print(f"Threshold variables: {results['dimensionality']['n_threshold_vars']}")
            print(f"Non-decision time variables: {results['dimensionality']['n_ndt_vars']}")
        print("-"*65)
        
        # Print MI results table header
        print(f"{'Type':<20} {'MI (bits)':<15} {'p-value':<15} {'Retention %':<15}")
        print("-"*65)
        
        # Print raw MI
        mi, p_value = results['mi']
        sig = self._get_significance_stars(p_value)
        print(f"{'Raw MI':<20} {mi:<15.3f} {p_value:<15.3f}{sig} {'N/A':<15}")
        
        # Print conditional MI results
        if 'conditional_mi' in results:
            for var_type, (cmi, p_value) in results['conditional_mi'].items():
                retention = results['retention'][var_type]
                sig = self._get_significance_stars(p_value)
                print(f"{'CMI ('+ var_type + ')':<20} {cmi:<15.3f} {p_value:<15.3f}{sig} {retention:<15.1f}")
        
        print("="*65)
        print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        print("MI = Mutual Information, CMI = Conditional Mutual Information")
        print("Retention % = Percentage of MI retained after controlling for variables")
    
    def _get_significance_stars(self, p_value):
        """Helper method to get significance stars for p-values"""
        if p_value < 0.001:
            return " ***"
        elif p_value < 0.01:
            return " **"
        elif p_value < 0.05:
            return " *"
        else:
            return ""

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
    analyzer = MutualInformationAnalyzer(df_merged)
    
    # Define control variables
    control_vars = ['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
    
    # Run analysis
    results = analyzer.analyze_alpha_reliability(
        feature='alpha_s1',
        target='alpha_s2',
        control_vars=control_vars,
        alpha=0.5,
        bandwidth=0.2
    )
    
    # Run bandwidth sensitivity analysis
    sensitivity_results = analyzer.bandwidth_sensitivity_analysis(
        x=analyzer.data['alpha_s1'].values,
        y=analyzer.data['alpha_s2'].values
    )
    
    # Plot bandwidth sensitivity
    analyzer.plot_bandwidth_sensitivity(sensitivity_results)
    
    # Plot results
    analyzer.plot_results(results, dataset_name="My Dataset")
