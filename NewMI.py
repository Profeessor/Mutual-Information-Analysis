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

class MutualInformationAnalyzer:
    """
    A comprehensive mutual information analyzer implementing multiple estimation methods
    suitable for small datasets, specifically designed for alpha parameter analysis.
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
        
    def gaussian_mi(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Estimate mutual information assuming Gaussian distribution.
        Suitable for small samples when normality assumption is reasonable.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable (e.g., alpha_s1)
        y : np.ndarray
            Second variable (e.g., alpha_s2)
            
        Returns:
        --------
        mi : float
            Estimated mutual information (in bits)
        p_value : float
            P-value from permutation test
        """
        # Calculate correlation coefficient
        r = np.corrcoef(x, y)[0, 1]
        
        # Calculate MI assuming Gaussian distribution (in bits)
        mi = -0.5 * np.log2(1 - r**2)
        
        # Permutation test
        n_permutations = 5000
        mi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            r_perm = np.corrcoef(x, y_perm)[0, 1]
            mi_null[i] = -0.5 * np.log2(1 - r_perm**2)
        
        # More robust p-value calculation
        p_value = (np.sum(mi_null >= mi) + 1) / (n_permutations + 1)
        return mi, p_value
    
    def adaptive_binning_mi(self, x: np.ndarray, y: np.ndarray, 
                          n_bins: int = None) -> Tuple[float, float]:
        """
        Estimate mutual information using adaptive binning.
        More robust for small samples than fixed binning.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
        n_bins : int, optional
            Number of bins. If None, uses sqrt(n_samples)
            
        Returns:
        --------
        mi : float
            Estimated mutual information (in bits)
        p_value : float
            P-value from permutation test
        """
        if n_bins is None:
            n_bins = int(np.sqrt(self.n_samples))
            
        # Adaptive binning based on data distribution
        x_bins = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        y_bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        
        # Calculate joint and marginal histograms
        joint_hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        x_hist, _ = np.histogram(x, bins=x_bins)
        y_hist, _ = np.histogram(y, bins=y_bins)
        
        # Normalize histograms to probabilities
        joint_pdf = joint_hist / self.n_samples
        x_pdf = x_hist / self.n_samples
        y_pdf = y_hist / self.n_samples
        
        # Calculate MI (in bits using log2)
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_pdf[i,j] > 0:
                    mi += joint_pdf[i,j] * np.log2(joint_pdf[i,j] / (x_pdf[i] * y_pdf[j]))
        
        # Permutation test
        n_permutations = 5000
        mi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            joint_hist_perm, _, _ = np.histogram2d(x, y_perm, bins=[x_bins, y_bins])
            joint_pdf_perm = joint_hist_perm / self.n_samples
            mi_perm = 0
            for j in range(n_bins):
                for k in range(n_bins):
                    if joint_pdf_perm[j,k] > 0:
                        mi_perm += joint_pdf_perm[j,k] * np.log2(joint_pdf_perm[j,k] / (x_pdf[j] * y_pdf[k]))
            mi_null[i] = mi_perm
        
        # More robust p-value calculation
        p_value = (np.sum(mi_null >= mi) + 1) / (n_permutations + 1)
        return mi, p_value
    
    def shrinkage_mi(self, x: np.ndarray, y: np.ndarray, 
                    alpha: float = 0.5) -> Tuple[float, float]:
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
            
        Returns:
        --------
        mi : float
            Estimated mutual information (in bits)
        p_value : float
            P-value from permutation test
        """
        # Calculate empirical MI using k-NN
        k = max(3, int(np.sqrt(self.n_samples)))
        kd_x = KernelDensity(bandwidth=0.2).fit(x.reshape(-1, 1))
        kd_y = KernelDensity(bandwidth=0.2).fit(y.reshape(-1, 1))
        kd_xy = KernelDensity(bandwidth=0.2).fit(np.column_stack([x, y]))
        
        log_px = kd_x.score_samples(x.reshape(-1, 1))
        log_py = kd_y.score_samples(y.reshape(-1, 1))
        log_pxy = kd_xy.score_samples(np.column_stack([x, y]))
        
        # Convert from natural log to log2 (bits)
        mi_empirical = np.mean(log_pxy - log_px - log_py) / np.log(2)
        
        # Calculate Gaussian MI (in bits)
        r = np.corrcoef(x, y)[0, 1]
        mi_gaussian = -0.5 * np.log2(1 - r**2)
        
        # Combine estimates
        mi = alpha * mi_empirical + (1 - alpha) * mi_gaussian
        
        # Permutation test using Gaussian MI only (to avoid recursion)
        n_permutations = 5000
        mi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            r_perm = np.corrcoef(x, y_perm)[0, 1]
            mi_null[i] = -0.5 * np.log2(1 - r_perm**2)
        
        # More robust p-value calculation
        p_value = (np.sum(mi_null >= mi) + 1) / (n_permutations + 1)
        return mi, p_value
    
    def conditional_mi(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
        """
        Calculate conditional mutual information I(X;Y|Z).
        
        Parameters:
        -----------
        x : np.ndarray
            First variable (alpha_s1)
        y : np.ndarray
            Second variable (alpha_s2)
        z : np.ndarray
            Conditioning variable (threshold or NDT)
            
        Returns:
        --------
        cmi : float
            Conditional mutual information (in bits)
        p_value : float
            P-value from permutation test
        """
        # Fit KDE for all variables
        kd_x = KernelDensity(bandwidth=0.2).fit(x.reshape(-1, 1))
        kd_y = KernelDensity(bandwidth=0.2).fit(y.reshape(-1, 1))
        kd_z = KernelDensity(bandwidth=0.2).fit(z.reshape(-1, 1))
        kd_xz = KernelDensity(bandwidth=0.2).fit(np.column_stack([x, z]))
        kd_yz = KernelDensity(bandwidth=0.2).fit(np.column_stack([y, z]))
        kd_xyz = KernelDensity(bandwidth=0.2).fit(np.column_stack([x, y, z]))
        
        # Calculate log probabilities
        log_px = kd_x.score_samples(x.reshape(-1, 1))
        log_py = kd_y.score_samples(y.reshape(-1, 1))
        log_pz = kd_z.score_samples(z.reshape(-1, 1))
        log_pxz = kd_xz.score_samples(np.column_stack([x, z]))
        log_pyz = kd_yz.score_samples(np.column_stack([y, z]))
        log_pxyz = kd_xyz.score_samples(np.column_stack([x, y, z]))
        
        # Calculate conditional mutual information (convert to bits by dividing by log(2))
        cmi = np.mean(log_pxyz + log_pz - log_pxz - log_pyz) / np.log(2)
        
        # Permutation test
        n_permutations = 5000
        cmi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            kd_xyz_perm = KernelDensity(bandwidth=0.2).fit(np.column_stack([x, y_perm, z]))
            log_pxyz_perm = kd_xyz_perm.score_samples(np.column_stack([x, y_perm, z]))
            cmi_null[i] = np.mean(log_pxyz_perm + log_pz - log_pxz - log_pyz) / np.log(2)
        
        # More robust p-value calculation
        p_value = (np.sum(cmi_null >= cmi) + 1) / (n_permutations + 1)
        return cmi, p_value
    
    def conditional_mi_all(self, x: np.ndarray, y: np.ndarray, 
                          controls: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Calculate conditional mutual information when controlling for all variables simultaneously.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable (alpha_s1)
        y : np.ndarray
            Second variable (alpha_s2)
        controls : Dict[str, np.ndarray]
            Dictionary of control variables
            
        Returns:
        --------
        cmi : float
            Conditional mutual information (in bits)
        p_value : float
            P-value from permutation test
        """
        # Stack all control variables
        z_names = list(controls.keys())
        z_values = np.column_stack([controls[name] for name in z_names])
        
        # Print diagnostic information
        print(f"\nDiagnostic Information for Conditional MI (All Parameters):")
        print(f"Number of control variables: {len(z_names)}")
        print(f"Control variables: {z_names}")
        print(f"Sample size: {len(x)}")
        print(f"Dimensionality of control matrix: {z_values.shape}")
        
        # Correlations between control variables
        print("\nCorrelations between control variables:")
        for i in range(len(z_names)):
            for j in range(i+1, len(z_names)):
                corr = np.corrcoef(z_values[:,i], z_values[:,j])[0,1]
                print(f"{z_names[i]} - {z_names[j]}: r = {corr:.4f}")
        
        # Fit KDE for all variables
        bandwidth = min(0.2, 0.5 * np.power(self.n_samples, -1.0/(z_values.shape[1] + 4)))
        print(f"Using bandwidth: {bandwidth:.4f}")
        
        kd_x = KernelDensity(bandwidth=bandwidth).fit(x.reshape(-1, 1))
        kd_y = KernelDensity(bandwidth=bandwidth).fit(y.reshape(-1, 1))
        kd_z = KernelDensity(bandwidth=bandwidth).fit(z_values)
        kd_xz = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([x, z_values]))
        kd_yz = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([y, z_values]))
        kd_xyz = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([x, y, z_values]))
        
        # Calculate log probabilities
        log_px = kd_x.score_samples(x.reshape(-1, 1))
        log_py = kd_y.score_samples(y.reshape(-1, 1))
        log_pz = kd_z.score_samples(z_values)
        log_pxz = kd_xz.score_samples(np.column_stack([x, z_values]))
        log_pyz = kd_yz.score_samples(np.column_stack([y, z_values]))
        log_pxyz = kd_xyz.score_samples(np.column_stack([x, y, z_values]))
        
        # Print summary statistics of log probabilities
        print("\nSummary statistics of log probabilities:")
        print(f"log_px: mean = {np.mean(log_px):.4f}, std = {np.std(log_px):.4f}")
        print(f"log_py: mean = {np.mean(log_py):.4f}, std = {np.std(log_py):.4f}")
        print(f"log_pz: mean = {np.mean(log_pz):.4f}, std = {np.std(log_pz):.4f}")
        print(f"log_pxz: mean = {np.mean(log_pxz):.4f}, std = {np.std(log_pxz):.4f}")
        print(f"log_pyz: mean = {np.mean(log_pyz):.4f}, std = {np.std(log_pyz):.4f}")
        print(f"log_pxyz: mean = {np.mean(log_pxyz):.4f}, std = {np.std(log_pxyz):.4f}")
        
        # Calculate conditional mutual information (convert to bits by dividing by log(2))
        cmi = np.mean(log_pxyz + log_pz - log_pxz - log_pyz) / np.log(2)
        
        # Print intermediate calculation steps
        print(f"\nIntermediate calculation steps:")
        print(f"Mean(log_pxyz + log_pz): {np.mean(log_pxyz + log_pz):.4f}")
        print(f"Mean(log_pxz + log_pyz): {np.mean(log_pxz + log_pyz):.4f}")
        print(f"Final CMI: {cmi:.4f} bits")
        
        # Calculate the components that contribute to CMI
        print("\nComponents contributing to CMI:")
        term1 = np.mean(log_pxyz)
        term2 = np.mean(log_pz)
        term3 = np.mean(log_pxz)
        term4 = np.mean(log_pyz)
        print(f"Mean(log_pxyz): {term1:.4f}")
        print(f"Mean(log_pz): {term2:.4f}")
        print(f"Mean(log_pxz): {term3:.4f}")
        print(f"Mean(log_pyz): {term4:.4f}")
        print(f"CMI = ({term1:.4f} + {term2:.4f} - {term3:.4f} - {term4:.4f}) / ln(2) = {cmi:.4f} bits")
        
        # Permutation test
        n_permutations = 5000
        cmi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            kd_xyz_perm = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([x, y_perm, z_values]))
            log_pxyz_perm = kd_xyz_perm.score_samples(np.column_stack([x, y_perm, z_values]))
            cmi_null[i] = np.mean(log_pxyz_perm + log_pz - log_pxz - log_pyz) / np.log(2)
        
        # More robust p-value calculation
        p_value = (np.sum(cmi_null >= cmi) + 1) / (n_permutations + 1)
        print(f"Permutation test p-value: {p_value:.4f}")
        
        return cmi, p_value

    def bootstrap_mi_ci(self, x: np.ndarray, y: np.ndarray, 
                      method: str = 'gaussian_mi', 
                      n_bootstrap: int = 1000, 
                      alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals for mutual information estimates.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
        method : str
            MI calculation method ('gaussian_mi', 'adaptive_binning_mi', or 'shrinkage_mi')
        n_bootstrap : int
            Number of bootstrap samples
        alpha : float
            Significance level for confidence intervals (default: 0.05 for 95% CI)
            
        Returns:
        --------
        lower_ci : float
            Lower bound of confidence interval
        upper_ci : float
            Upper bound of confidence interval
        """
        # Select the MI calculation method
        if method == 'gaussian_mi':
            mi_func = lambda a, b: self.gaussian_mi(a, b)[0]
        elif method == 'adaptive_binning_mi':
            mi_func = lambda a, b: self.adaptive_binning_mi(a, b)[0]
        elif method == 'shrinkage_mi':
            mi_func = lambda a, b: self.shrinkage_mi(a, b)[0]
        else:
            raise ValueError(f"Unknown MI method: {method}")
        
        # Perform bootstrap resampling
        mi_bootstrap = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.randint(0, len(x), len(x))
            x_sample = x[indices]
            y_sample = y[indices]
            
            # Calculate MI for this bootstrap sample
            mi_bootstrap[i] = mi_func(x_sample, y_sample)
        
        # Calculate confidence intervals
        lower_ci = np.percentile(mi_bootstrap, 100 * alpha / 2)
        upper_ci = np.percentile(mi_bootstrap, 100 * (1 - alpha / 2))
        
        return lower_ci, upper_ci

    def analyze_alpha_reliability(self, 
                                feature: str = 'alpha_s1',
                                target: str = 'alpha_s2',
                                control_vars: List[str] = None,
                                primary_method: str = 'adaptive_binning_mi',
                                calculate_bootstrap_ci: bool = False) -> Dict:
        """
        Comprehensive analysis of alpha parameter reliability using multiple MI methods.
        
        Parameters:
        -----------
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        control_vars : List[str]
            Variables to control for in the analysis
        primary_method : str
            The primary MI method to use for retention calculations ('gaussian_mi', 'adaptive_binning_mi', 'shrinkage_mi')
        calculate_bootstrap_ci : bool
            Whether to calculate bootstrap confidence intervals (computationally intensive)
            
        Returns:
        --------
        results : Dict
            Dictionary containing results from all MI estimation methods
        """
        results = {}
        
        # Extract variables
        x = self.data[feature].values
        y = self.data[target].values
        
        # Calculate raw MI using different methods
        results['gaussian_mi'] = self.gaussian_mi(x, y)
        results['adaptive_binning_mi'] = self.adaptive_binning_mi(x, y)
        results['shrinkage_mi'] = self.shrinkage_mi(x, y)
        
        # Store the primary method for reference
        results['primary_method'] = primary_method
        
        # Calculate bootstrap confidence intervals if requested
        if calculate_bootstrap_ci:
            results['bootstrap_ci'] = {}
            print("\nCalculating bootstrap confidence intervals (this may take a while)...")
            for method in ['gaussian_mi', 'adaptive_binning_mi', 'shrinkage_mi']:
                lower_ci, upper_ci = self.bootstrap_mi_ci(x, y, method=method)
                results['bootstrap_ci'][method] = (lower_ci, upper_ci)
                print(f"{method.replace('_mi', '').title()} MI: 95% CI [{lower_ci:.4f}, {upper_ci:.4f}] bits")
        
        # Calculate conditional MI if control variables are provided
        if control_vars:
            results['conditional_mi'] = {}
            for var in control_vars:
                z = self.data[var].values
                results['conditional_mi'][var] = self.conditional_mi(x, y, z)
            
            # Calculate retention percentage for individual variables using the selected primary method
            raw_mi = results[primary_method][0]
            cmi_values = [v[0] for v in results['conditional_mi'].values()]
            results['retention'] = {
                var: (cmi / raw_mi) * 100 
                for var, cmi in zip(control_vars, cmi_values)
            }
            
            # Calculate conditional MI controlling for all variables simultaneously
            control_dict = {var: self.data[var].values for var in control_vars}
            results['cond_mi_all'], results['p_value_all'] = self.conditional_mi_all(x, y, control_dict)
            
            # Calculate overall retention percentage using the selected primary method
            results['retention_all'] = (results['cond_mi_all'] / raw_mi) * 100
            
            # Check for KDE dimensionality issues by comparing joint and individual controls
            self._check_kde_consistency(results, control_vars, raw_mi)
        
        return results
    
    def _check_kde_consistency(self, results: Dict, control_vars: List[str], raw_mi: float) -> None:
        """
        Check for potential high-dimensional KDE issues by comparing joint and individual control results.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from analyze_alpha_reliability
        control_vars : List[str]
            List of control variables
        raw_mi : float
            Raw mutual information value
        """
        # Get the minimum retention from individual controls
        min_individual_retention = min(results['retention'].values())
        joint_retention = results['retention_all']
        
        # Compare joint vs individual - if joint is much lower, it might indicate KDE issues
        retention_ratio = joint_retention / min_individual_retention if min_individual_retention > 0 else float('inf')
        
        print("\nKDE Dimensionality Consistency Check:")
        print(f"Minimum retention from individual controls: {min_individual_retention:.2f}%")
        print(f"Joint retention (all controls): {joint_retention:.2f}%")
        print(f"Ratio (joint/min individual): {retention_ratio:.2f}")
        
        if retention_ratio < 0.5 and len(control_vars) > 2:
            print("\nWARNING: Joint CMI shows much lower retention than individual controls.")
            print("This might indicate high-dimensional KDE estimation issues.")
            print("Consider interpretation with caution due to the curse of dimensionality.")
            print("Recommendation: If sample size is small relative to the number of control variables,")
            print("consider alternative approaches like stepwise conditioning or regression-based methods.")
        else:
            print("\nNo major inconsistency detected between joint and individual CMI estimates.")
    
    def plot_results(self, results: Dict, feature: str = 'alpha_s1', 
                    target: str = 'alpha_s2') -> None:
        """
        Create visualizations of the MI analysis results.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze_alpha_reliability
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        """
        # Plot 1: Scatter plot and MI estimates
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Scatter plot with correlation
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.data, x=feature, y=target, alpha=0.6)
        plt.title(f'Alpha Parameters Across Sessions (n={self.n_samples})')
        
        # Plot 2: MI estimates comparison
        plt.subplot(1, 2, 2)
        mi_values = [results['gaussian_mi'][0], 
                    results['adaptive_binning_mi'][0],
                    results['shrinkage_mi'][0]]
        mi_labels = ['Gaussian MI', 'Adaptive Binning MI', 'Shrinkage MI']
        
        # Add p-values in the labels
        p_values = [results['gaussian_mi'][1], 
                   results['adaptive_binning_mi'][1],
                   results['shrinkage_mi'][1]]
        
        mi_labels = [f"{label}\n(p={p:.4f})" for label, p in zip(mi_labels, p_values)]
        
        # Plot bars
        bars = plt.bar(mi_labels, mi_values)
        
        # Add error bars for bootstrap CIs if available
        if 'bootstrap_ci' in results:
            # Create properly formatted yerr for matplotlib
            yerr = np.zeros((2, len(mi_values)))
            
            for i, method in enumerate(['gaussian_mi', 'adaptive_binning_mi', 'shrinkage_mi']):
                mi_value = mi_values[i]
                lower_ci, upper_ci = results['bootstrap_ci'][method]
                yerr[0, i] = mi_value - lower_ci  # lower error
                yerr[1, i] = upper_ci - mi_value  # upper error
            
            # Add error bars
            x_positions = range(len(mi_labels))
            plt.errorbar(x_positions, mi_values, yerr=yerr, fmt='none', color='black', capsize=5)
            
            # Update the title if bootstrap CIs are shown
            plt.title(f'Mutual Information Estimates with 95% CIs (n={self.n_samples})')
        else:
            plt.title(f'Mutual Information Estimates (n={self.n_samples})')
            
        plt.ylabel('Mutual Information (bits)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Check if we have conditional MI results
        if 'retention_all' in results:
            # Print the summary for clarity
            print("\nMutual Information Summary (All Parameters):")
            print(f"Primary Method: {results['primary_method']}")
            primary_raw_mi = results[results['primary_method']][0]
            print(f"Raw MI: {primary_raw_mi:.4f} bits (p={results[results['primary_method']][1]:.4f})")
            print(f"Conditional MI (all parameters): {results['cond_mi_all']:.4f} bits (p={results['p_value_all']:.4f})")
            print(f"Overall Retention: {results['retention_all']:.1f}%")
            
            primary_method_name = results['primary_method'].replace('_mi', '').title()
            
            # Create a pie chart showing the breakdown of information for the primary method
            plt.figure(figsize=(8, 8))
            labels = ['Unique to Alpha', 'Shared with\nThreshold & NDT']
            retention = results['retention_all']
            # Cap retention at 100% for visualization purposes
            retention = min(retention, 100)
            sizes = [retention, 100-retention]
            colors = ['#2ca02c', '#ff7f0e']  # Green and orange
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                    startangle=90, textprops={'fontsize': 14})
            plt.axis('equal')
            plt.title(f'Breakdown of Alpha Reliability (n={self.n_samples})\n(Based on {primary_method_name} MI in bits)', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            # Create separate pie charts for Adaptive Binning and Shrinkage MI if both are present
            methods_to_plot = ['adaptive_binning_mi', 'shrinkage_mi']
            if all(method in results for method in methods_to_plot) and 'retention_all' in results:
                for method in methods_to_plot:
                    if method != results['primary_method']:  # Skip if it's already the primary method
                        method_raw_mi = results[method][0]
                        method_retention = (results['cond_mi_all'] / method_raw_mi) * 100 if method_raw_mi > 0 else 0
                        method_retention = min(method_retention, 100)
                        
                        method_name = method.replace('_mi', '').title()
                        
                        plt.figure(figsize=(8, 8))
                        sizes = [method_retention, 100-method_retention]
                        
                        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                startangle=90, textprops={'fontsize': 14})
                        plt.axis('equal')
                        plt.title(f'Breakdown of Alpha Reliability (n={self.n_samples})\n(Based on {method_name} MI in bits)', fontsize=16)
                        plt.tight_layout()
                        plt.show()
            
            # Bar chart showing individual parameter contributions
            if 'retention' in results:
                plt.figure(figsize=(10, 6))
                vars_list = list(results['retention'].keys())
                retention_values = [results['retention'][var] for var in vars_list]
                
                # Get display names for variables
                display_names = []
                for var in vars_list:
                    if 'a_mean' in var:
                        display_names.append(f"Threshold ({var})")
                    elif 'ndt' in var:
                        display_names.append(f"Non-Decision Time ({var})")
                    else:
                        display_names.append(var)
                
                # Add the "all parameters" bar
                display_names.append("All Parameters")
                retention_values.append(results['retention_all'])
                
                # Create a bar chart with error bars
                plt.figure(figsize=(12, 6))
                bars = plt.bar(display_names, retention_values)
                
                # Color the "All Parameters" bar differently
                bars[-1].set_color('darkred')
                
                plt.axhline(y=100, color='blue', linestyle='--', alpha=0.5, label='100% Retention')
                plt.ylabel('Information Retention (%)', fontsize=12)
                plt.title(f'Information Retention After Controlling for Parameters (n={self.n_samples})\n(Based on {primary_method_name} MI in bits)', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.ylim(0, max(max(retention_values) * 1.1, 100))
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
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
    analyzer = MutualInformationAnalyzer(df_merged)
    
    # Define control variables
    control_vars = ['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']
    
    # Run analysis
    results = analyzer.analyze_alpha_reliability(
        feature='alpha_s1',
        target='alpha_s2',
        control_vars=control_vars
    )
    
    # Plot results
    analyzer.plot_results(results)
    
    # Print results
    print("\nMutual Information Estimates:")
    print(f"Gaussian MI: {results['gaussian_mi'][0]:.4f} bits (p={results['gaussian_mi'][1]:.4f})")
    print(f"Adaptive Binning MI: {results['adaptive_binning_mi'][0]:.4f} bits (p={results['adaptive_binning_mi'][1]:.4f})")
    print(f"Shrinkage MI: {results['shrinkage_mi'][0]:.4f} bits (p={results['shrinkage_mi'][1]:.4f})")
