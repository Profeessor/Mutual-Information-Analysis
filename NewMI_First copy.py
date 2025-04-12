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
            Estimated mutual information
        p_value : float
            P-value from permutation test
        """
        # Calculate correlation coefficient
        r = np.corrcoef(x, y)[0, 1]
        
        # Calculate MI assuming Gaussian distribution
        mi = -0.5 * np.log(1 - r**2)
        
        # Permutation test
        n_permutations = 5000
        mi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            r_perm = np.corrcoef(x, y_perm)[0, 1]
            mi_null[i] = -0.5 * np.log(1 - r_perm**2)
        
        p_value = np.mean(mi_null >= mi)
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
            Estimated mutual information
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
        
        # Calculate MI
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_pdf[i,j] > 0:
                    mi += joint_pdf[i,j] * np.log(joint_pdf[i,j] / (x_pdf[i] * y_pdf[j]))
        
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
                        mi_perm += joint_pdf_perm[j,k] * np.log(joint_pdf_perm[j,k] / (x_pdf[j] * y_pdf[k]))
            mi_null[i] = mi_perm
        
        p_value = np.mean(mi_null >= mi)
        return mi, p_value
    
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
            Conditional mutual information
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
        
        # Calculate conditional mutual information
        cmi = np.mean(log_pxyz + log_pz - log_pxz - log_pyz)
        
        # Permutation test
        n_permutations = 5000
        cmi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            kd_xyz_perm = KernelDensity(bandwidth=0.2).fit(np.column_stack([x, y_perm, z]))
            log_pxyz_perm = kd_xyz_perm.score_samples(np.column_stack([x, y_perm, z]))
            cmi_null[i] = np.mean(log_pxyz_perm + log_pz - log_pxz - log_pyz)
        
        p_value = np.mean(cmi_null >= cmi)
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
            Conditional mutual information
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
        
        # Calculate conditional mutual information
        cmi = np.mean(log_pxyz + log_pz - log_pxz - log_pyz)
        
        # Print intermediate calculation steps
        print(f"\nIntermediate calculation steps:")
        print(f"Mean(log_pxyz + log_pz): {np.mean(log_pxyz + log_pz):.4f}")
        print(f"Mean(log_pxz + log_pyz): {np.mean(log_pxz + log_pyz):.4f}")
        print(f"Final CMI: {cmi:.4f}")
        
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
        print(f"CMI = {term1:.4f} + {term2:.4f} - {term3:.4f} - {term4:.4f} = {cmi:.4f}")
        
        # Permutation test
        n_permutations = 5000
        cmi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            kd_xyz_perm = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([x, y_perm, z_values]))
            log_pxyz_perm = kd_xyz_perm.score_samples(np.column_stack([x, y_perm, z_values]))
            cmi_null[i] = np.mean(log_pxyz_perm + log_pz - log_pxz - log_pyz)
        
        p_value = np.mean(cmi_null >= cmi)
        print(f"Permutation test p-value: {p_value:.4f}")
        
        return cmi, p_value

    def conditional_mi_threshold(self, x: np.ndarray, y: np.ndarray,
                               threshold_controls: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Calculate conditional mutual information when controlling for threshold parameters only.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable (alpha_s1)
        y : np.ndarray
            Second variable (alpha_s2)
        threshold_controls : Dict[str, np.ndarray]
            Dictionary of threshold control variables (a_s1, a_s2)
            
        Returns:
        --------
        cmi : float
            Conditional mutual information
        p_value : float
            P-value from permutation test
        """
        # Stack all threshold control variables
        z_names = list(threshold_controls.keys())
        z_values = np.column_stack([threshold_controls[name] for name in z_names])
        
        # Print diagnostic information
        print(f"\nDiagnostic Information for Conditional MI (Threshold Parameters):")
        print(f"Number of threshold variables: {len(z_names)}")
        print(f"Threshold variables: {z_names}")
        print(f"Sample size: {len(x)}")
        print(f"Dimensionality of threshold matrix: {z_values.shape}")
        
        # Correlations between threshold variables
        print("\nCorrelations between threshold variables:")
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
        
        # Calculate conditional mutual information
        cmi = np.mean(log_pxyz + log_pz - log_pxz - log_pyz)
        
        # Print intermediate calculation steps
        print(f"\nIntermediate calculation steps:")
        print(f"Mean(log_pxyz + log_pz): {np.mean(log_pxyz + log_pz):.4f}")
        print(f"Mean(log_pxz + log_pyz): {np.mean(log_pxz + log_pyz):.4f}")
        print(f"Final Threshold CMI: {cmi:.4f}")
        
        # Permutation test
        n_permutations = 5000
        cmi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            kd_xyz_perm = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([x, y_perm, z_values]))
            log_pxyz_perm = kd_xyz_perm.score_samples(np.column_stack([x, y_perm, z_values]))
            cmi_null[i] = np.mean(log_pxyz_perm + log_pz - log_pxz - log_pyz)
        
        p_value = np.mean(cmi_null >= cmi)
        print(f"Permutation test p-value: {p_value:.4f}")
        
        return cmi, p_value

    def conditional_mi_ndt(self, x: np.ndarray, y: np.ndarray,
                        ndt_controls: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Calculate conditional mutual information when controlling for non-decision time parameters only.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable (alpha_s1)
        y : np.ndarray
            Second variable (alpha_s2)
        ndt_controls : Dict[str, np.ndarray]
            Dictionary of non-decision time control variables (ndt_s1, ndt_s2)
            
        Returns:
        --------
        cmi : float
            Conditional mutual information
        p_value : float
            P-value from permutation test
        """
        # Stack all NDT control variables
        z_names = list(ndt_controls.keys())
        z_values = np.column_stack([ndt_controls[name] for name in z_names])
        
        # Print diagnostic information
        print(f"\nDiagnostic Information for Conditional MI (NDT Parameters):")
        print(f"Number of NDT variables: {len(z_names)}")
        print(f"NDT variables: {z_names}")
        print(f"Sample size: {len(x)}")
        print(f"Dimensionality of NDT matrix: {z_values.shape}")
        
        # Correlations between NDT variables
        print("\nCorrelations between NDT variables:")
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
        
        # Calculate conditional mutual information
        cmi = np.mean(log_pxyz + log_pz - log_pxz - log_pyz)
        
        # Print intermediate calculation steps
        print(f"\nIntermediate calculation steps:")
        print(f"Mean(log_pxyz + log_pz): {np.mean(log_pxyz + log_pz):.4f}")
        print(f"Mean(log_pxz + log_pyz): {np.mean(log_pxz + log_pyz):.4f}")
        print(f"Final NDT CMI: {cmi:.4f}")
        
        # Permutation test
        n_permutations = 5000
        cmi_null = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_perm = np.random.permutation(y)
            kd_xyz_perm = KernelDensity(bandwidth=bandwidth).fit(np.column_stack([x, y_perm, z_values]))
            log_pxyz_perm = kd_xyz_perm.score_samples(np.column_stack([x, y_perm, z_values]))
            cmi_null[i] = np.mean(log_pxyz_perm + log_pz - log_pxz - log_pyz)
        
        p_value = np.mean(cmi_null >= cmi)
        print(f"Permutation test p-value: {p_value:.4f}")
        
        return cmi, p_value

    def analyze_alpha_reliability(self, 
                                feature: str = 'alpha_s1',
                                target: str = 'alpha_s2',
                                control_vars: List[str] = None,
                                primary_method: str = 'adaptive_binning_mi',
                                optimal_bandwidth: float = None) -> Dict:
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
        optimal_bandwidth : float, optional
            The bandwidth to use for shrinkage_mi if provided
            
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
        
        # Use optimal bandwidth if provided
        if optimal_bandwidth is not None and primary_method == 'shrinkage_mi':
            results['shrinkage_mi'] = self.shrinkage_mi(x, y, bandwidth=optimal_bandwidth)
            results['optimal_bandwidth'] = optimal_bandwidth
        else:
            results['shrinkage_mi'] = self.shrinkage_mi(x, y)
        
        # Store the primary method for reference
        results['primary_method'] = primary_method
        
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
            
            # Separate threshold and non-decision time variables
            threshold_vars = [var for var in control_vars if 'a_' in var]
            ndt_vars = [var for var in control_vars if 'ndt_' in var]
            
            # Calculate conditional MI for threshold variables only
            if threshold_vars:
                threshold_dict = {var: self.data[var].values for var in threshold_vars}
                results['cond_mi_threshold'], results['p_value_threshold'] = self.conditional_mi_threshold(x, y, threshold_dict)
                results['retention_threshold'] = (results['cond_mi_threshold'] / raw_mi) * 100
            
            # Calculate conditional MI for NDT variables only
            if ndt_vars:
                ndt_dict = {var: self.data[var].values for var in ndt_vars}
                results['cond_mi_ndt'], results['p_value_ndt'] = self.conditional_mi_ndt(x, y, ndt_dict)
                results['retention_ndt'] = (results['cond_mi_ndt'] / raw_mi) * 100
            
            # Calculate overall retention percentage using the selected primary method
            results['retention_all'] = (results['cond_mi_all'] / raw_mi) * 100
        
        return results
    
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
        plt.title('Alpha Parameters Across Sessions')
        
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
        
        plt.bar(mi_labels, mi_values)
        plt.title('Mutual Information Estimates')
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
            plt.title(f'Breakdown of Alpha Reliability\n(Based on {primary_method_name} MI)', fontsize=16)
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
                        plt.title(f'Breakdown of Alpha Reliability\n(Based on {method_name} MI)', fontsize=16)
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
                plt.title(f'Information Retention After Controlling for Parameters\n(Based on {primary_method_name} MI)', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.ylim(0, max(max(retention_values) * 1.1, 100))
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.show()
                
    def plot_parameter_specific_retentions(self, results: Dict, feature: str = 'alpha_s1', 
                               target: str = 'alpha_s2', dataset_name: str = None) -> None:
        """
        Create pie charts for threshold-only and NDT-only retention percentages.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze_alpha_reliability
        feature : str
            Column name for alpha from session 1
        target : str
            Column name for alpha from session 2
        dataset_name : str, optional
            Name of the dataset to include in the title
        """
        # Check if we have the necessary results
        if 'retention_threshold' not in results or 'retention_ndt' not in results:
            print("Error: Results do not contain threshold and NDT retention values.")
            return
        
        primary_method_name = results['primary_method'].replace('_mi', '').title()
        dataset_info = f" - {dataset_name}" if dataset_name else ""
        
        # Create a figure with two pie charts side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Common pie chart settings
        labels = ['Unique to Alpha', 'Shared with Parameters']
        colors = ['#2ca02c', '#ff7f0e']  # Green and orange
        
        # Plot Threshold-only pie chart
        threshold_retention = results['retention_threshold']
        # Cap retention at 100% for visualization purposes
        threshold_retention = min(threshold_retention, 100)
        threshold_sizes = [threshold_retention, 100-threshold_retention]
        
        axes[0].pie(threshold_sizes, labels=None, colors=colors, autopct='%1.1f%%', 
                startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        axes[0].axis('equal')
        axes[0].set_title(f'Controlling for Threshold Only\nRetention: {threshold_retention:.1f}%', fontsize=14)
        
        # Plot NDT-only pie chart
        ndt_retention = results['retention_ndt']
        # Cap retention at 100% for visualization purposes
        ndt_retention = min(ndt_retention, 100)
        ndt_sizes = [ndt_retention, 100-ndt_retention]
        
        axes[1].pie(ndt_sizes, labels=None, colors=colors, autopct='%1.1f%%', 
                startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        axes[1].axis('equal')
        axes[1].set_title(f'Controlling for NDT Only\nRetention: {ndt_retention:.1f}%', fontsize=14)
        
        # Create a legend for both plots
        fig.legend(labels, loc='center', bbox_to_anchor=(0.5, 0.1), ncol=2, fontsize=12)
        
        # Add overall title
        plt.suptitle(f'Breakdown of Alpha Reliability{dataset_info}\n(Based on {primary_method_name} MI)', 
                  fontsize=16, y=0.95)
        
        # Add text box with p-values
        if 'p_value_threshold' in results and 'p_value_ndt' in results:
            threshold_p = results['p_value_threshold']
            ndt_p = results['p_value_ndt']
            significance = []
            if threshold_p < 0.05:
                significance.append(f"Threshold MI significant (p={threshold_p:.4f})")
            else:
                significance.append(f"Threshold MI not significant (p={threshold_p:.4f})")
            if ndt_p < 0.05:
                significance.append(f"NDT MI significant (p={ndt_p:.4f})")
            else:
                significance.append(f"NDT MI not significant (p={ndt_p:.4f})")
            
            plt.figtext(0.5, 0.02, "\n".join(significance), ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for the legend
        plt.show()
        
        # Additionally, create a bar chart comparing all retention values
        if 'retention_all' in results:
            plt.figure(figsize=(9, 5))
            retention_values = [results['retention_threshold'], results['retention_ndt'], results['retention_all']]
            bar_labels = ['Threshold Only', 'NDT Only', 'All Parameters']
            
            bars = plt.bar(bar_labels, retention_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Retention')
            plt.ylabel('Information Retention (%)', fontsize=12)
            plt.title(f'Information Retention After Controlling for Parameters{dataset_info}', fontsize=14)
            plt.ylim(0, max(max(retention_values) * 1.1, 100))
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()

    def bandwidth_sensitivity_analysis(self, x: np.ndarray, y: np.ndarray,
                                    bandwidths: List[float] = None,
                                    alpha: float = 0.5) -> Dict:
        """
        Perform sensitivity analysis for different bandwidths in shrinkage_mi method.
        
        Parameters:
        -----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
        bandwidths : List[float], optional
            List of bandwidths to test. If None, uses default range based on Silverman's rule.
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
                base_bandwidth * 3
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
    
    def find_optimal_bandwidth(self, sensitivity_results: Dict) -> Dict:
        """
        Find the optimal bandwidth based on significance and stability.
        
        Parameters:
        -----------
        sensitivity_results : Dict
            Results from bandwidth_sensitivity_analysis
            
        Returns:
        --------
        Dict
            Dictionary containing optimal bandwidth and related metrics
        """
        # Find bandwidth with lowest p-value
        min_p_idx = np.argmin(sensitivity_results['p_values'])
        most_significant_bw = sensitivity_results['bandwidths'][min_p_idx]
        
        # Calculate scores that balance significance and stability
        bandwidth_scores = []
        
        for i, h in enumerate(sensitivity_results['bandwidths']):
            # Lower p-value is better
            p_score = 1 - sensitivity_results['p_values'][i]
            
            # Stability - how close this bandwidth's MI is to neighboring bandwidths
            if i > 0 and i < len(sensitivity_results['bandwidths'])-1:
                left_diff = abs(sensitivity_results['mi_values'][i] - sensitivity_results['mi_values'][i-1])
                right_diff = abs(sensitivity_results['mi_values'][i] - sensitivity_results['mi_values'][i+1])
                stability_score = 1 - ((left_diff + right_diff) / 2) / sensitivity_results['mi_values'][i]
            else:
                stability_score = 0.5  # Penalize edge bandwidths
            
            # Combined score (significance weighted more than stability)
            combined_score = 0.7 * p_score + 0.3 * stability_score
            bandwidth_scores.append(combined_score)
        
        # Find optimal bandwidth based on combined score
        optimal_idx = np.argmax(bandwidth_scores)
        optimal_bandwidth = sensitivity_results['bandwidths'][optimal_idx]
        
        # Look for elbow point in MI values
        mi_diff = np.diff(sensitivity_results['mi_values'])
        # Find where the rate of change in MI stabilizes
        if len(mi_diff) > 1:  # Need at least 3 bandwidths for this
            stabilization_idx = np.argmin(np.abs(mi_diff - np.median(mi_diff))) + 1
            stabilized_bandwidth = sensitivity_results['bandwidths'][stabilization_idx]
        else:
            stabilized_bandwidth = optimal_bandwidth
        
        # Calculate coefficient of variation around optimal point
        nearby_indices = [i for i, h in enumerate(sensitivity_results['bandwidths']) 
                        if 0.8*optimal_bandwidth <= h <= 1.2*optimal_bandwidth]
        if len(nearby_indices) > 1:
            nearby_mi = [sensitivity_results['mi_values'][i] for i in nearby_indices]
            stability_cv = np.std(nearby_mi) / np.mean(nearby_mi)
            stability_assessment = "High" if stability_cv < 0.05 else "Moderate" if stability_cv < 0.1 else "Low"
        else:
            stability_cv = None
            stability_assessment = "Insufficient data points around optimal bandwidth"
        
        return {
            'optimal_bandwidth': optimal_bandwidth,
            'mi_value': sensitivity_results['mi_values'][optimal_idx],
            'p_value': sensitivity_results['p_values'][optimal_idx],
            'most_significant_bandwidth': most_significant_bw,
            'stabilized_bandwidth': stabilized_bandwidth,
            'stability_cv': stability_cv,
            'stability_assessment': stability_assessment,
            'bandwidth_scores': bandwidth_scores
        }

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
    
    # First run bandwidth sensitivity analysis to find optimal bandwidth
    sensitivity_results = analyzer.bandwidth_sensitivity_analysis(
        x=analyzer.data['alpha_s1'].values, 
        y=analyzer.data['alpha_s2'].values
    )

    # Find optimal bandwidth
    optimal_results = analyzer.find_optimal_bandwidth(sensitivity_results)
    optimal_bandwidth = optimal_results['optimal_bandwidth']

    # Define control variables
    control_vars = ['a_mean_s1', 'a_mean_s2', 'ndt_mean_s1', 'ndt_mean_s2']

    # Run analysis with optimal bandwidth
    results = analyzer.analyze_alpha_reliability(
        feature='alpha_s1',
        target='alpha_s2',
        control_vars=control_vars,
        primary_method='shrinkage_mi',
        optimal_bandwidth=optimal_bandwidth
    )

    # Print the new results
    print("\nConditional MI Results:")
    print(f"All Parameters: {results['cond_mi_all']:.4f} bits (p={results['p_value_all']:.4f})")
    print(f"Threshold Only: {results['cond_mi_threshold']:.4f} bits (p={results['p_value_threshold']:.4f})")
    print(f"NDT Only: {results['cond_mi_ndt']:.4f} bits (p={results['p_value_ndt']:.4f})")

    print("\nRetention Percentages:")
    print(f"All Parameters: {results['retention_all']:.1f}%")
    print(f"Threshold Only: {results['retention_threshold']:.1f}%")
    print(f"NDT Only: {results['retention_ndt']:.1f}%")

    # Plot parameter-specific retentions
    analyzer.plot_parameter_specific_retentions(results)
