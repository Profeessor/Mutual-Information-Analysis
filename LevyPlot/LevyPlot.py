#!/usr/bin/env python
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import sys

# Custom alpha-stable random number generator
def alpha_stable_random(alpha, beta=0, n=1):
    """Generate random samples from an alpha-stable distribution"""
    # Generate uniform random variables
    u = np.random.uniform(-np.pi/2, np.pi/2, n)
    w = np.random.exponential(1, n)
    
    # Transform to alpha-stable
    if alpha == 1:
        # Special case for alpha = 1
        x = (2/np.pi) * ((np.pi/2 + beta*u) * np.tan(u) - 
                         beta * np.log(w * np.cos(u) / (np.pi/2 + beta*u)))
    else:
        # General case
        a_term = np.sin(alpha * (u + (beta * np.pi/(2*alpha))))
        b_term = (np.cos(u))**(1/alpha)
        c_term = np.cos(u - alpha * (u + (beta * np.pi/(2*alpha)))) / w
        
        x = a_term / b_term * (c_term)**((1-alpha)/alpha)
        
        # Adjust for alpha != 1
        if alpha != 1:
            x = x - beta * np.tan(np.pi * alpha / 2)
    
    return x[0] if n == 1 else x

def visualize_levy_flight_model():
    """Visualize the Lévy flight model with different alpha values."""
    
    print("Starting Lévy flight model simulation...")
    start_time = time.time()
    
    alpha_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    drift = 1.0         # Drift rate
    boundary = 2.0      # Upper threshold (lower threshold at 0)
    n_trials = 100000    # Number of trials to simulate for distribution
    max_time = 5.0      # Maximum simulation time
    dt = 0.001          # Time step
    ndt = 0.1           # Non-decision time
    
    # Results containers
    correct_rts = {alpha: [] for alpha in alpha_values}
    error_rts = {alpha: [] for alpha in alpha_values}
    sample_trajectories = {alpha: None for alpha in alpha_values}
    
    # For visualizing trajectories
    def simulate_trajectory(alpha, v, a, max_steps=5000):
        position = np.zeros(max_steps+1)
        position[0] = 0.5 * a  # Start in the middle
        
        for step in range(max_steps):
            # Generate alpha-stable noise using our custom function
            noise = alpha_stable_random(alpha) * dt**(1/alpha)
            
            # Update position
            position[step+1] = position[step] + v * dt + noise
            
            # Check if boundary is reached
            if position[step+1] >= a or position[step+1] <= 0:
                return position[:step+2], step+1
        
        return position, max_steps
    
    # Simulate decision-time distributions for all alpha values
    for alpha in alpha_values:
        print(f"Simulating alpha = {alpha}")
        sys.stdout.flush()  # Ensure output is flushed to the log in background mode
        
        # Generate sample trajectories
        trajectory, steps = simulate_trajectory(alpha, drift, boundary)
        times = np.arange(len(trajectory)) * dt
        sample_trajectories[alpha] = (times, trajectory)
        
        # Generate many trials for RT distributions
        for j in range(n_trials):
            if j % 5000 == 0:
                print(f"  Alpha {alpha}: {j}/{n_trials} trials completed")
                sys.stdout.flush()
            
            # For distributions, use same alpha but add a bit of randomness to other parameters
            v = drift * (1 + 0.1 * np.random.randn())
            
            # Generate noise increment from alpha-stable distribution
            position = 0.5 * boundary
            for step in range(int(max_time/dt)):
                noise = alpha_stable_random(alpha) * dt**(1/alpha)
                position += v * dt + noise
                position = np.clip(position, 0, boundary)  # Clip to boundaries
                
                if position >= boundary:  # Upper boundary (correct)
                    correct_rts[alpha].append(step * dt + ndt)
                    break
                elif position <= 0:  # Lower boundary (error)
                    error_rts[alpha].append(step * dt + ndt)
                    break
    
    print("Simulation completed, generating plot...")
    sys.stdout.flush()
    
    # Calculate maximum densities for scaling purposes
    max_correct_density = 0
    max_error_density = 0
    
    # Kernel density estimation
    from scipy import stats
    
    for alpha in alpha_values:
        if correct_rts[alpha]:
            # Use custom bandwidth for smoother KDE
            density = stats.gaussian_kde(correct_rts[alpha], bw_method=0.2)
            xs = np.linspace(0, max_time, 2000)  # More points for smoother curve
            max_density = max(density(xs))
            max_correct_density = max(max_correct_density, max_density)
            
        if error_rts[alpha]:
            # Use custom bandwidth for smoother KDE
            density = stats.gaussian_kde(error_rts[alpha], bw_method=0.2)
            xs = np.linspace(0, max_time, 2000)  # More points for smoother curve
            max_density = max(density(xs))
            max_error_density = max(max_error_density, max_density)
    
    # Plotting with perfect alignment
    plt.figure(figsize=(10, 12))
    
    # Create GridSpec with zero spacing between subplots
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0)
    
    # Custom colors to match the image (from blue to red)
    colors = ['#5000FF', '#7030FF', '#9060FF', '#C060A0', '#E04060', '#FF0000']
    
    # Top panel: Correct RTs
    ax1 = plt.subplot(gs[0])
    for i, alpha in enumerate(alpha_values):
        if correct_rts[alpha]:
            density = stats.gaussian_kde(correct_rts[alpha], bw_method=0.2)
            xs = np.linspace(0, max_time, 2000)
            ax1.plot(xs, density(xs), color=colors[i], label=f"α = {alpha}", linewidth=1.5)
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(0, max_correct_density)  # Set explicit y-limits
    ax1.set_ylabel('CRT Density')
    # Remove ticks and labels
    ax1.set_xticks([])  # Remove x-ticks
    ax1.set_yticks([])  # Remove y-ticks
    ax1.legend(loc='upper right', frameon=True, fancybox=True,fontsize=16)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)  # Hide bottom spine to connect to middle panel
    
    # Middle panel: Sample trajectories
    ax2 = plt.subplot(gs[1])
    for i, alpha in enumerate(alpha_values):
        times, positions = sample_trajectories[alpha]
        ax2.plot(times, positions, color=colors[i], linewidth=1.0)
    ax2.axhline(y=boundary, color='black', linestyle='-', alpha=0.2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax2.set_xlim(0, 2)  # Focus on the first 2 seconds
    ax2.set_ylim(0, boundary)  # Ensure y-axis exactly matches the boundaries
    ax2.set_ylabel('Trajectories')
    # Remove ticks and labels
    ax2.set_xticks([])  # Remove x-ticks
    ax2.set_yticks([])  # Remove y-ticks
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)  # Hide bottom spine to connect to bottom panel
    
    # Bottom panel: Error RTs (properly rotated 180 degrees)
    ax3 = plt.subplot(gs[2])
    for i, alpha in enumerate(alpha_values):
        if error_rts[alpha]:
            density = stats.gaussian_kde(error_rts[alpha], bw_method=0.2)
            xs = np.linspace(0, max_time, 2000)
            # Plot with negative values for proper 180-degree rotation over x-axis
            ax3.plot(xs, -density(xs), color=colors[i], linewidth=1.5)
    ax3.set_xlim(0, max_time)
    # Set y-limits to be exactly the negative of max density so it aligns perfectly
    ax3.set_ylim(-max_error_density, 0)
    # Remove ticks and labels
    ax3.set_xticks([])  # Remove x-ticks
    ax3.set_yticks([])  # Remove y-ticks
    # Remove the x-axis label
    ax3.set_xlabel('')
    ax3.set_ylabel('ERT Density')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)  # Hide bottom axis line
    
    # Adjust layout for perfect alignment
    plt.tight_layout()
    
    # Make post-layout adjustments to ensure perfect alignment
    # This moves the subplots closer together, eliminating any remaining gaps
    plt.subplots_adjust(hspace=0)
    
    # Save the figure without displaying it (good for background processes)
    plt.savefig('levy_flight_model_PY.png', dpi=300)
    
    # Don't call plt.show() in background mode
    
    elapsed_time = time.time() - start_time
    print(f"Completed! Plot saved as 'levy_flight_model_PY.png'")
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")
    
    return "Simulation complete! Results saved as 'levy_flight_model_PY.png'"

if __name__ == "__main__":
    # This allows the script to be run directly from the command line
    visualize_levy_flight_model()