import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

def plot_log_log(clim_file, nlim_file, switch_file, output_file='log_log_plot.png', min_value=1e-6):
    """
    Create a log-log plot of Delta S vs Non-Additivity.
    
    Parameters:
    -----------
    clim_file : str
        Path to the Clim fitness data file (CSV)
    nlim_file : str
        Path to the Nlim fitness data file (CSV)
    switch_file : str
        Path to the Switch fitness data file (CSV)
    output_file : str
        Path to save the output plot
    min_value : float
        Minimum value to use for log transformation (to avoid log(0))
    """
    # Read the CSV files
    clim_data = pd.read_csv(clim_file)
    nlim_data = pd.read_csv(nlim_file)
    switch_data = pd.read_csv(switch_file)
    
    # Print dataset sizes
    print(f"Clim dataset: {len(clim_data)} rows")
    print(f"Nlim dataset: {len(nlim_data)} rows")
    print(f"Switch dataset: {len(switch_data)} rows")
    
    # Check if datasets are aligned
    min_rows = min(len(clim_data), len(nlim_data), len(switch_data))
    if len(clim_data) != len(nlim_data) or len(clim_data) != len(switch_data):
        print(f"Warning: Datasets have different numbers of rows. Using first {min_rows} rows.")
    
    # Make sure we use only the rows that exist in all datasets
    clim_data = clim_data.iloc[:min_rows]
    nlim_data = nlim_data.iloc[:min_rows]
    switch_data = switch_data.iloc[:min_rows]
    
    # Use the known column for all fitness values
    clim_fitness = clim_data['Estimated_Fitness']
    nlim_fitness = nlim_data['Estimated_Fitness']
    switch_fitness = switch_data['Estimated_Fitness']
    
    # Calculate average of Clim and Nlim fitness
    avg_fitness = (clim_fitness + nlim_fitness) / 2
    
    # Calculate delta S and non-additivity
    delta_s = np.abs(clim_fitness - nlim_fitness)
    non_additivity = np.abs(switch_fitness - avg_fitness)
    
    # Create a DataFrame for plotting
    combined_data = pd.DataFrame({
        'Delta_S': delta_s,
        'Non_Additivity': non_additivity
    })
    
    # Filter out any NaN values
    combined_data = combined_data.dropna()
    
    # Replace zeros or very small values with min_value to avoid log(0) issues
    combined_data['Delta_S'] = combined_data['Delta_S'].apply(lambda x: max(x, min_value))
    combined_data['Non_Additivity'] = combined_data['Non_Additivity'].apply(lambda x: max(x, min_value))
    
    # Calculate correlations on log-transformed data
    log_delta_s = np.log10(combined_data['Delta_S'])
    log_non_additivity = np.log10(combined_data['Non_Additivity'])
    
    pearson_corr, pearson_p = pearsonr(log_delta_s, log_non_additivity)
    spearman_corr, spearman_p = spearmanr(combined_data['Delta_S'], combined_data['Non_Additivity'])
    
    print(f"Pearson correlation on log-transformed data: {pearson_corr:.3f} (p={pearson_p:.2e})")
    print(f"Spearman rank correlation: {spearman_corr:.3f} (p={spearman_p:.2e})")
    
    # Create the log-log plot
    plt.figure(figsize=(10, 8))
    
    # Set grid styling for log scales
    plt.grid(True, which="both", ls="-", alpha=0.7)
    
    # Create the scatter plot with log scales on both axes
    plt.loglog(
        combined_data['Delta_S'], 
        combined_data['Non_Additivity'],
        'o',  # Circle markers
        alpha=0.6,
        markersize=5,
        markeredgewidth=0.5,
        markeredgecolor='w'
    )
    
    # Add a power law fit (straight line on log-log plot)
    # y = A * x^b  ->  log(y) = log(A) + b*log(x)
    z = np.polyfit(log_delta_s, log_non_additivity, 1)
    b = z[0]  # Power law exponent
    A = 10**z[1]  # Coefficient
    
    # Generate points for the fit line
    x_range = np.logspace(
        np.log10(combined_data['Delta_S'].min()),
        np.log10(combined_data['Delta_S'].max()),
        100
    )
    y_fit = A * x_range**b
    
    plt.loglog(x_range, y_fit, 'r-', linewidth=2)
    
    # Customize the plot
    plt.xlabel('Delta S = |Clim Fitness - Nlim Fitness| (log scale)', fontsize=14)
    plt.ylabel('Non-Additivity = |Switch Fitness - Average Fitness| (log scale)', fontsize=14)
    plt.title(f'Log-Log Plot: Delta S vs Non-Additivity\nPower Law Fit: y = {A:.3e} * x^{b:.3f}', fontsize=16)
    
    # Add a text box with statistics
    stats_text = (
        f'Statistics:\n'
        f'Points: {len(combined_data)}\n'
        f'Log-Log Pearson: {pearson_corr:.3f} (p={pearson_p:.2e})\n'
        f'Spearman rank: {spearman_corr:.3f} (p={spearman_p:.2e})\n'
        f'Power law exponent: {b:.3f}'
    )
    
    plt.text(
        0.02, 0.02, stats_text, 
        transform=plt.gca().transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
        fontsize=12
    )
    
    # Add a line of equality (y = x)
    max_val = max(combined_data['Delta_S'].max(), combined_data['Non_Additivity'].max())
    min_val = min(combined_data['Delta_S'].min(), combined_data['Non_Additivity'].min())
    plt.loglog([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x')
    plt.legend(loc='lower right')
    
    # Add minor grid lines
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Log-Log plot saved to {output_file}')
    
    # Create a density hexbin version of the plot for better visualization of point density
    plt.figure(figsize=(10, 8))
    
    # Create a hexbin plot with log scales
    h = plt.hexbin(
        log_delta_s, 
        log_non_additivity, 
        gridsize=50, 
        bins='log',
        cmap='viridis'
    )
    
    # Add a colorbar
    cbar = plt.colorbar(h)
    cbar.set_label('log10(count)')
    
    # Add the power law fit line
    plt.plot(np.log10(x_range), np.log10(y_fit), 'r-', linewidth=2)
    
    # Customize the plot
    plt.xlabel('log10(Delta S)', fontsize=14)
    plt.ylabel('log10(Non-Additivity)', fontsize=14)
    plt.title(f'Log-Log Density Plot: Delta S vs Non-Additivity\nPower Law Fit: y = {A:.3e} * x^{b:.3f}', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_density.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Log-Log density plot saved to {output_file.replace(".png", "_density.png")}')

if __name__ == "__main__":
    # File paths
    clim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Clim_rep3_FitSeq.csv"
    nlim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Nlim_rep3_FitSeq.csv" 
    switch_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Switch_rep3_FitSeq.csv"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory/"
    output_file = os.path.join(output_dir, "delta_vs_nonadditivity_loglog_rep3.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate plots
    print("Generating log-log plots of Delta S vs Non-Additivity...")
    plot_log_log(clim_file, nlim_file, switch_file, output_file)
    print("\nPlots generated successfully!")
