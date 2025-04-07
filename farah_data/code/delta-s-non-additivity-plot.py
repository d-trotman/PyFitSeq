import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import os

def plot_delta_vs_nonadditivity(clim_file, nlim_file, switch_file, output_file='delta_vs_nonadditivity.png', axis_limit=None):
    """
    Plot delta S (|Clim - Nlim|) vs non-additivity (|Switch - avg(Clim,Nlim)|),
    centered at the origin (0,0).
    
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
    axis_limit : float or None
        The limit for both axes (±axis_limit). If None, will be determined from data.
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
        'Non_Additivity': non_additivity,
        'Clim_Fitness': clim_fitness,
        'Nlim_Fitness': nlim_fitness,
        'Switch_Fitness': switch_fitness,
        'Avg_Fitness': avg_fitness
    })
    
    # Filter out any NaN values
    combined_data = combined_data.dropna()
    
    # Calculate correlation
    correlation, p_value = pearsonr(combined_data['Delta_S'], combined_data['Non_Additivity'])
    
    # If axis_limit is not specified, determine from data
    if axis_limit is None:
        # Find the maximum value to set axis limits
        max_val = max(
            combined_data['Delta_S'].max(),
            combined_data['Non_Additivity'].max()
        )
        # Add 10% padding
        axis_limit = max_val * 1.1
        print(f"Automatically determined axis limit: {axis_limit:.4f}")
    else:
        # Filter data to the specified axis limit
        combined_data = combined_data[
            (combined_data['Delta_S'] <= axis_limit) & 
            (combined_data['Non_Additivity'] <= axis_limit)
        ]
        print(f"Data points within axis limits (±{axis_limit}): {len(combined_data)}")
    
    # Get ranges for plotting
    delta_min = combined_data['Delta_S'].min()
    delta_max = combined_data['Delta_S'].max()
    nonadd_min = combined_data['Non_Additivity'].min()
    nonadd_max = combined_data['Non_Additivity'].max()
    
    print(f"Delta S range: {delta_min:.4f} to {delta_max:.4f}")
    print(f"Non-Additivity range: {nonadd_min:.4f} to {nonadd_max:.4f}")
    
    # Create scatterplot
    plt.figure(figsize=(10, 8))
    
    # Set grid styling
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot
    plt.scatter(
        combined_data['Delta_S'],
        combined_data['Non_Additivity'],
        alpha=0.6,
        s=30,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Add a regression line
    if len(combined_data) > 1:
        x = combined_data['Delta_S']
        y = combined_data['Non_Additivity']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        # Calculate regression line using the full axis range
        x_line = np.array([0, axis_limit])
        plt.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    # Add reference lines at zero (though this doesn't make much sense for absolute values)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Customize the plot
    plt.xlabel('Delta S = |Clim Fitness - Nlim Fitness|', fontsize=14)
    plt.ylabel('Non-Additivity = |Switch Fitness - Average Fitness|', fontsize=14)
    plt.title(f'Delta S vs Non-Additivity\nCorrelation: {correlation:.3f}', fontsize=16)
    
    # Set axis limits
    plt.xlim(0, axis_limit)
    plt.ylim(0, axis_limit)
    
    # Add a text box with statistics
    stats_text = (
        f'Statistics:\n'
        f'Points shown: {len(combined_data)}\n'
        f'Correlation: {correlation:.3f} (p={p_value:.3e})\n'
        f'Delta S mean: {combined_data["Delta_S"].mean():.4f}\n'
        f'Non-Additivity mean: {combined_data["Non_Additivity"].mean():.4f}'
    )
    
    plt.text(
        0.02, 0.98, stats_text, 
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
        fontsize=12
    )
    
    # Add a line of equality (y = x)
    plt.plot([0, axis_limit], [0, axis_limit], 'k--', alpha=0.5, label='y = x')
    plt.legend(loc='lower right')
    
    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Plot saved to {output_file}')
    
    # Additional analysis: sign patterns of fitness values
    print("\nAdditional Analysis of Sign Patterns:")
    
    # Count strains where Clim and Nlim have the same or different signs
    same_sign_clim_nlim = ((clim_fitness > 0) & (nlim_fitness > 0)) | ((clim_fitness < 0) & (nlim_fitness < 0))
    diff_sign_clim_nlim = ((clim_fitness > 0) & (nlim_fitness < 0)) | ((clim_fitness < 0) & (nlim_fitness > 0))
    
    # Count strains where Switch and Avg have the same or different signs
    same_sign_switch_avg = ((switch_fitness > 0) & (avg_fitness > 0)) | ((switch_fitness < 0) & (avg_fitness < 0))
    diff_sign_switch_avg = ((switch_fitness > 0) & (avg_fitness < 0)) | ((switch_fitness < 0) & (avg_fitness > 0))
    
    print(f"Clim and Nlim same sign: {same_sign_clim_nlim.sum()} ({same_sign_clim_nlim.mean()*100:.1f}%)")
    print(f"Clim and Nlim different signs: {diff_sign_clim_nlim.sum()} ({diff_sign_clim_nlim.mean()*100:.1f}%)")
    print(f"Switch and Avg same sign: {same_sign_switch_avg.sum()} ({same_sign_switch_avg.mean()*100:.1f}%)")
    print(f"Switch and Avg different signs: {diff_sign_switch_avg.sum()} ({diff_sign_switch_avg.mean()*100:.1f}%)")

if __name__ == "__main__":
    # File paths - using the updated Switch file
    clim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Clim_rep1_FitSeq.csv"
    nlim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Nlim_rep1_FitSeq.csv" 
    switch_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Switch_rep1_FitSeq.csv"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/"
    output_file = os.path.join(output_dir, "delta_vs_nonadditivity.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Axis limit - can be set to None to automatically determine from data
    axis_limit = 0.05  # Adjust this value to zoom in/out as needed
    
    # Generate plot
    print(f"Generating Delta S vs Non-Additivity plot (axis limits = 0 to {axis_limit})...")
    plot_delta_vs_nonadditivity(clim_file, nlim_file, switch_file, output_file, axis_limit)
    print("\nPlot generated successfully!")
