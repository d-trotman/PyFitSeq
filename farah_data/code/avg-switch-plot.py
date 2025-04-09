import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import os

def plot_avg_vs_switch(clim_file, nlim_file, switch_file, output_file='avg_vs_switch.png', axis_limit=0.2):
    """
    Plot the average of Clim and Nlim fitness (x-axis) against the Switch fitness (y-axis),
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
    axis_limit : float
        The limit for the x-axis (±axis_limit)
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
    
    # Create a DataFrame for plotting
    combined_data = pd.DataFrame({
        'Avg_Fitness': avg_fitness,
        'Switch_Fitness': switch_fitness
    })
    
    # Filter out any NaN values
    combined_data = combined_data.dropna()
    
    # Calculate correlation
    correlation, p_value = pearsonr(combined_data['Avg_Fitness'], combined_data['Switch_Fitness'])
    
    # Filter data to the specified x-axis range
    plotting_data = combined_data[
        (combined_data['Avg_Fitness'] >= -axis_limit) & 
        (combined_data['Avg_Fitness'] <= axis_limit)
    ]
    
    print(f"Data points within axis limits (±{axis_limit}): {len(plotting_data)}")
    
    # Get fitness ranges for plotting
    switch_min = plotting_data['Switch_Fitness'].min()
    switch_max = plotting_data['Switch_Fitness'].max()
    avg_min = plotting_data['Avg_Fitness'].min()
    avg_max = plotting_data['Avg_Fitness'].max()
    
    print(f"Avg Fitness range: {avg_min:.4f} to {avg_max:.4f}")
    print(f"Switch Fitness range: {switch_min:.4f} to {switch_max:.4f}")
    
    # Create scatterplot
    plt.figure(figsize=(10, 8))
    
    # Set grid styling
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot
    plt.scatter(
        plotting_data['Avg_Fitness'],
        plotting_data['Switch_Fitness'],
        alpha=0.6,
        s=30,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Add a regression line
    if len(plotting_data) > 1:
        x = plotting_data['Avg_Fitness']
        y = plotting_data['Switch_Fitness']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        # Calculate regression line using the full axis range
        x_line = np.array([-axis_limit, axis_limit])
        plt.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    # Add reference lines at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Customize the plot
    plt.xlabel('Average Fitness (Clim + Nlim)/2', fontsize=14)
    plt.ylabel('Switch Fitness', fontsize=14)
    plt.title(f'Fitness Comparison: Average (Clim,Nlim) vs Switch\nCorrelation: {correlation:.3f}', fontsize=16)
    
    # Set x-axis limits
    plt.xlim(-axis_limit, axis_limit)
    
    # Find the maximum absolute value in the y-direction to make the plot symmetric
    y_max_abs = max(abs(switch_min), abs(switch_max))
    # Add 10% padding
    y_max_abs *= 1.1
    
    # Set symmetric y-axis limits centered at 0
    plt.ylim(-y_max_abs, y_max_abs)
    
    # Verify the center
    ax = plt.gca()
    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    print(f"X-axis center: {x_center} (should be close to 0)")
    print(f"Y-axis center: {y_center} (should be close to 0)")
    
    # Add a text box with statistics
    stats_text = (
        f'Statistics:\n'
        f'Points shown: {len(plotting_data)}\n'
        f'Correlation: {correlation:.3f} (p={p_value:.3e})\n'
        f'Avg Fitness mean: {plotting_data["Avg_Fitness"].mean():.4f}\n'
        f'Switch Fitness mean: {plotting_data["Switch_Fitness"].mean():.4f}'
    )
    
    plt.text(
        0.02, 0.02, stats_text, 
        transform=plt.gca().transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
        fontsize=12
    )
    
    # Calculate percentages in each quadrant (relative to zero)
    q1 = ((plotting_data['Avg_Fitness'] > 0) & (plotting_data['Switch_Fitness'] > 0)).mean() * 100
    q2 = ((plotting_data['Avg_Fitness'] < 0) & (plotting_data['Switch_Fitness'] > 0)).mean() * 100
    q3 = ((plotting_data['Avg_Fitness'] < 0) & (plotting_data['Switch_Fitness'] < 0)).mean() * 100
    q4 = ((plotting_data['Avg_Fitness'] > 0) & (plotting_data['Switch_Fitness'] < 0)).mean() * 100
    
    # Add quadrant labels
    quadrant_props = {
        'boxstyle': 'round,pad=0.5',
        'facecolor': 'white',
        'alpha': 0.7,
        'edgecolor': 'gray'
    }
    
    # Position labels in each quadrant - use axis limit and y_max_abs for positioning
    plt.text(axis_limit*0.6, y_max_abs*0.6, f'Q1: {q1:.1f}%', bbox=quadrant_props, fontsize=12)
    plt.text(-axis_limit*0.6, y_max_abs*0.6, f'Q2: {q2:.1f}%', bbox=quadrant_props, fontsize=12)
    plt.text(-axis_limit*0.6, -y_max_abs*0.6, f'Q3: {q3:.1f}%', bbox=quadrant_props, fontsize=12)
    plt.text(axis_limit*0.6, -y_max_abs*0.6, f'Q4: {q4:.1f}%', bbox=quadrant_props, fontsize=12)
    
    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Plot saved to {output_file}')

if __name__ == "__main__":
    # File paths - using the updated Switch file
    clim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Clim_rep1_FitSeq.csv"
    nlim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Nlim_rep1_FitSeq.csv" 
    switch_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Switch_rep1_FitSeq.csv"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory/"
    output_file = os.path.join(output_dir, "avgCN_vs_switch_rep1.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Axis limit
    axis_limit = 1
    
    # Generate plot
    print(f"Generating fitness comparison plot (axis limits = ±{axis_limit})...")
    plot_avg_vs_switch(clim_file, nlim_file, switch_file, output_file, axis_limit)
    print("\nPlot generated successfully!")
