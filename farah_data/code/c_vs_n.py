import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import os

def plot_fitness_comparison(clim_file, nlim_file, output_file, min_fitness=-1, axis_limit=1):
    """
    Plot fitness comparison between Clim and Nlim conditions with the origin perfectly centered.
    
    Parameters:
    -----------
    clim_file : str
        Path to the Clim fitness data file (CSV)
    nlim_file : str
        Path to the Nlim fitness data file (CSV)
    output_file : str
        Path to save the output plot
    min_fitness : float
        Minimum fitness threshold - values below this will be filtered out
    axis_limit : float
        The absolute limit for both axes (±axis_limit)
    """
    # Read the CSV files
    clim_data = pd.read_csv(clim_file)
    nlim_data = pd.read_csv(nlim_file)
    
    # Print column names for debugging
    print("Clim file columns:", clim_data.columns.tolist())
    print("Nlim file columns:", nlim_data.columns.tolist())
    
    # Find fitness columns - first look for 'Estimated_Fitness', then try numeric columns
    clim_fitness_col = None
    nlim_fitness_col = None
    
    # Check for Estimated_Fitness column
    if 'Estimated_Fitness' in clim_data.columns:
        clim_fitness_col = 'Estimated_Fitness'
    if 'Estimated_Fitness' in nlim_data.columns:
        nlim_fitness_col = 'Estimated_Fitness'
    
    # If not found, use the first numeric column (excluding read counts and other known non-fitness columns)
    if clim_fitness_col is None:
        for col in clim_data.columns:
            if pd.api.types.is_numeric_dtype(clim_data[col]) and not col.startswith('Estimated_Read'):
                clim_fitness_col = col
                print(f"Using '{col}' as Clim fitness column")
                break
    
    if nlim_fitness_col is None:
        for col in nlim_data.columns:
            if pd.api.types.is_numeric_dtype(nlim_data[col]) and not col.startswith('Estimated_Read'):
                nlim_fitness_col = col
                print(f"Using '{col}' as Nlim fitness column")
                break
    
    # Check if fitness columns were found
    if clim_fitness_col is None or nlim_fitness_col is None:
        raise ValueError("Could not identify fitness columns in the data files")
    
    # Extract fitness values
    clim_fitness = clim_data[clim_fitness_col]
    nlim_fitness = nlim_data[nlim_fitness_col]
    
    # Create a DataFrame for plotting
    combined_data = pd.DataFrame({
        'Clim_Fitness': clim_fitness,
        'Nlim_Fitness': nlim_fitness
    })
    
    # Print distribution information before filtering
    print("\nBefore filtering:")
    print(f"Clim Fitness range: {combined_data['Clim_Fitness'].min()} to {combined_data['Clim_Fitness'].max()}")
    print(f"Nlim Fitness range: {combined_data['Nlim_Fitness'].min()} to {combined_data['Nlim_Fitness'].max()}")
    print(f"Points near origin (both between -0.05 and 0.05): {((combined_data['Clim_Fitness'].abs() < 0.05) & (combined_data['Nlim_Fitness'].abs() < 0.05)).sum()}")
    
    # Filter out values below the minimum fitness threshold
    print(f"Original data points: {len(combined_data)}")
    filtered_data = combined_data[
        (combined_data['Clim_Fitness'] >= min_fitness) & 
        (combined_data['Nlim_Fitness'] >= min_fitness)
    ]
    print(f"After filtering (min fitness = {min_fitness}): {len(filtered_data)}")
    
    # Calculate correlation on filtered data
    correlation, p_value = pearsonr(
        filtered_data['Nlim_Fitness'], 
        filtered_data['Clim_Fitness']
    )
    
    # Further filter data to be within the axis limits for plotting
    plotting_data = filtered_data[
        (filtered_data['Clim_Fitness'] >= -axis_limit) & 
        (filtered_data['Clim_Fitness'] <= axis_limit) &
        (filtered_data['Nlim_Fitness'] >= -axis_limit) & 
        (filtered_data['Nlim_Fitness'] <= axis_limit)
    ]
    print(f"Data points within axis limits (±{axis_limit}): {len(plotting_data)}")
    
    # Create scatterplot directly
    plt.figure(figsize=(10, 8))
    
    # Set nice grid styling
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot
    plt.scatter(
        plotting_data['Nlim_Fitness'],
        plotting_data['Clim_Fitness'],
        alpha=0.6,
        s=30,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Add a regression line
    if len(plotting_data) > 1:
        x = plotting_data['Nlim_Fitness']
        y = plotting_data['Clim_Fitness']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        # Calculate regression line using the full axis range
        x_line = np.array([-axis_limit, axis_limit])
        plt.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    # Add reference lines at zero (make them more prominent)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Customize the plot
    plt.xlabel('Nlim Fitness', fontsize=14)
    plt.ylabel('Clim Fitness', fontsize=14)
    plt.title(f'Fitness Comparison: Clim vs Nlim\nCorrelation: {correlation:.3f}', fontsize=16)
    
    # Set SYMMETRIC axis limits 
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)
    
    # Add quadrant labels
    quadrant_props = {
        'boxstyle': 'round,pad=0.5',
        'facecolor': 'white',
        'alpha': 0.7,
        'edgecolor': 'gray'
    }
    
    # Calculate percentages in each quadrant (relative to zero)
    q1 = ((plotting_data['Nlim_Fitness'] > 0) & (plotting_data['Clim_Fitness'] > 0)).mean() * 100
    q2 = ((plotting_data['Nlim_Fitness'] < 0) & (plotting_data['Clim_Fitness'] > 0)).mean() * 100
    q3 = ((plotting_data['Nlim_Fitness'] < 0) & (plotting_data['Clim_Fitness'] < 0)).mean() * 100
    q4 = ((plotting_data['Nlim_Fitness'] > 0) & (plotting_data['Clim_Fitness'] < 0)).mean() * 100
    
    # Position labels in each quadrant
    offset = axis_limit * 0.7
    plt.text(offset/2, offset/2, f'Q1: {q1:.1f}%', bbox=quadrant_props, fontsize=12)
    plt.text(-offset/2, offset/2, f'Q2: {q2:.1f}%', bbox=quadrant_props, fontsize=12)
    plt.text(-offset/2, -offset/2, f'Q3: {q3:.1f}%', bbox=quadrant_props, fontsize=12)
    plt.text(offset/2, -offset/2, f'Q4: {q4:.1f}%', bbox=quadrant_props, fontsize=12)
    
    # Add a text box with statistics
    stats_text = (
        f'Statistics:\n'
        f'Points shown: {len(plotting_data)}\n'
        f'Correlation: {correlation:.3f}\n'
        f'Clim mean: {plotting_data["Clim_Fitness"].mean():.4f}\n'
        f'Nlim mean: {plotting_data["Nlim_Fitness"].mean():.4f}'
    )
    
    plt.text(
        0.02, 0.02, stats_text, 
        transform=plt.gca().transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
        fontsize=12
    )
    
    # Check if the origin is actually at the center
    ax = plt.gca()
    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    print(f"X-axis center: {x_center} (should be close to 0)")
    print(f"Y-axis center: {y_center} (should be close to 0)")
    
    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Plot saved to {output_file}')

if __name__ == "__main__":
    # File paths
    clim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Clim_rep3_FitSeq.csv"
    nlim_file = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/results_Nlim_rep3_FitSeq.csv"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory/"
    output_file = os.path.join(output_dir, "fitness_comparison_rep3.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Minimum fitness threshold (filter out values below this)
    min_fitness = -0.5
    
    # Axis limit
    axis_limit = 1
    
    # Generate plot
    print(f"Generating fitness comparison plot (axis limits = ±{axis_limit})...")
    plot_fitness_comparison(clim_file, nlim_file, output_file, min_fitness, axis_limit)
    print("\nPlot generated successfully!")