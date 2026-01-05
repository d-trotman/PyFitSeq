import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import os
from matplotlib.gridspec import GridSpec

def load_and_process_data(clim_file, nlim_file, switch_file):
    """
    Load and process data from the given CSV files.
    
    Parameters:
    -----------
    clim_file : str
        Path to the Clim fitness data file (CSV)
    nlim_file : str
        Path to the Nlim fitness data file (CSV)
    switch_file : str
        Path to the Switch fitness data file (CSV)
        
    Returns:
    --------
    combined_data : pandas.DataFrame
        DataFrame containing Avg_Fitness, Switch_Fitness and fitness values
    """
    # Read the CSV files
    try:
        clim_data = pd.read_csv(clim_file)
        nlim_data = pd.read_csv(nlim_file)
        switch_data = pd.read_csv(switch_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        raise
    
    # Print dataset sizes
    print(f"  Clim dataset: {len(clim_data)} rows")
    print(f"  Nlim dataset: {len(nlim_data)} rows")
    print(f"  Switch dataset: {len(switch_data)} rows")
    
    # Check if datasets are aligned
    min_rows = min(len(clim_data), len(nlim_data), len(switch_data))
    if len(clim_data) != len(nlim_data) or len(clim_data) != len(switch_data):
        print(f"  Warning: Datasets have different numbers of rows. Using first {min_rows} rows.")
    
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
        'Switch_Fitness': switch_fitness,
        'Clim_Fitness': clim_fitness,
        'Nlim_Fitness': nlim_fitness
    })
    
    # Filter out any NaN values
    combined_data = combined_data.dropna()
    print(f"  After filtering NaNs: {len(combined_data)} rows")
    
    return combined_data

def create_avg_switch_plot(combined_data, ax, title, display_limit=None):
    """
    Create an Average vs Switch fitness plot on the given axis.
    
    Parameters:
    -----------
    combined_data : pandas.DataFrame
        DataFrame containing Avg_Fitness and Switch_Fitness columns
    ax : matplotlib.axes.Axes
        The axis to plot on
    title : str
        Title for the plot
    display_limit : float or None
        The limit for displaying the x-axis (±display_limit). If None, autoscale to show all data.
        All data is used for calculations regardless.
    
    Returns:
    --------
    tuple: (correlation, p_value, slope, intercept, quadrant_percentages)
        Statistics used for the plot
    """
    # Calculate correlation using ALL data points
    correlation, p_value = pearsonr(combined_data['Avg_Fitness'], combined_data['Switch_Fitness'])
    
    # Calculate regression using ALL data points
    if len(combined_data) > 1:
        z = np.polyfit(combined_data['Avg_Fitness'], combined_data['Switch_Fitness'], 1)
        slope, intercept = z[0], z[1]
    else:
        slope, intercept = np.nan, np.nan
    
    # Set grid styling
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot with all data points
    scatter = ax.scatter(
        combined_data['Avg_Fitness'],
        combined_data['Switch_Fitness'],
        alpha=0.6,
        s=20,  # Smaller point size for the panel plot
        edgecolor='w',
        linewidth=0.5
    )
    
    # Determine data range
    xmin, xmax = combined_data['Avg_Fitness'].min(), combined_data['Avg_Fitness'].max()
    ymin, ymax = combined_data['Switch_Fitness'].min(), combined_data['Switch_Fitness'].max()
    
    # If no display limit is specified, autoscale to show all data
    if display_limit is None:
        # Find the maximum absolute value in x-direction
        max_abs_x = max(abs(xmin), abs(xmax))
        # Add 10% padding
        display_limit = max_abs_x * 1.1
    
    # Add a regression line using ALL data for calculation
    if not np.isnan(slope):
        # Extend the line across the full x data range
        x_line = np.array([xmin, xmax])
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5)
    
    # Add reference lines at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Average Fitness (Clim + Nlim)/2', fontsize=10)
    ax.set_ylabel('Switch Fitness', fontsize=10)
    
    if np.isnan(correlation):
        ax.set_title(f'{title}\nInsufficient data', fontsize=12)
    else:
        ax.set_title(f'{title}\nCorr: {correlation:.3f}', fontsize=12)
    
    # Set axis limits to show all data
    ax.set_xlim(xmin - abs(xmin)*0.05, xmax + abs(xmax)*0.05)
    
    # Find the maximum absolute value in the y-direction to make the plot symmetric
    y_max_abs = max(abs(ymin), abs(ymax))
    # Add 10% padding
    y_max_abs *= 1.1
    
    # Set symmetric y-axis limits centered at 0
    ax.set_ylim(-y_max_abs, y_max_abs)
    
    # Calculate percentages in each quadrant (relative to zero) using ALL data
    q1 = ((combined_data['Avg_Fitness'] > 0) & (combined_data['Switch_Fitness'] > 0)).mean() * 100
    q2 = ((combined_data['Avg_Fitness'] < 0) & (combined_data['Switch_Fitness'] > 0)).mean() * 100
    q3 = ((combined_data['Avg_Fitness'] < 0) & (combined_data['Switch_Fitness'] < 0)).mean() * 100
    q4 = ((combined_data['Avg_Fitness'] > 0) & (combined_data['Switch_Fitness'] < 0)).mean() * 100
    
    # Add quadrant labels
    quadrant_props = {
        'boxstyle': 'round,pad=0.3',
        'facecolor': 'white',
        'alpha': 0.7,
        'edgecolor': 'gray'
    }
    
    # Position labels in each quadrant based on data range
    pos_x = (xmax - 0) / 2  # Midpoint of positive x
    neg_x = (0 - xmin) / 2  # Midpoint of negative x 
    pos_y = (y_max_abs - 0) / 2  # Midpoint of positive y
    neg_y = (0 - y_max_abs) / 2  # Midpoint of negative y
    
    # Adjust position to be visible
    pos_x = min(pos_x, xmax * 0.8)
    neg_x = max(neg_x * -1, xmin * 0.8)
    
    # Add quadrant labels
    ax.text(pos_x, pos_y, f'Q1: {q1:.1f}%', bbox=quadrant_props, fontsize=8)
    ax.text(neg_x, pos_y, f'Q2: {q2:.1f}%', bbox=quadrant_props, fontsize=8)
    ax.text(neg_x, neg_y, f'Q3: {q3:.1f}%', bbox=quadrant_props, fontsize=8)
    ax.text(pos_x, neg_y, f'Q4: {q4:.1f}%', bbox=quadrant_props, fontsize=8)
    
    # Add a text box with statistics
    if not np.isnan(correlation):
        stats_text = (
            f'n={len(combined_data)}\n'
            f'r={correlation:.3f} (p={p_value:.2e})\n'
            f'y={slope:.3f}x+{intercept:.3f}\n'
            f'Data range: ({xmin:.2f},{ymin:.2f}) to ({xmax:.2f},{ymax:.2f})'
        )
        
        ax.text(
            0.03, 0.97, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
            fontsize=8
        )
    
    quadrant_percentages = {
        'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4
    }
    
    return correlation, p_value, slope, intercept, quadrant_percentages

def calculate_average_data(data_by_rep):
    """
    Calculate average values across replicates.
    
    Parameters:
    -----------
    data_by_rep : dict
        Dictionary with replicate names as keys and DataFrames as values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with average values across replicates
    """
    # Create a list to store aligned data
    aligned_data = []
    
    # For each strain index (assuming they're aligned across replicates)
    max_idx = min([len(df) for df in data_by_rep.values()])
    
    for idx in range(max_idx):
        # Extract values for this strain from each replicate
        strain_values = {}
        
        for rep, df in data_by_rep.items():
            if idx < len(df):
                for col in ['Clim_Fitness', 'Nlim_Fitness', 'Switch_Fitness']:
                    strain_values.setdefault(col, []).append(df.iloc[idx][col])
        
        # Calculate averages for this strain
        avg_strain = {}
        for col, values in strain_values.items():
            avg_strain[col] = np.mean(values) if values else np.nan
        
        # Add to aligned data
        if avg_strain:
            aligned_data.append(avg_strain)
    
    # Convert to DataFrame
    avg_df = pd.DataFrame(aligned_data)
    
    # Calculate derived metrics
    if not avg_df.empty:
        avg_df['Avg_Fitness'] = (avg_df['Clim_Fitness'] + avg_df['Nlim_Fitness']) / 2
    
    # Filter out NaN values
    avg_df = avg_df.dropna()
    
    return avg_df

def calculate_replicate_correlations(data_by_rep):
    """Calculate correlations between replicates."""
    rep_pairs = [('rep1', 'rep2'), ('rep1', 'rep3'), ('rep2', 'rep3')]
    rep_metrics = {}
    
    for rep1, rep2 in rep_pairs:
        # For each pair, we need to create a common index based on the original positions
        # in the datasets, not the filtered dataset indices which might be different
        
        # We'll create new dataframes with reset indices to simplify the merge
        data1 = data_by_rep[rep1].reset_index(drop=True).copy()
        data2 = data_by_rep[rep2].reset_index(drop=True).copy()
        
        # Create a common dataset up to the shortest length
        min_len = min(len(data1), len(data2))
        common_data1 = data1.iloc[:min_len]
        common_data2 = data2.iloc[:min_len]
        
        # Filter out rows where either dataset has NaN values
        valid_indices = ~(common_data1[['Avg_Fitness', 'Switch_Fitness']].isna().any(axis=1) | 
                         common_data2[['Avg_Fitness', 'Switch_Fitness']].isna().any(axis=1))
        
        common_data1 = common_data1[valid_indices]
        common_data2 = common_data2[valid_indices]
        
        if len(common_data1) > 1:
            # Calculate correlations for average and switch fitness
            avg_fitness_corr, avg_fitness_p = pearsonr(
                common_data1['Avg_Fitness'], 
                common_data2['Avg_Fitness']
            )
            switch_corr, switch_p = pearsonr(
                common_data1['Switch_Fitness'], 
                common_data2['Switch_Fitness']
            )
            
            # Also calculate correlations for raw fitness values
            clim_corr, clim_p = pearsonr(common_data1['Clim_Fitness'], common_data2['Clim_Fitness'])
            nlim_corr, nlim_p = pearsonr(common_data1['Nlim_Fitness'], common_data2['Nlim_Fitness'])
            
            rep_metrics[f"{rep1}_vs_{rep2}"] = {
                'avg_fitness_corr': avg_fitness_corr,
                'avg_fitness_p': avg_fitness_p,
                'switch_corr': switch_corr,
                'switch_p': switch_p,
                'clim_corr': clim_corr,
                'nlim_corr': nlim_corr,
                'n_common': len(common_data1)
            }
        else:
            rep_metrics[f"{rep1}_vs_{rep2}"] = {
                'avg_fitness_corr': np.nan,
                'avg_fitness_p': np.nan,
                'switch_corr': np.nan,
                'switch_p': np.nan,
                'clim_corr': np.nan,
                'nlim_corr': np.nan,
                'n_common': len(common_data1) if 'common_data1' in locals() else 0
            }
    
    return rep_metrics

def calculate_coefficient_of_variation(metrics, quadrant_data):
    """Calculate coefficient of variation for metrics across replicates."""
    cv_metrics = {}
    
    # For main correlation and slope/intercept
    for param in ['correlation', 'slope', 'intercept']:
        values = [metrics[rep][param] for rep in ['rep1', 'rep2', 'rep3'] 
                 if not np.isnan(metrics[rep][param])]
        
        if len(values) > 1 and np.mean(values) != 0:
            cv = np.std(values) / np.mean(values)
            cv_metrics[f'{param}_cv'] = cv
        else:
            cv_metrics[f'{param}_cv'] = np.nan
    
    # For quadrant percentages
    for q in ['q1', 'q2', 'q3', 'q4']:
        values = [quadrant_data[rep][q] for rep in ['rep1', 'rep2', 'rep3']]
        
        if len(values) > 1 and np.mean(values) != 0:
            cv = np.std(values) / np.mean(values)
            cv_metrics[f'{q}_cv'] = cv
        else:
            cv_metrics[f'{q}_cv'] = np.nan
    
    return cv_metrics

def format_metrics_text(metrics, rep_metrics, cv_metrics, quadrant_data):
    """Format all metrics into a readable text block."""
    
    # Individual replicate metrics
    metrics_text = "Individual Replicate Metrics (Avg Fitness vs Switch):\n"
    for rep in ['rep1', 'rep2', 'rep3', 'average']:
        rep_name = 'Average' if rep == 'average' else f'Replicate {rep[-1]}'
        if not np.isnan(metrics[rep]['correlation']):
            metrics_text += f"  {rep_name}: r={metrics[rep]['correlation']:.3f} (p={metrics[rep]['p_value']:.2e})\n"
            metrics_text += f"    Slope={metrics[rep]['slope']:.3f}, Intercept={metrics[rep]['intercept']:.3f}\n"
        else:
            metrics_text += f"  {rep_name}: Insufficient data\n"
    
    # Quadrant distribution
    metrics_text += "\nQuadrant Distribution Analysis:\n"
    for rep in ['rep1', 'rep2', 'rep3', 'average']:
        rep_name = 'Average' if rep == 'average' else f'Replicate {rep[-1]}'
        q_data = quadrant_data[rep]
        metrics_text += f"  {rep_name}: Q1={q_data['q1']:.1f}%, Q2={q_data['q2']:.1f}%, Q3={q_data['q3']:.1f}%, Q4={q_data['q4']:.1f}%\n"
    
    # Correlation between replicates
    metrics_text += "\nReproducibility (Correlations Between Replicates):\n"
    for pair, values in rep_metrics.items():
        rep1, rep2 = pair.split('_vs_')
        rep1_name = f"Rep {rep1[-1]}"
        rep2_name = f"Rep {rep2[-1]}"
        
        if not np.isnan(values['avg_fitness_corr']):
            metrics_text += f"  {rep1_name} vs {rep2_name} (n={values['n_common']}):\n"
            metrics_text += f"    Avg Fitness: r={values['avg_fitness_corr']:.3f} (p={values['avg_fitness_p']:.2e})\n"
            metrics_text += f"    Switch Fitness: r={values['switch_corr']:.3f} (p={values['switch_p']:.2e})\n"
            
            # Add the raw fitness correlations if available
            if 'clim_corr' in values and not np.isnan(values['clim_corr']):
                metrics_text += f"    Individual correlations:\n"
                metrics_text += f"      Clim fitness: r={values['clim_corr']:.3f}\n"
                metrics_text += f"      Nlim fitness: r={values['nlim_corr']:.3f}\n"
        else:
            metrics_text += f"  {rep1_name} vs {rep2_name}: Insufficient common data points\n"
    
    # Coefficient of variation
    metrics_text += "\nVariability (Coefficient of Variation Across Replicates):\n"
    
    # First handle the correlation/slope metrics
    for metric, cv in cv_metrics.items():
        if metric.endswith('_cv'):
            metric_name = metric.replace('_cv', '').capitalize()
            if not np.isnan(cv):
                metrics_text += f"  {metric_name}: CV={cv:.3f}\n"
            else:
                metrics_text += f"  {metric_name}: Cannot calculate CV\n"
    
    # Then handle the quadrant percentages
    metrics_text += "  Quadrant percentages:\n"
    for q in ['q1', 'q2', 'q3', 'q4']:
        cv_key = f'{q}_cv'
        if cv_key in cv_metrics and not np.isnan(cv_metrics[cv_key]):
            metrics_text += f"    {q.upper()}: CV={cv_metrics[cv_key]:.3f}\n"
        else:
            metrics_text += f"    {q.upper()}: Cannot calculate CV\n"
    
    return metrics_text

def plot_all_replicates_plus_average(base_path, output_file, display_limit=None):
    """
    Create a panel plot with Avg Fitness vs Switch Fitness for all three replicates
    plus a plot of the average values.
    
    Parameters:
    -----------
    base_path : str
        Base path to the directory containing CSV files
    output_file : str
        Path to save the output plot
    display_limit : float or None
        The limit for displaying the x-axis (±display_limit). If None, autoscale to show all data.
    """
    # Define file paths for all replicates
    file_paths = {
        'rep1': {
            'clim': os.path.join(base_path, 'results_Clim_rep1_FitSeq.csv'),
            'nlim': os.path.join(base_path, 'results_Nlim_rep1_FitSeq.csv'),
            'switch': os.path.join(base_path, 'results_Switch_rep1_FitSeq.csv')
        },
        'rep2': {
            'clim': os.path.join(base_path, 'results_Clim_rep2_FitSeq.csv'),
            'nlim': os.path.join(base_path, 'results_Nlim_rep2_FitSeq.csv'),
            'switch': os.path.join(base_path, 'results_Switch_rep2_FitSeq.csv')
        },
        'rep3': {
            'clim': os.path.join(base_path, 'results_Clim_rep3_FitSeq.csv'),
            'nlim': os.path.join(base_path, 'results_Nlim_rep3_FitSeq.csv'),
            'switch': os.path.join(base_path, 'results_Switch_rep3_FitSeq.csv')
        }
    }
    
    # Load and process data for each replicate
    data_by_rep = {}
    
    print("Loading and processing data for each replicate...")
    for rep, files in file_paths.items():
        print(f"\nProcessing {rep}:")
        data = load_and_process_data(files['clim'], files['nlim'], files['switch'])
        data_by_rep[rep] = data
    
    # Calculate averaged data across replicates
    print("\nCalculating average across replicates...")
    avg_data = calculate_average_data(data_by_rep)
    print(f"  Average dataset: {len(avg_data)} rows after filtering NaNs")
    
    # Create a figure with a 3x2 grid for the panel plot:
    # - Top row: Rep 1, Rep 2
    # - Middle row: Rep 3, Average
    # - Bottom row: Metrics panel (spans both columns)
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 2, figure=fig, wspace=0.25, hspace=0.35, height_ratios=[1, 1, 0.6])
    
    # Dictionary to store correlation values and other metrics
    metrics = {}
    quadrant_data = {}
    
    # Plot replicates 1 and 2 in the top row
    for i, rep in enumerate(['rep1', 'rep2']):
        ax = fig.add_subplot(gs[0, i])
        
        correlation, p_value, slope, intercept, quad_percentages = create_avg_switch_plot(
            data_by_rep[rep], ax, f'Replicate {i+1}', display_limit
        )
        
        metrics[rep] = {
            'correlation': correlation,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'n_points': len(data_by_rep[rep])
        }
        quadrant_data[rep] = quad_percentages
    
    # Plot replicate 3 in the middle row, left column
    ax_rep3 = fig.add_subplot(gs[1, 0])
    correlation, p_value, slope, intercept, quad_percentages = create_avg_switch_plot(
        data_by_rep['rep3'], ax_rep3, 'Replicate 3', display_limit
    )
    
    metrics['rep3'] = {
        'correlation': correlation,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'n_points': len(data_by_rep['rep3'])
    }
    quadrant_data['rep3'] = quad_percentages
    
    # Plot the average data in the middle row, right column
    ax_avg = fig.add_subplot(gs[1, 1])
    correlation, p_value, slope, intercept, quad_percentages = create_avg_switch_plot(
        avg_data, ax_avg, 'Average of Replicates', display_limit
    )
    
    metrics['average'] = {
        'correlation': correlation,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'n_points': len(avg_data)
    }
    quadrant_data['average'] = quad_percentages
    
    # Add a title to the entire figure
    fig.suptitle('Average (Clim,Nlim) vs Switch Fitness Across Replicates', fontsize=16, y=0.98)
    
    # Calculate correlation between replicates
    rep_metrics = calculate_replicate_correlations(data_by_rep)
    
    # Calculate coefficient of variation
    cv_metrics = calculate_coefficient_of_variation(metrics, quadrant_data)
    
    # Combine all metrics into a text block
    metrics_text = format_metrics_text(metrics, rep_metrics, cv_metrics, quadrant_data)
    
    # Create a dedicated panel for metrics in the bottom row
    ax_metrics = fig.add_subplot(gs[2, :])
    ax_metrics.axis('off')  # Hide the axes
    ax_metrics.set_title("Comparison Metrics", fontsize=14)
    
    # Add the metrics text to the dedicated panel
    ax_metrics.text(0.02, 0.99, metrics_text, fontsize=10, 
                   va='top', ha='left', transform=ax_metrics.transAxes,
                   bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 5})
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPanel plot saved to: {output_file}")
    print("\nComparison Metrics Summary:")
    print(metrics_text)

if __name__ == "__main__":
    # Base path for CSV files 
    base_path = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory"
    output_file = os.path.join(output_dir, "avgCN_vs_switch_panel_plot.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Use None for display_limit to autoscale and show all data points
    display_limit = None
    
    # Generate panel plot
    print("Generating Average vs Switch fitness panel plot (autoscaling to show all data)...")
    plot_all_replicates_plus_average(base_path, output_file, display_limit)
    print("\nPanel plot generated successfully!")