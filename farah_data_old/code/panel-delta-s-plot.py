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
        DataFrame containing Delta_S, Non_Additivity and fitness values
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
    print(f"  After filtering NaNs: {len(combined_data)} rows")
    
    return combined_data

def create_delta_vs_nonadditivity_plot(combined_data, ax, title, axis_limit=0.05):
    """
    Create a Delta S vs Non-Additivity plot on the given axis.
    
    Parameters:
    -----------
    combined_data : pandas.DataFrame
        DataFrame containing Delta_S and Non_Additivity columns
    ax : matplotlib.axes.Axes
        The axis to plot on
    title : str
        Title for the plot
    axis_limit : float
        The limit for both axes (±axis_limit)
    
    Returns:
    --------
    tuple: (correlation, p_value, filtered_data)
        Statistics and filtered data used for the plot
    """
    # Filter data to the specified axis limit
    filtered_data = combined_data[
        (combined_data['Delta_S'] <= axis_limit) & 
        (combined_data['Non_Additivity'] <= axis_limit)
    ]
    
    # Calculate correlation
    if len(filtered_data) > 1:
        correlation, p_value = pearsonr(filtered_data['Delta_S'], filtered_data['Non_Additivity'])
    else:
        correlation, p_value = float('nan'), float('nan')
        print(f"Warning: Not enough data points for {title} after filtering")
    
    # Set grid styling
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot
    ax.scatter(
        filtered_data['Delta_S'],
        filtered_data['Non_Additivity'],
        alpha=0.6,
        s=20,  # Smaller point size for the panel plot
        edgecolor='w',
        linewidth=0.5
    )
    
    # Add a regression line
    if len(filtered_data) > 1:
        x = filtered_data['Delta_S']
        y = filtered_data['Non_Additivity']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        # Calculate regression line using the full axis range
        x_line = np.array([0, axis_limit])
        ax.plot(x_line, p(x_line), 'r-', linewidth=1.5)
    
    # Add reference lines at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Delta S = |Clim - Nlim|', fontsize=10)
    ax.set_ylabel('Non-Additivity = |Switch - Avg|', fontsize=10)
    
    if np.isnan(correlation):
        ax.set_title(f'{title}\nInsufficient data', fontsize=12)
    else:
        ax.set_title(f'{title}\nCorr: {correlation:.3f}', fontsize=12)
    
    # Set axis limits
    ax.set_xlim(0, axis_limit)
    ax.set_ylim(0, axis_limit)
    
    # Add a text box with statistics
    if not np.isnan(correlation):
        stats_text = (
            f'n={len(filtered_data)}\n'
            f'r={correlation:.3f} (p={p_value:.2e})\n'
            f'ΔS mean: {filtered_data["Delta_S"].mean():.4f}\n'
            f'NA mean: {filtered_data["Non_Additivity"].mean():.4f}'
        )
        
        ax.text(
            0.03, 0.97, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
            fontsize=8
        )
    
    # Add a line of equality (y = x)
    ax.plot([0, axis_limit], [0, axis_limit], 'k--', alpha=0.5)
    
    return correlation, p_value, filtered_data

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
        avg_df['Delta_S'] = np.abs(avg_df['Clim_Fitness'] - avg_df['Nlim_Fitness'])
        avg_df['Non_Additivity'] = np.abs(avg_df['Switch_Fitness'] - avg_df['Avg_Fitness'])
    
    # Filter out NaN values
    avg_df = avg_df.dropna()
    
    return avg_df

def calculate_replicate_correlations(filtered_data_by_rep):
    """Calculate correlations between replicates for Delta_S and Non_Additivity."""
    rep_pairs = [('rep1', 'rep2'), ('rep1', 'rep3'), ('rep2', 'rep3')]
    rep_metrics = {}
    
    for rep1, rep2 in rep_pairs:
        # For each pair, we need to create a common index based on the original positions
        # in the datasets, not the filtered dataset indices which might be different
        
        # We'll create new dataframes with reset indices to simplify the merge
        data1 = filtered_data_by_rep[rep1].reset_index(drop=True).copy()
        data2 = filtered_data_by_rep[rep2].reset_index(drop=True).copy()
        
        # Create a common dataset up to the shortest length
        min_len = min(len(data1), len(data2))
        common_data1 = data1.iloc[:min_len]
        common_data2 = data2.iloc[:min_len]
        
        # Filter out rows where either dataset has NaN values
        valid_indices = ~(common_data1[['Delta_S', 'Non_Additivity']].isna().any(axis=1) | 
                         common_data2[['Delta_S', 'Non_Additivity']].isna().any(axis=1))
        
        common_data1 = common_data1[valid_indices]
        common_data2 = common_data2[valid_indices]
        
        if len(common_data1) > 1:
            # Calculate correlations
            delta_s_corr, delta_s_p = pearsonr(common_data1['Delta_S'], common_data2['Delta_S'])
            nonadd_corr, nonadd_p = pearsonr(common_data1['Non_Additivity'], common_data2['Non_Additivity'])
            
            # Also calculate correlations for raw fitness values
            clim_corr, clim_p = pearsonr(common_data1['Clim_Fitness'], common_data2['Clim_Fitness'])
            nlim_corr, nlim_p = pearsonr(common_data1['Nlim_Fitness'], common_data2['Nlim_Fitness'])
            switch_corr, switch_p = pearsonr(common_data1['Switch_Fitness'], common_data2['Switch_Fitness'])
            
            rep_metrics[f"{rep1}_vs_{rep2}"] = {
                'delta_s_corr': delta_s_corr,
                'delta_s_p': delta_s_p,
                'nonadd_corr': nonadd_corr,
                'nonadd_p': nonadd_p,
                'clim_corr': clim_corr,
                'nlim_corr': nlim_corr, 
                'switch_corr': switch_corr,
                'n_common': len(common_data1)
            }
        else:
            rep_metrics[f"{rep1}_vs_{rep2}"] = {
                'delta_s_corr': np.nan,
                'delta_s_p': np.nan,
                'nonadd_corr': np.nan,
                'nonadd_p': np.nan,
                'clim_corr': np.nan,
                'nlim_corr': np.nan,
                'switch_corr': np.nan,
                'n_common': len(common_data1) if 'common_data1' in locals() else 0
            }
    
    return rep_metrics

def calculate_coefficient_of_variation(metrics):
    """Calculate coefficient of variation for Delta_S and Non_Additivity means."""
    delta_s_means = [metrics[rep]['delta_s_mean'] for rep in ['rep1', 'rep2', 'rep3'] 
                    if not np.isnan(metrics[rep]['delta_s_mean'])]
    
    nonadd_means = [metrics[rep]['non_additivity_mean'] for rep in ['rep1', 'rep2', 'rep3']
                   if not np.isnan(metrics[rep]['non_additivity_mean'])]
    
    cv_metrics = {}
    
    if delta_s_means:
        cv_metrics['delta_s_cv'] = np.std(delta_s_means) / np.mean(delta_s_means) if np.mean(delta_s_means) != 0 else np.nan
    else:
        cv_metrics['delta_s_cv'] = np.nan
        
    if nonadd_means:
        cv_metrics['nonadd_cv'] = np.std(nonadd_means) / np.mean(nonadd_means) if np.mean(nonadd_means) != 0 else np.nan
    else:
        cv_metrics['nonadd_cv'] = np.nan
    
    return cv_metrics

def format_metrics_text(metrics, rep_metrics, cv_metrics):
    """Format all metrics into a readable text block."""
    
    # Individual replicate metrics
    metrics_text = "Individual Replicate Metrics (Delta S vs Non-Additivity):\n"
    for rep in ['rep1', 'rep2', 'rep3', 'average']:
        rep_name = 'Average' if rep == 'average' else f'Replicate {rep[-1]}'
        if not np.isnan(metrics[rep]['correlation']):
            metrics_text += f"  {rep_name}: r={metrics[rep]['correlation']:.3f} (p={metrics[rep]['p_value']:.2e}, n={metrics[rep]['n_points']})\n"
        else:
            metrics_text += f"  {rep_name}: Insufficient data\n"
    
    # Correlation between replicates
    metrics_text += "\nReproducibility (Correlations Between Replicates):\n"
    for pair, values in rep_metrics.items():
        rep1, rep2 = pair.split('_vs_')
        rep1_name = f"Rep {rep1[-1]}"
        rep2_name = f"Rep {rep2[-1]}"
        
        if not np.isnan(values['delta_s_corr']):
            metrics_text += f"  {rep1_name} vs {rep2_name} (n={values['n_common']}):\n"
            metrics_text += f"    Delta_S: r={values['delta_s_corr']:.3f} (p={values['delta_s_p']:.2e})\n"
            metrics_text += f"    Non-Additivity: r={values['nonadd_corr']:.3f} (p={values['nonadd_p']:.2e})\n"
            
            # Add the raw fitness correlations if available
            if 'clim_corr' in values and not np.isnan(values['clim_corr']):
                metrics_text += f"    Raw fitness correlations:\n"
                metrics_text += f"      Clim fitness: r={values['clim_corr']:.3f}\n"
                metrics_text += f"      Nlim fitness: r={values['nlim_corr']:.3f}\n"
                metrics_text += f"      Switch fitness: r={values['switch_corr']:.3f}\n"
        else:
            metrics_text += f"  {rep1_name} vs {rep2_name}: Insufficient common data points\n"
    
    # Coefficient of variation
    metrics_text += "\nVariability (Coefficient of Variation Across Replicates):\n"
    if not np.isnan(cv_metrics['delta_s_cv']):
        metrics_text += f"  Delta_S mean CV: {cv_metrics['delta_s_cv']:.3f}\n"
    else:
        metrics_text += "  Delta_S mean CV: Cannot calculate\n"
        
    if not np.isnan(cv_metrics['nonadd_cv']):
        metrics_text += f"  Non-Additivity mean CV: {cv_metrics['nonadd_cv']:.3f}\n"
    else:
        metrics_text += "  Non-Additivity mean CV: Cannot calculate\n"
    
    return metrics_text

def plot_all_replicates_plus_average(base_path, output_file, axis_limit=0.05):
    """
    Create a panel plot with Delta S vs Non-Additivity for all three replicates
    plus a plot of the average values.
    
    Parameters:
    -----------
    base_path : str
        Base path to the directory containing CSV files
    output_file : str
        Path to save the output plot
    axis_limit : float
        The limit for both axes (±axis_limit)
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
    filtered_data_by_rep = {}
    
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
    
    # Plot replicates 1 and 2 in the top row
    for i, rep in enumerate(['rep1', 'rep2']):
        ax = fig.add_subplot(gs[0, i])
        
        correlation, p_value, filtered_data = create_delta_vs_nonadditivity_plot(
            data_by_rep[rep], ax, f'Replicate {i+1}', axis_limit
        )
        
        filtered_data_by_rep[rep] = filtered_data
        
        metrics[rep] = {
            'correlation': correlation,
            'p_value': p_value,
            'delta_s_mean': filtered_data['Delta_S'].mean() if not filtered_data.empty else np.nan,
            'non_additivity_mean': filtered_data['Non_Additivity'].mean() if not filtered_data.empty else np.nan,
            'n_points': len(filtered_data)
        }
    
    # Plot replicate 3 in the middle row, left column
    ax_rep3 = fig.add_subplot(gs[1, 0])
    correlation, p_value, filtered_data = create_delta_vs_nonadditivity_plot(
        data_by_rep['rep3'], ax_rep3, 'Replicate 3', axis_limit
    )
    
    filtered_data_by_rep['rep3'] = filtered_data
    
    metrics['rep3'] = {
        'correlation': correlation,
        'p_value': p_value,
        'delta_s_mean': filtered_data['Delta_S'].mean() if not filtered_data.empty else np.nan,
        'non_additivity_mean': filtered_data['Non_Additivity'].mean() if not filtered_data.empty else np.nan,
        'n_points': len(filtered_data)
    }
    
    # Plot the average data in the middle row, right column
    ax_avg = fig.add_subplot(gs[1, 1])
    correlation, p_value, filtered_avg_data = create_delta_vs_nonadditivity_plot(
        avg_data, ax_avg, 'Average of Replicates', axis_limit
    )
    
    metrics['average'] = {
        'correlation': correlation,
        'p_value': p_value,
        'delta_s_mean': filtered_avg_data['Delta_S'].mean() if not filtered_avg_data.empty else np.nan,
        'non_additivity_mean': filtered_avg_data['Non_Additivity'].mean() if not filtered_avg_data.empty else np.nan,
        'n_points': len(filtered_avg_data)
    }
    
    # Add a title to the entire figure
    fig.suptitle('Delta S vs Non-Additivity Across Replicates', fontsize=16, y=0.98)
    
    # Calculate correlation between replicates
    rep_metrics = calculate_replicate_correlations(filtered_data_by_rep)
    
    # Calculate coefficient of variation
    cv_metrics = calculate_coefficient_of_variation(metrics)
    
    # Combine all metrics into a text block
    metrics_text = format_metrics_text(metrics, rep_metrics, cv_metrics)
    
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
    
    # Perform additional analysis: sign patterns of fitness values
    print("\nAdditional Analysis of Sign Patterns for Each Replicate:")
    for rep, data in data_by_rep.items():
        print(f"\n{rep.upper()} Sign Pattern Analysis:")
        clim_fitness = data['Clim_Fitness']
        nlim_fitness = data['Nlim_Fitness']
        switch_fitness = data['Switch_Fitness']
        avg_fitness = data['Avg_Fitness']
        
        # Count strains where Clim and Nlim have the same or different signs
        same_sign_clim_nlim = ((clim_fitness > 0) & (nlim_fitness > 0)) | ((clim_fitness < 0) & (nlim_fitness < 0))
        diff_sign_clim_nlim = ((clim_fitness > 0) & (nlim_fitness < 0)) | ((clim_fitness < 0) & (nlim_fitness > 0))
        
        # Count strains where Switch and Avg have the same or different signs
        same_sign_switch_avg = ((switch_fitness > 0) & (avg_fitness > 0)) | ((switch_fitness < 0) & (avg_fitness < 0))
        diff_sign_switch_avg = ((switch_fitness > 0) & (avg_fitness < 0)) | ((switch_fitness < 0) & (avg_fitness > 0))
        
        print(f"  Clim and Nlim same sign: {same_sign_clim_nlim.sum()} ({same_sign_clim_nlim.mean()*100:.1f}%)")
        print(f"  Clim and Nlim different signs: {diff_sign_clim_nlim.sum()} ({diff_sign_clim_nlim.mean()*100:.1f}%)")
        print(f"  Switch and Avg same sign: {same_sign_switch_avg.sum()} ({same_sign_switch_avg.mean()*100:.1f}%)")
        print(f"  Switch and Avg different signs: {diff_sign_switch_avg.sum()} ({diff_sign_switch_avg.mean()*100:.1f}%)")

if __name__ == "__main__":
    # Base path for CSV files 
    base_path = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory"
    output_file = os.path.join(output_dir, "delta_vs_nonadditivity_panel_plot.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Axis limit - can be adjusted as needed
    axis_limit = 1  # Adjust this value to zoom in/out as needed
    
    # Generate panel plot
    print(f"Generating Delta S vs Non-Additivity panel plot (axis limits = 0 to {axis_limit})...")
    plot_all_replicates_plus_average(base_path, output_file, axis_limit)
    print("\nPanel plot generated successfully!")