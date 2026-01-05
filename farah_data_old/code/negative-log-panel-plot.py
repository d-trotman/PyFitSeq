import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
from matplotlib.gridspec import GridSpec

def load_and_process_data(clim_file, nlim_file, switch_file, min_value=1e-10):
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
    min_value : float
        Minimum value to use for -log transformation (to avoid -log(0))
        
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
    
    # Replace zeros or very small values with min_value to avoid -log(0) issues
    combined_data['Delta_S'] = combined_data['Delta_S'].apply(lambda x: max(x, min_value))
    combined_data['Non_Additivity'] = combined_data['Non_Additivity'].apply(lambda x: max(x, min_value))
    
    return combined_data

def create_negative_log_plot(combined_data, ax, title, display_limit=None):
    """
    Create a -log10 transformed plot of Delta S vs Non-Additivity on the given axis.
    
    Parameters:
    -----------
    combined_data : pandas.DataFrame
        DataFrame containing Delta_S and Non_Additivity columns
    ax : matplotlib.axes.Axes
        The axis to plot on
    title : str
        Title for the plot
    display_limit : float or None
        Maximum limit for displaying the -log10 transformed axes. If None, determined from data.
        This only affects display, all data points are used for statistics.
    
    Returns:
    --------
    tuple: (pearson_corr, pearson_p, spearman_corr, spearman_p, transformed_data)
        Statistics and transformed data used for the plot
    """
    # Apply -log10 transformation to all data
    neg_log_delta_s = -np.log10(combined_data['Delta_S'])
    neg_log_non_additivity = -np.log10(combined_data['Non_Additivity'])
    
    # Create transformed DataFrame with all data points
    transformed_data = combined_data.copy()
    transformed_data['Neg_Log_Delta_S'] = neg_log_delta_s
    transformed_data['Neg_Log_Non_Additivity'] = neg_log_non_additivity
    
    # Calculate correlations using ALL data points
    pearson_corr, pearson_p = pearsonr(neg_log_delta_s, neg_log_non_additivity)
    spearman_corr, spearman_p = spearmanr(combined_data['Delta_S'], combined_data['Non_Additivity'])
    
    # Set grid styling
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot with all data points
    scatter = ax.scatter(
        neg_log_delta_s,
        neg_log_non_additivity,
        alpha=0.6,
        s=20,  # Smaller point size for the panel plot
        edgecolor='w',
        linewidth=0.5
    )
    
    # Add a linear fit using ALL data
    if len(transformed_data) > 1:
        x = neg_log_delta_s
        y = neg_log_non_additivity
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Determine x range for the fit line (using full range of data)
        x_min = min(neg_log_delta_s)
        x_max = max(neg_log_delta_s)
        x_line = np.array([x_min, x_max])
        
        ax.plot(x_line, p(x_line), 'r-', linewidth=1.5)
        
        # Calculate equation for the fit line (slope and intercept)
        slope = z[0]
        intercept = z[1]
    else:
        slope = np.nan
        intercept = np.nan
    
    # Customize the plot
    ax.set_xlabel('-log₁₀(Delta S)', fontsize=10)
    ax.set_ylabel('-log₁₀(Non-Additivity)', fontsize=10)
    
    if np.isnan(pearson_corr):
        ax.set_title(f'{title}\nInsufficient data', fontsize=12)
    else:
        ax.set_title(f'{title}\nPearson r: {pearson_corr:.3f}', fontsize=12)
    
    # Count points that would be out of display range
    if display_limit is not None:
        points_in_display = sum((neg_log_delta_s <= display_limit) & (neg_log_non_additivity <= display_limit))
        points_outside_display = len(neg_log_delta_s) - points_in_display
        
        # Set axis limits for display only (not filtering data)
        ax.set_xlim(0, display_limit)
        ax.set_ylim(0, display_limit)
    else:
        # If no display limit, calculate reasonable limits from data
        max_display = max(max(neg_log_delta_s), max(neg_log_non_additivity)) * 1.05
        ax.set_xlim(0, max_display)
        ax.set_ylim(0, max_display)
        points_outside_display = 0
    
    # Add a text box with statistics
    if not np.isnan(pearson_corr):
        if not np.isnan(slope):
            fit_eq = f'y = {slope:.3f}x + {intercept:.3f}'
        else:
            fit_eq = 'Insufficient data for fit'
        
        # Note: Including stats about how many points are beyond display limits
        if points_outside_display > 0:
            viz_note = f'Showing {points_in_display}/{len(transformed_data)} points\n({points_outside_display} beyond display limits)'
        else:
            viz_note = f'All {len(transformed_data)} points visible'
            
        stats_text = (
            f'n={len(transformed_data)}\n'
            f'Pearson r={pearson_corr:.3f} (p={pearson_p:.2e})\n'
            f'Spearman ρ={spearman_corr:.3f} (p={spearman_p:.2e})\n'
            f'Linear fit: {fit_eq}\n'
            f'{viz_note}'
        )
        
        ax.text(
            0.03, 0.97, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
            fontsize=8
        )
    
    # Add a line of equality (y = x)
    if display_limit is not None:
        ax.plot([0, display_limit], [0, display_limit], 'k--', alpha=0.5)
    else:
        max_val = max(max(neg_log_delta_s), max(neg_log_non_additivity))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    return pearson_corr, pearson_p, spearman_corr, spearman_p, transformed_data

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
    
    # Replace zeros with minimum value for log transformation
    min_value = 1e-10
    avg_df['Delta_S'] = avg_df['Delta_S'].apply(lambda x: max(x, min_value))
    avg_df['Non_Additivity'] = avg_df['Non_Additivity'].apply(lambda x: max(x, min_value))
    
    return avg_df

def calculate_replicate_correlations(transformed_data_by_rep):
    """Calculate correlations between replicates for transformed Delta_S and Non_Additivity."""
    rep_pairs = [('rep1', 'rep2'), ('rep1', 'rep3'), ('rep2', 'rep3')]
    rep_metrics = {}
    
    for rep1, rep2 in rep_pairs:
        # For each pair, we need to create a common index based on the original positions
        # in the datasets, not the filtered dataset indices which might be different
        
        # We'll create new dataframes with reset indices to simplify the merge
        data1 = transformed_data_by_rep[rep1].reset_index(drop=True).copy()
        data2 = transformed_data_by_rep[rep2].reset_index(drop=True).copy()
        
        # Create a common dataset up to the shortest length
        min_len = min(len(data1), len(data2))
        common_data1 = data1.iloc[:min_len]
        common_data2 = data2.iloc[:min_len]
        
        # Filter out rows where either dataset has NaN values
        valid_indices = ~(common_data1[['Neg_Log_Delta_S', 'Neg_Log_Non_Additivity']].isna().any(axis=1) | 
                         common_data2[['Neg_Log_Delta_S', 'Neg_Log_Non_Additivity']].isna().any(axis=1))
        
        common_data1 = common_data1[valid_indices]
        common_data2 = common_data2[valid_indices]
        
        if len(common_data1) > 1:
            # Calculate correlations for transformed data
            delta_s_corr, delta_s_p = pearsonr(
                common_data1['Neg_Log_Delta_S'], 
                common_data2['Neg_Log_Delta_S']
            )
            nonadd_corr, nonadd_p = pearsonr(
                common_data1['Neg_Log_Non_Additivity'], 
                common_data2['Neg_Log_Non_Additivity']
            )
            
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
    """Calculate coefficient of variation for metrics across replicates."""
    # Extract values for different metrics across replicates
    metric_values = {
        'pearson_corr': [],
        'spearman_corr': [],
        'slope': []
    }
    
    for rep in ['rep1', 'rep2', 'rep3']:
        if not np.isnan(metrics[rep]['pearson_corr']):
            metric_values['pearson_corr'].append(metrics[rep]['pearson_corr'])
        if not np.isnan(metrics[rep]['spearman_corr']):
            metric_values['spearman_corr'].append(metrics[rep]['spearman_corr'])
        if not np.isnan(metrics[rep]['slope']):
            metric_values['slope'].append(metrics[rep]['slope'])
    
    cv_metrics = {}
    
    # Calculate coefficient of variation for each metric
    for metric, values in metric_values.items():
        if len(values) > 1:
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.nan
            cv_metrics[f'{metric}_cv'] = cv
        else:
            cv_metrics[f'{metric}_cv'] = np.nan
    
    return cv_metrics

def format_metrics_text(metrics, rep_metrics, cv_metrics):
    """Format all metrics into a readable text block."""
    
    # Individual replicate metrics
    metrics_text = "Individual Replicate Metrics (-log Transform):\n"
    for rep in ['rep1', 'rep2', 'rep3', 'average']:
        rep_name = 'Average' if rep == 'average' else f'Replicate {rep[-1]}'
        if not np.isnan(metrics[rep]['pearson_corr']):
            metrics_text += f"  {rep_name}:\n"
            metrics_text += f"    Pearson r={metrics[rep]['pearson_corr']:.3f} (p={metrics[rep]['pearson_p']:.2e})\n"
            metrics_text += f"    Spearman ρ={metrics[rep]['spearman_corr']:.3f} (p={metrics[rep]['spearman_p']:.2e})\n"
            if not np.isnan(metrics[rep]['slope']):
                metrics_text += f"    Linear fit: y = {metrics[rep]['slope']:.3f}x + {metrics[rep]['intercept']:.3f}\n"
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
            metrics_text += f"    -log(Delta_S): r={values['delta_s_corr']:.3f} (p={values['delta_s_p']:.2e})\n"
            metrics_text += f"    -log(Non-Additivity): r={values['nonadd_corr']:.3f} (p={values['nonadd_p']:.2e})\n"
            
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
    for metric, cv in cv_metrics.items():
        if not np.isnan(cv):
            metric_name = metric.replace('_cv', '').replace('_', ' ').title()
            metrics_text += f"  {metric_name}: CV={cv:.3f}\n"
        else:
            metric_name = metric.replace('_cv', '').replace('_', ' ').title()
            metrics_text += f"  {metric_name}: Cannot calculate CV\n"
    
    return metrics_text

def plot_all_replicates_plus_average(base_path, output_file, min_value=1e-10):
    """
    Create a panel plot with -log transformed Delta S vs Non-Additivity for all three replicates
    plus a plot of the average values.
    
    Parameters:
    -----------
    base_path : str
        Base path to the directory containing CSV files
    output_file : str
        Path to save the output plot
    min_value : float
        Minimum value to use for -log transformation (to avoid -log(0))
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
    transformed_data_by_rep = {}
    
    print("Loading and processing data for each replicate...")
    for rep, files in file_paths.items():
        print(f"\nProcessing {rep}:")
        data = load_and_process_data(files['clim'], files['nlim'], files['switch'], min_value)
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
    
    # Find the max value across all datasets for consistent axis limits
    # This is now used only for display purposes, not for filtering data
    max_neg_log_values = []
    
    for rep, data in data_by_rep.items():
        neg_log_delta_s = -np.log10(data['Delta_S'])
        neg_log_non_additivity = -np.log10(data['Non_Additivity'])
        max_neg_log_values.extend([neg_log_delta_s.max(), neg_log_non_additivity.max()])
    
    # Add the average dataset values too
    neg_log_delta_s_avg = -np.log10(avg_data['Delta_S'])
    neg_log_non_additivity_avg = -np.log10(avg_data['Non_Additivity'])
    max_neg_log_values.extend([neg_log_delta_s_avg.max(), neg_log_non_additivity_avg.max()])
    
    # Determine a reasonable display limit for visualization
    # We'll find the 95th percentile of the -log values to avoid extreme outliers dominating the display
    all_values = np.array(max_neg_log_values)
    display_limit = np.percentile(all_values, 95) * 1.1  # 10% padding
    
    # Also track the absolute max for reporting
    absolute_max = max(max_neg_log_values)
    
    print(f"Display limit for -log plots set to: {display_limit:.2f}")
    print(f"Maximum -log value in any dataset: {absolute_max:.2f}")
    print(f"Note: All data points will be used for statistics and regression,")
    print(f"but display will zoom to show 95% of points clearly")
    
    # Plot replicates 1 and 2 in the top row
    for i, rep in enumerate(['rep1', 'rep2']):
        ax = fig.add_subplot(gs[0, i])
        
        pearson_corr, pearson_p, spearman_corr, spearman_p, transformed_data = create_negative_log_plot(
            data_by_rep[rep], ax, f'Replicate {i+1}', display_limit=display_limit
        )
        
        transformed_data_by_rep[rep] = transformed_data
        
        # Extract slope and intercept from the linear fit
        if len(transformed_data) > 1:
            x = transformed_data['Neg_Log_Delta_S']
            y = transformed_data['Neg_Log_Non_Additivity']
            z = np.polyfit(x, y, 1)
            slope = z[0]
            intercept = z[1]
        else:
            slope = np.nan
            intercept = np.nan
        
        metrics[rep] = {
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'slope': slope,
            'intercept': intercept,
            'n_points': len(transformed_data)
        }
    
    # Plot replicate 3 in the middle row, left column
    ax_rep3 = fig.add_subplot(gs[1, 0])
    pearson_corr, pearson_p, spearman_corr, spearman_p, transformed_data = create_negative_log_plot(
        data_by_rep['rep3'], ax_rep3, 'Replicate 3', display_limit=display_limit
    )
    
    transformed_data_by_rep['rep3'] = transformed_data
    
    # Extract slope and intercept
    if len(transformed_data) > 1:
        x = transformed_data['Neg_Log_Delta_S']
        y = transformed_data['Neg_Log_Non_Additivity']
        z = np.polyfit(x, y, 1)
        slope = z[0]
        intercept = z[1]
    else:
        slope = np.nan
        intercept = np.nan
    
    metrics['rep3'] = {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'slope': slope,
        'intercept': intercept,
        'n_points': len(transformed_data)
    }
    
    # Plot the average data in the middle row, right column
    ax_avg = fig.add_subplot(gs[1, 1])
    pearson_corr, pearson_p, spearman_corr, spearman_p, transformed_avg_data = create_negative_log_plot(
        avg_data, ax_avg, 'Average of Replicates', display_limit=display_limit
    )
    
    # Extract slope and intercept for average data
    if len(transformed_avg_data) > 1:
        x = transformed_avg_data['Neg_Log_Delta_S']
        y = transformed_avg_data['Neg_Log_Non_Additivity']
        z = np.polyfit(x, y, 1)
        slope = z[0]
        intercept = z[1]
    else:
        slope = np.nan
        intercept = np.nan
    
    metrics['average'] = {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'slope': slope,
        'intercept': intercept,
        'n_points': len(transformed_avg_data)
    }
    
    # Add a title to the entire figure
    fig.suptitle('-log₁₀ Transformation: Delta S vs Non-Additivity Across Replicates', fontsize=16, y=0.98)
    
    # Calculate correlation between replicates
    rep_metrics = calculate_replicate_correlations(transformed_data_by_rep)
    
    # Calculate coefficient of variation for metrics across replicates
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
    
    print(f"\nPanel plot with -log transformation saved to: {output_file}")
    print("\nComparison Metrics Summary:")
    print(metrics_text)
    
    # Create a density plot version as well
    create_density_panel_plot(transformed_data_by_rep, transformed_avg_data, metrics, 
                             output_file.replace('.png', '_density.png'), display_limit)

def create_density_panel_plot(transformed_data_by_rep, transformed_avg_data, metrics, output_file, display_limit=None):
    """
    Create a panel of density hexbin plots for the -log transformed data.
    
    Parameters:
    -----------
    transformed_data_by_rep : dict
        Dictionary with replicate names as keys and DataFrames with transformed data as values
    transformed_avg_data : pandas.DataFrame
        DataFrame with transformed average data
    metrics : dict
        Dictionary with metrics for each replicate
    output_file : str
        Path to save the output plot
    display_limit : float or None
        Maximum limit for displaying the -log10 transformed axes. If None, determined from data.
        This only affects display, all data points are used for statistics.
    """
    # Create a figure with a 3x2 grid for the panel plot (same layout as scatter plot)
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 2, figure=fig, wspace=0.25, hspace=0.35, height_ratios=[1, 1, 0.6])
    
    # Plot replicates 1 and 2 in the top row
    for i, rep in enumerate(['rep1', 'rep2']):
        ax = fig.add_subplot(gs[0, i])
        
        data = transformed_data_by_rep[rep]
        x = data['Neg_Log_Delta_S']
        y = data['Neg_Log_Non_Additivity']
        
        # Create hexbin plot using ALL data points (no filtering)
        h = ax.hexbin(x, y, gridsize=40, cmap='viridis', mincnt=1)
        
        # Add colorbar
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_label('Count')
        
        # Customize the plot
        ax.set_xlabel('-log₁₀(Delta S)', fontsize=10)
        ax.set_ylabel('-log₁₀(Non-Additivity)', fontsize=10)
        
        # Add a linear fit line calculated from ALL data points
        if not np.isnan(metrics[rep]['slope']):
            # For display purposes, limit the line to the axis limits
            if display_limit is not None:
                x_range = np.array([0, display_limit])
            else:
                # Use the full data range
                x_range = np.array([0, max(x)])
                
            y_fit = metrics[rep]['slope'] * x_range + metrics[rep]['intercept']
            ax.plot(x_range, y_fit, 'r-', linewidth=1.5)
        
        # Count points that would be out of display range
        if display_limit is not None:
            points_in_display = sum((x <= display_limit) & (y <= display_limit))
            points_outside_display = len(x) - points_in_display
            
            # Add a note about points outside display limits
            if points_outside_display > 0:
                viz_note = f'{points_in_display}/{len(x)} points within display limits'
                ax.text(0.03, 0.03, viz_note, transform=ax.transAxes, fontsize=8,
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
        
        ax.set_title(f'Replicate {i+1} Density Plot\nPearson r: {metrics[rep]["pearson_corr"]:.3f}', fontsize=12)
        
        # Set axis limits for display only
        if display_limit is not None:
            ax.set_xlim(0, display_limit)
            ax.set_ylim(0, display_limit)
        else:
            # Calculate reasonable limits from the data
            max_display = max(max(x), max(y)) * 1.05
            ax.set_xlim(0, max_display)
            ax.set_ylim(0, max_display)
        
        # Add a line of equality
        if display_limit is not None:
            ax.plot([0, display_limit], [0, display_limit], 'k--', alpha=0.5)
        else:
            max_val = max(max(x), max(y))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Plot replicate 3 in the middle row, left column
    ax_rep3 = fig.add_subplot(gs[1, 0])
    
    data = transformed_data_by_rep['rep3']
    x = data['Neg_Log_Delta_S']
    y = data['Neg_Log_Non_Additivity']
    
    # Create hexbin plot for rep3 (no filtering)
    h = ax_rep3.hexbin(x, y, gridsize=40, cmap='viridis', mincnt=1)
    
    # Add colorbar
    cbar = plt.colorbar(h, ax=ax_rep3)
    cbar.set_label('Count')
    
    # Customize the plot
    ax_rep3.set_xlabel('-log₁₀(Delta S)', fontsize=10)
    ax_rep3.set_ylabel('-log₁₀(Non-Additivity)', fontsize=10)
    
    # Add linear fit line
    if not np.isnan(metrics['rep3']['slope']):
        if display_limit is not None:
            x_range = np.array([0, display_limit])
        else:
            x_range = np.array([0, max(x)])
            
        y_fit = metrics['rep3']['slope'] * x_range + metrics['rep3']['intercept']
        ax_rep3.plot(x_range, y_fit, 'r-', linewidth=1.5)
    
    # Count points that would be out of display range
    if display_limit is not None:
        points_in_display = sum((x <= display_limit) & (y <= display_limit))
        points_outside_display = len(x) - points_in_display
        
        # Add a note about points outside display limits
        if points_outside_display > 0:
            viz_note = f'{points_in_display}/{len(x)} points within display limits'
            ax_rep3.text(0.03, 0.03, viz_note, transform=ax_rep3.transAxes, fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    
    ax_rep3.set_title(f'Replicate 3 Density Plot\nPearson r: {metrics["rep3"]["pearson_corr"]:.3f}', fontsize=12)
    
    # Set axis limits for display only
    if display_limit is not None:
        ax_rep3.set_xlim(0, display_limit)
        ax_rep3.set_ylim(0, display_limit)
    else:
        max_display = max(max(x), max(y)) * 1.05
        ax_rep3.set_xlim(0, max_display)
        ax_rep3.set_ylim(0, max_display)
    
    # Add a line of equality
    if display_limit is not None:
        ax_rep3.plot([0, display_limit], [0, display_limit], 'k--', alpha=0.5)
    else:
        max_val = max(max(x), max(y))
        ax_rep3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Plot the average data in the middle row, right column
    ax_avg = fig.add_subplot(gs[1, 1])
    
    x = transformed_avg_data['Neg_Log_Delta_S']
    y = transformed_avg_data['Neg_Log_Non_Additivity']
    
    # Create hexbin plot for average data (no filtering)
    h = ax_avg.hexbin(x, y, gridsize=40, cmap='viridis', mincnt=1)
    
    # Add colorbar
    cbar = plt.colorbar(h, ax=ax_avg)
    cbar.set_label('Count')
    
    # Customize the plot
    ax_avg.set_xlabel('-log₁₀(Delta S)', fontsize=10)
    ax_avg.set_ylabel('-log₁₀(Non-Additivity)', fontsize=10)
    
    # Add linear fit line
    if not np.isnan(metrics['average']['slope']):
        if display_limit is not None:
            x_range = np.array([0, display_limit])
        else:
            x_range = np.array([0, max(x)])
            
        y_fit = metrics['average']['slope'] * x_range + metrics['average']['intercept']
        ax_avg.plot(x_range, y_fit, 'r-', linewidth=1.5)
    
    # Count points that would be out of display range
    if display_limit is not None:
        points_in_display = sum((x <= display_limit) & (y <= display_limit))
        points_outside_display = len(x) - points_in_display
        
        # Add a note about points outside display limits
        if points_outside_display > 0:
            viz_note = f'{points_in_display}/{len(x)} points within display limits'
            ax_avg.text(0.03, 0.03, viz_note, transform=ax_avg.transAxes, fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
    
    ax_avg.set_title(f'Average of Replicates Density Plot\nPearson r: {metrics["average"]["pearson_corr"]:.3f}', fontsize=12)
    
    # Set axis limits for display only
    if display_limit is not None:
        ax_avg.set_xlim(0, display_limit)
        ax_avg.set_ylim(0, display_limit)
    else:
        max_display = max(max(x), max(y)) * 1.05
        ax_avg.set_xlim(0, max_display)
        ax_avg.set_ylim(0, max_display)
    
    # Add a title to the entire figure
    fig.suptitle('-log₁₀ Transformation: Delta S vs Non-Additivity Density Plots', fontsize=16, y=0.98)
    
    # Add an explanation panel in the bottom row
    ax_explanation = fig.add_subplot(gs[2, :])
    ax_explanation.axis('off')  # Hide the axes
    ax_explanation.set_title("About Density Plots", fontsize=14)
    
    explanation_text = (
        "Density Visualization Explanation:\n\n"
        "These hexbin plots show the density of data points using a color scale, with darker colors indicating "
        "higher concentrations of points. This visualization helps to reveal patterns in regions where many points overlap.\n\n"
        "Key observations:\n"
        "• Areas with higher point density appear as darker hexagons\n"
        "• The red line shows the linear fit of the -log transformed data\n"
        "• The dashed line shows where -log(Delta S) = -log(Non-Additivity)\n"
        "• The -log transformation emphasizes small values near zero, making them appear as larger numbers\n"
        "• Points farther right/up represent smaller original values (closer to zero)\n\n"
        "When interpreting these plots, remember that the -log transformation inverts the scale: "
        "smaller values in the original data appear as larger values in the transformed space."
    )
    
    ax_explanation.text(0.02, 0.99, explanation_text, fontsize=10, 
                      va='top', ha='left', transform=ax_explanation.transAxes,
                      bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 5})
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Density panel plot with -log transformation saved to: {output_file}")

if __name__ == "__main__":
    # Base path for CSV files 
    base_path = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory"
    output_file = os.path.join(output_dir, "delta_vs_nonadditivity_neglog_panel_plot.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate panel plot with -log transformation
    print("Generating -log₁₀ transformation panel plots of Delta S vs Non-Additivity...")
    plot_all_replicates_plus_average(base_path, output_file)
    print("\nAll plots generated successfully!")