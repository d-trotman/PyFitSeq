import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
from matplotlib.gridspec import GridSpec

def load_data(base_path, condition, rep_number):
    """
    Load fitness data for a specific condition and replicate.
    
    Parameters:
    -----------
    base_path : str
        Base path to the directory containing CSV files
    condition : str
        Condition name ('Clim', 'Nlim', or 'Switch')
    rep_number : int
        Replicate number (1, 2, or 3)
        
    Returns:
    --------
    pandas.Series
        Series containing fitness values
    """
    file_path = os.path.join(base_path, f"results_{condition}_rep{rep_number}_FitSeq.csv")
    
    try:
        data = pd.read_csv(file_path)
        fitness = data['Estimated_Fitness']
        print(f"  Loaded {condition} rep{rep_number}: {len(fitness)} rows")
        return fitness
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def create_filtered_comparison_plot(data1, data2, ax, title, condition, rep1, rep2, threshold=0.05):
    """
    Create a scatter plot comparing two replicates of the same condition,
    filtering out points where both values have absolute fitness < threshold.
    
    Parameters:
    -----------
    data1 : pandas.Series
        Fitness data for first replicate
    data2 : pandas.Series
        Fitness data for second replicate
    ax : matplotlib.axes.Axes
        The axis to plot on
    title : str
        Title for the plot
    condition : str
        Condition name ('Clim', 'Nlim', or 'Switch')
    rep1 : int
        First replicate number
    rep2 : int
        Second replicate number
    threshold : float
        Absolute fitness threshold for filtering (default: 0.05)
    
    Returns:
    --------
    dict: Statistics and counts for the filtered and unfiltered data
    """
    # Create a DataFrame of aligned data points
    # Take the minimum length in case they have different numbers of rows
    min_len = min(len(data1), len(data2))
    
    combined_data = pd.DataFrame({
        f'Rep{rep1}': data1.iloc[:min_len].reset_index(drop=True),
        f'Rep{rep2}': data2.iloc[:min_len].reset_index(drop=True)
    })
    
    # Remove NaN values before any filtering
    combined_data = combined_data.dropna()
    total_points = len(combined_data)
    
    # Store the unfiltered correlations for comparison
    unfiltered_corr, unfiltered_p = pearsonr(combined_data[f'Rep{rep1}'], combined_data[f'Rep{rep2}'])
    unfiltered_spearman, unfiltered_spearman_p = spearmanr(combined_data[f'Rep{rep1}'], combined_data[f'Rep{rep2}'])
    
    # Apply threshold filter - keep points where at least one replicate has |fitness| >= threshold
    filtered_data = combined_data[
        (abs(combined_data[f'Rep{rep1}']) >= threshold) | 
        (abs(combined_data[f'Rep{rep2}']) >= threshold)
    ]
    
    # Count how many points were filtered out
    filtered_out = total_points - len(filtered_data)
    
    # Calculate Pearson correlation on filtered data
    if len(filtered_data) > 1:
        correlation, p_value = pearsonr(filtered_data[f'Rep{rep1}'], filtered_data[f'Rep{rep2}'])
        spearman_corr, spearman_p = spearmanr(filtered_data[f'Rep{rep1}'], filtered_data[f'Rep{rep2}'])
    else:
        correlation, p_value = np.nan, np.nan
        spearman_corr, spearman_p = np.nan, np.nan
        print(f"Warning: No points left after filtering for {condition} {rep1} vs {rep2}")
    
    # Calculate regression line on filtered data
    if len(filtered_data) > 1:
        z = np.polyfit(filtered_data[f'Rep{rep1}'], filtered_data[f'Rep{rep2}'], 1)
        slope, intercept = z[0], z[1]
    else:
        slope, intercept = np.nan, np.nan
    
    # Set grid styling
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot with filtered data
    scatter = ax.scatter(
        filtered_data[f'Rep{rep1}'],
        filtered_data[f'Rep{rep2}'],
        alpha=0.8,
        s=25,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Determine data range from filtered data
    if len(filtered_data) > 0:
        xmin, xmax = filtered_data[f'Rep{rep1}'].min(), filtered_data[f'Rep{rep1}'].max()
        ymin, ymax = filtered_data[f'Rep{rep2}'].min(), filtered_data[f'Rep{rep2}'].max()
    else:
        xmin, xmax, ymin, ymax = -threshold, threshold, -threshold, threshold
    
    # Add a regression line for filtered data
    if not np.isnan(slope):
        # Extend the line across the full x data range
        x_line = np.array([xmin, xmax])
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5)
    
    # Add a line of perfect equality (y = x)
    all_min = min(xmin, ymin)
    all_max = max(xmax, ymax)
    ax.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.5, label='y = x')
    
    # Add threshold reference lines
    ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)
    ax.axhline(y=-threshold, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)
    ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)
    ax.axvline(x=-threshold, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)
    
    # Add reference lines at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.4, linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel(f'{condition} Rep{rep1} Fitness', fontsize=10)
    ax.set_ylabel(f'{condition} Rep{rep2} Fitness', fontsize=10)
    
    # Set axis limits to show all data with a little padding
    padding_x = max((xmax - xmin) * 0.05, 0.01)
    padding_y = max((ymax - ymin) * 0.05, 0.01)
    ax.set_xlim(xmin - padding_x, xmax + padding_x)
    ax.set_ylim(ymin - padding_y, ymax + padding_y)
    
    # Use a common title format with filtered point count
    if np.isnan(correlation):
        ax.set_title(f'{title}\nNo data after filtering', fontsize=11)
    else:
        percent_kept = (len(filtered_data) / total_points) * 100
        ax.set_title(f'{title}\nFiltered r: {correlation:.3f} ({percent_kept:.1f}% of points)', fontsize=10)
    
    # Add a text box with statistics
    if not np.isnan(correlation):
        if not np.isnan(slope):
            fit_eq = f'y = {slope:.3f}x + {intercept:.3f}'
        else:
            fit_eq = 'Insufficient data for fit'
            
        stats_text = (
            f'Filtered data (|fitness| ≥ {threshold}):\n'
            f'  Points: {len(filtered_data)}/{total_points} ({percent_kept:.1f}%)\n'
            f'  Pearson r={correlation:.3f} (p={p_value:.2e})\n'
            f'  Spearman ρ={spearman_corr:.3f} (p={spearman_p:.2e})\n\n'
            f'Unfiltered correlations:\n'
            f'  Pearson r={unfiltered_corr:.3f} (p={unfiltered_p:.2e})\n'
            f'  Spearman ρ={unfiltered_spearman:.3f}'
        )
        
        ax.text(
            0.03, 0.97, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
            fontsize=8
        )
    
    # Return statistics for both filtered and unfiltered data
    stats = {
        'total_points': total_points,
        'filtered_points': len(filtered_data),
        'percent_kept': percent_kept if 'percent_kept' in locals() else 0,
        'filtered_pearson_r': correlation,
        'filtered_pearson_p': p_value,
        'filtered_spearman_r': spearman_corr,
        'filtered_spearman_p': spearman_p,
        'unfiltered_pearson_r': unfiltered_corr,
        'unfiltered_pearson_p': unfiltered_p,
        'unfiltered_spearman_r': unfiltered_spearman,
        'unfiltered_spearman_p': unfiltered_spearman_p,
        'slope': slope,
        'intercept': intercept
    }
    
    return stats

def create_replicate_comparison_panel(base_path, output_file, threshold=0.05):
    """
    Create a panel of plots comparing replicates within each condition,
    filtering out points where both values have absolute fitness < threshold.
    
    Parameters:
    -----------
    base_path : str
        Base path to the directory containing CSV files
    output_file : str
        Path to save the output plot
    threshold : float
        Absolute fitness threshold for filtering (default: 0.05)
    """
    # Load all data
    print(f"Loading data for all conditions and replicates (will filter |fitness| < {threshold})...")
    
    # Dictionary to store fitness data
    fitness_data = {}
    
    # Load data for each condition and replicate
    for condition in ['Clim', 'Nlim', 'Switch']:
        fitness_data[condition] = {}
        for rep in [1, 2, 3]:
            fitness_data[condition][rep] = load_data(base_path, condition, rep)
    
    # Create a figure with a 4-row grid:
    # - Rows 1-3: 3×3 grid of plots for condition comparisons
    # - Row 4: Statistics summary panel spanning all columns
    fig = plt.figure(figsize=(16, 18))
    
    # Use GridSpec for more control over spacing
    gs = GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.3, height_ratios=[1, 1, 1, 0.5])
    
    # List of conditions and replicate comparisons
    conditions = ['Nlim', 'Clim', 'Switch']
    rep_comparisons = [(1, 2), (1, 3), (2, 3)]
    
    # Dictionary to store metrics
    metrics = {}
    
    # Create plots for each condition and replicate comparison
    for i, condition in enumerate(conditions):
        metrics[condition] = {}
        
        for j, (rep1, rep2) in enumerate(rep_comparisons):
            # Get the subplot
            ax = fig.add_subplot(gs[i, j])
            
            # Plot title
            title = f'{condition}: Rep{rep1} vs Rep{rep2}'
            
            # Create the comparison plot with filtering
            stats = create_filtered_comparison_plot(
                fitness_data[condition][rep1],
                fitness_data[condition][rep2],
                ax, title, condition, rep1, rep2, threshold
            )
            
            # Store metrics
            metrics[condition][f'rep{rep1}_vs_rep{rep2}'] = stats
    
    # Add a main title to the figure
    fig.suptitle(f'Replicate Comparisons Within Each Condition\nFiltered to Show Points with |Fitness| ≥ {threshold}', 
                fontsize=18, y=0.98)
    
    # Create a summary of metrics
    summary_text = "Reproducibility Summary (Filtered Data):\n\n"
    
    # Organize by condition
    for condition in conditions:
        summary_text += f"{condition} Replicates (|fitness| ≥ {threshold}):\n"
        
        # Calculate average filtering stats across all comparisons for this condition
        total_points = [stats['total_points'] for stats in metrics[condition].values()]
        filtered_points = [stats['filtered_points'] for stats in metrics[condition].values()]
        avg_total = np.mean(total_points)
        avg_filtered = np.mean(filtered_points)
        avg_percent = (avg_filtered / avg_total) * 100 if avg_total > 0 else 0
        
        summary_text += f"  Average points kept: {avg_filtered:.0f}/{avg_total:.0f} ({avg_percent:.1f}%)\n"
        
        # Add individual comparison metrics
        for comparison, stats in metrics[condition].items():
            rep1, rep2 = comparison.split('_vs_')
            if not np.isnan(stats['filtered_pearson_r']):
                improvement = stats['filtered_pearson_r'] - stats['unfiltered_pearson_r']
                summary_text += (
                    f"  {rep1.capitalize()} vs {rep2.capitalize()}: "
                    f"r={stats['filtered_pearson_r']:.3f} ({stats['percent_kept']:.1f}% of points, "
                    f"Δr={improvement:+.3f})\n"
                )
            else:
                summary_text += f"  {rep1.capitalize()} vs {rep2.capitalize()}: Insufficient data after filtering\n"
        
        # Add unfiltered vs filtered correlation comparison
        filtered_rs = [stats['filtered_pearson_r'] for stats in metrics[condition].values() 
                     if not np.isnan(stats['filtered_pearson_r'])]
        unfiltered_rs = [stats['unfiltered_pearson_r'] for stats in metrics[condition].values()]
        
        if filtered_rs:
            avg_filtered_r = np.mean(filtered_rs)
            avg_unfiltered_r = np.mean(unfiltered_rs)
            improvement = avg_filtered_r - avg_unfiltered_r
            summary_text += f"  Average correlation: r={avg_filtered_r:.3f} (Δr={improvement:+.3f} vs unfiltered)\n"
        else:
            summary_text += f"  Average correlation: Insufficient data after filtering\n"
            
        summary_text += "\n"
    
    # Calculate which condition has the most reproducible replicates after filtering
    condition_avg_filtered_corrs = {}
    for condition in conditions:
        filtered_rs = [stats['filtered_pearson_r'] for stats in metrics[condition].values() 
                     if not np.isnan(stats['filtered_pearson_r'])]
        if filtered_rs:
            condition_avg_filtered_corrs[condition] = np.mean(filtered_rs)
    
    if condition_avg_filtered_corrs:
        most_reproducible = max(condition_avg_filtered_corrs, key=condition_avg_filtered_corrs.get)
        summary_text += (
            f"Most reproducible condition after filtering: "
            f"{most_reproducible} (avg r={condition_avg_filtered_corrs[most_reproducible]:.3f})\n"
        )
    
    # Add additional analysis comparing filtered vs. unfiltered
    summary_text += "\nEffect of Filtering on Correlations:\n"
    
    # Calculate average improvement across all conditions
    all_improvements = []
    all_spearman_improvements = []
    
    for condition in conditions:
        for stats in metrics[condition].values():
            if not np.isnan(stats['filtered_pearson_r']) and not np.isnan(stats['unfiltered_pearson_r']):
                improvement = stats['filtered_pearson_r'] - stats['unfiltered_pearson_r']
                all_improvements.append(improvement)
                
                spearman_improvement = stats['filtered_spearman_r'] - stats['unfiltered_spearman_r']
                all_spearman_improvements.append(spearman_improvement)
    
    if all_improvements:
        avg_improvement = np.mean(all_improvements)
        summary_text += f"  Average Pearson correlation improvement: Δr={avg_improvement:+.3f}\n"
    
    if all_spearman_improvements:
        avg_spearman_improvement = np.mean(all_spearman_improvements)
        summary_text += f"  Average Spearman correlation improvement: Δρ={avg_spearman_improvement:+.3f}\n"
    
    # Add a dedicated panel for the statistics summary that spans all columns in the bottom row
    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis('off')  # Hide the axes
    ax_stats.set_title(f"Reproducibility Metrics (Filtered at |fitness| ≥ {threshold})", fontsize=14)
    
    # Add the summary text to the dedicated panel
    ax_stats.text(0.02, 0.99, summary_text, 
                 va='top', ha='left', transform=ax_stats.transAxes,
                 fontsize=11, bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 10})
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Leave space for main title
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFiltered panel plot saved to: {output_file}")
    print(f"Applied threshold: |fitness| ≥ {threshold}")
    print("\nReproducibility Summary:")
    print(summary_text)

if __name__ == "__main__":
    # Base path for CSV files 
    base_path = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory"
    output_file = os.path.join(output_dir, "filtered_replicate_comparison_panel_plot.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Threshold for filtering (only show points with |fitness| >= threshold)
    threshold = 0.05
    
    # Generate panel plot with filtering
    print(f"Generating panel plot comparing replicates within each condition...")
    create_replicate_comparison_panel(base_path, output_file, threshold)
    print("\nFiltered panel plot generated successfully!")