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

def create_replicate_comparison_plot(data1, data2, ax, title, condition, rep1, rep2):
    """
    Create a scatter plot comparing two replicates of the same condition.
    
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
    
    Returns:
    --------
    tuple: (correlation, p_value, slope, intercept)
        Statistics used for the plot
    """
    # Create a DataFrame of aligned data points
    # Take the minimum length in case they have different numbers of rows
    min_len = min(len(data1), len(data2))
    
    combined_data = pd.DataFrame({
        f'Rep{rep1}': data1.iloc[:min_len].reset_index(drop=True),
        f'Rep{rep2}': data2.iloc[:min_len].reset_index(drop=True)
    })
    
    # Remove NaN values
    combined_data = combined_data.dropna()
    
    # Calculate Pearson correlation
    correlation, p_value = pearsonr(combined_data[f'Rep{rep1}'], combined_data[f'Rep{rep2}'])
    
    # Calculate Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(combined_data[f'Rep{rep1}'], combined_data[f'Rep{rep2}'])
    
    # Calculate regression line
    if len(combined_data) > 1:
        z = np.polyfit(combined_data[f'Rep{rep1}'], combined_data[f'Rep{rep2}'], 1)
        slope, intercept = z[0], z[1]
    else:
        slope, intercept = np.nan, np.nan
    
    # Set grid styling
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create the scatter plot
    scatter = ax.scatter(
        combined_data[f'Rep{rep1}'],
        combined_data[f'Rep{rep2}'],
        alpha=0.6,
        s=25,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Determine data range
    xmin, xmax = combined_data[f'Rep{rep1}'].min(), combined_data[f'Rep{rep1}'].max()
    ymin, ymax = combined_data[f'Rep{rep2}'].min(), combined_data[f'Rep{rep2}'].max()
    
    # Add a regression line
    if not np.isnan(slope):
        # Extend the line across the full x data range
        x_line = np.array([xmin, xmax])
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5)
    
    # Add a line of perfect equality (y = x)
    all_min = min(xmin, ymin)
    all_max = max(xmax, ymax)
    ax.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.5, label='y = x')
    
    # Add reference lines at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.4, linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel(f'{condition} Rep{rep1} Fitness', fontsize=10)
    ax.set_ylabel(f'{condition} Rep{rep2} Fitness', fontsize=10)
    
    # Set axis limits to show all data with a little padding
    padding_x = (xmax - xmin) * 0.05
    padding_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - padding_x, xmax + padding_x)
    ax.set_ylim(ymin - padding_y, ymax + padding_y)
    
    # Use a common title format
    if np.isnan(correlation):
        ax.set_title(f'{title}\nInsufficient data', fontsize=11)
    else:
        ax.set_title(f'{title}\nPearson r: {correlation:.3f}', fontsize=11)
    
    # Add a text box with statistics
    if not np.isnan(correlation):
        if not np.isnan(slope):
            fit_eq = f'y = {slope:.3f}x + {intercept:.3f}'
        else:
            fit_eq = 'Insufficient data for fit'
            
        stats_text = (
            f'n={len(combined_data)}\n'
            f'Pearson r={correlation:.3f} (p={p_value:.2e})\n'
            f'Spearman ρ={spearman_corr:.3f} (p={spearman_p:.2e})\n'
            f'{fit_eq}'
        )
        
        ax.text(
            0.03, 0.97, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5},
            fontsize=8
        )
    
    return correlation, p_value, spearman_corr, spearman_p, slope, intercept

def create_replicate_comparison_panel(base_path, output_file):
    """
    Create a panel of plots comparing replicates within each condition.
    
    Parameters:
    -----------
    base_path : str
        Base path to the directory containing CSV files
    output_file : str
        Path to save the output plot
    """
    # Load all data
    print("Loading data for all conditions and replicates...")
    
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
    gs = GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.3, height_ratios=[1, 1, 1, 0.4])
    
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
            
            # Create the comparison plot
            correlation, p_value, spearman_corr, spearman_p, slope, intercept = create_replicate_comparison_plot(
                fitness_data[condition][rep1],
                fitness_data[condition][rep2],
                ax, title, condition, rep1, rep2
            )
            
            # Store metrics
            metrics[condition][f'rep{rep1}_vs_rep{rep2}'] = {
                'pearson_r': correlation,
                'pearson_p': p_value,
                'spearman_r': spearman_corr,
                'spearman_p': spearman_p,
                'slope': slope,
                'intercept': intercept,
                'n_points': len(fitness_data[condition][rep1].dropna().iloc[:len(fitness_data[condition][rep2])])
            }
    
    # Add a main title to the figure
    fig.suptitle('Replicate Comparisons Within Each Condition', fontsize=18, y=0.98)
    
    # Create a summary of metrics
    summary_text = "Reproducibility Summary (Pearson r):\n\n"
    
    # Organize by condition
    for condition in conditions:
        summary_text += f"{condition} Replicates:\n"
        for comparison, stats in metrics[condition].items():
            rep1, rep2 = comparison.split('_vs_')
            summary_text += f"  {rep1.capitalize()} vs {rep2.capitalize()}: r={stats['pearson_r']:.3f} (p={stats['pearson_p']:.2e}, n={stats['n_points']})\n"
        summary_text += "\n"
    
    # Calculate average correlation within each condition
    summary_text += "Average Correlations Within Conditions:\n"
    for condition in conditions:
        correlations = [stats['pearson_r'] for stats in metrics[condition].values() 
                       if not np.isnan(stats['pearson_r'])]
        if correlations:
            avg_corr = np.mean(correlations)
            summary_text += f"  {condition}: {avg_corr:.3f}\n"
        else:
            summary_text += f"  {condition}: Insufficient data\n"
    
    # Calculate which condition has the most reproducible replicates
    condition_avg_corrs = {}
    for condition in conditions:
        correlations = [stats['pearson_r'] for stats in metrics[condition].values() 
                       if not np.isnan(stats['pearson_r'])]
        if correlations:
            condition_avg_corrs[condition] = np.mean(correlations)
    
    if condition_avg_corrs:
        most_reproducible = max(condition_avg_corrs, key=condition_avg_corrs.get)
        summary_text += f"\nMost reproducible condition: {most_reproducible} (avg r={condition_avg_corrs[most_reproducible]:.3f})\n"
    
    # Add a dedicated panel for the statistics summary that spans all columns in the bottom row
    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis('off')  # Hide the axes
    ax_stats.set_title("Reproducibility Metrics", fontsize=14)
    
    # Add the summary text to the dedicated panel
    ax_stats.text(0.02, 0.99, summary_text, 
                 va='top', ha='left', transform=ax_stats.transAxes,
                 fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 10})
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Leave space for main title
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPanel plot saved to: {output_file}")
    print("\nReproducibility Summary:")
    print(summary_text)

if __name__ == "__main__":
    # Base path for CSV files 
    base_path = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific"
    
    # Output directory
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/memory"
    output_file = os.path.join(output_dir, "replicate_comparison_panel_plot.png")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate panel plot
    print("Generating panel plot comparing replicates within each condition...")
    create_replicate_comparison_panel(base_path, output_file)
    print("\nPanel plot generated successfully!")