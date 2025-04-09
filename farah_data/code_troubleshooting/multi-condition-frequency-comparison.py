import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import glob

def compare_frequencies(condition, replicate, input_dir, output_dir):
    """
    Compare -log10 transformed estimated frequencies from model with actual frequencies
    for a specific condition and replicate.
    
    Parameters:
    -----------
    condition : str
        Experimental condition (Clim, Nlim, or Switch)
    replicate : int
        Replicate number (1, 2, or 3)
    input_dir : str
        Directory containing input files
    output_dir : str
        Directory where output files will be saved
    """
    # File paths
    estimated_file = os.path.join(input_dir, f'results_{condition}_rep{replicate}_FitSeq.csv')
    actual_file = os.path.join(input_dir, f'pyfitseq_input_{condition}_rep{replicate}.csv')
    
    # Check if files exist
    if not os.path.exists(estimated_file):
        print(f"Error: Estimated data file not found at {estimated_file}")
        return None
        
    if not os.path.exists(actual_file):
        print(f"Error: Actual data file not found at {actual_file}")
        return None
    
    # Load the data
    print(f"Loading data files for {condition} replicate {replicate}...")
    print(f"  Estimated file: {estimated_file}")
    print(f"  Actual file: {actual_file}")
    
    estimated_df = pd.read_csv(estimated_file)
    
    # For actual data, we don't have headers
    actual_df = pd.read_csv(actual_file, header=None)
    
    # Add strain_id to actual data for merging
    actual_df['strain_id'] = range(len(actual_df))
    
    # Print shapes to verify
    print(f"Estimated data shape: {estimated_df.shape}")
    print(f"Actual data shape: {actual_df.shape}")
    
    # Identify timepoint columns in estimated data
    timepoint_cols = [col for col in estimated_df.columns if col.startswith('Estimated_Read_Number_t')]
    n_timepoints = len(timepoint_cols)
    
    print(f"Detected {n_timepoints} timepoints in estimated data")
    
    # Ensure actual data has at least the same number of timepoints
    if actual_df.shape[1] - 1 < n_timepoints:  # -1 for strain_id column
        print(f"Warning: Actual data has fewer columns ({actual_df.shape[1] - 1}) than expected timepoints ({n_timepoints})")
        n_timepoints = min(n_timepoints, actual_df.shape[1] - 1)
        print(f"Limiting analysis to {n_timepoints} timepoints")
    
    # Calculate frequencies and prepare data for plotting
    print("Calculating frequencies...")
    
    # Store results for each timepoint
    all_plot_data = []
    correlations = []
    point_counts = []
    
    # Find global min/max for consistent plotting
    global_min = float('inf')
    global_max = float('-inf')
    
    # Process each timepoint
    for t in range(n_timepoints):
        # Column names
        est_col = f"Estimated_Read_Number_t{t}"
        act_col = t  # Column index in actual_df
        
        # Calculate totals for frequency calculation
        est_total = estimated_df[est_col].sum()
        act_total = actual_df[act_col].sum()
        
        print(f"Timepoint {t}: Estimated total={est_total}, Actual total={act_total}")
        
        # Calculate frequencies
        est_freq = estimated_df[est_col] / est_total
        act_freq = actual_df[act_col] / act_total
        
        # Combine data
        plot_data = pd.DataFrame({
            'strain_id': range(len(estimated_df)),
            'est_count': estimated_df[est_col],
            'act_count': actual_df[act_col],
            'est_freq': est_freq,
            'act_freq': act_freq
        })
        
        # Filter for non-zero values in both datasets
        valid_data = plot_data[(plot_data['est_count'] > 0) & (plot_data['act_count'] > 0)].copy()
        
        # Calculate -log transformed frequencies (using base 10)
        valid_data['-log_est_freq'] = -np.log10(valid_data['est_freq'])
        valid_data['-log_act_freq'] = -np.log10(valid_data['act_freq'])
        
        # Store data
        all_plot_data.append(valid_data)
        
        # Calculate correlation
        if len(valid_data) > 2:
            corr = np.corrcoef(valid_data['-log_act_freq'], valid_data['-log_est_freq'])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)
        
        point_counts.append(len(valid_data))
        
        # Update global min/max
        if len(valid_data) > 0:
            local_min = min(valid_data['-log_est_freq'].min(), valid_data['-log_act_freq'].min())
            local_max = max(valid_data['-log_est_freq'].max(), valid_data['-log_act_freq'].max())
            
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)
    
    # Add some padding to min/max
    global_min = np.floor(global_min) - 0.5
    global_max = np.ceil(global_max) + 0.5
    
    print(f"Plotting range: {global_min:.2f} to {global_max:.2f}")
    
    # Create condition-specific output directory
    condition_dir = os.path.join(output_dir, condition.lower())
    os.makedirs(condition_dir, exist_ok=True)
    
    # Create plots
    print("Creating plots...")
    
    # 1. Create individual plots for each timepoint
    for t in range(n_timepoints):
        data = all_plot_data[t]
        corr = correlations[t]
        count = point_counts[t]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Skip if no valid data
        if len(data) == 0:
            plt.text(0.5, 0.5, "No valid data points", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax.transAxes, fontsize=14)
            plt.title(f"Timepoint {t} - No valid data")
            continue
        
        # Plot diagonal line (y=x)
        ax.plot([global_min, global_max], [global_min, global_max], 
                'r--', linewidth=1, alpha=0.7, label='y=x')
        
        # Plot data points
        ax.scatter(data['-log_act_freq'], data['-log_est_freq'], 
                  alpha=0.6, s=15, color='#1e88e5',
                  label=f'Strains (n={count}, r={corr:.3f})')
        
        # Set labels and title
        ax.set_xlabel('-log₁₀(Actual Frequency)')
        ax.set_ylabel('-log₁₀(Estimated Frequency)')
        ax.set_title(f'{condition} Rep{replicate} - T{t}: -log₁₀(Estimated) vs -log₁₀(Actual) Frequencies')
        
        # Set consistent axes
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Add text with additional stats
        coverage = count / len(estimated_df) * 100
        ax.text(0.05, 0.95, 
                f"Total strains: {len(estimated_df)}\nCoverage: {coverage:.1f}%",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Save figure
        plt.tight_layout()
        output_file = os.path.join(condition_dir, f"{condition}_rep{replicate}_t{t}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Saved figure to {output_file}")
        plt.close()
    
    # 2. Create a summary figure with correlation across timepoints
    plt.figure(figsize=(10, 6))
    
    # Plot correlation values
    plt.plot(range(n_timepoints), correlations, 'o-', linewidth=2, markersize=8, color='#1e88e5')
    
    # Add data point counts as text
    for t in range(n_timepoints):
        if not np.isnan(correlations[t]):
            plt.text(t, correlations[t] + 0.02, str(point_counts[t]), 
                    ha='center', va='bottom', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Timepoint')
    plt.ylabel('Correlation Coefficient (r)')
    plt.title(f'{condition} Rep{replicate}: Correlation Between -log₁₀(Estimated) and -log₁₀(Actual) Frequencies')
    
    # Set x-ticks
    plt.xticks(range(n_timepoints), [f'T{t}' for t in range(n_timepoints)])
    
    # Set y-limits
    plt.ylim(0, 1.05)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add annotation explaining correlation
    plt.figtext(0.5, 0.01, 
               "Note: Only strains with non-zero reads in both datasets are included.\nCorrelation is between -log₁₀(estimated frequency) and -log₁₀(actual frequency).\nHigher -log values indicate rarer strains.",
               ha='center', fontsize=9)
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    summary_file = os.path.join(condition_dir, f"{condition}_rep{replicate}_correlation_summary.png")
    plt.savefig(summary_file, dpi=300)
    print(f"Saved summary to {summary_file}")
    
    # 3. Create a multi-panel figure showing all timepoints
    if n_timepoints > 0:
        # Calculate grid dimensions
        n_cols = min(5, n_timepoints)
        n_rows = (n_timepoints + n_cols - 1) // n_cols  # Ceiling division
        
        fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
        
        # Create a grid of subplots
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Plot each timepoint
        for t in range(n_timepoints):
            row = t // n_cols
            col = t % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            data = all_plot_data[t]
            
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title(f"T{t}")
                continue
            
            # Plot diagonal line
            ax.plot([global_min, global_max], [global_min, global_max], 
                    'r--', linewidth=1, alpha=0.5)
            
            # Plot data points
            ax.scatter(data['-log_act_freq'], data['-log_est_freq'], 
                      alpha=0.5, s=5, color='#1e88e5')
            
            # Set title with correlation
            ax.set_title(f"T{t} (r={correlations[t]:.3f}, n={point_counts[t]})")
            
            # Set consistent axes
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.2)
            
            # Only add x and y labels for specific subplots
            if row == n_rows - 1:
                ax.set_xlabel('-log₁₀(Actual)')
            
            if col == 0:
                ax.set_ylabel('-log₁₀(Estimated)')
        
        # Add overall title
        fig.suptitle(f'{condition} Rep{replicate}: -log₁₀(Estimated) vs -log₁₀(Actual) Frequencies', 
                    fontsize=14)
        
        # Add annotation
        plt.figtext(0.5, 0.01, 
                   "Note: Only strains with non-zero reads in both estimated and actual datasets are included.\nHigher values indicate rarer strains.",
                   ha='center', fontsize=9)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        grid_file = os.path.join(condition_dir, f"{condition}_rep{replicate}_all_timepoints_grid.png")
        plt.savefig(grid_file, dpi=300)
        print(f"Saved grid plot to {grid_file}")
    
    return {
        'condition': condition,
        'replicate': replicate,
        'n_timepoints': n_timepoints,
        'correlations': correlations,
        'point_counts': point_counts
    }

def create_condition_summary(condition, results, output_dir):
    """Create summary plot for all replicates of a condition"""
    
    plt.figure(figsize=(12, 6))
    
    # Find max timepoints across all replicates
    max_timepoints = max(result['n_timepoints'] for result in results)
    
    # Plot correlation values for each replicate
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, result in enumerate(results):
        rep_num = result['replicate']
        timepoints = range(result['n_timepoints'])
        correlations = result['correlations']
        
        plt.plot(timepoints, correlations, 
                 marker=markers[i], linestyle='-', linewidth=2, markersize=8, 
                 color=colors[i], label=f'Replicate {rep_num}')
    
    # Add labels and title
    plt.xlabel('Timepoint')
    plt.ylabel('Correlation Coefficient (r)')
    plt.title(f'{condition}: Correlation Between Estimated and Actual Frequencies Across Replicates')
    
    # Set x-ticks
    plt.xticks(range(max_timepoints), [f'T{t}' for t in range(max_timepoints)])
    
    # Set y-limits
    plt.ylim(0, 1.05)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Save figure
    condition_dir = os.path.join(output_dir, condition.lower())
    os.makedirs(condition_dir, exist_ok=True)
    
    summary_file = os.path.join(condition_dir, f"{condition}_all_replicates_summary.png")
    plt.tight_layout()
    plt.savefig(summary_file, dpi=300)
    print(f"Saved condition summary to {summary_file}")

def main():
    """Main function to run analysis on all conditions and replicates"""
    
    # Define input and output directories - adjust these paths as needed
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/model_validation/"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List of conditions and replicates to analyze
    conditions = ['Clim', 'Nlim', 'Switch']
    replicates = [1, 2, 3]
    
    # Store results for summary plots
    all_results = {}
    
    # Process each condition and replicate
    for condition in conditions:
        condition_results = []
        for replicate in replicates:
            print(f"\n{'='*50}")
            print(f"Processing {condition} Replicate {replicate}")
            print(f"{'='*50}\n")
            
            result = compare_frequencies(condition, replicate, input_dir, output_dir)
            if result:
                condition_results.append(result)
        
        # Create condition summary if we have results
        if condition_results:
            all_results[condition] = condition_results
            create_condition_summary(condition, condition_results, output_dir)
    
    # Create overall heatmap summary
    create_overall_summary(all_results, output_dir)
    
    print("\nAnalysis complete!")

def create_overall_summary(all_results, output_dir):
    """Create an overall summary heatmap of correlations across all conditions and replicates"""
    
    # Extract data for heatmap
    conditions = list(all_results.keys())
    
    # Calculate average correlation for each condition/replicate
    avg_correlations = {}
    for condition in conditions:
        avg_correlations[condition] = []
        for result in all_results[condition]:
            # Calculate average correlation, ignoring NaN values
            corrs = result['correlations']
            valid_corrs = [c for c in corrs if not np.isnan(c)]
            if valid_corrs:
                avg = np.mean(valid_corrs)
            else:
                avg = np.nan
            avg_correlations[condition].append((result['replicate'], avg))
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Bar plot of average correlations
    bar_width = 0.25
    index = np.arange(len(conditions))
    
    for i in range(3):  # Assuming 3 replicates
        rep_data = []
        for condition in conditions:
            # Find data for this replicate, or use NaN if not available
            rep_val = next((avg for rep, avg in avg_correlations[condition] if rep == i+1), np.nan)
            rep_data.append(rep_val)
        
        plt.bar(index + i * bar_width, rep_data, bar_width,
                label=f'Replicate {i+1}', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Condition')
    plt.ylabel('Average Correlation Coefficient (r)')
    plt.title('Average Correlation Between Estimated and Actual Frequencies\nAcross All Conditions and Replicates')
    
    plt.xticks(index + bar_width, conditions)
    plt.ylim(0, 1.0)
    
    # Add grid and legend
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    
    # Save figure
    summary_file = os.path.join(output_dir, "overall_correlation_summary.png")
    plt.tight_layout()
    plt.savefig(summary_file, dpi=300)
    print(f"Saved overall summary to {summary_file}")

if __name__ == "__main__":
    main()