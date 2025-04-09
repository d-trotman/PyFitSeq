import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def compare_estimated_vs_actual_frequencies():
    """
    Compare ln-transformed estimated frequencies from model with ln-transformed 
    actual frequencies from raw data for replicate 1.
    """
    # Define input and output directories - adjust these paths as needed
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/model_validation/"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths - construct full paths
    estimated_file = os.path.join(input_dir, 'results_Clim_rep1_FitSeq.csv')
    actual_file = os.path.join(input_dir, 'pyfitseq_input_Clim_rep1.csv')
    
    # Check if files exist
    if not os.path.exists(estimated_file):
        print(f"Error: Estimated data file not found at {estimated_file}")
        print(f"Current working directory: {os.getcwd()}")
        return
        
    if not os.path.exists(actual_file):
        print(f"Error: Actual data file not found at {actual_file}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Load the data
    print(f"Loading data files...")
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
    
    # Verify we have the expected columns in estimated data
    expected_columns = [f"Estimated_Read_Number_t{t}" for t in range(10)]
    missing_columns = [col for col in expected_columns if col not in estimated_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing expected columns in estimated data: {missing_columns}")
    
    # Calculate frequencies and prepare data for plotting
    print("Calculating frequencies...")
    
    # Number of timepoints
    n_timepoints = 10
    
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
    global_min = np.floor(global_min) - 1
    global_max = np.ceil(global_max) + 1
    
    print(f"Plotting range: {global_min:.2f} to {global_max:.2f}")
    
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
        ax.set_title(f'Timepoint {t}: -log₁₀(Estimated) vs -log₁₀(Actual) Frequencies')
        
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
        output_file = os.path.join(output_dir, f"estimated_vs_actual_t{t}.png")
        plt.savefig(output_file, dpi=300)
        print(f"Saved figure to {output_file}")
        plt.close()
    
    # 2. Create a summary figure with correlation across timepoints
    plt.figure(figsize=(10, 6))
    
    # Plot correlation values
    plt.plot(range(n_timepoints), correlations, 'o-', linewidth=2, markersize=8, color='#1e88e5')
    
    # Add data point counts as text
    for t in range(n_timepoints):
        plt.text(t, correlations[t] + 0.02, str(point_counts[t]), 
                ha='center', va='bottom', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Timepoint')
    plt.ylabel('Correlation Coefficient (r)')
    plt.title('Correlation Between Estimated and Actual Frequencies Across Timepoints')
    
    # Set x-ticks
    plt.xticks(range(n_timepoints), [f'T{t}' for t in range(n_timepoints)])
    
    # Set y-limits
    plt.ylim(0, 1.05)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add annotation explaining correlation
    plt.figtext(0.5, 0.01, 
               "Note: Only strains with non-zero reads in both datasets are included.\nCorrelation is between ln(estimated frequency) and ln(actual frequency).",
               ha='center', fontsize=9)
    
    # Save figure
    plt.tight_layout()
    summary_file = os.path.join(output_dir, "correlation_summary.png")
    plt.savefig(summary_file, dpi=300)
    print(f"Saved summary to {summary_file}")
    
    # 3. Create a multi-panel figure (2x5 grid) showing all timepoints
    fig = plt.figure(figsize=(15, 8))
    
    # Create a grid of subplots
    gs = GridSpec(2, 5, figure=fig)
    
    # Plot each timepoint
    for t in range(n_timepoints):
        row = t // 5
        col = t % 5
        
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
        if row == 1:
            ax.set_xlabel('-log₁₀(Actual)')
        
        if col == 0:
            ax.set_ylabel('-log₁₀(Estimated)')
    
    # Add overall title
    fig.suptitle('Comparison of -log₁₀(Estimated) vs -log₁₀(Actual) Frequencies Across All Timepoints', 
                fontsize=14)
    
    # Add annotation
    plt.figtext(0.5, 0.01, 
               "Note: Only strains with non-zero reads in both estimated and actual datasets are included.\nHigher values indicate rarer strains.",
               ha='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    grid_file = os.path.join(output_dir, "all_timepoints_grid.png")
    plt.savefig(grid_file, dpi=300)
    print(f"Saved grid plot to {grid_file}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    compare_estimated_vs_actual_frequencies()