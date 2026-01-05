import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import matplotlib.cm as cm

def analyze_yeast_replicates_exclude_zeros():
    # Define input and output directories
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/troubleshooting/frequency_no_zeros"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths for the three replicates
    file_paths = [
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep1.csv'),
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep2.csv'), 
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep3.csv')
    ]
    
    # Load data from each replicate file
    replicates = []
    for i, file_path in enumerate(file_paths):
        try:
            # Read with header=None since there are no column headers
            df = pd.read_csv(file_path, header=None)
            
            # Print shape to confirm all columns are loaded
            print(f"Replicate {i+1} data shape: {df.shape}")
            
            # Add a strain ID column
            df['strain_id'] = range(len(df))
            replicates.append(df)
            
            # Verify we have all time columns plus strain_id
            print(f"Replicate {i+1} columns after adding strain_id: {df.columns.tolist()}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Calculate frequencies for each replicate, replacing zeros with NaN
    freq_dfs = []
    
    for i, df in enumerate(replicates):
        # Create frequency dataframe
        freq_df = pd.DataFrame()
        freq_df['strain_id'] = df['strain_id']
        
        # Get all time columns (numeric columns excluding strain_id)
        time_cols = [col for col in df.columns if col != 'strain_id']
        print(f"Replicate {i+1} time columns: {time_cols}")
        
        # Track zeros for reporting
        zero_counts = {}
        
        # Calculate raw frequencies for each time point
        for col in time_cols:
            # Replace zeros with NaN before calculating frequencies
            filtered_series = df[col].replace(0, np.nan)
            
            # Count and report zeros
            zero_count = (df[col] == 0).sum()
            zero_counts[col] = zero_count
            print(f"Replicate {i+1}, timepoint {col}: {zero_count} strains with zero reads excluded ({zero_count/len(df)*100:.1f}%)")
            
            # Calculate total reads for this timepoint (excluding zeros)
            total_reads = filtered_series.sum()
            
            if total_reads > 0:
                # Calculate raw frequency (zeros will be NaN)
                freq_df[col] = filtered_series / total_reads
            else:
                # If there are no reads at all (extremely rare case)
                print(f"Warning: Zero total reads at time point {col} in replicate {i+1}")
                freq_df[col] = np.nan
        
        freq_dfs.append(freq_df)
    
    # Calculate ln(freq_i+1/freq_i) for consecutive time points
    ln_ratio_dfs = []
    
    for i, freq_df in enumerate(freq_dfs):
        # Create a new dataframe for ln ratios
        ln_ratio_df = pd.DataFrame()
        ln_ratio_df['strain_id'] = freq_df['strain_id']
        
        # Get time columns (all columns except strain_id)
        time_cols = [col for col in freq_df.columns if col != 'strain_id']
        
        # Make sure time columns are sorted properly if they're numeric
        if all(isinstance(col, int) for col in time_cols):
            time_cols.sort()
        
        print(f"Replicate {i+1} has {len(time_cols)} time points, expecting {len(time_cols)-1} transitions")
        
        # Track excluded strains for reporting
        excluded_counts = {}
        
        # Calculate for each pair of consecutive time points
        for j in range(len(time_cols) - 1):
            current_col = time_cols[j]
            next_col = time_cols[j+1]
            ratio_col = f"ln_ratio_{j}_to_{j+1}"
            
            # Calculate ln ratio - NaN will propagate where either timepoint has zero reads
            with np.errstate(divide='ignore', invalid='ignore'):
                ln_ratio_df[ratio_col] = np.log(freq_df[next_col] / freq_df[current_col])
            
            # Count excluded strains (those with NaN)
            excluded_count = ln_ratio_df[ratio_col].isna().sum()
            excluded_counts[ratio_col] = excluded_count
            print(f"Replicate {i+1}, transition {j} → {j+1}: {excluded_count} strains excluded due to zeros ({excluded_count/len(ln_ratio_df)*100:.1f}%)")
        
        # Verify ratio columns were created correctly
        ratio_cols = [col for col in ln_ratio_df.columns if col.startswith('ln_ratio')]
        print(f"Replicate {i+1} ratio columns: {ratio_cols}")
        
        ln_ratio_dfs.append(ln_ratio_df)
    
    # Get all combinations of replicates
    replicate_pairs = list(combinations(range(len(ln_ratio_dfs)), 2))
    
    # Create plots for each replicate pair
    for pair_idx, (rep1_idx, rep2_idx) in enumerate(replicate_pairs):
        plt.figure(figsize=(12, 10))
        
        # Get ratio columns (should be 9 transitions for 10 time points)
        ratio_cols1 = [col for col in ln_ratio_dfs[rep1_idx].columns if col.startswith('ln_ratio')]
        ratio_cols2 = [col for col in ln_ratio_dfs[rep2_idx].columns if col.startswith('ln_ratio')]
        
        # Find common ratio columns between the two replicates
        common_ratio_cols = sorted(set(ratio_cols1) & set(ratio_cols2))
        print(f"Common ratio columns between Rep {rep1_idx+1} and Rep {rep2_idx+1}: {common_ratio_cols}")
        
        # Number of transitions to plot
        num_transitions = len(common_ratio_cols)
        
        # Create colormap
        colormap = cm.get_cmap('viridis', num_transitions)
        
        # Track min/max for axis scaling
        all_min = float('inf')
        all_max = float('-inf')
        
        # For storing correlation values and valid data point counts
        correlations = []
        valid_counts = []
        
        # Plot each transition with a different color
        for t, ratio_col in enumerate(common_ratio_cols):
            # Extract time points from ratio column name
            transition_label = ratio_col.replace('ln_ratio_', 'T').replace('_to_', '→T')
            
            # Merge data based on strain ID
            merged_df = pd.merge(
                ln_ratio_dfs[rep1_idx][['strain_id', ratio_col]],
                ln_ratio_dfs[rep2_idx][['strain_id', ratio_col]],
                on='strain_id',
                suffixes=('_rep1', '_rep2')
            )
            
            # Get the data for plotting, dropna to exclude NaN values
            valid_df = merged_df.dropna()
            valid_count = len(valid_df)
            valid_counts.append(valid_count)
            
            # Skip if no valid data points
            if valid_count == 0:
                print(f"No valid data points for {transition_label}, skipping.")
                correlations.append(np.nan)
                continue
                
            x = valid_df[f"{ratio_col}_rep1"]
            y = valid_df[f"{ratio_col}_rep2"]
            
            # Skip if we have too few data points for correlation
            if len(x) <= 1:
                print(f"Too few data points for {transition_label}, skipping.")
                correlations.append(np.nan)
                continue
            
            # Update global min/max
            current_min = min(x.min(), y.min())
            current_max = max(x.max(), y.max())
            
            # Handle possibly infinite values from logarithm
            if np.isfinite(current_min):
                all_min = min(all_min, current_min)
            if np.isfinite(current_max):
                all_max = max(all_max, current_max)
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            correlations.append(corr)
            
            # Report data exclusion
            all_strains = len(merged_df)
            excluded = all_strains - valid_count
            print(f"{transition_label}: {valid_count} valid strains used, {excluded} excluded ({excluded/all_strains*100:.1f}%)")
            
            # Plot with specific color for this transition
            plt.scatter(x, y, alpha=0.4, s=5, color=colormap(t/num_transitions), 
                      label=f"{transition_label} (r={corr:.2f}, n={valid_count})")
        
        # Skip rest of plotting if no valid data at all
        if all(np.isnan(corr) for corr in correlations):
            plt.close()
            print(f"No valid correlations for Rep {rep1_idx+1} vs Rep {rep2_idx+1}, skipping plot.")
            continue
            
        # Ensure we have valid min/max for plotting
        if all_min == float('inf') or all_max == float('-inf'):
            # Set default range if no valid data
            all_min, all_max = -5, 5
        
        # Add diagonal line
        buffer = (all_max - all_min) * 0.05  # 5% buffer on each end
        plt.plot([all_min-buffer, all_max+buffer], [all_min-buffer, all_max+buffer], 
                'r--', linewidth=1, label='y=x')
        
        # Set labels and title
        plt.xlabel(f"Replicate {rep1_idx+1} ln(frequency ratio)")
        plt.ylabel(f"Replicate {rep2_idx+1} ln(frequency ratio)")
        plt.title(f"Fitness Comparison: Replicate {rep1_idx+1} vs Replicate {rep2_idx+1}\n(Excluding Zero Reads)")
        
        # Add legend - place it outside the plot area
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Make axes square for better visualization
        plt.axis('square')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Tight layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Save the figure
        output_file = os.path.join(output_dir, f"replicate_{rep1_idx+1}_vs_{rep2_idx+1}_all_transitions_no_zeros.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
        
        # Create a summary plot for this replicate pair showing correlation over time
        plt.figure(figsize=(10, 5))
        
        # Create a subplot grid for correlation and data count
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Time points for x-axis
        time_points = [f"{i} → {i+1}" for i in range(len(correlations))]
        
        # Plot correlation values
        ax1.plot(time_points, correlations, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.grid(True, linestyle='--')
        ax1.set_ylabel('Correlation Coefficient (r)')
        ax1.set_title(f'Comparison: Replicate {rep1_idx+1} vs Replicate {rep2_idx+1} (Excluding Zero Reads)')
        ax1.set_ylim(0, 1.0)
        
        # Plot valid data point counts
        ax2.bar(time_points, valid_counts, color='green', alpha=0.7)
        ax2.grid(True, linestyle='--', axis='y')
        ax2.set_xlabel('Time Point Transition')
        ax2.set_ylabel('Valid Data Points')
        ax2.set_ylim(bottom=0)
        
        # Rotate x-axis labels
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save correlation trend plot
        trend_file = os.path.join(output_dir, f"replicate_{rep1_idx+1}_vs_{rep2_idx+1}_correlation_trend_no_zeros.png")
        plt.savefig(trend_file, dpi=300)
        print(f"Saved correlation trend to {trend_file}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    analyze_yeast_replicates_exclude_zeros()