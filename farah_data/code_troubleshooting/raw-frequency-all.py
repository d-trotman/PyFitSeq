import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os
import matplotlib.cm as cm

def analyze_yeast_replicates_all_timepoints():
    # Define input and output directories
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/troubleshooting/frequency"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths for the three replicates
    file_paths = [
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep1.csv'),
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep2.csv'), 
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep3.csv')
    ]
    
    # Load data from each replicate file - with explicit debugging
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
            
            # Verify we have all 10 time columns plus strain_id
            print(f"Replicate {i+1} columns after adding strain_id: {df.columns.tolist()}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Calculate frequencies with strain-specific pseudocounts for each replicate
    freq_dfs = []
    # Parameter alpha controls the pseudocount strength (recommended: 0.1-1.0)
    alpha = 0.5
    
    for i, df in enumerate(replicates):
        # Create frequency dataframe
        freq_df = pd.DataFrame()
        freq_df['strain_id'] = df['strain_id']
        
        # Get all time columns (numeric columns excluding strain_id)
        time_cols = [col for col in df.columns if col != 'strain_id']
        print(f"Replicate {i+1} time columns: {time_cols}")
        
        # Number of strains
        num_strains = len(df)
        
        # Calculate frequencies with pseudocounts for each time point
        for col in time_cols:
            # Calculate total reads for this timepoint
            total_reads = df[col].sum()
            
            if total_reads > 0:
                # Calculate strain-specific pseudocount based on sequencing depth
                pseudocount = total_reads * alpha / num_strains
                
                # Add pseudocount to raw counts, then calculate adjusted frequency
                adjusted_counts = df[col] + pseudocount
                adjusted_total = total_reads + (pseudocount * num_strains)
                
                # Store the adjusted frequency
                freq_df[col] = adjusted_counts / adjusted_total
                
                # Log the pseudocount for this timepoint
                print(f"Replicate {i+1}, timepoint {col}: Using pseudocount of {pseudocount:.2f} (total reads: {total_reads})")
            else:
                # If there are no reads at all (extremely rare case)
                print(f"Warning: Zero total reads at time point {col} in replicate {i+1}")
                # Assign equal frequencies to all strains
                freq_df[col] = 1.0 / num_strains
        
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
        
        # Calculate for each pair of consecutive time points
        for j in range(len(time_cols) - 1):
            current_col = time_cols[j]
            next_col = time_cols[j+1]
            ratio_col = f"ln_ratio_{j}_to_{j+1}"
            
            # Since we've already handled zeros with pseudocounts, we can directly take ln ratios
            ln_ratio_df[ratio_col] = np.log(freq_df[next_col] / freq_df[current_col])
        
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
        
        # For storing correlation values
        correlations = []
        
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
            
            # Get the data for plotting
            x = merged_df[f"{ratio_col}_rep1"]
            y = merged_df[f"{ratio_col}_rep2"]
            
            # Update global min/max
            all_min = min(all_min, x.min(), y.min())
            all_max = max(all_max, x.max(), y.max())
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            correlations.append(corr)
            
            # Plot with specific color for this transition
            plt.scatter(x, y, alpha=0.4, s=5, color=colormap(t/num_transitions), 
                      label=f"{transition_label} (r={corr:.2f})")
        
        # Add diagonal line
        buffer = (all_max - all_min) * 0.05  # 5% buffer on each end
        plt.plot([all_min-buffer, all_max+buffer], [all_min-buffer, all_max+buffer], 
                'r--', linewidth=1, label='y=x')
        
        # Set labels and title
        plt.xlabel(f"Replicate {rep1_idx+1} ln(frequency ratio)")
        plt.ylabel(f"Replicate {rep2_idx+1} ln(frequency ratio)")
        plt.title(f"Fitness Comparison: Replicate {rep1_idx+1} vs Replicate {rep2_idx+1}")
        
        # Add legend - place it outside the plot area
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Make axes square for better visualization
        plt.axis('square')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Tight layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Save the figure
        output_file = os.path.join(output_dir, f"replicate_{rep1_idx+1}_vs_{rep2_idx+1}_all_transitions.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
        
        # Create a summary plot for this replicate pair showing correlation over time
        plt.figure(figsize=(10, 5))
        time_points = [f"{i} → {i+1}" for i in range(len(correlations))]
        plt.plot(time_points, correlations, 'o-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--')
        plt.xlabel('Time Point Transition')
        plt.ylabel('Correlation Coefficient (r)')
        plt.title(f'Correlation Over Time: Replicate {rep1_idx+1} vs Replicate {rep2_idx+1}')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save correlation trend plot
        trend_file = os.path.join(output_dir, f"replicate_{rep1_idx+1}_vs_{rep2_idx+1}_correlation_trend.png")
        plt.savefig(trend_file, dpi=300)
        print(f"Saved correlation trend to {trend_file}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    analyze_yeast_replicates_all_timepoints()