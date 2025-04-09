import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os

def analyze_yeast_replicates():
    # Define input and output directories
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/troubleshooting/frequency"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths for the three replicates with full path
    file_paths = [
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep1.csv'),
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep2.csv'), 
        os.path.join(input_dir, 'pyfitseq_input_Clim_rep3.csv')
    ]
    
    # Load data from each replicate file - with no headers
    replicates = []
    for i, file_path in enumerate(file_paths):
        try:
            # Read with header=None since there are no column headers
            df = pd.read_csv(file_path, header=None)
            # Add a strain ID column (0-based index)
            df['strain_id'] = range(len(df))
            replicates.append(df)
            print(f"Loaded Replicate {i+1} from {file_path} with {len(df)} strains and {len(df.columns)-1} time points")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if len(replicates) < 2:
        print("Need at least two replicates for comparison")
        return
    
    # Calculate frequencies for each replicate
    freq_dfs = []
    for i, df in enumerate(replicates):
        # Create frequency dataframe with artificial strain IDs
        freq_df = pd.DataFrame()
        freq_df['strain_id'] = df['strain_id']
        
        # Time columns are all numeric columns except the strain_id
        time_cols = [col for col in df.columns if col != 'strain_id']
        
        # Calculate frequencies for each time point
        for col in time_cols:
            total_reads = df[col].sum()
            if total_reads > 0:
                freq_df[col] = df[col] / total_reads
            else:
                freq_df[col] = 0
                print(f"Warning: Zero total reads at time point {col} in replicate {i+1}")
        
        freq_dfs.append(freq_df)
        print(f"Calculated frequencies for Replicate {i+1}")
    
    # Calculate ln(freq_i+1/freq_i) for consecutive time points
    ln_ratio_dfs = []
    
    for i, freq_df in enumerate(freq_dfs):
        # Create a new dataframe for ln ratios
        ln_ratio_df = pd.DataFrame()
        ln_ratio_df['strain_id'] = freq_df['strain_id']
        
        # Get time columns (all columns except strain_id)
        time_cols = [col for col in freq_df.columns if col != 'strain_id']
        
        # Calculate for each pair of consecutive time points
        for j in range(len(time_cols) - 1):
            current_col = time_cols[j]
            next_col = time_cols[j+1]
            ratio_col = f"ln_ratio_{j}_to_{j+1}"
            
            # Add small value to prevent division by zero or log(0)
            epsilon = 1e-10
            ln_ratio_df[ratio_col] = np.log((freq_df[next_col] + epsilon) / (freq_df[current_col] + epsilon))
        
        ln_ratio_dfs.append(ln_ratio_df)
        print(f"Replicate {i+1} has {len(time_cols)-1} transitions")
        
        # Print some stats about the first and last transition to verify data
        first_ratio = f"ln_ratio_0_to_1"
        last_ratio = f"ln_ratio_{len(time_cols)-2}_to_{len(time_cols)-1}"
        if first_ratio in ln_ratio_df.columns:
            print(f"Replicate {i+1} {first_ratio} stats: mean={ln_ratio_df[first_ratio].mean():.4f}, std={ln_ratio_df[first_ratio].std():.4f}")
        if last_ratio in ln_ratio_df.columns:
            print(f"Replicate {i+1} {last_ratio} stats: mean={ln_ratio_df[last_ratio].mean():.4f}, std={ln_ratio_df[last_ratio].std():.4f}")
    
    # Get all combinations of replicates
    # This gives pairs like (0,1), (0,2), (1,2) for indices, which corresponds to reps 1-2, 1-3, 2-3
    replicate_pairs = list(combinations(range(len(ln_ratio_dfs)), 2))
    print(f"Replicate pairs for comparison: {[(i+1, j+1) for i, j in replicate_pairs]}")
    
    # Determine number of time transitions (should be 9 for 10 time points)
    num_transitions = min([len([col for col in df.columns if col.startswith('ln_ratio')]) 
                          for df in ln_ratio_dfs])
    print(f"Number of transitions to plot: {num_transitions}")
    
    # Create panel plot with one row for each replicate pair
    fig, axes = plt.subplots(len(replicate_pairs), num_transitions, 
                            figsize=(4*num_transitions, 4*len(replicate_pairs)),
                            squeeze=False)  # Important: prevents dimension reduction for single row/col
    
    # Plot each comparison
    for row, (rep1_idx, rep2_idx) in enumerate(replicate_pairs):
        print(f"Plotting row {row+1}: Replicate {rep1_idx+1} vs Replicate {rep2_idx+1}")
        
        for col in range(num_transitions):
            ax = axes[row, col]
            
            # Get ratio column for this transition
            ratio_col = f"ln_ratio_{col}_to_{col+1}"
            
            # Merge data based on strain ID
            merged_df = pd.merge(
                ln_ratio_dfs[rep1_idx][['strain_id', ratio_col]],
                ln_ratio_dfs[rep2_idx][['strain_id', ratio_col]],
                on='strain_id',
                suffixes=('_rep1', '_rep2')
            )
            print(f"  Transition {col} to {col+1}: {len(merged_df)} strains matched")
            
            # Plot scatter plot
            x = merged_df[f"{ratio_col}_rep1"]
            y = merged_df[f"{ratio_col}_rep2"]
            ax.scatter(x, y, alpha=0.5, s=3)  # Smaller points due to many strains
            
            # Add y=x diagonal line
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add correlation coefficient
            corr = np.corrcoef(x, y)[0, 1]
            ax.annotate(f"r = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
            
            # Set labels with explicit mention of replicate numbers
            ax.set_ylabel(f"Replicate {rep2_idx+1}")
            ax.set_xlabel(f"Replicate {rep1_idx+1}")
            
            ax.set_title(f"T{col} to T{col+1}")
    
    # Add clear row labels on the right side
    for row, (rep1_idx, rep2_idx) in enumerate(replicate_pairs):
        fig.text(1.01, (row + 0.5) / len(replicate_pairs), 
                f"Rep {rep1_idx+1} vs Rep {rep2_idx+1}", 
                va='center', ha='left',
                transform=fig.transFigure,
                fontsize=12, fontweight='bold')
    
    plt.suptitle("Comparison of ln(frequency ratios) between replicates", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # Save the figure to the output directory
    output_file = os.path.join(output_dir, "replicate_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_file}")
    
    # Also save a higher resolution version
    output_file_hi_res = os.path.join(output_dir, "replicate_comparison_hires.png")
    plt.savefig(output_file_hi_res, dpi=600, bbox_inches='tight')
    print(f"Saved high-resolution figure to {output_file_hi_res}")
    
    plt.show()

if __name__ == "__main__":
    analyze_yeast_replicates()