import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from scipy.stats import pearsonr

# Define paths
base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
fluctuating_dir = os.path.join(base_dir, "farah_data/outputs/fluctuating_env")
counts_file = os.path.join(base_dir, "farah_data/outputs/barcode_counts.csv")
plots_dir = os.path.join(fluctuating_dir, "plots")

# Create plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Define fluctuating environments
fluctuating_conditions = ["Switch", "PulseAS", "PulseGln"]

# Function to calculate "actual" fitness (log2 fold change per hour)
def calculate_actual_fitness(counts_df, segment_data, gene_columns):
    """
    Calculate actual fitness based on fold change within a segment
    """
    # Get first and last time point in the segment
    first_tp = segment_data['TimePoint'].min()
    last_tp = segment_data['TimePoint'].max()
    time_diff = last_tp - first_tp  # in hours
    
    # Get data for those time points
    first_data = segment_data[segment_data['TimePoint'] == first_tp]
    last_data = segment_data[segment_data['TimePoint'] == last_tp]
    
    # Calculate log2 fold change for each gene
    fold_changes = {}
    for gene in gene_columns:
        try:
            # Get counts, adding small constant to avoid division by zero
            first_count = float(first_data[gene].values[0]) + 1e-10
            last_count = float(last_data[gene].values[0]) + 1e-10
            
            # Calculate log2 fold change
            log2_fc = np.log2(last_count / first_count)
            
            # Normalize by time (hours)
            if time_diff > 0:
                normalized_fc = log2_fc / time_diff
            else:
                normalized_fc = 0
                
            fold_changes[gene] = normalized_fc
        except Exception as e:
            print(f"Error calculating fold change for {gene}: {e}")
            fold_changes[gene] = 0
            
    return fold_changes

# Load count data
print(f"Reading counts file: {counts_file}")
counts_df = pd.read_csv(counts_file)
print(f"Shape of data: {counts_df.shape}")

# Get gene columns
gene_columns = [col for col in counts_df.columns if col not in ['TimePoint', 'Condition', 'Replicate']]
print(f"Found {len(gene_columns)} genes")

# Find all result files from the fluctuating environment analysis
result_files = glob.glob(os.path.join(fluctuating_dir, "results_*_FitSeq.csv"))
print(f"Found {len(result_files)} result files")

if len(result_files) == 0:
    print("No fitness result files found. Make sure PyFitSeq analysis has completed.")
    exit(1)

# Create summary dataframe to store results
summary_data = {
    'Condition': [],
    'Replicate': [], 
    'Segment': [],
    'Start_Time': [],
    'End_Time': [],
    'Correlation': [],
    'P_Value': [],
    'Num_Genes': []
}

# Create figure for combined plot
plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plot_idx = 1

# Process each result file
for result_file in result_files:
    # Extract condition, replicate, and segment from filename
    filename = os.path.basename(result_file)
    parts = filename.replace("_FitSeq.csv", "").split("_")
    
    # Handle different filename formats
    if len(parts) < 3:
        print(f"Skipping file with unexpected name format: {filename}")
        continue
    
    condition = parts[1]
    
    # Check if this is a fluctuating environment
    if condition not in fluctuating_conditions:
        print(f"Skipping non-fluctuating condition: {condition}")
        continue
    
    # Parse replicate number
    replicate_str = parts[2].replace("rep", "")
    try:
        replicate = int(replicate_str)
    except ValueError:
        print(f"Could not parse replicate number from: {replicate_str}")
        continue
    
    # Check if there's a segment identifier
    segment = None
    segment_id = "full"
    if len(parts) > 3 and "segment" in parts[3]:
        segment_id = parts[3].replace("segment", "")
        try:
            segment = int(segment_id)
        except ValueError:
            print(f"Could not parse segment number from: {segment_id}")
            continue
    
    print(f"\nProcessing {condition} replicate {replicate} segment {segment_id}...")
    
    # Load PyFitSeq results
    try:
        fitness_df = pd.read_csv(result_file)
        if "Estimated_Fitness" not in fitness_df.columns:
            print(f"Error: 'Estimated_Fitness' column not found in {result_file}")
            continue
        
        # Get estimated fitness values
        estimated_fitness = fitness_df["Estimated_Fitness"].values
        print(f"Loaded {len(estimated_fitness)} estimated fitness values")
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        continue
    
    # Get data for this condition and replicate
    cond_rep_df = counts_df[(counts_df['Condition'] == condition) & 
                            (counts_df['Replicate'] == replicate)]
    
    if len(cond_rep_df) == 0:
        print(f"No count data found for {condition} replicate {replicate}")
        continue
    
    # Determine time range for this segment
    if segment is not None:
        # Define segment boundaries based on 30-hour cycles
        env_switches = [0, 30, 60, 90, 120, 150, 180, 210, 240]
        if segment < len(env_switches) - 1:
            start_time = env_switches[segment]
            end_time = env_switches[segment + 1]
            
            # Get data for this segment
            segment_data = cond_rep_df[(cond_rep_df['TimePoint'] >= start_time) & 
                                      (cond_rep_df['TimePoint'] < end_time)]
        else:
            print(f"Invalid segment number: {segment}")
            continue
    else:
        # Use all data for this condition/replicate
        segment_data = cond_rep_df
        start_time = segment_data['TimePoint'].min()
        end_time = segment_data['TimePoint'].max()
    
    if len(segment_data) < 2:
        print(f"Not enough data points in segment")
        continue
    
    print(f"Segment time range: {start_time} to {end_time} hours")
    print(f"Timepoints in segment: {sorted(segment_data['TimePoint'].unique())}")
    
    # Calculate actual fitness based on fold change
    actual_fitness_dict = calculate_actual_fitness(counts_df, segment_data, gene_columns)
    actual_fitness = np.array([actual_fitness_dict[gene] for gene in gene_columns])
    
    # Remove any NaN or infinite values
    valid_indices = np.isfinite(actual_fitness) & np.isfinite(estimated_fitness)
    if np.sum(valid_indices) < 10:
        print(f"Too few valid genes with finite fitness values")
        continue
    
    valid_actual = actual_fitness[valid_indices]
    valid_estimated = estimated_fitness[valid_indices]
    
    # Calculate correlation
    correlation, p_value = pearsonr(valid_estimated, valid_actual)
    print(f"Correlation: {correlation:.3f} (p={p_value:.3e})")
    
    # Add to summary data
    summary_data['Condition'].append(condition)
    summary_data['Replicate'].append(replicate)
    summary_data['Segment'].append(segment_id)
    summary_data['Start_Time'].append(start_time)
    summary_data['End_Time'].append(end_time)
    summary_data['Correlation'].append(correlation)
    summary_data['P_Value'].append(p_value)
    summary_data['Num_Genes'].append(np.sum(valid_indices))
    
    # Create scatter plot for this segment
    plt.subplot(3, 4, plot_idx)
    plt.scatter(valid_actual, valid_estimated, alpha=0.5, s=20)
    
    # Add regression line
    try:
        m, b = np.polyfit(valid_actual, valid_estimated, 1)
        x_range = np.linspace(min(valid_actual), max(valid_actual), 100)
        plt.plot(x_range, m * x_range + b, 'r-', alpha=0.7)
    except Exception as e:
        print(f"Error fitting regression line: {e}")
    
    # Add reference diagonal line
    lims = [
        min(min(valid_actual), min(valid_estimated)),
        max(max(valid_actual), max(valid_estimated))
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Actual Fitness (Log2 FC/hr)')
    plt.ylabel('Estimated Fitness')
    plt.title(f'{condition} Rep{replicate} {start_time}-{end_time}hr\nr={correlation:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Create individual plot file with more details
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_actual, valid_estimated, alpha=0.5, s=20)
    
    # Add regression line
    try:
        m, b = np.polyfit(valid_actual, valid_estimated, 1)
        x_range = np.linspace(min(valid_actual), max(valid_actual), 100)
        plt.plot(x_range, m * x_range + b, 'r-', alpha=0.7)
    except Exception as e:
        print(f"Error fitting regression line: {e}")
    
    # Add reference diagonal line
    lims = [
        min(min(valid_actual), min(valid_estimated)),
        max(max(valid_actual), max(valid_estimated))
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    
    # Add zero lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Add correlation text
    plt.text(0.05, 0.95, f"r = {correlation:.3f}\np = {p_value:.3e}\nn = {np.sum(valid_indices)}", 
             transform=plt.gca().transAxes, ha='left', va='top', fontsize=12)
    
    # Set labels and title
    plt.xlabel('Actual Fitness (Log2 Fold Change per Hour)')
    plt.ylabel('Estimated Fitness (PyFitSeq)')
    plt.title(f'{condition} Replicate {replicate}: {start_time}-{end_time} hours')
    plt.grid(True, alpha=0.3)
    
    # Save individual plot
    individual_plot_file = os.path.join(plots_dir, f"{condition}_rep{replicate}_segment{segment_id}_fitness_comparison.png")
    plt.tight_layout()
    plt.savefig(individual_plot_file, dpi=300)
    plt.close()
    print(f"Saved individual plot to: {individual_plot_file}")
    
    # Increment plot index for combined figure
    plot_idx += 1
    
    # If we've filled the 3x4 grid, save and create a new figure
    if plot_idx > 12:
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plot_idx = 1

# Save the combined plot
if plot_idx > 1:  # Only save if at least one plot was added
    plt.tight_layout()
    combined_plot_file = os.path.join(plots_dir, "combined_fitness_comparison.png")
    plt.savefig(combined_plot_file, dpi=300)
    plt.close()
    print(f"\nSaved combined plot to: {combined_plot_file}")

# Create summary dataframe and save as CSV
summary_df = pd.DataFrame(summary_data)
summary_file = os.path.join(plots_dir, "fitness_correlation_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"Saved summary data to: {summary_file}")

# Create heatmap of correlations
if len(summary_df) > 0:
    # Reshape data for heatmap
    pivot_df = summary_df.pivot_table(
        index='Condition', 
        columns=['Segment'], 
        values='Correlation',
        aggfunc='mean'  # Average across replicates
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, center=0, 
                fmt='.3f', linewidths=0.5)
    plt.title('Correlation Between Estimated and Actual Fitness by Environment Segment')
    
    # Save heatmap
    heatmap_file = os.path.join(plots_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_file}")

print("\nAnalysis complete!")
