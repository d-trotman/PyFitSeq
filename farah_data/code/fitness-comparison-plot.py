import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from scipy.stats import pearsonr

# Define paths
base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
results_dir = os.path.join(base_dir, "farah_data/outputs/env_specific")
counts_file = os.path.join(base_dir, "farah_data/outputs/barcode_counts.csv")
plots_dir = os.path.join(results_dir, "plots")

# Create plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Function to calculate fold change (observed fitness)
def calculate_fold_change(df, gene_columns):
    # Get first and last time point
    t_first = df['TimePoint'].min()
    t_last = df['TimePoint'].max()
    
    # Get counts at first and last time point
    first_tp = df[df['TimePoint'] == t_first]
    last_tp = df[df['TimePoint'] == t_last]
    
    # Calculate log2 fold change for each gene
    fold_changes = {}
    for gene in gene_columns:
        # Avoid division by zero by adding a small constant
        first_count = first_tp[gene].values[0] + 0.1
        last_count = last_tp[gene].values[0] + 0.1
        
        # Log2 fold change
        log2_fc = np.log2(last_count / first_count)
        
        # Normalize by time (generations)
        generations = t_last - t_first
        if generations > 0:
            normalized_fc = log2_fc / generations
        else:
            normalized_fc = log2_fc
            
        fold_changes[gene] = normalized_fc
    
    return fold_changes

# Load original count data
print(f"Reading counts file: {counts_file}")
try:
    counts_df = pd.read_csv(counts_file)
    gene_columns = [col for col in counts_df.columns if col not in ['TimePoint', 'Condition', 'Replicate']]
    print(f"Found {len(gene_columns)} genes in count data")
except Exception as e:
    print(f"Error loading counts file: {e}")
    gene_columns = []

# Find all fitness result files
result_files = glob.glob(os.path.join(results_dir, "results_*_FitSeq.csv"))
print(f"Found {len(result_files)} fitness result files")

if len(result_files) == 0:
    print("No fitness result files found. Make sure PyFitSeq analysis has completed.")

# Process each condition and replicate
for result_file in result_files:
    # Extract condition and replicate from filename
    filename = os.path.basename(result_file)
    # The pattern is "results_CONDITION_repNUMBER_FitSeq.csv"
    parts = filename.replace("_FitSeq.csv", "").split("_")
    if len(parts) < 3:
        print(f"Skipping file with unexpected name format: {filename}")
        continue
        
    condition = parts[1]
    replicate = parts[2].replace("rep", "")
    
    print(f"\nProcessing {condition} replicate {replicate}...")
    
    # Load the fitness estimates
    try:
        fitness_df = pd.read_csv(result_file)
        # Check if the file has the expected columns
        if "Estimated_Fitness" not in fitness_df.columns:
            print(f"Error: 'Estimated_Fitness' column not found in {result_file}")
            continue
            
        # Get estimated fitness values
        estimated_fitness = fitness_df["Estimated_Fitness"].values
        print(f"Loaded {len(estimated_fitness)} estimated fitness values")
        
        # Calculate observed fitness (log2 fold change)
        condition_rep_df = counts_df[(counts_df['Condition'] == condition) & 
                                    (counts_df['Replicate'] == int(replicate))]
        
        if len(condition_rep_df) == 0:
            print(f"No count data found for {condition} replicate {replicate}")
            continue
            
        fold_changes = calculate_fold_change(condition_rep_df, gene_columns)
        observed_fitness = np.array([fold_changes[gene] for gene in gene_columns])
        print(f"Calculated {len(observed_fitness)} observed fitness values")
        
        # Calculate correlation
        correlation, p_value = pearsonr(estimated_fitness, observed_fitness)
        print(f"Correlation: {correlation:.3f} (p={p_value:.3e})")
        
        # Create scatter plot
        plt.figure(figsize=(8, 8))
        
        # Plot data points
        plt.scatter(observed_fitness, estimated_fitness, alpha=0.5, s=15)
        
        # Add diagonal line
        min_val = min(min(observed_fitness), min(estimated_fitness))
        max_val = max(max(observed_fitness), max(estimated_fitness))
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.7)
        
        # Add regression line
        m, b = np.polyfit(observed_fitness, estimated_fitness, 1)
        x_range = np.linspace(min(observed_fitness), max(observed_fitness), 100)
        plt.plot(x_range, m * x_range + b, '-', color='red', alpha=0.7)
        
        # Add zero lines
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        # Add correlation text
        plt.text(0.05, 0.95, f"r = {correlation:.3f}", transform=plt.gca().transAxes, 
                ha='left', va='top', fontsize=12)
        
        # Set labels and title
        plt.xlabel('Observed Fitness (Log2 Fold Change per Generation)')
        plt.ylabel('Estimated Fitness (PyFitSeq)')
        plt.title(f'{condition} Replicate {replicate}: Estimated vs Observed Fitness')
        plt.grid(True, alpha=0.3)
        
        # Make axes equal and square
        plt.axis('equal')
        
        # Save the plot
        plot_file = os.path.join(plots_dir, f"{condition}_rep{replicate}_fitness_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved plot to: {plot_file}")
        
        # Create histogram of fitness values
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        plt.hist(estimated_fitness, bins=50, alpha=0.5, label='Estimated Fitness')
        plt.hist(observed_fitness, bins=50, alpha=0.5, label='Observed Fitness')
        
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Fitness')
        plt.ylabel('Frequency')
        plt.title(f'{condition} Replicate {replicate}: Fitness Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the histogram
        hist_file = os.path.join(plots_dir, f"{condition}_rep{replicate}_fitness_histogram.png")
        plt.tight_layout()
        plt.savefig(hist_file, dpi=300)
        plt.close()
        print(f"Saved histogram to: {hist_file}")
        
    except Exception as e:
        print(f"Error processing {result_file}: {e}")

# Create comparison across all conditions
try:
    # Create a dataframe to hold all results
    all_results = pd.DataFrame(index=gene_columns)
    
    # For each condition and replicate, add the fitness values
    for result_file in result_files:
        filename = os.path.basename(result_file)
        parts = filename.replace("_FitSeq.csv", "").split("_")
        if len(parts) < 3:
            continue
            
        condition = parts[1]
        replicate = parts[2].replace("rep", "")
        key = f"{condition}_rep{replicate}"
        
        # Load the fitness estimates
        fitness_df = pd.read_csv(result_file)
        if "Estimated_Fitness" not in fitness_df.columns:
            continue
            
        # Add to dataframe
        all_results[key] = fitness_df["Estimated_Fitness"].values
    
    # Calculate correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = all_results.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, fmt='.2f')
    plt.title('Correlation of Fitness Estimates Across Conditions and Replicates')
    
    # Save the correlation matrix
    corr_file = os.path.join(plots_dir, "fitness_correlation_matrix.png")
    plt.tight_layout()
    plt.savefig(corr_file, dpi=300)
    plt.close()
    print(f"\nSaved correlation matrix to: {corr_file}")
    
except Exception as e:
    print(f"\nError creating correlation matrix: {e}")

print("\nAll plots have been created successfully!")
