import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

# Define directories
base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
results_dir = os.path.join(base_dir, "farah_data/outputs/env_specific")
plots_dir = os.path.join(results_dir, "plots")

# Create plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Find all result files
result_files = glob.glob(os.path.join(results_dir, "results_*_FitSeq.csv"))
print(f"Found {len(result_files)} result files")

if len(result_files) == 0:
    print("No result files found. Make sure PyFitSeq analysis has completed.")
    exit(1)

# Load all fitness results into a dictionary
fitness_data = {}
all_genes = set()

for result_file in result_files:
    # Extract condition and replicate from filename
    filename = os.path.basename(result_file)
    # The pattern is "results_CONDITION_repNUMBER_FitSeq.csv"
    parts = filename.replace("_FitSeq.csv", "").split("_")
    condition = parts[1]
    replicate = parts[2].replace("rep", "")
    
    print(f"Loading {condition} replicate {replicate}...")
    
    # Load the data
    try:
        df = pd.read_csv(result_file)
        # The first column should be Estimated_Fitness
        if "Estimated_Fitness" in df.columns:
            fitness_values = df["Estimated_Fitness"].values
            key = f"{condition}_{replicate}"
            fitness_data[key] = fitness_values
            
            # Keep track of the number of genes
            all_genes.update(range(len(fitness_values)))
            
            print(f"  Loaded {len(fitness_values)} genes")
            print(f"  Fitness range: {np.min(fitness_values):.4f} to {np.max(fitness_values):.4f}")
            print(f"  Mean fitness: {np.mean(fitness_values):.4f}")
        else:
            print(f"  Error: 'Estimated_Fitness' column not found in {result_file}")
    except Exception as e:
        print(f"  Error loading {result_file}: {e}")

# Check if we have data to plot
if len(fitness_data) == 0:
    print("No fitness data was loaded. Check the result files.")
    exit(1)

# Get all conditions and replicates
conditions = sorted(list(set([k.split("_")[0] for k in fitness_data.keys()])))
num_conditions = len(conditions)
print(f"\nFound {num_conditions} conditions: {conditions}")

# Create a custom colormap for the correlation heatmap
colors = plt.cm.RdBu_r(np.linspace(0, 1, 100))
custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors)

# 1. Overview Plot - Distribution of fitness values by condition
plt.figure(figsize=(12, 8))

for i, condition in enumerate(conditions):
    condition_data = []
    labels = []
    
    for key in fitness_data:
        if key.startswith(condition + "_"):
            condition_data.append(fitness_data[key])
            labels.append(key.split("_")[1])  # This is the replicate number
    
    positions = np.array(range(len(condition_data))) + (i * (len(condition_data) + 2))
    violin_parts = plt.violinplot(condition_data, positions=positions, widths=0.8, showmeans=True)
    
    # Set violin colors
    for pc in violin_parts['bodies']:
        pc.set_facecolor(plt.cm.tab10(i % 10))
        pc.set_alpha(0.7)
    
    # Add labels for each replicate
    for pos, label in zip(positions, labels):
        plt.text(pos, plt.ylim()[0] * 0.98, f"Rep {label}", 
                 ha='center', va='top', fontsize=9, rotation=45)

# Add condition labels
for i, condition in enumerate(conditions):
    mid_pos = i * (len([k for k in fitness_data.keys() if k.startswith(condition + "_")]) + 2) + 1
    plt.text(mid_pos, plt.ylim()[1] * 0.95, condition, 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.ylabel('Fitness')
plt.title('Distribution of Fitness Values by Condition and Replicate')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "fitness_distributions.png"), dpi=300)
plt.close()

# 2. Correlation Matrix of all datasets
all_keys = sorted(fitness_data.keys())
n_keys = len(all_keys)

# Create a correlation matrix
correlation_matrix = np.zeros((n_keys, n_keys))
pvalues_matrix = np.zeros((n_keys, n_keys))

for i, key1 in enumerate(all_keys):
    for j, key2 in enumerate(all_keys):
        if len(fitness_data[key1]) == len(fitness_data[key2]):
            # Calculate Pearson correlation
            corr, pval = pearsonr(fitness_data[key1], fitness_data[key2])
            correlation_matrix[i, j] = corr
            pvalues_matrix[i, j] = pval
        else:
            print(f"Warning: {key1} and {key2} have different lengths")
            correlation_matrix[i, j] = np.nan

# Plot correlation matrix
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap=custom_cmap, vmin=-1, vmax=1)

# Add correlation values
for i in range(n_keys):
    for j in range(n_keys):
        color = "white" if abs(correlation_matrix[i, j]) > 0.5 else "black"
        plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", 
                 ha="center", va="center", color=color, fontsize=8)

# Add asterisks for significant correlations
for i in range(n_keys):
    for j in range(n_keys):
        if pvalues_matrix[i, j] < 0.001:
            plt.text(j, i+0.3, "***", ha="center", va="center", color="black", fontsize=8)
        elif pvalues_matrix[i, j] < 0.01:
            plt.text(j, i+0.3, "**", ha="center", va="center", color="black", fontsize=8)
        elif pvalues_matrix[i, j] < 0.05:
            plt.text(j, i+0.3, "*", ha="center", va="center", color="black", fontsize=8)

plt.colorbar(label="Pearson Correlation")
plt.xticks(range(n_keys), all_keys, rotation=90)
plt.yticks(range(n_keys), all_keys)
plt.title("Correlation Matrix of Fitness Across Conditions and Replicates")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "fitness_correlation_matrix.png"), dpi=300)
plt.close()

# 3. Replicate consistency plots (scatter plots between replicates of same condition)
for condition in conditions:
    # Get keys for this condition
    condition_keys = [k for k in all_keys if k.startswith(condition + "_")]
    
    if len(condition_keys) < 2:
        print(f"Not enough replicates for condition {condition} to plot")
        continue
    
    # Plot pairwise comparisons of replicates
    n_pairs = len(condition_keys) * (len(condition_keys) - 1) // 2
    fig, axes = plt.subplots(1, n_pairs, figsize=(n_pairs * 5, 5))
    
    # Handle case with only one pair (axes is not a list)
    if n_pairs == 1:
        axes = [axes]
    
    pair_idx = 0
    for i, key1 in enumerate(condition_keys):
        for j, key2 in enumerate(condition_keys):
            if i < j:  # Ensure each pair is plotted only once
                ax = axes[pair_idx]
                rep1 = key1.split("_")[1]
                rep2 = key2.split("_")[1]
                
                # Calculate correlation
                corr, _ = pearsonr(fitness_data[key1], fitness_data[key2])
                
                # Create scatter plot
                ax.scatter(fitness_data[key1], fitness_data[key2], alpha=0.5, s=15)
                ax.set_xlabel(f"Replicate {rep1} Fitness")
                ax.set_ylabel(f"Replicate {rep2} Fitness")
                
                # Add correlation line
                m, b = np.polyfit(fitness_data[key1], fitness_data[key2], 1)
                x_range = np.linspace(min(fitness_data[key1]), max(fitness_data[key1]), 100)
                ax.plot(x_range, m * x_range + b, '-', color='red', alpha=0.7)
                
                # Add identity line
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                min_val = min(xlim[0], ylim[0])
                max_val = max(xlim[1], ylim[1])
                ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5)
                
                # Set equal limits
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                
                # Add correlation text
                ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, 
                        ha='left', va='top', fontsize=12)
                
                # Add zero lines
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
                
                pair_idx += 1
    
    plt.suptitle(f"{condition} - Fitness Comparisons Between Replicates")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{condition}_replicate_comparison.png"), dpi=300)
    plt.close()

# 4. Find genes with consistent fitness effects across conditions
# First, create a DataFrame with all fitness values
fitness_df = pd.DataFrame(index=range(max(all_genes) + 1))

for key, values in fitness_data.items():
    # Make sure the array length matches the number of genes
    if len(values) == len(fitness_df):
        fitness_df[key] = values
    else:
        print(f"Warning: {key} has {len(values)} genes, but expected {len(fitness_df)}")
        # Create a padded array
        padded = np.full(len(fitness_df), np.nan)
        padded[:len(values)] = values
        fitness_df[key] = padded

# Calculate mean fitness per condition
for condition in conditions:
    condition_cols = [col for col in fitness_df.columns if col.startswith(condition + "_")]
    fitness_df[f"{condition}_mean"] = fitness_df[condition_cols].mean(axis=1)

# Calculate overall mean and standard deviation across conditions
condition_means = [col for col in fitness_df.columns if col.endswith("_mean")]
fitness_df["overall_mean"] = fitness_df[condition_means].mean(axis=1)
fitness_df["overall_std"] = fitness_df[condition_means].std(axis=1)

# Identify top beneficial and detrimental genes
# - Consistent = high absolute mean, low std
# - Condition-specific = high std
n_top = 20

# Find genes with most negative fitness across all conditions
most_detrimental = fitness_df.sort_values("overall_mean").head(n_top)
print("\nTop detrimental genes (lowest fitness across all conditions):")
for idx, row in most_detrimental.iterrows():
    print(f"Gene {idx}: Mean fitness = {row['overall_mean']:.4f}, Std = {row['overall_std']:.4f}")

# Find genes with most positive fitness across all conditions
most_beneficial = fitness_df.sort_values("overall_mean", ascending=False).head(n_top)
print("\nTop beneficial genes (highest fitness across all conditions):")
for idx, row in most_beneficial.iterrows():
    print(f"Gene {idx}: Mean fitness = {row['overall_mean']:.4f}, Std = {row['overall_std']:.4f}")

# Find genes with most variable fitness (condition-specific)
most_variable = fitness_df.sort_values("overall_std", ascending=False).head(n_top)
print("\nMost condition-specific genes (highest standard deviation across conditions):")
for idx, row in most_variable.iterrows():
    condition_values = [f"{condition}: {row[f'{condition}_mean']:.4f}" for condition in conditions 
                        if f"{condition}_mean" in row.index]
    print(f"Gene {idx}: Std = {row['overall_std']:.4f}, Means: {', '.join(condition_values)}")

# 5. Create a heatmap of top genes across conditions
top_genes = pd.concat([most_detrimental, most_beneficial, most_variable])
top_genes = top_genes[~top_genes.index.duplicated(keep='first')]  # Remove duplicates

# Create a matrix for the heatmap
heatmap_data = top_genes[condition_means].copy()
heatmap_data.columns = [col.replace("_mean", "") for col in heatmap_data.columns]
heatmap_data = heatmap_data.sort_values("overall_mean", key=abs, ascending=False)

# Plot the heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(heatmap_data, cmap=custom_cmap, center=0, 
            cbar_kws={'label': 'Fitness'}, yticklabels=heatmap_data.index)
plt.title("Top Genes Across Conditions")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "top_genes_heatmap.png"), dpi=300)
plt.close()

print(f"\nAll plots have been saved to: {plots_dir}")
print("\nSummary of analyses:")
print(f"- Generated fitness distributions for {num_conditions} conditions")
print(f"- Created correlation matrix across all datasets")
print(f"- Generated replicate comparison plots for each condition")
print(f"- Identified top consistent and condition-specific genes")
print(f"- Created heatmap of top genes across conditions")
