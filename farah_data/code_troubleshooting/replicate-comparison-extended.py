import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import os

def read_csv_files(data_dir, condition, replicates=[1, 2, 3]):
    """
    Read CSV files for a specific condition and multiple replicates.
    """
    data = {}
    for rep in replicates:
        file_path = os.path.join(data_dir, f"pyfitseq_input_{condition}_rep{rep}.csv")
        try:
            df = pd.read_csv(file_path, header=None)
            data[f"rep{rep}"] = df
            print(f"Loaded {file_path}: Shape {df.shape}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return data

def compare_replicates_extended(data_dir, output_dir, condition="Clim", replicates=[1, 2, 3]):
    """
    Create extensive visualizations to compare replicates before model processing.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    data = read_csv_files(data_dir, condition, replicates)
    
    # Check if we have data for all replicates
    all_replicates_present = all(f"rep{rep}" in data for rep in replicates)
    if not all_replicates_present:
        available_reps = [rep for rep in replicates if f"rep{rep}" in data]
        print(f"Missing data for some replicates. Only using: {available_reps}")
        replicates = available_reps
    
    if len(replicates) < 2:
        print(f"Not enough replicates for condition {condition} to perform comparison.")
        return None
    
    # Calculate sums for each strain across all time points
    sums = {}
    for rep, df in data.items():
        sums[rep] = df.sum(axis=1)
    
    # Create a results figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'{condition} Condition: Replicate Variability Analysis', fontsize=16)
    
    # 1. Log-transformed scatter plots (for first two replicates)
    ax1 = fig.add_subplot(331)
    rep1 = f"rep{replicates[0]}"
    rep2 = f"rep{replicates[1]}"
    ax1.scatter(np.log10(sums[rep1] + 1), np.log10(sums[rep2] + 1), alpha=0.5, s=5)
    ax1.set_xlabel(f'Log10({rep1} Reads + 1)')
    ax1.set_ylabel(f'Log10({rep2} Reads + 1)')
    ax1.set_title('Log-Transformed Read Counts')
    
    # Add x=y line
    max_val = max(np.log10(sums[rep1] + 1).max(), np.log10(sums[rep2] + 1).max())
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.3)
    
    # 2. Bland-Altman plot (MA plot)
    ax2 = fig.add_subplot(332)
    mean_reads = (np.log10(sums[rep1] + 1) + np.log10(sums[rep2] + 1)) / 2
    diff_reads = np.log10(sums[rep1] + 1) - np.log10(sums[rep2] + 1)
    ax2.scatter(mean_reads, diff_reads, alpha=0.5, s=5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Mean Log10 Reads')
    ax2.set_ylabel(f'Difference ({rep1} - {rep2})')
    ax2.set_title('Bland-Altman Plot')
    
    # Calculate mean and standard deviation of differences
    mean_diff = np.mean(diff_reads)
    std_diff = np.std(diff_reads)
    ax2.axhline(y=mean_diff, color='g', linestyle='-')
    ax2.axhline(y=mean_diff + 1.96*std_diff, color='g', linestyle='--')
    ax2.axhline(y=mean_diff - 1.96*std_diff, color='g', linestyle='--')
    
    # 3. Coefficient of Variation vs Mean
    ax3 = fig.add_subplot(333)
    
    # Calculate mean and CV across replicates for each strain
    means = []
    cvs = []
    
    for i in range(len(sums[rep1])):
        values = [sums[f"rep{rep}"][i] for rep in replicates if i < len(sums[f"rep{rep}"])]
        if values:  # Make sure we have values to calculate with
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val > 0 else 0
            
            means.append(mean_val)
            cvs.append(cv)
    
    ax3.scatter(np.log10(np.array(means) + 1), cvs, alpha=0.5, s=5)
    ax3.set_xlabel('Log10(Mean Reads + 1)')
    ax3.set_ylabel('Coefficient of Variation')
    ax3.set_title('Variability vs. Abundance')
    
    # 4. Histogram of relative differences
    ax4 = fig.add_subplot(334)
    rel_diff = []
    
    for i in range(min(len(sums[rep1]), len(sums[rep2]))):
        if sums[rep1][i] > 0 and sums[rep2][i] > 0:  # Avoid division by zero
            rd = 100 * abs(sums[rep1][i] - sums[rep2][i]) / ((sums[rep1][i] + sums[rep2][i])/2)
            rel_diff.append(rd)
    
    ax4.hist(rel_diff, bins=50, alpha=0.7)
    ax4.set_xlabel('Percent Difference')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Distribution of % Differences Between {rep1} and {rep2}')
    
    # 5. Time-course plots for select strains
    ax5 = fig.add_subplot(335)
    
    # Select a few random strains with reasonable read counts
    if means:
        np.random.seed(42)
        threshold = np.percentile(means, 75)  # Top 25% by abundance
        high_abundance_strains = [i for i, m in enumerate(means) if m > threshold]
        if high_abundance_strains:
            selected_strains = np.random.choice(high_abundance_strains, 
                                               min(5, len(high_abundance_strains)), 
                                               replace=False)
            
            # Make time-course plots
            markers = ['o', 's', '^', 'd', 'x']
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            
            for idx, strain in enumerate(selected_strains):
                if idx >= len(markers) or idx >= len(colors):
                    continue  # Skip if we run out of markers or colors
                    
                for rep_idx, rep in enumerate(replicates[:2]):  # Just the first two replicates
                    rep_name = f"rep{rep}"
                    if strain < len(data[rep_name]):
                        strain_data = data[rep_name].iloc[strain]
                        timepoints = range(len(strain_data))
                        ax5.plot(timepoints, strain_data, 
                                marker=markers[idx], color=colors[idx], 
                                linestyle='-' if rep_idx == 0 else '--',
                                alpha=0.7, label=f'Strain {strain}, {rep_name}')
    
    ax5.set_xlabel('Time Point')
    ax5.set_ylabel('Read Count')
    ax5.set_title('Time-course of Selected Strains')
    ax5.legend(loc='upper left', fontsize='small')
    
    # 6. Spearman rank correlation between replicates
    ax6 = fig.add_subplot(336)
    
    # Calculate correlation matrix
    corr_matrix = np.zeros((len(replicates), len(replicates)))
    for i, rep1_idx in enumerate(replicates):
        for j, rep2_idx in enumerate(replicates):
            if i <= j:  # Only calculate upper triangle
                rep1_name = f"rep{rep1_idx}"
                rep2_name = f"rep{rep2_idx}"
                
                # Make sure we have data for both replicates
                min_length = min(len(sums[rep1_name]), len(sums[rep2_name]))
                
                if min_length > 0:
                    corr, _ = spearmanr(sums[rep1_name][:min_length], sums[rep2_name][:min_length])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr  # Matrix is symmetric
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=0.9, vmax=1.0,
               xticklabels=[f'Rep{r}' for r in replicates],
               yticklabels=[f'Rep{r}' for r in replicates], ax=ax6)
    ax6.set_title('Spearman Rank Correlation')
    
    # 7. PCA of replicates (if we have at least 2 replicates)
    ax7 = fig.add_subplot(337)
    
    if len(replicates) >= 2:
        # Find the minimum number of strains across all replicates
        min_strains = min(len(sums[f"rep{rep}"]) for rep in replicates)
        
        # Prepare data for PCA - transpose to have strains as features
        pca_data = np.column_stack([sums[f"rep{rep}"][:min_strains] for rep in replicates])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)
        
        # Plot results
        ax7.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=10)
        ax7.set_xlabel('PC1')
        ax7.set_ylabel('PC2')
        ax7.set_title('PCA of Strain Abundances')
    else:
        ax7.text(0.5, 0.5, "Not enough replicates for PCA", 
                 horizontalalignment='center', verticalalignment='center')
    
    # 8. Quantile-quantile plot
    ax8 = fig.add_subplot(338)
    
    # Sort data for Q-Q plot
    min_length = min(len(sums[rep1]), len(sums[rep2]))
    sorted_rep1 = np.sort(np.log10(sums[rep1][:min_length] + 1))
    sorted_rep2 = np.sort(np.log10(sums[rep2][:min_length] + 1))
    
    ax8.scatter(sorted_rep1, sorted_rep2, alpha=0.5, s=5)
    
    # Add x=y line
    max_val = max(sorted_rep1.max(), sorted_rep2.max())
    ax8.plot([0, max_val], [0, max_val], 'r--', alpha=0.3)
    
    ax8.set_xlabel(f'{rep1} Quantiles (Log10)')
    ax8.set_ylabel(f'{rep2} Quantiles (Log10)')
    ax8.set_title('Q-Q Plot')
    
    # 9. Consistency across time points
    ax9 = fig.add_subplot(339)
    
    # Find the minimum number of time points (columns) across replicates
    min_timepoints = min(data[rep1].shape[1], data[rep2].shape[1])
    print(f"Using {min_timepoints} time points for correlation analysis")
    
    # Calculate per-timepoint correlations
    tp_correlations = []
    
    for tp in range(min_timepoints):
        # Get the specified column for each replicate
        rep1_tp_data = data[rep1].iloc[:, tp]
        rep2_tp_data = data[rep2].iloc[:, tp]
        
        # Calculate correlation only if we have sufficient non-zero values
        non_zero_mask = (rep1_tp_data > 0) & (rep2_tp_data > 0)
        if sum(non_zero_mask) > 10:  # Require at least 10 valid data points
            corr, _ = spearmanr(rep1_tp_data, rep2_tp_data)
            tp_correlations.append(corr)
        else:
            tp_correlations.append(np.nan)
    
    # Plot only non-nan values
    valid_tps = [i for i, c in enumerate(tp_correlations) if not np.isnan(c)]
    valid_corrs = [tp_correlations[i] for i in valid_tps]
    
    if valid_corrs:
        ax9.plot(valid_tps, valid_corrs, 'o-')
        ax9.set_xlabel('Time Point')
        ax9.set_ylabel('Spearman Correlation')
        ax9.set_title(f'{rep1}-{rep2} Correlation at Each Time Point')
        
        # Set y-axis limits based on actual correlations
        min_corr = max(0.5, min(valid_corrs) - 0.05)
        ax9.set_ylim(min_corr, 1.0)
    else:
        ax9.text(0.5, 0.5, "Insufficient data for time point correlation", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"{condition}_extended_analysis.png"), dpi=300)
    
    # Calculate statistics for a summary report
    stats = {
        'Total strains': len(sums[rep1]),
        'Mean reads rep1': np.mean(sums[rep1]),
        'Mean reads rep2': np.mean(sums[rep2]),
        'Pearson correlation': np.corrcoef(sums[rep1][:min_length], sums[rep2][:min_length])[0, 1],
        'Spearman correlation': spearmanr(sums[rep1][:min_length], sums[rep2][:min_length])[0],
        'Mean % difference': np.mean(rel_diff) if rel_diff else np.nan,
        'Median % difference': np.median(rel_diff) if rel_diff else np.nan,
        'Low count strains (< 10 reads)': sum(1 for x in means if x < 10),
        'High count strains (> 1000 reads)': sum(1 for x in means if x > 1000)
    }
    
    # Save statistics to file
    with open(os.path.join(output_dir, f"{condition}_summary_stats.txt"), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Analysis complete for {condition}. Files saved to {output_dir}")
    return fig

def analyze_all_conditions(data_dir, output_dir):
    """
    Run extended analysis for all conditions.
    """
    conditions = ['Clim', 'Nlim', 'Switch']
    
    for condition in conditions:
        print(f"Analyzing {condition}...")
        compare_replicates_extended(data_dir, output_dir, condition)
    
    print("All analyses complete!")

if __name__ == "__main__":
    # Set your input and output directories
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/troubleshooting"
    
    # Run analysis for all conditions
    analyze_all_conditions(input_dir, output_dir)
