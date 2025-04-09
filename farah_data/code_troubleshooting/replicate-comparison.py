import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def read_and_sum_csv(file_path):
    """
    Read a CSV file and sum the read counts across all time points for each strain.
    """
    try:
        # Read the CSV file (no headers)
        df = pd.read_csv(file_path, header=None)
        
        # Calculate sum of each row (strain)
        row_sums = df.sum(axis=1)
        
        return row_sums
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.Series()

def calculate_correlation(x, y):
    """
    Calculate Pearson correlation coefficient between two series.
    """
    return np.corrcoef(x, y)[0, 1]

def create_scatter_plot(ax, x_data, y_data, title, xlabel, ylabel):
    """
    Create a scatter plot on the given axis.
    """
    correlation = calculate_correlation(x_data, y_data)
    
    # Create scatter plot
    ax.scatter(x_data, y_data, alpha=0.5, s=5)
    
    # Add a line of best fit
    max_val = max(x_data.max(), y_data.max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nCorrelation: {correlation:.4f}")
    
    # Make axes equal for better visualization
    ax.set_aspect('equal')
    
    return correlation

def compare_replicates(data_dir, output_dir):
    """
    Compare replicates for each condition and create scatter plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    conditions = ['Clim', 'Nlim', 'Switch']
    replicates = [1, 2, 3]
    
    # Dictionary to store results
    results = {}
    correlations = {}
    
    # Read and process all files
    for condition in conditions:
        results[condition] = {}
        correlations[condition] = {}
        
        for rep in replicates:
            file_path = os.path.join(data_dir, f"pyfitseq_input_{condition}_rep{rep}.csv")
            results[condition][f"rep{rep}"] = read_and_sum_csv(file_path)
    
    # Create figure for all plots
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 3, figure=fig)
    
    # Create scatter plots for each condition
    for i, condition in enumerate(conditions):
        print(f"\nCondition: {condition}")
        
        rep1 = results[condition]["rep1"]
        rep2 = results[condition]["rep2"]
        rep3 = results[condition]["rep3"]
        
        # Make sure all replicates have the same length (use the minimum length)
        min_length = min(len(rep1), len(rep2), len(rep3))
        rep1 = rep1[:min_length]
        rep2 = rep2[:min_length]
        rep3 = rep3[:min_length]
        
        # Create plots for each comparison
        ax1 = fig.add_subplot(gs[i, 0])
        corr1 = create_scatter_plot(ax1, rep1, rep2, f"{condition}: Rep1 vs Rep2", "Rep1", "Rep2")
        correlations[condition]["rep1_vs_rep2"] = corr1
        print(f"Rep1 vs Rep2 correlation: {corr1:.4f}")
        
        ax2 = fig.add_subplot(gs[i, 1])
        corr2 = create_scatter_plot(ax2, rep1, rep3, f"{condition}: Rep1 vs Rep3", "Rep1", "Rep3")
        correlations[condition]["rep1_vs_rep3"] = corr2
        print(f"Rep1 vs Rep3 correlation: {corr2:.4f}")
        
        ax3 = fig.add_subplot(gs[i, 2])
        corr3 = create_scatter_plot(ax3, rep2, rep3, f"{condition}: Rep2 vs Rep3", "Rep2", "Rep3")
        correlations[condition]["rep2_vs_rep3"] = corr3
        print(f"Rep2 vs Rep3 correlation: {corr3:.4f}")
    
    # Add a main title
    fig.suptitle('S. cerevisiae Strain Read Count Comparisons Between Replicates', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    output_path = os.path.join(output_dir, "replicate_comparisons.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")
    
    # Create a summary table of correlations
    summary = pd.DataFrame({
        'Condition': conditions,
        'Rep1 vs Rep2': [correlations[c]["rep1_vs_rep2"] for c in conditions],
        'Rep1 vs Rep3': [correlations[c]["rep1_vs_rep3"] for c in conditions],
        'Rep2 vs Rep3': [correlations[c]["rep2_vs_rep3"] for c in conditions]
    })
    
    print("\nCorrelation Summary:")
    print(summary.to_string(index=False, float_format="%.4f"))
    
    # Save the summary table
    summary_path = os.path.join(output_dir, "correlation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    return fig, correlations

if __name__ == "__main__":
    # Specified input and output directories
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/troubleshooting"
    
    # Call the main function with the specified directories
    compare_replicates(data_dir=input_dir, output_dir=output_dir)

    # Uncomment to display the plot interactively
    # plt.show()