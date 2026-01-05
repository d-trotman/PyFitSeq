import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Define paths
base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
input_file = os.path.join(base_dir, "farah_data/outputs/pyfitseq_input.csv")
original_counts = os.path.join(base_dir, "farah_data/outputs/barcode_counts.csv")
output_file = os.path.join(base_dir, "farah_data/outputs/pyfitseq_input_corrected.csv")
results_file = os.path.join(base_dir, "farah_data/outputs/farah_results_FitSeq.csv")

print("Checking PyFitSeq input file format...")

# Load the pyfitseq input file
try:
    pyfitseq_input = np.loadtxt(input_file, delimiter=',')
    print(f"Successfully loaded input file: {input_file}")
    print(f"Shape: {pyfitseq_input.shape}")
    print(f"Number of genes: {pyfitseq_input.shape[0]}")
    print(f"Number of time points: {pyfitseq_input.shape[1]}")
except Exception as e:
    print(f"Error loading input file: {e}")

# Load the original counts data for comparison
try:
    counts_df = pd.read_csv(original_counts)
    gene_columns = [col for col in counts_df.columns if col not in ['TimePoint', 'Condition', 'Replicate']]
    print(f"\nOriginal data has {len(gene_columns)} genes and {counts_df['TimePoint'].nunique()} time points")
except Exception as e:
    print(f"Error loading original counts: {e}")

# Load the results file
try:
    results_df = pd.read_csv(results_file)
    print(f"\nResults file has {results_df.shape[0]} genes and {results_df.shape[1]} columns")
    print("First few columns:", results_df.columns[:5].tolist())
    
    # Check estimated read counts for each time point
    read_count_cols = [col for col in results_df.columns if 'Estimated_Read_Number' in col]
    print(f"Found {len(read_count_cols)} time points in results")
    
    # Print a summary of the fitness results
    print(f"\nEstimated Fitness statistics:")
    print(f"Mean: {results_df['Estimated_Fitness'].mean():.4f}")
    print(f"Median: {results_df['Estimated_Fitness'].median():.4f}")
    print(f"Min: {results_df['Estimated_Fitness'].min():.4f}")
    print(f"Max: {results_df['Estimated_Fitness'].max():.4f}")
    
except Exception as e:
    print(f"Error loading results file: {e}")

# Create a plot to visualize the input data structure
try:
    plt.figure(figsize=(12, 6))
    
    # Plot the first 10 genes across all time points
    for i in range(min(10, pyfitseq_input.shape[0])):
        plt.plot(pyfitseq_input[i, :], marker='o', label=f"Gene {i+1}")
    
    plt.title('Read Counts for First 10 Genes Across Time Points')
    plt.xlabel('Time Point')
    plt.ylabel('Read Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plot_file = os.path.join(base_dir, "farah_data/outputs/input_data_check.png")
    plt.savefig(plot_file)
    print(f"\nPlot saved to: {plot_file}")
    
    # Also create a heatmap of a sample of the input data
    plt.figure(figsize=(12, 8))
    sample_size = min(100, pyfitseq_input.shape[0])
    plt.imshow(pyfitseq_input[:sample_size, :], aspect='auto', cmap='viridis')
    plt.colorbar(label='Read Count')
    plt.title(f'Heatmap of First {sample_size} Genes Across Time Points')
    plt.xlabel('Time Point')
    plt.ylabel('Gene Index')
    
    # Save the heatmap
    heatmap_file = os.path.join(base_dir, "farah_data/outputs/input_data_heatmap.png")
    plt.savefig(heatmap_file)
    print(f"Heatmap saved to: {heatmap_file}")
    
except Exception as e:
    print(f"Error creating plots: {e}")

# Check if we need to reformat the input
print("\nChecking if input needs reformatting...")

# If the number of genes in the input doesn't match the original data,
# or if the number of time points is wrong, reformat the data
if (pyfitseq_input.shape[1] != counts_df['TimePoint'].nunique() or 
    pyfitseq_input.shape[0] != len(gene_columns)):
    
    print("Input format appears incorrect. Reformatting...")
    
    # Create a proper input file with genes as rows and time points as columns
    # Start with an empty DataFrame
    timepoints = sorted(counts_df['TimePoint'].unique())
    new_input = np.zeros((len(gene_columns), len(timepoints)))
    
    for i, gene in enumerate(gene_columns):
        for j, tp in enumerate(timepoints):
            # Get counts for this gene at this time point
            new_input[i, j] = counts_df[counts_df['TimePoint'] == tp][gene].mean()
    
    # Save the corrected input
    np.savetxt(output_file, new_input, delimiter=',', fmt='%g')
    print(f"Corrected input saved to: {output_file}")
    
    # Show command to run PyFitSeq with corrected input
    timepoints_str = ' '.join(map(str, timepoints))
    print(f"\nRun PyFitSeq with this command:")
    print(f"python3 {os.path.join(base_dir, 'pyfitseq.py')} -i {output_file} -t {timepoints_str} -o farah_results_corrected")
else:
    print("Input format appears correct!")
    print("\nVerify the time points used in your PyFitSeq command:")
    timepoints = sorted(counts_df['TimePoint'].unique())
    timepoints_str = ' '.join(map(str, timepoints))
    print(f"python3 {os.path.join(base_dir, 'pyfitseq.py')} -i {input_file} -t {timepoints_str} -o farah_results")
