import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define paths
base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
counts_file = os.path.join(base_dir, "farah_data/outputs/barcode_counts.csv")
output_dir = os.path.join(base_dir, "farah_data/outputs/env_specific")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the barcode counts
print(f"Reading file: {counts_file}")
counts_df = pd.read_csv(counts_file)

# Display unique conditions/environments
conditions = counts_df['Condition'].unique()
print(f"Found {len(conditions)} unique conditions/environments: {conditions}")

# Get timepoints
timepoints = sorted(counts_df['TimePoint'].unique())
print(f"Found {len(timepoints)} timepoints: {timepoints}")

# Get gene columns (all columns except TimePoint, Condition, Replicate)
gene_columns = [col for col in counts_df.columns if col not in ['TimePoint', 'Condition', 'Replicate']]
print(f"Found {len(gene_columns)} genes")

# Process each condition separately
for condition in conditions:
    print(f"\nProcessing condition: {condition}")
    
    # Filter for this condition
    condition_df = counts_df[counts_df['Condition'] == condition]
    
    # Check for replicates
    replicates = condition_df['Replicate'].unique()
    print(f"  Found {len(replicates)} replicates: {replicates}")
    
    # Process each replicate separately
    for replicate in replicates:
        print(f"  Processing replicate: {replicate}")
        
        # Filter for this replicate
        replicate_df = condition_df[condition_df['Replicate'] == replicate]
        
        # Sort by timepoint
        replicate_df = replicate_df.sort_values('TimePoint')
        
        # Check if we have all timepoints
        replicate_timepoints = replicate_df['TimePoint'].unique()
        print(f"    Found {len(replicate_timepoints)} timepoints: {replicate_timepoints}")
        
        # Skip if we don't have at least 3 timepoints (minimum for reliable fitness estimates)
        if len(replicate_timepoints) < 3:
            print(f"    Skipping: Not enough timepoints for fitness estimation")
            continue
        
        # Create PyFitSeq input for this condition and replicate
        pyfitseq_data = []
        
        for gene in gene_columns:
            gene_counts = []
            for tp in sorted(replicate_timepoints):
                # Get the read count for this gene at this timepoint
                count = replicate_df[replicate_df['TimePoint'] == tp][gene].values[0]
                gene_counts.append(count)
            pyfitseq_data.append(gene_counts)
        
        # Convert to numpy array
        pyfitseq_array = np.array(pyfitseq_data)
        
        # Output filename for this condition and replicate
        output_file = os.path.join(output_dir, f"pyfitseq_input_{condition}_rep{replicate}.csv")
        
        # Save to CSV without header
        np.savetxt(output_file, pyfitseq_array, delimiter=',', fmt='%g')
        print(f"    Saved PyFitSeq input to: {output_file}")
        
        # Generate PyFitSeq command
        timepoints_str = ' '.join(map(str, sorted(replicate_timepoints)))
        cmd = f"python3 {os.path.join(base_dir, 'pyfitseq.py')} -i {output_file} -t {timepoints_str} -o {os.path.join(output_dir, f'results_{condition}_rep{replicate}')}"
        
        # Save command to a shell script
        script_file = os.path.join(output_dir, f"run_pyfitseq_{condition}_rep{replicate}.sh")
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(cmd + "\n")
        
        # Make the script executable
        os.chmod(script_file, 0o755)
        print(f"    Created run script: {script_file}")
        
        # Create a simple overview plot of this dataset
        plt.figure(figsize=(10, 6))
        
        # Plot the first 10 genes across all time points
        for i in range(min(10, pyfitseq_array.shape[0])):
            plt.plot(sorted(replicate_timepoints), pyfitseq_array[i, :], marker='o', label=f"Gene {i+1}")
        
        plt.title(f'Read Counts for First 10 Genes - {condition} Rep{replicate}')
        plt.xlabel('Time Point')
        plt.ylabel('Read Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"data_preview_{condition}_rep{replicate}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"    Created preview plot: {plot_file}")

# Create a master script to run all the individual scripts
master_script = os.path.join(output_dir, "run_all_pyfitseq.sh")
with open(master_script, 'w') as f:
    f.write("#!/bin/bash\n\n")
    f.write("# Master script to run all PyFitSeq analyses\n\n")
    
    for condition in conditions:
        for replicate in counts_df[counts_df['Condition'] == condition]['Replicate'].unique():
            script_file = f"run_pyfitseq_{condition}_rep{replicate}.sh"
            if os.path.exists(os.path.join(output_dir, script_file)):
                f.write(f"echo 'Running {condition} replicate {replicate}...'\n")
                f.write(f"./{script_file}\n")
                f.write("echo 'Done!'\n\n")

os.chmod(master_script, 0o755)
print(f"\nCreated master script: {master_script}")
print("\nNext steps:")
print("1. Change to the output directory:")
print(f"   cd {output_dir}")
print("2. Run the master script to analyze all conditions:")
print("   ./run_all_pyfitseq.sh")
print("3. Or run individual analyses with:")
print("   ./run_pyfitseq_CONDITION_repNUMBER.sh")
