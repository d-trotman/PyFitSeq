import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define paths
base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
counts_file = os.path.join(base_dir, "farah_data/outputs/barcode_counts.csv")
output_dir = os.path.join(base_dir, "farah_data/outputs/fluctuating_env")
plots_dir = os.path.join(output_dir, "plots")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Define fluctuating environment parameters
# For each condition, define when environment switches occur (in hours)
env_switches = {
    "Switch": [0, 30, 60, 90, 120, 150, 180, 210, 240],  # Environment changes every 30h
    "PulseAS": [0, 30, 60, 90, 120, 150, 180, 210, 240], # Environment changes every 30h
    "PulseGln": [0, 30, 60, 90, 120, 150, 180, 210, 240], # Environment changes every 30h
    "Clim": [],  # Constant environment
    "Nlim": []   # Constant environment
}

# Load count data
print(f"Reading counts file: {counts_file}")
counts_df = pd.read_csv(counts_file)
print(f"Shape of data: {counts_df.shape}")

# Get timepoints, conditions, and gene columns
timepoints = sorted(counts_df['TimePoint'].unique())
conditions = sorted(counts_df['Condition'].unique())
gene_columns = [col for col in counts_df.columns if col not in ['TimePoint', 'Condition', 'Replicate']]
print(f"Found {len(gene_columns)} genes")
print(f"Found timepoints: {timepoints}")
print(f"Found conditions: {conditions}")

# Function to segment data by environment phase
def segment_by_environment(df, condition, timepoints):
    """Segment data based on environment changes"""
    
    # Get switches for this condition
    switches = env_switches.get(condition, [])
    
    # If no switches defined or constant environment, return all data as one segment
    if not switches:
        return {0: df.copy()}
    
    # Create segments
    segments = {}
    for i in range(len(switches) - 1):
        start_time = switches[i]
        end_time = switches[i+1]
        
        # Get data for this segment
        segment_data = df[(df['TimePoint'] >= start_time) & (df['TimePoint'] < end_time)]
        
        # Only include segment if it has at least 2 timepoints
        if segment_data['TimePoint'].nunique() >= 2:
            segments[i] = segment_data
    
    return segments

# Process each condition and replicate
for condition in conditions:
    print(f"\nProcessing condition: {condition}")
    
    # Get all replicates for this condition
    replicates = sorted(counts_df[counts_df['Condition'] == condition]['Replicate'].unique())
    
    for replicate in replicates:
        print(f"  Processing replicate: {replicate}")
        
        # Get data for this condition and replicate
        cond_rep_df = counts_df[(counts_df['Condition'] == condition) & 
                               (counts_df['Replicate'] == replicate)]
        
        # Segment data by environment phase
        segments = segment_by_environment(cond_rep_df, condition, timepoints)
        
        # Create PyFitSeq input files for each segment
        for segment_id, segment_data in segments.items():
            # Only process if we have at least 2 timepoints
            if segment_data['TimePoint'].nunique() < 2:
                print(f"    Segment {segment_id} has fewer than 2 timepoints, skipping")
                continue
                
            segment_timepoints = sorted(segment_data['TimePoint'].unique())
            print(f"    Segment {segment_id}: Timepoints {segment_timepoints}")
            
            # Create PyFitSeq input for this segment
            pyfitseq_data = []
            
            for gene in gene_columns:
                gene_counts = []
                for tp in segment_timepoints:
                    # Get count for this gene at this timepoint
                    count = segment_data[segment_data['TimePoint'] == tp][gene].values[0]
                    gene_counts.append(count)
                pyfitseq_data.append(gene_counts)
            
            # Convert to numpy array
            pyfitseq_array = np.array(pyfitseq_data)
            
            # Create output filename
            if len(segments) > 1:
                output_file = os.path.join(output_dir, f"pyfitseq_input_{condition}_rep{replicate}_segment{segment_id}.csv")
            else:
                output_file = os.path.join(output_dir, f"pyfitseq_input_{condition}_rep{replicate}.csv")
            
            # Save to CSV
            np.savetxt(output_file, pyfitseq_array, delimiter=',', fmt='%g')
            print(f"    Saved PyFitSeq input to: {output_file}")
            
            # Create script to run PyFitSeq
            if len(segments) > 1:
                script_file = os.path.join(output_dir, f"run_pyfitseq_{condition}_rep{replicate}_segment{segment_id}.sh")
                results_prefix = f"results_{condition}_rep{replicate}_segment{segment_id}"
            else:
                script_file = os.path.join(output_dir, f"run_pyfitseq_{condition}_rep{replicate}.sh")
                results_prefix = f"results_{condition}_rep{replicate}"
            
            # Convert timepoints to generational times (assuming generations)
            # Here we convert to relative hours from the start of the segment
            relative_timepoints = [t - segment_timepoints[0] for t in segment_timepoints]
            timepoints_str = ' '.join(map(str, relative_timepoints))
            
            with open(script_file, 'w') as f:
                f.write("#!/bin/bash\n\n")
                f.write(f"# Run PyFitSeq for {condition} replicate {replicate} segment {segment_id}\n")
                f.write(f"python3 {os.path.join(base_dir, 'PyFitSeq/pyfitseq.py')} \\\n")
                f.write(f"  -i {output_file} \\\n")
                f.write(f"  -t {timepoints_str} \\\n")
                f.write(f"  -o {os.path.join(output_dir, results_prefix)}\n")
            
            # Make script executable
            os.chmod(script_file, 0o755)
            print(f"    Created run script: {script_file}")
            
            # Create a plot to visualize this segment's data
            plt.figure(figsize=(10, 6))
            
            # Plot counts for first 10 genes
            for i in range(min(10, len(gene_columns))):
                gene = gene_columns[i]
                plt.plot(segment_data['TimePoint'], segment_data[gene], marker='o', label=f"{gene}")
            
            plt.title(f"{condition} Rep{replicate} Segment {segment_id}")
            plt.xlabel("Time (hours)")
            plt.ylabel("Count")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add vertical lines for environment changes
            for switch in env_switches.get(condition, []):
                if switch > min(segment_timepoints) and switch < max(segment_timepoints):
                    plt.axvline(x=switch, color='r', linestyle='--', alpha=0.5)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            plt.tight_layout()
            
            # Save plot
            if len(segments) > 1:
                plot_file = os.path.join(plots_dir, f"{condition}_rep{replicate}_segment{segment_id}_preview.png")
            else:
                plot_file = os.path.join(plots_dir, f"{condition}_rep{replicate}_preview.png")
            
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"    Created preview plot: {plot_file}")

# Create master script to run all analyses
master_script = os.path.join(output_dir, "run_all_pyfitseq.sh")
with open(master_script, 'w') as f:
    f.write("#!/bin/bash\n\n")
    f.write("# Master script to run all PyFitSeq analyses for fluctuating environments\n\n")
    
    for sh_file in [f for f in os.listdir(output_dir) if f.startswith('run_pyfitseq_') and f.endswith('.sh')]:
        f.write(f"echo 'Running {sh_file}...'\n")
        f.write(f"./{sh_file}\n")
        f.write("echo 'Done!'\n\n")

os.chmod(master_script, 0o755)
print(f"\nCreated master script: {master_script}")

# Create a visualization to compare old vs new segmentation approach
plt.figure(figsize=(14, 10))

# Create a 2x3 grid of subplots for the first few conditions and replicates
for i, condition in enumerate(conditions[:min(len(conditions), 6)]):
    plt.subplot(2, 3, i+1)
    
    # Get first replicate for this condition
    replicate = counts_df[counts_df['Condition'] == condition]['Replicate'].unique()[0]
    
    # Get data
    cond_rep_df = counts_df[(counts_df['Condition'] == condition) & 
                           (counts_df['Replicate'] == replicate)]
    
    # Plot gene count trajectories
    for j, gene in enumerate(gene_columns[:5]):  # First 5 genes
        plt.plot(cond_rep_df['TimePoint'], cond_rep_df[gene], marker='o', label=f"Gene {j+1}")
    
    # Add vertical lines for environment changes
    for switch in env_switches.get(condition, []):
        plt.axvline(x=switch, color='r', linestyle='--', alpha=0.5, 
                   label="Environment Switch" if switch == env_switches.get(condition, [])[1] else "")
    
    plt.title(f"{condition} Rep{replicate}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if i == 0:  # Only add legend for first plot
        plt.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "environment_segments_overview.png"), dpi=300)
plt.close()

print("\nCreated overview plot of segmentation strategy.")
print("\nNext steps:")
print("1. Change to the fluctuating environment directory:")
print(f"   cd {output_dir}")
print("2. Run the master script to analyze all segments:")
print("   ./run_all_pyfitseq.sh")
print("3. After PyFitSeq completes, visualize the results for each environment phase separately")
