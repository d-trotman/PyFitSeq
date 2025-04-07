import pandas as pd
import os
import numpy as np

# Define paths
base_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
input_file = os.path.join(base_dir, "farah_data/outputs/barcode_counts.csv")
output_file = os.path.join(base_dir, "farah_data/outputs/pyfitseq_input.csv")

# Read the CSV file
print(f"Reading file: {input_file}")
df = pd.read_csv(input_file)

# Display the structure of the data
print("\nOriginal data structure:")
print(f"Columns: {df.columns[:10].tolist()}...")  # Show first 10 columns
print(f"Shape: {df.shape}")

# Extract experiment metadata
timepoints = df['TimePoint'].unique()
conditions = df['Condition'].unique()
replicates = df['Replicate'].unique()

print(f"\nUnique TimePoints: {timepoints}")
print(f"Unique Conditions: {conditions}")
print(f"Unique Replicates: {replicates}")

# Get gene column names (everything except TimePoint, Condition, Replicate)
gene_columns = [col for col in df.columns if col not in ['TimePoint', 'Condition', 'Replicate']]
print(f"\nNumber of genes: {len(gene_columns)}")

# For PyFitSeq, we need to create a matrix where:
# - Each row is a gene
# - Each column is a time point
# - Cell values are read counts

# Let's create a transposed dataframe
# First, filter data if needed (e.g., select one condition)
if len(conditions) > 1:
    print(f"\nMultiple conditions found: {conditions}")
    print(f"Using first condition: {conditions[0]}")
    df_filtered = df[df['Condition'] == conditions[0]]
else:
    df_filtered = df

# Sort by TimePoint to ensure columns are in correct order
df_filtered = df_filtered.sort_values('TimePoint')

# Create a new dataframe for PyFitSeq
# For each gene, create a row with read counts for each time point
pyfitseq_data = []

for gene in gene_columns:
    gene_counts = []
    for timepoint in sorted(timepoints):
        # Get the read count for this gene at this timepoint
        # If there are multiple replicates, average them or select one
        count = df_filtered[df_filtered['TimePoint'] == timepoint][gene].mean()
        gene_counts.append(count)
    pyfitseq_data.append(gene_counts)

# Convert to numpy array and save as CSV without headers or indices
pyfitseq_array = np.array(pyfitseq_data)
print(f"\nTransformed data shape: {pyfitseq_array.shape}")
print(f"First few genes and their counts across timepoints:")
print(pyfitseq_array[:5, :])

# Save to CSV without header
np.savetxt(output_file, pyfitseq_array, delimiter=',', fmt='%g')
print(f"\nSaved transformed data to: {output_file}")

# Display the PyFitSeq command to use
print("\nUse this command to run PyFitSeq:")
timepoints_str = ' '.join(map(str, sorted(timepoints)))
print(f"python3 {os.path.join(base_dir, 'pyfitseq.py')} -i {output_file} -t {timepoints_str} -o farah_results")
