import pandas as pd
import numpy as np
import os
import argparse
from itertools import combinations

def analyze_all_timepoints(input_dir, output_dir, replicate_ids=None, num_timepoints=10):
    """
    Analyze all specified timepoints across multiple replicates to identify outliers and irregularities.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory where output files will be saved
    replicate_ids : list or None
        List of replicate IDs to analyze (e.g., [1, 2, 3]). If None, all files matching pattern will be processed.
    num_timepoints : int
        Number of timepoints to analyze in each file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the file pattern
    file_pattern = 'pyfitseq_input_Clim_rep'
    
    # Get all CSV files in the input directory
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and file_pattern in f]
    
    # Filter files based on replicate_ids if provided
    if replicate_ids:
        files_to_process = [f for f in all_files if any(f'rep{rep}' in f for rep in replicate_ids)]
    else:
        files_to_process = all_files
    
    # Check if we found any files
    if not files_to_process:
        print(f"No matching files found in {input_dir}")
        return
    
    print(f"Found {len(files_to_process)} files to process: {', '.join(files_to_process)}")
    
    # Process each file (replicate)
    for file_name in files_to_process:
        file_path = os.path.join(input_dir, file_name)
        replicate_id = file_name.split('rep')[1].split('.')[0]  # Extract replicate ID
        
        print(f"\n{'=' * 50}")
        print(f"Processing replicate {replicate_id}: {file_name}")
        print(f"{'=' * 50}")
        
        # Create replicate-specific output directory
        rep_output_dir = os.path.join(output_dir, f"replicate_{replicate_id}")
        os.makedirs(rep_output_dir, exist_ok=True)
        
        # Load the data
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, header=None)
        
        # Verify we have enough columns for the requested timepoints
        available_timepoints = min(num_timepoints, len(df.columns))
        if available_timepoints < num_timepoints:
            print(f"Warning: Requested {num_timepoints} timepoints but file only has {available_timepoints}.")
        
        # Add row information
        df['row_idx'] = df.index + 1  # +1 to match spreadsheet row numbers (1-indexed)
        
        # Analyze each pair of consecutive timepoints
        analyze_consecutive_timepoints(df, available_timepoints, rep_output_dir)
        
        # Analyze all timepoints together
        analyze_all_timepoints_together(df, available_timepoints, rep_output_dir)


def analyze_consecutive_timepoints(df, num_timepoints, output_dir):
    """Analyze consecutive pairs of timepoints to identify outliers."""
    print("\nAnalyzing consecutive timepoint pairs...")
    
    # For each consecutive pair of timepoints
    for t1 in range(num_timepoints - 1):
        t2 = t1 + 1
        print(f"\nAnalyzing timepoints {t1} → {t2}")
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame()
        analysis_df['row_number'] = df['row_idx']
        analysis_df['reads_t1'] = df[t1]
        analysis_df['reads_t2'] = df[t2]
        
        # Calculate column sums
        t1_total_reads = df[t1].sum()
        t2_total_reads = df[t2].sum()
        
        print(f"Total reads at timepoint {t1}: {t1_total_reads:,}")
        print(f"Total reads at timepoint {t2}: {t2_total_reads:,}")
        
        # Calculate frequencies
        analysis_df['freq_t1'] = df[t1] / t1_total_reads
        analysis_df['freq_t2'] = df[t2] / t2_total_reads
        
        # Calculate ratios and ln ratios with protection against zero
        epsilon = 1e-10  # Small value to prevent division by zero
        analysis_df['freq_ratio'] = (analysis_df['freq_t2'] + epsilon) / (analysis_df['freq_t1'] + epsilon)
        analysis_df['ln_freq_ratio'] = np.log(analysis_df['freq_ratio'])
        
        # Sort by absolute ln frequency ratio to find extreme values
        analysis_df['abs_ln_freq_ratio'] = analysis_df['ln_freq_ratio'].abs()
        analysis_df = analysis_df.sort_values('abs_ln_freq_ratio', ascending=False)
        
        # Print statistics
        print("\nStatistics for ln frequency ratios:")
        print(f"Mean: {analysis_df['ln_freq_ratio'].mean():.4f}")
        print(f"Median: {analysis_df['ln_freq_ratio'].median():.4f}")
        print(f"Min: {analysis_df['ln_freq_ratio'].min():.4f}")
        print(f"Max: {analysis_df['ln_freq_ratio'].max():.4f}")
        print(f"Number of values > 5: {(analysis_df['ln_freq_ratio'] > 5).sum()}")
        print(f"Number of values < -5: {(analysis_df['ln_freq_ratio'] < -5).sum()}")
        
        # Extract top 50 most extreme values
        extreme_values = analysis_df.head(50)
        
        # Save results to CSV
        pair_dir = os.path.join(output_dir, f"timepoints_{t1}_to_{t2}")
        os.makedirs(pair_dir, exist_ok=True)
        
        output_file = os.path.join(pair_dir, f"ln_ratio_analysis_t{t1}_t{t2}.csv")
        analysis_df.to_csv(output_file, index=False)
        print(f"Saved full analysis to {output_file}")
        
        # Save extreme values to a separate CSV
        extreme_file = os.path.join(pair_dir, f"top50_extreme_values_t{t1}_t{t2}.csv")
        extreme_values.to_csv(extreme_file, index=False)
        print(f"Saved top 50 extreme values to {extreme_file}")
        
        # Zero read analysis
        analyze_zero_reads(analysis_df, t1, t2, pair_dir)
        
        # Read count distribution
        analyze_read_distribution(df, t1, t2, pair_dir)


def analyze_all_timepoints_together(df, num_timepoints, output_dir):
    """Analyze trends across all timepoints together."""
    print("\nAnalyzing all timepoints together...")
    
    # Create a directory for all-timepoint analysis
    all_tp_dir = os.path.join(output_dir, "all_timepoints")
    os.makedirs(all_tp_dir, exist_ok=True)
    
    # Create a comprehensive DataFrame with all timepoints
    comprehensive_df = pd.DataFrame()
    comprehensive_df['row_number'] = df['row_idx']
    
    # Add read counts for each timepoint
    for tp in range(num_timepoints):
        comprehensive_df[f'reads_t{tp}'] = df[tp]
    
    # Calculate total reads per timepoint
    total_reads = {}
    for tp in range(num_timepoints):
        total_reads[tp] = df[tp].sum()
        print(f"Total reads at timepoint {tp}: {total_reads[tp]:,}")
    
    # Add frequencies for each timepoint
    for tp in range(num_timepoints):
        comprehensive_df[f'freq_t{tp}'] = df[tp] / total_reads[tp]
    
    # Calculate CV (coefficient of variation) of frequencies across timepoints
    freq_cols = [f'freq_t{tp}' for tp in range(num_timepoints)]
    comprehensive_df['freq_mean'] = comprehensive_df[freq_cols].mean(axis=1)
    comprehensive_df['freq_std'] = comprehensive_df[freq_cols].std(axis=1)
    comprehensive_df['freq_cv'] = comprehensive_df['freq_std'] / (comprehensive_df['freq_mean'] + 1e-10)
    
    # Calculate number of zero reads per strain
    for tp in range(num_timepoints):
        comprehensive_df[f'zero_t{tp}'] = (comprehensive_df[f'reads_t{tp}'] == 0).astype(int)
    
    comprehensive_df['zero_count'] = comprehensive_df[[f'zero_t{tp}' for tp in range(num_timepoints)]].sum(axis=1)
    
    # Sort by CV to find strains with most variable frequencies
    high_variance_df = comprehensive_df.sort_values('freq_cv', ascending=False)
    
    # Save the comprehensive analysis
    comp_file = os.path.join(all_tp_dir, "comprehensive_timepoint_analysis.csv")
    comprehensive_df.to_csv(comp_file, index=False)
    print(f"Saved comprehensive analysis to {comp_file}")
    
    # Save top variable strains
    var_file = os.path.join(all_tp_dir, "top50_variable_strains.csv")
    high_variance_df.head(50).to_csv(var_file, index=False)
    print(f"Saved top 50 highly variable strains to {var_file}")
    
    # Analyze strains with many zero reads
    many_zeros_df = comprehensive_df.sort_values('zero_count', ascending=False)
    zeros_file = os.path.join(all_tp_dir, "strains_with_many_zeros.csv")
    many_zeros_df[many_zeros_df['zero_count'] > 0].to_csv(zeros_file, index=False)
    print(f"Saved analysis of strains with zero reads to {zeros_file}")
    
    # Analyze pairwise correlations between timepoints
    correlation_analysis(df, num_timepoints, all_tp_dir)


def analyze_zero_reads(analysis_df, t1, t2, output_dir):
    """Analyze cases with zero reads at one or both timepoints."""
    zero_t1 = analysis_df[analysis_df['reads_t1'] == 0]
    zero_t2 = analysis_df[analysis_df['reads_t2'] == 0]
    zero_both = analysis_df[(analysis_df['reads_t1'] == 0) & (analysis_df['reads_t2'] == 0)]
    
    print(f"\nZero read counts analysis:")
    print(f"Strains with zero reads at timepoint {t1}: {len(zero_t1)}")
    print(f"Strains with zero reads at timepoint {t2}: {len(zero_t2)}")
    print(f"Strains with zero reads at both timepoints: {len(zero_both)}")
    
    # Save zero read analysis
    if len(zero_t1) > 0:
        zero_t1.to_csv(os.path.join(output_dir, f"zero_reads_t{t1}.csv"), index=False)
    
    if len(zero_t2) > 0:
        zero_t2.to_csv(os.path.join(output_dir, f"zero_reads_t{t2}.csv"), index=False)
    
    if len(zero_both) > 0:
        zero_both.to_csv(os.path.join(output_dir, f"zero_reads_both.csv"), index=False)


def analyze_read_distribution(df, t1, t2, output_dir):
    """Analyze the distribution of read counts at two timepoints."""
    for tp_idx, tp_name in [(t1, f"timepoint {t1}"), (t2, f"timepoint {t2}")]:
        reads = df[tp_idx]
        print(f"\nRead count distribution at {tp_name}:")
        print(reads.describe())
        
        # Calculate percentages of strains with different read counts
        zero_pct = (reads == 0).mean() * 100
        one_pct = (reads == 1).mean() * 100
        low_pct = ((reads >= 2) & (reads <= 5)).mean() * 100
        med_pct = ((reads >= 6) & (reads <= 10)).mean() * 100
        high_pct = (reads > 10).mean() * 100
        
        print(f"0 reads: {zero_pct:.2f}%")
        print(f"1 read: {one_pct:.2f}%")
        print(f"2-5 reads: {low_pct:.2f}%")
        print(f"6-10 reads: {med_pct:.2f}%")
        print(f">10 reads: {high_pct:.2f}%")
        
        # Save distribution data
        dist_df = pd.DataFrame({
            'read_count_range': ['0', '1', '2-5', '6-10', '>10'],
            'percentage': [zero_pct, one_pct, low_pct, med_pct, high_pct],
            'count': [
                (reads == 0).sum(),
                (reads == 1).sum(),
                ((reads >= 2) & (reads <= 5)).sum(),
                ((reads >= 6) & (reads <= 10)).sum(),
                (reads > 10).sum()
            ]
        })
        
        dist_file = os.path.join(output_dir, f"read_distribution_t{tp_idx}.csv")
        dist_df.to_csv(dist_file, index=False)
        print(f"Saved read distribution to {dist_file}")


def correlation_analysis(df, num_timepoints, output_dir):
    """Analyze pairwise correlations between all timepoints."""
    print("\nAnalyzing correlations between timepoints...")
    
    # Create a correlation matrix DataFrame
    corr_matrix = pd.DataFrame(index=range(num_timepoints), columns=range(num_timepoints))
    
    # Fill the correlation matrix
    for i, j in combinations(range(num_timepoints), 2):
        # Calculate Pearson correlation
        corr_val = df[i].corr(df[j])
        corr_matrix.loc[i, j] = corr_val
        corr_matrix.loc[j, i] = corr_val  # Mirror the value
    
    # Fill diagonal with 1.0
    for i in range(num_timepoints):
        corr_matrix.loc[i, i] = 1.0
    
    # Save correlation matrix
    corr_file = os.path.join(output_dir, "timepoint_correlations.csv")
    corr_matrix.to_csv(corr_file)
    print(f"Saved timepoint correlations to {corr_file}")
    
    # Print the correlation summary
    print("\nTimepoint correlation summary:")
    print(corr_matrix)


def main():
    parser = argparse.ArgumentParser(description='Analyze PyFitSeq outputs for outliers across multiple timepoints and replicates.')
    parser.add_argument('--input_dir', type=str, 
                        default="/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/",
                        help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str, 
                        default="/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/troubleshooting/frequency",
                        help='Directory where output files will be saved')
    parser.add_argument('--replicates', type=str, default=None,
                        help='Comma-separated list of replicate IDs to analyze (e.g., "1,2,3")')
    parser.add_argument('--timepoints', type=int, default=10,
                        help='Number of timepoints to analyze in each file')
    
    args = parser.parse_args()
    
    # Process replicate IDs if provided
    replicate_ids = None
    if args.replicates:
        replicate_ids = [int(rep.strip()) for rep in args.replicates.split(',')]
    
    # Run the analysis
    analyze_all_timepoints(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        replicate_ids=replicate_ids,
        num_timepoints=args.timepoints
    )


if __name__ == "__main__":
    main()