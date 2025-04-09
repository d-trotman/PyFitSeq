import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def calculate_initial_fitness_estimates(read_num_seq, t_seq, fitness_type='m', regression_num=2):
    """
    Calculate the initial fitness estimates using the same method as PyFitSeq.
    
    Parameters:
    -----------
    read_num_seq : numpy.ndarray
        Read numbers for each strain at each time point
    t_seq : numpy.ndarray
        Time points in number of generations
    fitness_type : str, optional
        Type of fitness: 'w' for Wrightian, 'm' for Malthusian (default)
    regression_num : int, optional
        Number of points used in linear regression (default: 2)
    
    Returns:
    --------
    numpy.ndarray
        Initial fitness estimates for each strain
    """
    # Ensure read counts are float and replace zeros with small value
    read_num_seq = read_num_seq.astype(float)
    read_num_seq[read_num_seq == 0] = 1e-1
    
    # Calculate read frequencies
    read_depth_seq = np.sum(read_num_seq, axis=0)
    read_freq_seq = read_num_seq / read_depth_seq
    
    # Calculate initial fitness estimates
    if fitness_type == 'w':  # Wrightian fitness
        if regression_num == 2:
            x0_tempt = np.power(np.true_divide(read_freq_seq[:, 1], read_freq_seq[:, 0]), 
                               1 / (t_seq[1] - t_seq[0])) - 1
        else:
            from scipy.stats import linregress
            x0_tempt = [linregress(t_seq[0:regression_num], 
                                  np.log(read_freq_seq[i, 0:regression_num])).slope 
                       for i in range(read_freq_seq.shape[0])]
            x0_tempt = np.exp(x0_tempt) - 1
        
        # Normalization
        x0 = (1 + x0_tempt) / (1 + np.dot(read_freq_seq[:, 0], x0_tempt)) - 1
    
    elif fitness_type == 'm':  # Malthusian fitness
        if regression_num == 2:
            x0_tempt = np.true_divide(read_freq_seq[:, 1] - read_freq_seq[:, 0], 
                                    t_seq[1] - t_seq[0])
        else:
            from scipy.stats import linregress
            x0_tempt = [linregress(t_seq[0:regression_num], 
                                  np.log(read_freq_seq[i, 0:regression_num])).slope 
                       for i in range(read_freq_seq.shape[0])]
        
        # Normalization
        x0 = x0_tempt - np.dot(read_freq_seq[:, 0], x0_tempt)
    
    return x0

def compare_initial_fitness_estimates(data_dir, output_dir, condition, t_seq, 
                                     fitness_type='m', regression_num=2, replicates=[1, 2, 3]):
    """
    Compare initial fitness estimates between replicates of a condition.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory to save output files
    condition : str
        Condition to analyze (e.g., 'Clim', 'Nlim', 'Switch')
    t_seq : list or numpy.ndarray
        Time points in number of generations
    fitness_type : str, optional
        Type of fitness: 'w' for Wrightian, 'm' for Malthusian (default)
    regression_num : int, optional
        Number of points used in linear regression (default: 2)
    replicates : list, optional
        List of replicate numbers to compare (default: [1, 2, 3])
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert t_seq to numpy array if it's not already
    t_seq = np.array(t_seq, dtype=float)
    
    # Load data and calculate initial fitness estimates for each replicate
    fitness_estimates = {}
    read_data = {}
    
    for rep in replicates:
        file_path = os.path.join(data_dir, f"pyfitseq_input_{condition}_rep{rep}.csv")
        try:
            # Load read counts
            read_data[f"rep{rep}"] = np.array(pd.read_csv(file_path, header=None), dtype=float)
            
            # Handle zeros and small values
            for i in range(regression_num):
                pos_zero = np.where(read_data[f"rep{rep}"][:, i] < 1)
                read_data[f"rep{rep}"][pos_zero, i] = 1
                
            # Calculate initial fitness estimates
            fitness_estimates[f"rep{rep}"] = calculate_initial_fitness_estimates(
                read_data[f"rep{rep}"], t_seq, fitness_type, regression_num)
            
            print(f"Processed {file_path}: {len(fitness_estimates[f'rep{rep}'])} strains")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Calculate correlation between replicates
    correlations = {}
    for i, rep1 in enumerate(replicates):
        for j, rep2 in enumerate(replicates):
            if i < j:  # Only calculate for unique pairs
                rep1_name = f"rep{rep1}"
                rep2_name = f"rep{rep2}"
                
                if rep1_name in fitness_estimates and rep2_name in fitness_estimates:
                    # Find common strains
                    min_strains = min(len(fitness_estimates[rep1_name]), 
                                     len(fitness_estimates[rep2_name]))
                    
                    # Calculate correlations
                    pearson_r, pearson_p = pearsonr(
                        fitness_estimates[rep1_name][:min_strains], 
                        fitness_estimates[rep2_name][:min_strains])
                    
                    spearman_r, spearman_p = spearmanr(
                        fitness_estimates[rep1_name][:min_strains], 
                        fitness_estimates[rep2_name][:min_strains])
                    
                    # Store results
                    correlations[f"{rep1_name} vs {rep2_name}"] = {
                        "Pearson R": pearson_r,
                        "Pearson p-value": pearson_p,
                        "Spearman R": spearman_r,
                        "Spearman p-value": spearman_p
                    }
                    
                    print(f"Correlation between {rep1_name} and {rep2_name}:")
                    print(f"  Pearson R: {pearson_r:.4f} (p-value: {pearson_p:.4e})")
                    print(f"  Spearman R: {spearman_r:.4f} (p-value: {spearman_p:.4e})")
    
    # Create visualization - scatter plots for each pair of replicates
    if len(fitness_estimates) >= 2:
        # Determine number of comparison plots needed
        num_comparisons = sum(1 for i, rep1 in enumerate(replicates) 
                              for j, rep2 in enumerate(replicates) if i < j 
                              and f"rep{rep1}" in fitness_estimates and f"rep{rep2}" in fitness_estimates)
        
        if num_comparisons > 0:
            # Create subplots
            fig, axes = plt.subplots(1, num_comparisons, figsize=(5*num_comparisons, 5))
            if num_comparisons == 1:
                axes = [axes]  # Make it iterable if only one comparison
            
            # Create scatter plots
            plot_idx = 0
            for i, rep1 in enumerate(replicates):
                for j, rep2 in enumerate(replicates):
                    if i < j:  # Only plot for unique pairs
                        rep1_name = f"rep{rep1}"
                        rep2_name = f"rep{rep2}"
                        
                        if rep1_name in fitness_estimates and rep2_name in fitness_estimates:
                            # Find common strains
                            min_strains = min(len(fitness_estimates[rep1_name]), 
                                             len(fitness_estimates[rep2_name]))
                            
                            # Get fitness values for plotting
                            x = fitness_estimates[rep1_name][:min_strains]
                            y = fitness_estimates[rep2_name][:min_strains]
                            
                            # Create scatter plot
                            axes[plot_idx].scatter(x, y, alpha=0.5, s=5)
                            
                            # Add x=y line
                            min_val = min(np.min(x), np.min(y))
                            max_val = max(np.max(x), np.max(y))
                            axes[plot_idx].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3)
                            
                            # Set labels and title
                            axes[plot_idx].set_xlabel(f'{rep1_name} Fitness')
                            axes[plot_idx].set_ylabel(f'{rep2_name} Fitness')
                            correlation_text = f"Pearson R: {correlations[f'{rep1_name} vs {rep2_name}']['Pearson R']:.4f}\nSpearman R: {correlations[f'{rep1_name} vs {rep2_name}']['Spearman R']:.4f}"
                            axes[plot_idx].set_title(f'{rep1_name} vs {rep2_name}\n{correlation_text}')
                            
                            # Increment plot index
                            plot_idx += 1
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{condition}_initial_fitness_comparison.png"), dpi=300)
            print(f"Saved comparison plot to {os.path.join(output_dir, f'{condition}_initial_fitness_comparison.png')}")
        
        # Also save the fitness estimates to CSV for further analysis
        for rep_name, fitness in fitness_estimates.items():
            df = pd.DataFrame(fitness, columns=['Initial_Fitness'])
            df.to_csv(os.path.join(output_dir, f"{condition}_{rep_name}_initial_fitness.csv"), index=True)
            print(f"Saved {rep_name} fitness estimates to {os.path.join(output_dir, f'{condition}_{rep_name}_initial_fitness.csv')}")
    
    # Create a summary heatmap of correlations
    if len(correlations) > 0:
        # Prepare correlation matrix
        corr_matrix = np.zeros((len(replicates), len(replicates)))
        for i, rep1 in enumerate(replicates):
            for j, rep2 in enumerate(replicates):
                rep1_name = f"rep{rep1}"
                rep2_name = f"rep{rep2}"
                
                if i == j:
                    corr_matrix[i, j] = 1.0  # Diagonal is always 1.0
                elif i < j and f"{rep1_name} vs {rep2_name}" in correlations:
                    corr_val = correlations[f"{rep1_name} vs {rep2_name}"]["Pearson R"]
                    corr_matrix[i, j] = corr_val
                    corr_matrix[j, i] = corr_val  # Matrix is symmetric
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='viridis', 
                   xticklabels=[f'Rep{r}' for r in replicates],
                   yticklabels=[f'Rep{r}' for r in replicates])
        plt.title(f'{condition} Initial Fitness Estimates - Pearson Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{condition}_initial_fitness_correlation_heatmap.png"), dpi=300)
        print(f"Saved correlation heatmap to {os.path.join(output_dir, f'{condition}_initial_fitness_correlation_heatmap.png')}")
    
    return fitness_estimates, correlations

def analyze_all_conditions(data_dir, output_dir, t_seq, fitness_type='m', regression_num=2):
    """
    Analyze initial fitness estimates for all conditions.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory to save output files
    t_seq : list or numpy.ndarray
        Time points in number of generations
    fitness_type : str, optional
        Type of fitness: 'w' for Wrightian, 'm' for Malthusian (default)
    regression_num : int, optional
        Number of points used in linear regression (default: 2)
    """
    conditions = ['Clim', 'Nlim', 'Switch']
    
    for condition in conditions:
        print(f"\n===== Analyzing {condition} =====")
        compare_initial_fitness_estimates(data_dir, output_dir, condition, 
                                         t_seq, fitness_type, regression_num)
    
    print("\nAll analyses complete!")

if __name__ == "__main__":
    # Set your input and output directories
    input_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific/"
    output_dir = "/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/troubleshooting"
    
    # Define time points (adjust these to match your experiment)
    t_seq = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216]
    
    # Set fitness type and regression number to match your PyFitSeq settings
    fitness_type = 'm'  # 'm' for Malthusian, 'w' for Wrightian
    regression_num = 2  # Default in PyFitSeq
    
    # Run analysis for all conditions
    analyze_all_conditions(input_dir, output_dir, t_seq, fitness_type, regression_num)
