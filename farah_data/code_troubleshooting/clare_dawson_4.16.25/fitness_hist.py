#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:08:20 2025

@author: ca3258
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import pearsonr

#%%
fit_data_path = '../../outputs/env_specific/'

switch1_fit = pd.read_csv(f'{fit_data_path}results_Switch_rep1_FitSeq.csv',index_col=0)
switch2_fit = pd.read_csv(f'{fit_data_path}results_Switch_rep2_FitSeq.csv',index_col=0)
switch3_fit = pd.read_csv(f'{fit_data_path}results_Switch_rep3_FitSeq.csv',index_col=0)

Clim1_fit = pd.read_csv(f'{fit_data_path}results_Clim_rep1_FitSeq.csv',index_col=0)
Clim2_fit = pd.read_csv(f'{fit_data_path}results_Clim_rep2_FitSeq.csv',index_col=0)
Clim3_fit = pd.read_csv(f'{fit_data_path}results_Clim_rep3_FitSeq.csv',index_col=0)

Nlim1_fit = pd.read_csv(f'{fit_data_path}results_Nlim_rep1_FitSeq.csv',index_col=0)
Nlim2_fit = pd.read_csv(f'{fit_data_path}results_Nlim_rep2_FitSeq.csv',index_col=0)
Nlim3_fit = pd.read_csv(f'{fit_data_path}results_Nlim_rep3_FitSeq.csv',index_col=0)

plt.hist(switch3_fit.index,bins=100)
plt.xlim([-.2,.2])
plt.show()

plt.scatter(switch1_fit.index,switch3_fit.index)
plt.show()

#%%
def plot_3combs(data,low_thr):
    # Plots all three combinations of three replicates against each other, 
    # not including fitness less than low_thr
    
    # Create all pairwise combinations
    combinations = list(itertools.combinations(data.items(), 2))

    # Set up subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)

    # Generate all pairwise combinations
    for ax, ((name1, data1), (name2, data2)) in zip(axes, combinations):
        # Filter where both values are >= low threshold
        mask = (data1.index.values >= low_thr) & (data2.index.values >= low_thr)
        x = data1.index[mask]
        y = data2.index[mask]

        # Plot
        ax.scatter(x, y)

        # One-to-one line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
        
        # Calculate Pearson r
        r, p_value = pearsonr(x, y)

        # Labels and legend
        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_title(f'{name1} vs {name2}, r={r:.2f}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
        


#%% Plot all replicates:

low_thr = -0.5

nlims = {
    "Nlim1": Nlim1_fit,
    "Nlim2": Nlim2_fit,
    "Nlim3": Nlim3_fit,
}

clims = {
    "Clim1": Clim1_fit,
    "Clim2": Clim2_fit,
    "Clim3": Clim3_fit,
}

switch = {
    "switch1": switch1_fit,
    "switch2": switch2_fit,
    "switch3": switch3_fit,
}

plot_3combs(nlims,low_thr)
plot_3combs(clims,low_thr)
plot_3combs(switch,low_thr)

#%% Most correlated fitnesses: Clim 1/3, Nlim 2/3, Switch 1/2
# Average only these pairs, then make a data frame including those averages:

Clim_ave = np.mean([Clim1_fit.index, Clim3_fit.index],0)
Nlim_ave = np.mean([Nlim2_fit.index, Nlim3_fit.index],0)
switch_ave = np.mean([switch1_fit.index, switch2_fit.index],0)

ave_static = (Clim_ave+Nlim_ave)/2
fit_diff = np.abs(Clim_ave-Nlim_ave)
nonadd = np.abs(switch_ave-ave_static)

nonadd_df = pd.DataFrame({'Clim_ave':Clim_ave.tolist(),
                          'Nlim_ave':Nlim_ave.tolist(),
                          'switch_ave':switch_ave.tolist(),
                          'fit_diff':fit_diff.tolist(),
                          'nonadd':nonadd.tolist()})

# More strict low threshold to exclude anything off:
low_thr = -0.3
mask = (nonadd_df['Clim_ave'].values >= low_thr) & \
       (nonadd_df['Nlim_ave'].values >= low_thr) & \
       (nonadd_df['switch_ave'].values >= low_thr)
       
# Plot C vs. N fitnesses:       
x = nonadd_df['Clim_ave'][mask]
y = nonadd_df['Nlim_ave'][mask]
r, p_value = pearsonr(x, y)
plt.scatter(x,y,alpha=0.1)
plt.xlabel('Clim ave')
plt.ylabel('Nlim ave')
plt.title(f'r={r:.2f}')
plt.show()

# Plot average of C,N fitnesses vs. switch fitness:     
x = (nonadd_df['Clim_ave'][mask]+nonadd_df['Nlim_ave'][mask])/2
y = nonadd_df['switch_ave'][mask]
r, p_value = pearsonr(x, y)
plt.scatter(x,y,alpha=0.1)
plt.xlabel('static ave')
plt.ylabel('switch ave')
plt.title(f'r={r:.2f}')
plt.show()

# Plot C/N fitness difference vs. non-additivity (switch-C/N ave difference):     
x = nonadd_df['fit_diff'][mask]
y = nonadd_df['nonadd'][mask]
r, p_value = pearsonr(x, y)
plt.scatter(x,y,alpha=0.1)
plt.xlabel('fit diff')
plt.ylabel('nonadditivity')
plt.title(f'r={r:.2f}')
plt.show()

