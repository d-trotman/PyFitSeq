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
        


#%%

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


