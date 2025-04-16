#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:08:20 2025

@author: ca3258
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
fit_data_path = '../../outputs/env_specific/'

switch1_fit = pd.read_csv(f'{fit_data_path}results_Switch_rep1_FitSeq.csv',index_col=0)
switch2_fit = pd.read_csv(f'{fit_data_path}results_Switch_rep2_FitSeq.csv',index_col=0)
switch3_fit = pd.read_csv(f'{fit_data_path}results_Switch_rep3_FitSeq.csv',index_col=0)

plt.hist(switch3_fit.index,bins=100)
plt.xlim([-.2,.2])
plt.show()

plt.scatter(switch1_fit.index,switch3_fit.index)
plt.show()