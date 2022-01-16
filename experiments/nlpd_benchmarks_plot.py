#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarks nlpd plot

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import RESULTS_DIR

filename = 'benchmarks_nlpd.csv'
df = pd.read_csv(filename, sep=',')
fig = plt.figure(figsize=(16,4))
_ax = fig.add_subplot(111, frame_on=False)
_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
colors = ['b', 'r', 'g', 'orange', 'steelblue','magenta']
model_cols = ['svgp_', 'sgpr_', 'sgpr_hmc_']
models = ['SVGP', 'SGPR', 'SGPR + HMC']
#models = df.columns[3:]
for _ in range(5): # iterate over subplots
    n_plot = _ + 1
    ax = fig.add_subplot(1,5,n_plot) 
    for m in range(3): # iterate over models within each subplot
        mean = df[[col for col in df if col.startswith(model_cols[m])][0]][_]
        se = df[[col for col in df if col.startswith(model_cols[m])][1]][_]
        ax.errorbar(x=mean, y=m, xerr=2*se, c=colors[m], marker='o', barsabove=True, capsize=4)
    ax.set_title(df['dataset_name'][_], fontsize='small')
    if (n_plot == 1 or n_plot == 5):
        ax.set_yticks(ticks=np.arange(6))
        ax.set_yticklabels(labels=models)
    elif (n_plot == 4 or n_plot == 8):
        ax.set_yticks(ticks=np.arange(6))
        ax.set_yticklabels(labels=models)
        ax.yaxis.set_ticks_position('right')
    else:
       ax.set_yticks(ticks=np.arange(6))
       ax.tick_params(axis='y', labelcolor="none", left=False)
       ax.grid(True)
        
plt.suptitle('Neg. log predictive density across benchmarks for UCI Regression datasets', fontsize='small')
_ax.set_xlabel('Test NLPD', fontsize='small')
plt.savefig(RESULTS_DIR/'nlpd_uci.png')