#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampler bar plot

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import RESULTS_DIR

filename = 'sampler_runtimes.csv'
df = pd.read_csv(filename, sep=',')

dataset_name = ['Boston', 'Concrete', 'Energy', 'WineRed', 'Yacht']
sgpr_hmc = np.array(df[df['Model'] == 'SGPR + HMC'][dataset_name])
joint_hmc = np.array(df[df['Model'] == 'JointHMC'][dataset_name])
gpr_hmc = np.array(df[df['Model'] == 'GPR + HMC'][dataset_name])

sgpr_hmc_se = np.array(df[[col for col in df if col.endswith('se')]])[0]
joint_hmc_se = np.array(df[[col for col in df if col.endswith('se')]])[1]
gpr_hmc_se = np.array(df[[col for col in df if col.endswith('se')]])[2]

x = np.arange(len(dataset_name))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/3, gpr_hmc.squeeze(), width, yerr=gpr_hmc_se, label='GPR + HMC', color='orange',capsize=5, ecolor='black')
rects2 = ax.bar(x + width/2, joint_hmc.squeeze(), width, yerr=joint_hmc_se, label='JointHMC', color='steelblue')
rects3 = ax.bar(x + width, sgpr_hmc.squeeze(), width, yerr=sgpr_hmc_se, label='SGPR + HMC', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Total sampling seconds')
ax.set_title('Wall clock seconds for total MCMC samples')
plt.xticks(x, dataset_name)
ax.legend()
ax.set_yscale('log')
fig.tight_layout()
plt.savefig(RESULTS_DIR/'sampling_seconds.png')

