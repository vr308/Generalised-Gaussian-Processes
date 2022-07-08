#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis plot

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import RESULTS_DIR

plt.style.use('seaborn')

sgpr_hmc = [0.0936, 0.0924, 0.0913, 0.0912, 0.0907]
joint = [0.114,0.1006,0.0971,0.0965, 0.0964]

plt.figure(figsize=(6,4))
plt.subplot(121)
plt.plot(sgpr_hmc,'bo-', label='SGPR + HMC')
plt.plot(joint,'ro-', label='JointHMC')
plt.legend(fontsize='small')
plt.title('RMSE')
plt.xticks(ticks=[0,1,2,3, 4], labels=['M=100', 'M=200', 'M=300', 'M=400', 'M=500'], fontsize='small')

sgpr_hmc = [-0.9207, -0.935, -0.941, -0.952, -0.9557]
joint = [-0.899,-0.908,-0.912,-0.915, -0.9170]

plt.subplot(122)
plt.plot(sgpr_hmc,color='purple', marker='o', label='SGPR + HMC')
plt.plot(joint,color='magenta',marker='o', label='JointHMC')
plt.legend(fontsize='small')
plt.title('NLPD')
plt.xticks(ticks=[0,1,2,3, 4], labels=['M=100', 'M=200', 'M=300', 'M=400', 'M=500'], fontsize='small')
plt.ylim(-1, -0.85)