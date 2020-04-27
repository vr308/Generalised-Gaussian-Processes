#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:24:25 2020

@author: vr308
"""

from matplotlib import pyplot as plt
import sys
import csv
import numpy as np
import gpflow
import os
import pandas as pd


df = pd.read_csv(os.getcwd() + '/Data/banana.csv', delimiter=',')

Xtrain = np.array(df[['V1', 'V2']])
Ytrain = np.array(df['Class'])


def gridParams():
    mins = [-3.25,-2.85 ]
    maxs = [ 3.65, 3.4 ]
    nGrid = 50
    xspaced = np.linspace( mins[0], maxs[0], nGrid )
    yspaced = np.linspace( mins[1], maxs[1], nGrid )
    xx, yy = np.meshgrid( xspaced, yspaced )
    Xplot = np.vstack((xx.flatten(),yy.flatten())).T
    return mins, maxs, xx, yy, Xplot

def plot(m, ax):
    col1 = '#0172B2'
    col2 = '#CC6600'
    mins, maxs, xx, yy, Xplot = gridParams()
    p = m.predict_y(Xplot)[0]
    ax.plot(Xtrain[:,0][Ytrain[:,0]==1], Xtrain[:,1][Ytrain[:,0]==1], 'o', color=col1, mew=0, alpha=0.5)
    ax.plot(Xtrain[:,0][Ytrain[:,0]==0], Xtrain[:,1][Ytrain[:,0]==0], 'o', color=col2, mew=0, alpha=0.5)
    if hasattr(m, 'Z'):
        Z = m.Z.read_value()
        ax.plot(Z[:,0], Z[:,1], 'ko', mew=0, ms=4)
        ax.set_title('m={}'.format(Z.shape[0]))
    else:
        ax.set_title('full')
    ax.contour(xx, yy, p.reshape(*xx.shape), [0.5], colors='k', linewidths=1.8, zorder=100)

# Setup the experiment and plotting.
Ms = [4, 8, 16, 32, 64]

# Run sparse classification with increasing number of inducing points
models = []
for index, num_inducing in enumerate(Ms):
    # kmeans for selecting Z
    from scipy.cluster.vq import kmeans
    Z = kmeans(Xtrain, num_inducing)[0]

    m = gpflow.models.VGP( (Xtrain[0:50], Ytrain[0:50]), kernel=gpflow.kernels.RBF(2),
        likelihood=gpflow.likelihoods.Bernoulli())
    
    # Initially fix the hyperparameters.
    m.feature.set_trainable(False)
    gpflow.train.ScipyOptimizer().minimize(m, maxiter=20)

    # Unfix the hyperparameters.
    m.feature.set_trainable(True)
    gpflow.train.ScipyOptimizer(options=dict(maxiter=200)).minimize(m)
    models.append(m)