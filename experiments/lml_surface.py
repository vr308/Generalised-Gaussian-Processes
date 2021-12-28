#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LML surface with a ridge / non-identifiability / high aleatoric uncertainty

"""
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from matplotlib import cm
import time as time
import matplotlib.pylab as plt
from theano.tensor.nlinalg import matrix_inverse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel
from matplotlib.colors import LogNorm
import seaborn as sns
from theano.tensor.nlinalg import matrix_inverse
import csv
import scipy.stats as st

if __name__ == "__main__":


    rng = np.random.RandomState(0)
    X_all = rng.uniform(0, 10, 40)[:, np.newaxis]
    
    X_small = np.random.choice(X_all.ravel(), replace=False, size=20)
    X_large = np.random.choice(X_all.ravel(), replace=False, size=30)
    
    X = X_small[:,None]
    
    f = np.exp(np.cos((0.4 - X[:,0])))
    y = f + rng.normal(0, 0.5, X.shape[0])
    
    X_star = np.linspace(0, 10, 100)
    f_star = np.exp(np.cos((0.4 - X_star)))
    
    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e4)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    gp1 = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(X, y)
    
    y_mean, y_cov = gp1.predict(X_star[:, np.newaxis], return_cov=True)
    plt.plot(X_star, y_mean, 'r', lw=1, zorder=9)
    plt.fill_between(X_star, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.2, color='k')
    plt.plot(X_star, f_star, 'k', lw=2, zorder=9)
    plt.scatter(X[:, 0], y, c='k', s=10)
    #plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
    #          % (kernel, gp.kernel_,
    #             gp.log_marginal_likelihood(gp.kernel_.theta)))
    lml = np.round(gp1.log_marginal_likelihood(gp1.kernel_.theta), 3)
    plt.title('LML:' + str(lml), fontsize='small')
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    plt.xlabel('X',fontsize='small')
    plt.ylabel('y',fontsize='small')
    plt.tight_layout()
    

