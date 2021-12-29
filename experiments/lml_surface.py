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

def generate_gp_latent(X_all, mean, cov, size):

    return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval(), size=size)

def generate_gp_training(X_all, f_all, n_train, noise_sd, uniform):

    if uniform == True:
         X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
         print('in here')
         pdf = 0.5*st.norm.pdf(X_all, 2, 1) + 0.5*st.norm.pdf(X_all, 8, 1)
         prob = pdf/np.sum(pdf)
         X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())

    train_index = []
    for i in X:
        train_index.append(X_all.ravel().tolist().index(i))

    X_train = np.empty(shape=(len(f_all), n_train))
    y_train = np.empty(shape=(len(f_all), n_train))
    for j in np.arange(len(f_all)):
        X = X_all[train_index]
        f = f_all[j][train_index]
        y = f + np.random.normal(0, scale=noise_sd, size=n_train)
        X_train[j] = X.ravel()
        y_train[j] = y.ravel()
    return X_train, y_train

if __name__ == "__main__":
    
    n_star = 500
    xmin = 0
    xmax = 10

    X_all = np.linspace(xmin, xmax,n_star)[:,None]

    # A mean function that is zero everywhere

    mean = pm.gp.mean.Zero()

    # Kernel Hyperparameters

    sig_sd_true = 5.0
    lengthscale_true = 2.0
    noise_sd_true = np.sqrt(1)
    uniform = True


    cov = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(1, lengthscale_true)

    # This will change the shape of the function

    f_all = np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval(), size=1)
    
    X_10, y_10 = generate_gp_training(X_all, f_all, 10, noise_sd_true, uniform)
    X_15, y_15 = generate_gp_training(X_all, f_all, 15, noise_sd_true, uniform)
    X_20, y_20 = generate_gp_training(X_all, f_all, 10, noise_sd_true, uniform)

    # Data attributes

    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]

    snr = np.round(sig_sd_true**2/noise_sd_true**2)
    
    # LML Surface as a function of training set size

    # GP fit on 10 data points

    X = X_20.ravel()
    y = y_20.ravel()

    kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e4)) \
            + WhiteKernel(noise_level=1,noise_level_bounds="fixed")

    gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, n_restarts_optimizer=5).fit(X[:,None], y)
    print(gp.kernel_)

    #gp.log_marginal_likelihood((-2.302, -2.302))

    theta0 = np.logspace(-1, 1.5, 60)
    theta1 = np.logspace(-1, 1.2, 60)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    #theta = np.log([Theta0[i,j], Theta1[i, j]])
    LML = [[gp.log_marginal_likelihood(np.log([Theta0[i,j], Theta1[i, j]]))
            for i in range(60)] for j in range(60)]
    LML = np.round(np.array(LML).T,2)
    vmin, vmax = (-LML).min(), (-LML).max()
    vmax = 100
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=2)
    plt.figure(figsize=(8,4))
    plt.contourf(Theta0, Theta1, -LML,
                levels=level, alpha=1, extend='min')
    C = plt.contour(Theta0, Theta1, -LML, levels=level, colors='black', linewidth=.2)
    plt.clabel(C, levels=level, fontsize=10)
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(y=lengthscale_true, color='r')
    plt.axvline(x=sig_sd_true**2, color='r')
    plt.scatter(gp.kernel_.k1.k1.constant_value, gp.kernel_.k1.k2.length_scale, marker='x', color='red')
    plt.ylabel("Length-scale", fontsize='x-small')
    plt.xlabel("Sig var", fontsize='x-small')
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')
    plt.title("Negative Log-marginal-likelihood", fontsize='x-small')


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
    

