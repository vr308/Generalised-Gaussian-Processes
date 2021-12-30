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
    cov_per = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.Periodic(1, period=2, ls=2)#*pm.gp.cov.ExpQuad(1, 1)


    # This will change the shape of the function

    f_all = np.random.multivariate_normal(mean(X_all).eval(), cov=cov_per(X_all, X_all).eval(), size=1)
    
    X_10, y_10 = generate_gp_training(X_all, f_all, 20, noise_sd_true, uniform)
    X_15, y_15 = generate_gp_training(X_all, f_all, 50, noise_sd_true, uniform)
    X_40, y_40 = generate_gp_training(X_all, f_all, 90, noise_sd_true, uniform)
    
    X_train = [X_10, X_15, X_40]
    y_train = [y_10, y_15, y_40]
    sizes = [20,50,90]

    # Data attributes

    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]

    snr = np.round(sig_sd_true**2/noise_sd_true**2)
    
    # LML Surface as a function of training set size
    
    plt.figure(figsize=(8,4))
    # GP fit on 10 data points
    
    for i in [0,1,2]:
        
        X = X_train[i].ravel()
        y = y_train[i].ravel()
        
        plt.subplot(1,3,i+1)
    
        kernel = 100 * PER(length_scale=1, length_scale_bounds=(1e-2, 1e4), periodicity=1.0) \
                + WhiteKernel(noise_level=1,noise_level_bounds="fixed")
    
        gp = GaussianProcessRegressor(kernel=kernel,
                                          alpha=0.0, n_restarts_optimizer=5).fit(X[:,None], y)
        print(gp.kernel_)
    
        #gp.log_marginal_likelihood((-2.302, -2.302))
    
        theta0 = np.logspace(-2, 2.0, 100)
        theta1 = np.logspace(-1, 2, 100)
        Theta0, Theta1 = np.meshgrid(theta0, theta1)
        #theta = np.log([Theta0[i,j], Theta1[i, j]])
        LML = [[gp.log_marginal_likelihood(np.log([25.0, Theta0[i,j], Theta1[i, j]]))
                for i in range(100)] for j in range(100)]
        LML = np.round(np.array(LML).T,2)
        vmin, vmax = (-LML).min(), (-LML).max()
        vmax = 100
        level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=2)
        #level = np.arange(22.70,23,0.01)
        #plt.figure(figsize=(8,4))
        plt.contourf(Theta0, Theta1, -LML,
                    levels=level, alpha=1, extend='both', nchunk=0)
        C = plt.contour(Theta0, Theta1, -LML, levels=level[::10], colors='black', linewidth=.05, alpha=0.5, nchunk=0)
        plt.clabel(C, fontsize=5)
        id_flat = np.where(-LML < level[1])
        #plt.scatter(Theta0[0][id_flat[1]], Theta1[:,0][id_flat[0]], marker='x', color='red', s=1)
    
        #lml_grid_flat = np.repeat(-LML[id_flat], repeats=len(id_flat[0])).reshape(len(id_flat[0]),len(id_flat[0]))
        #plt.contourf(Theta0[0][id_flat[1]], Theta1[:,0][id_flat[0]], lml_grid_flat, levels=np.unique(lml_grid_flat), colors='black', linewidth=.2, nchunk=0)
        plt.xscale("log")
        plt.yscale("log")
        #plt.axhline(y=lengthscale_true, color='r')
        #plt.axvline(x=sig_sd_true**2, color='r')
        #plt.scatter(sig_sd_true**2, lengthscale_true, color='r', marker='x')
        #plt.scatter(gp.kernel_.k1.k1.constant_value, gp.kernel_.k1.k2.length_scale, marker='x', color='blue')
        plt.xlabel("Lengthscale", fontsize='x-small')
        plt.ylabel("Periodicity", fontsize='x-small')
        plt.xticks(fontsize='small')
        plt.yticks(fontsize='small')
        plt.title('Train size = ' + str(sizes[i]), fontsize='small')
    
    plt.suptitle("Neg. log marginal likelihood (Sig var vs. Lengthscale)" + '\n' + 'Model Mismatch', fontsize='small')


