#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Experiments with synthetic data to analyse ML-II failure modes
with sensitivity to training size, dimensions and input dist. 

=======
Created on Thu Jan 30 15:13:07 2020

@author: vidhi
>>>>>>> 1eab4badae1ecc2ddac7a7bbf4014f76b74afb8b
"""

import numpy as np
import pymc3 as pm
import scipy.stats as st
import matplotlib.pylab as plt
import theano.tensor as tt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck,ExpSineSquared as PER, WhiteKernel
from pathlib import Path
import datetime
import os
BASE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, RationalQuadratic as RQ, Matern, ExpSineSquared as PER, WhiteKernel

def generate_gp_latent(X_all, mean, cov, size):

    return np.random.multivariate_normal(mean(X_all).eval(), cov=cov(X_all, X_all).eval(), size=size)

def get_default_x_coords(dim):
    
    xmin = -5
    xmax = 5
    
    if dim == 1:
        return np.linspace(xmin, xmax, 400)[:,None]
    
    elif dim == 2:
          xx, yy = np.meshgrid(np.linspace(xmin, xmax, 20),
                                 np.linspace(xmin, xmax, 20))
          X_eval = np.vstack((xx.reshape(-1), 
                                yy.reshape(-1))).T
          return X_eval
    
    else:
        return np.random.uniform(xmin, xmax, size=(400, dim))
    

def generate_gp_training(n_train, dim, hypers, uniform, num_datasets):
    
    X_all = get_default_x_coords(dim)
    
    # A mean function that is zero everywhere
    mean = pm.gp.mean.Zero()
    
    cov = pm.gp.cov.Constant(hypers[0]**2)*pm.gp.cov.ExpQuad(dim, hypers[1])
    
    # sample a latent functions from a GP with known cov
    f_all = generate_gp_latent(X_all, mean, cov, size=num_datasets)
    
    grid_indices = np.arange(len(X_all))
    
    if uniform == True:
         index = np.random.choice(grid_indices, n_train, replace=False)
    else:
         pdf = 0.5*st.norm.pdf(grid_indices, 30, 30) + 0.5*st.norm.pdf(grid_indices, 300, 30)
         prob = pdf/np.sum(pdf)
         index = np.random.choice(grid_indices, n_train, replace=False,  p=prob.ravel())

    y_train = np.empty(shape=(num_datasets, n_train))
    X = X_all[index]

    for j in np.arange(len(f_all)):
        f = f_all[j][index]
        y = f + np.random.normal(0, scale=hypers[2], size=n_train)
        y_train[j] = y.ravel()
        
    return X, y_train

def single_run():
    results_dict = {'kernel': kernel, 'num_train' : input_size, 'dimension' : test_dimension,
                'top_1': top_1, 'top_3': top_3, 'top_5': top_5}
    with open(filename, "a") as fp:
        json.dump(results_dict, fp, indent=4)
        fp.writelines('\n')

def main(args: argparse.Namespace) -> None:
    
    np.random.seed(57)

    date_str = datetime.now().strftime('%b%d')
    save_dir =  BASE_PATH / "results" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / "ml_recovery.txt"
    filename = f"{results_path}_.json"
    
    if Path(filename).exists():
      overwrite = yes_or_no(f"{filename} will be overwritten. Proceed?")
      if overwrite:
          import os
          os.remove(filename)      
      else:
          return;

    for size, dim, noise, uni in product(input_size, dims, noise_sd, uniform):
        single_run(
                 size,
                 dim, 
                 noise,
                filename) 


if __name__ == "__main__":
    
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

    X_train = np.empty(shape=(100,n_train))
    y_train = np.empty(shape=(100,n_train))
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

    hypers = [sig_sd_true, lengthscale_true, noise_sd_true]

    snr = np.round(sig_sd_true**2/noise_sd_true**2)


    # Generating datasets
    num_sizes = [10, 15, 20, 25, 40, 60, 80, 100, 120]
    dimensions = [1, 2]
    noise_sd = ['fixed', 'learn']
    uniform = [True, False]
    
    cov = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(1, lengthscale_true)

    # This will change the shape of the function

    f_all = generate_gp_latent(X_all, mean, cov, size=100)

    # Data attributes

    noise_sd_true = np.sqrt(1)

    hyp = [sig_sd_true, lengthscale_true, noise_sd_true]

    snr = np.round(sig_sd_true**2/noise_sd_true**2)

    uniform = True

    # Generating datasets

    X_10, y_10 = generate_gp_training(X_all, f_all, 10, noise_sd_true, uniform)
    X_15, y_15 = generate_gp_training(X_all, f_all, 15, noise_sd_true, uniform)
    X_20, y_20 = generate_gp_training(X_all, f_all, 20, noise_sd_true, uniform)
    X_25, y_25 = generate_gp_training(X_all, f_all, 25, noise_sd_true, uniform)
    X_40, y_40 = generate_gp_training(X_all, f_all, 40, noise_sd_true, uniform)
    X_60, y_60 = generate_gp_training(X_all, f_all, 60, noise_sd_true, uniform)
    X_80, y_80 = generate_gp_training(X_all, f_all, 80, noise_sd_true, uniform)
    X_100, y_100 = generate_gp_training(X_all, f_all, 100, noise_sd_true, uniform)
    X_120, y_120 = generate_gp_training(X_all, f_all, 120, noise_sd_true, uniform)

    X = [X_10,  X_15, X_20, X_25, X_40, X_60, X_80, X_100, X_120]
    y = [y_10,  y_15, y_20, y_25, y_40, y_60, y_80, y_100, y_120]

    seq = [10,15,20,25, 40,60, 80, 100, 120]

    nd = len(seq)

    # Sanity check data
    plt.figure()
    plt.plot(X_all, f_all[10])
    plt.plot(X_120[10], y_120[10], 'bo')

    # 100 datasets of each of the 12 sizes

    s = np.empty(shape=(nd,100))
    ls = np.empty(shape=(nd,100))

    # sklearn - learning - unconstrained optim and fixed noise

    for j in np.arange(nd):

        print('Analysing data-sets of size ' + str(np.shape(X[j])[1]))

        for i in np.arange(100):

            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=1, noise_level_bounds="fixed")

            gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, n_restarts_optimizer=10).fit(X[j][i][:,None], y[j][i])

            ls[j][i] = gp.kernel_.k1.k2.length_scale
            s[j][i] = np.sqrt(gp.kernel_.k1.k1.constant_value)

     # sklearn - learning - unconstrained optim and noise not fixed

    ls_noise =  np.empty(shape=(nd,100))
    s_noise =  np.empty(shape=(nd,100))
    noise =  np.empty(shape=(nd,100))

    for j in np.arange(nd):

        print('Analysing data-sets of size ' + str(np.shape(X[j])[1]))

        for i in np.arange(100):

            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=1)

            gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, n_restarts_optimizer=10).fit(X[j][i][:,None], y[j][i])

            ls_noise[j][i] = gp.kernel_.k1.k2.length_scale
            s_noise[j][i] = np.sqrt(gp.kernel_.k1.k1.constant_value)
            noise[j][i] = np.sqrt(gp.kernel_.k2.noise_level)


    # Plotting
    ls_mean_nf = np.mean(ls, axis=1)
    ls_mean_n = np.mean(ls_noise, axis=1)
    plt.plot(seq, np.log(ls_mean_nf), 'bo-', label='Noise level fixed')
    plt.plot(seq, np.log(ls_mean_n), 'go-', label='Noise level estimated')
    plt.axhline(y=np.log(lengthscale_true), color='r', label='True lengthscale')
    plt.title("Learning the lenghtscale under ML-II" + '\n' + 'Avg. across 100 datasets per training set size', fontsize='small')
    plt.legend(fontsize='small')
    plt.xlabel('Training set size', fontsize='small')
    plt.ylabel('Avg. estimated log-lengthscale', fontsize='small')

    noise_means = np.mean(noise, axis=1)
    noise_std = np.std(noise, axis=1)
    plt.figure()
    plt.plot(seq, noise_means, 'o-', label='Noise level')
    #plt.errorbar(seq, noise_means, yerr=noise_std)
    plt.title("Learning the noise level under ML-II" + '\n' + 'Avg. across 100 datasets per training set size', fontsize='small')
    plt.legend(fontsize='small')
    plt.axhline(y=noise_sd_true, color='r', label='True lengthscale')
    plt.xlabel('Training set size', fontsize='small')
    plt.ylabel('Avg. estimated noise std', fontsize='small')


    #  Generating datasets as a function of noise var

    n_star = 500

    xmin = 0
    xmax = 10

    X_all = np.linspace(xmin, xmax,n_star)[:,None]

    # A mean function that is zero everywhere

    mean = pm.gp.mean.Zero()

    # Kernel Hyperparameters

    sig_sd_true = 10.0
    lengthscale_true = 2.0

    cov = pm.gp.cov.Constant(sig_sd_true**2)*pm.gp.cov.ExpQuad(1, lengthscale_true)

    # This will change the shape of the function

    f_all = generate_gp_latent(X_all, mean, cov, size=100)

    # Data attributes

    noise_sd_low = np.sqrt(1)
    noise_sd_med1 = np.sqrt(3)
    noise_sd_med2 = np.sqrt(6)
    noise_sd_high = np.sqrt(9)

    uniform = True

    X_nlow, y_nlow = generate_gp_training(X_all, f_all, 30, noise_sd_low, uniform)
    X_nmed1, y_nmed1 = generate_gp_training(X_all, f_all, 30, noise_sd_med1, uniform)
    X_nmed2, y_nmed2 = generate_gp_training(X_all, f_all, 30, noise_sd_med2, uniform)
    X_nhi, y_nhi = generate_gp_training(X_all, f_all, 30, noise_sd_high, uniform)

    X_n = [X_nlow, X_nmed1, X_nmed2, X_nhi]
    y_n = [y_nlow, y_nmed1, y_nmed2, y_nhi]

    # Sanity checking the data

    key = 14

    # Low noise
    plt.figure()
    plt.plot(X_all, f_all[key])
    plt.plot(X_nlow[key], y_nlow[key], 'bo')
    plt.plot(X_nmed1[key], y_nmed1[key], 'ro')
    plt.plot(X_nhi[key], y_nhi[key], 'go')

    seq_n = [1, 3, 6, 9]
    nd = len(seq_n)

    s_n = np.empty(shape=(nd,100))
    ls_n = np.empty(shape=(nd,100))

    # sklearn - learning - unconstrained optim and fixed noise

    for j in np.arange(nd):

        print('Analysing data-sets with noise var ' + str(seq_n[j]))

        for i in np.arange(100):

            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) \
            + WhiteKernel(noise_level=seq_n[j], noise_level_bounds="fixed")

            gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, n_restarts_optimizer=5).fit(X_n[j][i][:,None], y_n[j][i])
            ls_n[j][i] = gp.kernel_.k1.k2.length_scale
            s_n[j][i] = np.sqrt(gp.kernel_.k1.k1.constant_value)


    # sklearn - learning - unconstrained optim and noise est.

    s_nest = np.empty(shape=(nd,100))
    ls_nest = np.empty(shape=(nd,100))
    noise_est = np.empty(shape=(nd, 100))

    for j in np.arange(nd):

        print('Analysing data-sets with noise var ' + str(seq_n[j]))

        for i in np.arange(100):

            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) \
            + WhiteKernel(noise_level=seq_n[j])

            gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, n_restarts_optimizer=5).fit(X_n[j][i][:,None], y_n[j][i])
            ls_nest[j][i] = gp.kernel_.k1.k2.length_scale
            s_nest[j][i] = np.sqrt(gp.kernel_.k1.k1.constant_value)
            noise_est[j][i] = np.sqrt(gp.kernel_.k2.noise_level)

    # Plotting
    ls_mean_noise = np.mean(ls_n, axis=1)
    noise_est_mean = np.mean(noise_est, axis=1)
    ls_mean_nest = np.mean(ls_nest, axis=1)
    plt.plot(seq_n, np.log(ls_mean_nest), 'go-')
    #plt.plot(seq_n, np.log(ls_mean_nest), color='cyan', marker='o', label='Noise level estimated')
    plt.axhline(y=np.log(lengthscale_true), color='r', label='True lengthscale')
    plt.title("Learning the lengthscale under ML-II" + '\n' + 'Avg. across 100 datasets per noise level', fontsize='small')
    plt.xticks(seq_n)
    plt.legend(fontsize='small')
    plt.xlabel('Noise var ' + r'$\sigma^{2}$', fontsize='small')
    plt.ylabel('Avg. estimated log-lengthscale', fontsize='small')

    # LML Surface as a function of training set size

    # GP fit on 10 data points

    X_trial = X_10[16]
    y_trial = y_10[16]

    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) \
            + WhiteKernel(noise_level=1,noise_level_bounds="fixed")

    gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, n_restarts_optimizer=5).fit(X_trial[:,None], y_trial)
    print(gp.kernel_)

    #gp.log_marginal_likelihood((-2.302, -2.302))

    theta0 = np.logspace(-1, 4, 50)
    theta1 = np.logspace(-1, 4, 50)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    #theta = np.log([Theta0[i,j], Theta1[i, j]])
    LML = [[gp.log_marginal_likelihood(np.log([Theta0[i,j], Theta1[i, j]]))
            for i in range(50)] for j in range(50)]
    LML = np.array(LML).T
    vmin, vmax = (-LML).min(), (-LML).max()
    vmax = 100
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=4)
    plt.figure(figsize=(8,4))
    plt.subplot(122)
    plt.contourf(Theta0, Theta1, -LML,
                levels=level, alpha=1, extend='both')
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
    #plt.colorbar()

    # LML Surface as a function of increasing noise

    X_trial = X_nlow[7]
    y_trial = y_nlow[7]

    kernel = Ck(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) \
            + WhiteKernel(noise_level=9)

    gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, n_restarts_optimizer=5).fit(X_trial[:,None], y_trial)
    print(gp.kernel_)

    #gp.log_marginal_likelihood((-2.302, -2.302))

    theta0 = np.logspace(-1, 5, 50)
    theta1 = np.logspace(-1, 5, 50)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    #theta = np.log([Theta0[i,j], Theta1[i, j]])
    LML = [[gp.log_marginal_likelihood(np.log([ gp.kernel_.k1.k1.constant_value ,Theta0[i,j], Theta1[i, j]]))
            for i in range(50)] for j in range(50)]
    LML = np.array(LML).T
    vmin, vmax = (-LML).min(), (-LML).max()
    vmax = 100
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 100), decimals=4)
    #plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.contourf(Theta0, Theta1, -LML,
                levels=level, alpha=1, extend='both')
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(y=1, color='r')
    plt.axvline(x=lengthscale_true, color='r')
    plt.scatter(gp.kernel_.k1.k2.length_scale, gp.kernel_.k2.noise_level,marker='x', color='red')
    plt.ylabel("Noise var", fontsize='x-small')
    plt.xlabel("Length-scale", fontsize='x-small')
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')
    plt.title("Negative Log-marginal-likelihood (low noise)", fontsize='x-small')
    #plt.colorbar()

















