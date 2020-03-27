#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:31:06 2020

@author: vidhi

1d and 2d GPC examples ML-II / HMC / VI
"""


import pymc3 as pm
import theano.tensor as tt
import numpy as np
from sampled import sampled
import sys
import random
import matplotlib.pylab as plt
from pymc3.gp.util import plot_gp_dist
from pymc3.gp.util import cholesky
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# link function
def invlogit(x, eps=sys.float_info.epsilon):
      return (1.0 + 2.0 * eps) / (1.0 + np.exp(-x)) + eps
  
def generate_gp_training(X_all, f_all, n_train, uniform, seed):
    
    np.random.seed(seed)
    
    if uniform == True:
             X = np.random.choice(X_all.ravel(), n_train, replace=False)
    else:
             pdf = 0.5*st.norm.pdf(X_all, 1, 0.5) + 0.3*st.norm.pdf(X_all, 4.5, 1)
             prob = pdf/np.sum(pdf)
             X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())

    train_index = []
    for i in X:
          train_index.append(X_all.ravel().tolist().index(i))
    X = X_all[train_index]
    f = f_all[train_index]
    y = pm.Bernoulli.dist(p=invlogit(f)).random()
    return X, y, f, train_index
    
if __name__== "__main__":

    seed = 4321
    np.random.seed(seed)
    
    # number of data points
    n = 200

    # x locations
    X_all = np.linspace(0, 5, n)

    # true hypers
    l_true = 0.5
    n_true = 2.0
    cov_func = n_true**2 * pm.gp.cov.ExpQuad(1, l_true)
    K = cov_func(X_all[:,None]).eval()

    # zero mean function
    mean = np.zeros(n)

    # sample from the gp prior - true function
    f_all = np.random.multivariate_normal(mean, K + 1e-6 * np.eye(n), 1).flatten()

    # Binary observations for classification 
    y = pm.Bernoulli.dist(p=invlogit(f_all)).random()
    
    fig = plt.figure(figsize=(12,5))
    ax = fig.gca()
    ax.plot(X_all, invlogit(f_all), 'dodgerblue', lw=2, label="True rate")
    ax.plot(X_all, y, 'ko', ms=3, label="Observed data")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    plt.legend()
        
    # Create training data sets of increasing size
    
    X_20, y_20, f_20, id_20 = generate_gp_training(X_all, f_all, 20, uniform=True, seed=seed)
    X_40, y_40, f_40, id_40 = generate_gp_training(X_all, f_all, 40, uniform=True, seed=seed)
    X_80, y_80, f_80, id_80 = generate_gp_training(X_all, f_all, 80, uniform=True, seed=seed)
    X_100, y_100, f_100, id_100 = generate_gp_training(X_all, f_all, 100, uniform=True, seed=seed)
    
    # ML-II
    
    
    
    # HMC Model 
    with pm.Model() as model:
    
            # covariance function
            log_l = pm.Normal('log_l', mu=0, sigma=3)
            log_s = pm.Normal('log_s', mu=0, sigma=3)
         
            l = pm.Deterministic('l', pm.math.exp(log_l))
            s = pm.Deterministic('s', pm.math.exp(log_s))
           
            cov = s**2 * pm.gp.cov.ExpQuad(1, l)
    
            gp = pm.gp.Latent(cov_func=cov)
    
            # make gp prior
            f = gp.prior("f", X=X_40[:,None], reparameterize=True)
            
            # logit link and Bernoulli likelihood
            p = pm.Deterministic("p", pm.math.invlogit(f))
            y_ = pm.Bernoulli("y", p=p, observed=y_40)
    
    with model:
        
            step_theta = pm.step_methods.HamiltonianMC(vars=[log_l, log_s], path_length=2)
            step_latent = pm.step_methods.HamiltonianMC(vars=[f], path_length=10)
            trace_hmc_40 = pm.sample(draws=200, tune=400, chains=1, step=[step_theta, step_latent])
        
    with model:
        
            step_theta = pm.step_methods.NUTS(vars=[log_l, log_s])
            step_latent = pm.step_methods.NUTS(vars=[f])
            trace_nuts_100_split = pm.sample(draws=200, tune=400, chains=1, step=[step_theta, step_latent])
            trace_nuts_100_joint = pm.sample(draws=200, tune=400, chains=1)

            
    
    plt.figure(figsize=(10,4))
    plt.subplot(131)
    plt.plot(X_40, trace_nuts_100_split['p'].T, 'bo', markersize=1)
    plt.plot(X_all, invlogit(f_all), 'k', lw=2, label="True rate")
    plt.subplot(132)
    plt.plot(X_40, trace_nuts_100_joint['p'].T, 'ro', markersize=1)
    plt.plot(X_all, invlogit(f_all), 'k', lw=2, label="True rate")
    plt.subplot(133)
    plt.plot(X_40, trace_hmc_40['p'].T, 'go', markersize=2)

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(X_40, plt.mean(trace_hmc_40['p'], axis=0), 'bo', markersize=2)
    plt.plot(X_all, invlogit(f_all), 'k', lw=2, label="True rate")
    plt.subplot(122)
    plt.plot(X_40, plt.mean(trace_nuts_40['p'], axis=0), 'ro', markersize=1)
    plt.plot(X_all, invlogit(f_all), 'k', lw=2, label="True rate")
            
  
        
plt.figure()
plt.plot(X_100, trace_hmc_100['f'].T, 'bo', markersize=2)      



n_pred = 500
X_new = np.linspace(0, 5.0, n_pred)[:,None]

with model:
    f_pred = gp.conditional("f_pred", X_new)

with model:
    pred_samples = pm.sample_posterior_predictive(trace_hmc, vars=[f_pred], samples=1000)

pm.traceplot(trace, var_names=['l', 'n']);

# plot the results
fig = plt.figure(figsize=(12,5));
ax = fig.gca()

# plot the samples from the gp posterior with samples and shading
plot_gp_dist(ax, invlogit(trace["f"]), x, plot_samples=True, palette='Blues');
plot_gp_dist(ax, invlogit(pred_samples['f_pred']), X_new)

# plot the data (with some jitter) and the true latent function
plt.plot(x, invlogit(f_true), "k", lw=2, label="True f");
plt.plot(x, y, 'ok', ms=3, alpha=0.5, label="Observed data");

# axis labels and title
plt.xlabel("X");
plt.ylabel("True f(x)");
plt.title("HMC");
plt.legend();







# Testing the relationship between f_rotated_ and f

# f_test = np.empty(shape=(len(trace_hmc),len(x)))
# for i in np.arange(len(trace_hmc)):
#     print(i)
#     s_test = float(trace_hmc['s'][i])
#     l_test = float(trace_hmc['l'][i])
#     cov_test = s_test**2 * pm.gp.cov.ExpQuad(1, l_test)
#     K_test = cov_test(x[:,None]) + 1e-6 * np.eye(len(x))
   
#     v = trace_hmc['f_rotated_'][i]
#     f = cholesky(K_test).dot(v)
#     f_test[i] = f.eval()