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


if __name__== "__main__":

    np.random.seed(4321)
    
    # number of data points
    n = 200

    # x locations
    x = np.linspace(0, 5, n)

    # true covariance
    l_true = 0.5
    n_true = 2.0
    cov_func = n_true**2 * pm.gp.cov.ExpQuad(1, l_true)
    K = cov_func(x[:,None]).eval()

    # zero mean function
    mean = np.zeros(n)

    # sample from the gp prior
    f_true = np.random.multivariate_normal(mean, K + 1e-6 * np.eye(n), 1).flatten()

    # Binary observations for classification 
    y = pm.Bernoulli.dist(p=invlogit(f_true)).random()
    
    fig = plt.figure(figsize=(12,5))
    ax = fig.gca()
    ax.plot(x, invlogit(f_true), 'dodgerblue', lw=2, label="True rate")
    ax.plot(x, y, 'ko', ms=3, label="Observed data")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    plt.legend()
    
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
        f = gp.prior("f", X=x[:,None], reparameterize=True)
        
        # logit link and Bernoulli likelihood
        p = pm.Deterministic("p", pm.math.invlogit(f))
        y_ = pm.Bernoulli("y", p=p, observed=y)

        trace_hmc = pm.sample(draws=100, tune=200, chains=1)
        #trace_ess = pm.sample()


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