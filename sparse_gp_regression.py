#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308
"""

import pymc3 as pm
import theano.tensor as tt
import matplotlib.pylab as plt
import numpy as np
from pymc3.gp.util import plot_gp_dist

# set the seed
np.random.seed(4321)

def optimal_variational_q_u(theta):
    
    return

def marginalised_variational_q_u(trace):
    
    return 


n = 2000 # The number of data points
X = 10*np.sort(np.random.rand(n))[:,None]

# Define the true covariance function and its parameters
l_true = 1.0
n_true = 3.0
cov_func = n_true**2 * pm.gp.cov.Matern52(1, l_true)

# A mean function that is zero everywhere
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
# Note that we have to call `eval()` because PyMC3 built on top of Theano
f_true = np.random.multivariate_normal(mean_func(X).eval(),
                                       cov_func(X).eval() + 1e-8*np.eye(n), 1).flatten()

# The observed data is the latent function plus a small amount of IID Gaussian noise
# The standard deviation of the noise is `sigma`
sig_true = 2.0
y = f_true + sig_true * np.random.randn(n)

## Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(X, f_true, "dodgerblue", lw=3, label="True f");
ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data");
ax.set_xlabel("X"); ax.set_ylabel("The true f(x)"); 
plt.legend();

#usual
Xu_init = 10*np.random.rand(20)

with pm.Model() as model:
    l = pm.Gamma("l", alpha=2, beta=1)
    n = pm.HalfCauchy("n", beta=5)
    noise_sd = pm.HalfCauchy("noise_sd", beta=5)

    cov = n**2 * pm.gp.cov.Matern52(1, l)
    gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")
    
    Xu = pm.gp.util.kmeans_inducing_points(20, X)

    # initialize 20 inducing points with K-means
    y_ = np.exp(gp.marginal_likelihood("y", X=X, Xu=Xu, y=y, noise=noise_sd))
    
    trace = pm.sample(1000)
    
X_new = np.linspace(-1, 11, 200)[:,None]

# add the GP conditional to the model, given the new X values
with model:
    f_pred = gp.conditional("f_pred", X_new)

with model:
    pred_samples = pm.sample_posterior_predictive(trace, vars=[f_pred], samples=1000)
    

# plot the results
fig = plt.figure(figsize=(12,5)); 
ax = fig.gca()

# plot the samples from the gp posterior with samples and shading
plot_gp_dist(ax, pred_samples["f_pred"], X_new);

# plot the data and the true latent function
plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data");
plt.plot(X, f_true, "dodgerblue", lw=3, label="True f");
plt.plot(Xu, 10*np.ones(Xu.shape[0]), "cx", ms=10, label="Inducing point locations")

# axis labels and title
plt.xlabel("X"); 
plt.ylim([-13,13]);
plt.title("Posterior distribution over $f(x)$ at the observed values"); 
plt.legend();
    