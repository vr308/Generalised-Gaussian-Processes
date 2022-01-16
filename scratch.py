#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

# set the seed
np.random.seed(1)

n = 2000  # The number of data points
X = 10 * np.sort(np.random.rand(n))[:, None]

# Define the true covariance function and its parameters
ℓ_true = 1.0
η_true = 3.0
cov_func = η_true ** 2 * pm.gp.cov.Matern52(1, ℓ_true)

# A mean function that is zero everywhere
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
# Note that we have to call `eval()` because PyMC3 built on top of Theano
f_true = np.random.multivariate_normal(
    mean_func(X).eval(), cov_func(X).eval() + 1e-8 * np.eye(n), 1
).flatten()

# The observed data is the latent function plus a small amount of IID Gaussian noise
# The standard deviation of the noise is `sigma`
σ_true = 2.0
y = f_true + σ_true * np.random.randn(n)

## Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()
ax.plot(X, f_true, "dodgerblue", lw=3, label="True f")
ax.plot(X, y, "ok", ms=3, alpha=0.5, label="Data")
ax.set_xlabel("X")
ax.set_ylabel("The true f(x)")
plt.legend();

Xu_init = 10 * np.random.rand(20)

with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=2, beta=1)
    η = pm.HalfCauchy("η", beta=5)

    cov = η ** 2 * pm.gp.cov.Matern52(1, ℓ)
    gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")

    # initialize 20 inducing points with K-means
    # gp.util
    Xu = pm.Flat("Xu", shape=20, testval=Xu_init)

    σ = pm.HalfCauchy("σ", beta=5)
    y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, y=y, noise=σ)

    trace = pm.sample(tune=200, draws=100, chains=1)

# add the GP conditional to the model, given the new X values
with model:
    f_pred = gp.conditional("f_pred", X_new)

# To use the MAP values, you can just replace the trace with a length-1 list with `mp`
with model:
    pred_samples = pm.sample_posterior_predictive(trace, vars=[f_pred], samples=1000)














import math
import torch
import gpytorch
import numpy as np
from typing import Dict, List, Type
from matplotlib import pyplot as plt
import utils.dataset as uci_datasets
from utils.dataset import Dataset
from utils.metrics import *

class ExperimentName:
    def __init__(self, base):
        self.s = base

    def add(self, name, value):
        self.s += f"_{name}-{value}"
        return self

    def get(self):
        return self.s
    
def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)

dataset = get_dataset_class('Boston')(split=8, prop=0.9)
X_train, Y_train, X_test, Y_test = dataset.X_train, dataset.Y_train, dataset.X_test, dataset.Y_test
   
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))    
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train, Y_train.flatten(), likelihood)

model.train()
likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

   #grad_params = get_trainable_param_names(self)

   #trace_states = []
losses = []
num_steps = 2000
for j in range(num_steps):
      optimizer.zero_grad()
      output = model.forward(X_train)
      loss = -mll(output, Y_train).sum()
      losses.append(loss.item())
      loss.backward()
      if j%1000 == 0:
                print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %s   noise: %.3f ' % (
                j + 1, num_steps, loss.item(),
                model.covar_module.outputscale.item(),
                model.covar_module.base_kernel.lengthscale,
                model.likelihood.noise.item()))
      optimizer.step()


model.eval()
model.likelihood.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad():
    y_star = model.likelihood(model(X_test))   

print(np.round(rmse(y_star.loc, Y_test, dataset.Y_std).item(), 4)) 
print(np.round(nlpd(y_star, Y_test, dataset.Y_std).item(), 4)) 

# for i in np.arange(5):
#     for param_name, param in model.named_parameters():
#          model = ExactGPModel(train_x, train_y, likelihood)
#          print(f'Parameter name: {param_name:42} value = {param.item()}')