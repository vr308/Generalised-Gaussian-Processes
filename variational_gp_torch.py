#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:19:00 2020

@author: vr308
"""

import torch
import gpytorch
import math
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_swiss_roll, make_moons
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from gpytorch.variational import VariationalStrategy

default_seed = 4321

# Create dataset 
X, y = make_moons(noise=0.3, random_state=0)
Xt = torch.from_numpy(X).float()
yt = torch.from_numpy(y).float()

# Create evalutaion grid
h = 0.05
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
X_eval = np.vstack((xx.reshape(-1), 
                    yy.reshape(-1))).T
X_eval = torch.from_numpy(X_eval).float()

class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=True
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


# Initialize model and likelihood
model = GPClassificationModel(Xt)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
training_iterations = 300
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the number of training datapoints
mll = gpytorch.mlls.VariationalELBO(likelihood, 
                                    model, 
                                    10, 
                                    combine_terms=False)

for i in range(training_iterations):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(Xt)
    # Calc loss and backprop gradients
    log_lik, kl_div, log_prior = mll(output, yt)
    loss = -(log_lik - kl_div + log_prior)
    #loss = -mll(output, yt)
    loss.backward()
    
    print('Iter %d/%d - Loss: %.3f lengthscale: %.3f outputscale: %.3f' % (
        i + 1, training_iterations, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.covar_module.outputscale.item() # There is no noise in the Bernoulli likelihood
    ))
    
    optimizer.step()

# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():    
    # Get classification predictions
    observed_pred = likelihood(model(X_eval))

    p = observed_pred.mean.numpy()
    Z_gpy = p.reshape(xx.shape)
    

# Initialize fig and axes for plot
f  = plt.figure( figsize=(10, 10))
ax = f.add_subplot(1,1,1)
ax.contourf(xx,yy,Z_gpy, levels=20)
ax.scatter(X[y == 0,0], X[y == 0,1])
ax.scatter(X[y == 1,0], X[y == 1,1])
ax.set_title('Variational ELBO')
plt.show()