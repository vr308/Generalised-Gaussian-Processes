#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVGP Model gpytorch

"""

import gpytorch as gpytorch
import torch as torch
import numpy as np
import tqdm
import math
import urllib.request
from math import floor
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

class StochasticVariationalGP(ApproximateGP):
    
    """The sparse GP class for regression with the uncollapsed stochastic bound.
         The parameters of q(u) \sim N(m, S) are learnt explicitly. 
    """
      
    def __init__(self, inducing_points): 
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':

X = torch.linspace(-5,5,1000)[:,None]
y = torch.sin(X).flatten() + 0.4*torch.randn(len(X))

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()
    
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

inducing_points = train_x[:100, :]
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

num_epochs = 100
losses = []
epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    print('Finished 1 loop')
    minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=True)
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        losses.append(loss)
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

model.eval()
likelihood.eval()
means = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in train_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])
means = means[1:]

plt.figure()
plt.plot(train_x, means, 'r-')
plt.plot(train_x, train_y, 'bo')
