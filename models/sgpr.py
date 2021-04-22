#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:35:19 2021

@author: vidhi

SGPR (Titsias)
"""

import gpytorch 
import torch
import numpy as np
from prettytable import PrettyTable
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

N = 10000  # Number of training observations

X = torch.randn(N) * 2 - 1  # X values
Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

class SparseGPR(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        
        """The sparse GP class for regression with the collapsed bound.
        """
        super(SparseGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def get_trainable_param_names(self):
        
        ''' Prints a list of parameters (model + variational) which will be 
        learnt in the process of optimising the objective '''
        
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SparseGPR(X, Y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(1000):
  # Zero backprop gradients
  optimizer.zero_grad()
  # Get output from model
  output = model(X)
  # Calc loss and backprop derivatives
  loss = -mll(output, Y)
  loss.backward()
  #print('Iter %d/%d - Loss: %.3f' % (i + 1, 1000, loss.item()))
  print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, 1000, loss.item(),
       model.base_covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()))
  optimizer.step()
