#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

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