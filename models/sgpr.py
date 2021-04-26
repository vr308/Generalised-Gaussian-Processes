#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGPR (Titsias)
"""

import gpytorch 
import torch
import numpy as np
from prettytable import PrettyTable
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

N = 1000  # Number of training observations

X = torch.randn(N) * 2 - 1  # X values
Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

plt.plot(X, Y, 'bx')

class SparseGPR(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, Z_init):
        
        """The sparse GP class for regression with the collapsed bound.
           q*(u) is implicit. 
        """
        super(SparseGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
   
            
      
    
    def train(self, likelihood, combine_terms=True):
        
        self.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        losses = []
        for i in range(1000):
          # Zero backprop gradients
          optimizer.zero_grad()
          # Get output from model
          output = self(X)
          # Calc loss and backprop derivatives
          loss = -mll(output, Y)
          losses.append(loss)
          loss.backward()
          if i%100 == 0:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, 1000, loss.item(),
                    self.base_covar_module.base_kernel.lengthscale.item(),
                    likelihood.noise.item()))
          optimizer.step()
          return model, losses, trace
    
    def optimization_trace(self):
        return;
        
    def optimal_q_u(self):
       return self(self.covar_module.inducing_points)
    
    def posterior_predictive(model, test_x):
        
        model.eval()
        likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_star = likelihood(model(test_x))
        return y_star
        
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
        

# Initial inducing points
Z_init = torch.randn(12)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SparseGPR(X, Y, likelihood, Z_init)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

test_x = torch.linspace(-8, 8, 1000)

losses, trace = model.train(likelihood, combine_terms=True)
y_pred = model.test(test_x)


plt.plot(X, Y, 'bx')
plt.plot(model.covar_module.inducing_points.detach(), [-1.5]*12, 'ro', label='optimised')
plt.plot(test_x, observed_pred.loc.detach(), 'r-')
plt.plot(test_x, mu_u.detach(), 'g-')

# q*(u) - mean
# Z = model.covar_module.inducing_points
# K_nm = model.base_covar_module(X,Z).evaluate()
# K_mm = model.covar_module._inducing_mat
# noise = likelihood.noise
# K_mn_K_nm = torch.matmul(K_nm.T,K_nm)
# W = (K_mm + K_mn_K_nm/noise).inverse()
# lhs = K_mm/noise
# rhs = torch.matmul(K_nm.T, Y)
# l1 = torch.matmul(lhs, W)
# final = torch.matmul(l1, rhs)

# K_star_m = model.base_covar_module(test_x, Z).evaluate()
# K_mm_inv = model.covar_module._inducing_mat.inverse()
# H = torch.matmul(K_star_m, K_mm_inv)
# mu_u = torch.matmul(H, final)

