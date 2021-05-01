#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGPR (Titsias)

"""

#TODO: Collect optimisation trace

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


class SparseGPR(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, Z_init):
        
        """The sparse GP class for regression with the collapsed bound.
           q*(u) is implicit. 
        """
        super(SparseGPR, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.inducin_points = Z_init
        self.num_inducing = len(Z_init)
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=likelihood)

    def forward(self, x): # returns ?
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def elbo(self, output, y):
        
        Knn = model.base_covar_module(self.train_x).evaluate()
        Knm = model.base_covar_module(self.train_x, Z_init).evaluate()
        lhs = torch.matmul(Knm, self.covar_module._inducing_mat.inverse())
        Qnn = torch.matmul(lhs, Knm.evaluate().T)
        p_y = model.likelihood(output)
        expected_log_lik = p_y.log_prob(y)
        shape = Knn.shape[:-1]
        noise_diag = self.likelihood._shaped_noise_covar(shape).diag()
        diag = Knn.diag() - Qnn.diag()
        trace_term = 0.5*(diag/noise_diag).sum() 
    
        return expected_log_lik, trace_term
            
    def train_model(self, likelihood, combine_terms=True):
        
        self.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        losses = []
        for i in range(1000):
          optimizer.zero_grad()
          output = self(self.train_x)
          if combine_terms:
              loss = -mll(output, self.train_y)
          else:
              loss = -self.elbo(output, self.train_y)
          losses.append(loss)
          loss.backward()
          if i%100 == 0:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, 1000, loss.item(),
                    self.base_covar_module.base_kernel.lengthscale.item(),
                    likelihood.noise.item()))
          optimizer.step()
        return losses
    
    def optimization_trace(self):
        return;
        
    def optimal_q_u(self):
       return self(self.covar_module.inducing_points)
    
    def posterior_predictive(self, test_x):
        
        ''' Returns the posterior predictive multivariate normal '''
        
        self.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_star = likelihood(self(test_x))
        return y_star
    
    def visualise_posterior(self, test_x, y_star):
    
        ''' Visualising posterior predictive '''
        
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        lower, upper = y_star.confidence_region()
        ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'kx')
        ax.plot(test_x.numpy(), y_star.mean.numpy(), 'b-')
        ax.plot(self.covar_module.inducing_points.detach(), [-2.5]*self.num_inducing, 'rx')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Train', 'Mean', 'Inducing inputs', r'$\pm$2\sigma'])
        plt.show()

    def visualise_train(self):
        
        ''' Visualise training points '''
        
        plt.figure()
        plt.plot(self.train_x, self.train_y, 'bx', label='Train')
        plt.legend()

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
        
    def neg_test_log_likelihood(self, y_star, test_y):
        
         lpd = y_star.log_prob(test_y)
         # return the average
         return -torch.mean(lpd).detach()
    
    def rmse(self, y_star, test_y):
        
       return torch.sqrt(torch.mean((y_star.loc - test_y)**2)).detach()


if __name__ == '__main__':
    
    N = 1000  # Number of training observations

    X = torch.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

    # Initial inducing points
    Z_init = torch.randn(12)
    
    # Initialise model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SparseGPR(X, Y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        
    # Train
    losses = model.train_model(likelihood, combine_terms=True)

    # Test 
    test_x = torch.linspace(-8, 8, 1000)
    test_y = func(test_x)
    
    y_star = model.posterior_predictive(test_x)
    
    # Visualise 
    
    model.visualise_posterior(test_x, y_star)
    
    # Compute metrics
    rmse = model.rmse(y_star, test_y)
    nll = model.neg_test_log_likelihood(y_star, test_y)
    
    # Plot predictive variances 
    plt.figure()
    plt.plot(y_star.covariance_matrix.diag().detach())


# Verify: elbo, q*(u), p(f*|y)

# # q*(u) - mean
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
# pred_mean = torch.matmul(H, final)

# l1 = torch.matmul(K_mm, W)
# sigma = torch.matmul(l1, K_mm)

# sigma = model.optimal_q_u().covariance_matrix
# Knn = model.base_covar_module(model.train_x).evaluate()
# lhs = torch.matmul(K_nm, model.covar_module._inducing_mat.inverse())
# Qnn = torch.matmul(lhs, K_nm.evaluate().T)
# K_ss = model.base_covar_module(test_x)
# lh = torch.matmul(K_star_m, K_mm_inv)
# third_term = torch.matmul(torch.matmul(lh, sigma), lh.T)

# pred_covar = K_ss.evaluate() - torch.matmul(H, K_star_m.T) + third_term

