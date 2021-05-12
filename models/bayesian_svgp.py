#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian SVGP

"""

#TODO: Mixture Posterior Predictive

import gpytorch as gpytorch
import torch as torch
import numpy as np
import tqdm
from math import floor
from tqdm import trange
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from gpytorch.models import ApproximateGP
from torch.distributions import kl_divergence
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls.added_loss_term import AddedLossTerm

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

class LogHyperVariationalDist(gpytorch.Module):
    
     def __init__(self, hyper_dim, hyper_prior, n, data_dim):
        super().__init__()
        
        self.hyper_dim = hyper_dim
        self.hyper_prior = hyper_prior
        self.n = n
        self.data_dim = data_dim

        # Global variational params
        self.q_mu = torch.nn.Parameter(torch.randn(hyper_dim))
        self.q_log_sigma = torch.nn.Parameter(torch.randn(hyper_dim))     
        # This will add the KL divergence KL(q(theta) || p(theta)) to the loss
        self.register_added_loss_term("theta_kl")

     def forward(self):
        # Variational distribution over the hyper variable q(x)
        q_theta = torch.distributions.Normal(self.q_mu, torch.nn.functional.softplus(self.q_log_sigma))
        theta_kl = kl_gaussian_loss_term(q_theta, self.hyper_prior, self.n, self.data_dim)
        self.update_added_loss_term('theta_kl', theta_kl)  # Update the KL term
        return q_theta.rsample()
    
class kl_gaussian_loss_term(AddedLossTerm):
    
    def __init__(self, q_theta, hyper_prior, n, data_dim):
        self.q_theta = q_theta
        self.p_theta = hyper_prior
        self.n = n
        self.data_dim = data_dim
        
    def loss(self): 
        kl_per_latent_dim = kl_divergence(self.q_theta, self.p_theta).sum(axis=0) # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum()/self.n # scalar
        # inside the forward method of variational ELBO, 
        # the added loss terms are expanded (using add_) to take the same 
        # shape as the log_lik term (has shape data_dim)
        # so they can be added together. Hence, we divide by data_dim to avoid 
        # overcounting the kl term
        return (kl_per_point/self.data_dim)


class BayesianStochasticVariationalGP(ApproximateGP):
    
    """ The sparse GP class for regression with the uncollapsed stochastic bound.
         The parameters of q(u) \sim N(m, S) are learnt explicitly. 
    """
      
    def __init__(self, train_x, train_y, likelihood, Z_init): 
        
        # Locations Z corresponding to u, they can be randomly initialized or 
        # regularly placed.
        self.inducing_inputs = Z_init
        self.num_inducing = len(Z_init)
        self.n = len(train_y)
        self.data_dim = train_x.shape[1]
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(self.num_inducing) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(BayesianStochasticVariationalGP, self).__init__(q_f)
        self.likelihood = likelihood
        self.train_x = train_x
        self.train_y = train_y
       
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Hyperparameter Variational distribution
        hyper_prior_mean = torch.Tensor([0])
        hyper_dim = len(hyper_prior_mean)

        log_hyper_prior = NormalPrior(hyper_prior_mean, torch.ones_like(hyper_prior_mean))
        self.log_theta = LogHyperVariationalDist(hyper_dim, log_hyper_prior, self.n, self.data_dim)
        
    def forward(self, x, log_theta=1.0):
        mean_x = self.mean_module(x)
        theta = torch.nn.functional.softplus(log_theta)
        self.update_covar_module_at_theta(theta)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def update_covar_module_at_theta(self, theta):
        self.covar_module.base_kernel.lengthscale = theta 
        return self.covar_module
    
    def sample_variational_log_hyper(self):
        
        return self.log_theta()
    
    def get_inducing_prior(self):
        
        Kmm = self.covar_module._inducing_mat
        return torch.distributions.MultivariateNormal(ZeroMean(), Kmm)
            
    def train_model(self, optimizer, train_loader, minibatch_size=100, num_epochs=25, combine_terms=True):
        
        self.train()
        self.likelihood.train()
        elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=len(self.train_y))
        
        losses = []
        iterator = trange(1000, leave=True)
        for i in iterator:
            batch_index = self._get_batch_idx(batch_size=200)
            optimizer.zero_grad()
            log_hyper_sample = self.sample_variational_log_hyper()
            #print('hyper_sample :' + str(log_hyper_sample))
            output = self(self.train_x[batch_index], log_theta=log_hyper_sample)
            #print(self.covar_module.base_kernel.lengthscale)
            loss = -elbo(output, self.train_y[batch_index]).sum()
            losses.append(loss.item())
            if i%100 == 0:
                  print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %.3f   noise: %.3f' % (
                  i + 1, 1000, loss.item(),
                  self.base_covar_module.outputscale.item(),
                  self.base_covar_module.base_kernel.lengthscale.item(),
                  self.likelihood.noise.item()))
            loss.backward()
            optimizer.step()
        return losses

    def _get_batch_idx(self, batch_size):
           
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
    
    def optimization_trace(self):
        return;
        
    def mixture_posterior_predictive(self, test_x):
        
        ''' Returns the posterior predictive multivariate normal '''
        
        self.eval()
        self.likelihood.eval()
        
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_star = self.likelihood(self(test_x))
        return y_star
    

# if __name__ == '__main__':

#     N = 1000  # Number of training observations

#     X = torch.randn(N) * 2 - 1  # X values
#     Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

#     train_n = int(floor(0.8 * len(X)))
#     train_x = X[:train_n][:,None]
#     train_y = Y[:train_n].contiguous()
    
#     test_x = X[train_n:][:,None]
#     test_y = Y[train_n:].contiguous()
        
#     train_dataset = TensorDataset(train_x, train_y)
#     train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    
#     test_dataset = TensorDataset(test_x, test_y)
#     test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
#     # Initial inducing points
#     Z_init = torch.randn(25)
    
#     likelihood = gpytorch.likelihoods.BernoulliLikelihood()
#     model = BayesianStochasticVariationalGP(train_x, train_y, likelihood, Z_init)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


#     # Train
#     #losses = model.train_model(likelihood, optimizer, train_loader, 
#     #                          minibatch_size=100, num_epochs=100,  combine_terms=True)
    
#     model.train()
#     likelihood.train()
#     mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))
    


#     # Test 
#     test_x = torch.linspace(-8, 8, 1000)
#     test_y = func(test_x)
    
#     y_star = model.posterior_predictive(test_x)
    
#     # Visualise 
    
#     model.visualise_posterior(test_x, y_star)
    
#     # Compute metrics
#     rmse = model.rmse(y_star, test_y)
#     nll = model.neg_test_log_likelihood(y_star, test_y)
    


   