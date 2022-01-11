#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian SVGP - Extension to Hensman et al (2015) with variational treatment of hyperparameters

"""

import gpytorch as gpytorch
import torch as torch
import numpy as np
from tqdm import tqdm
from math import floor
from models.svgp import StochasticVariationalGP
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

     def forward(self, num_samples):
        # Variational distribution over the hyper variable q(x)
        q_theta = torch.distributions.Normal(self.q_mu, torch.nn.functional.softplus(self.q_log_sigma))
        theta_kl = kl_gaussian_loss_term(q_theta, self.hyper_prior, self.n, self.data_dim)
        self.update_added_loss_term('theta_kl', theta_kl)  # Update the KL term
        return q_theta.rsample(sample_shape=torch.Size([num_samples]))
    
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
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        
        #self.covar_module.raw_outputscale.register_prior(NormalPrior(0,2.0))
        #self.covar_module.raw_outputscale
        
        # Hyperparameter variational distribution
        
        hyper_dim = self.data_dim + 2 # lengthscale per dim, sig var and noise var
        hyper_prior_mean = torch.ones(hyper_dim)
        log_hyper_prior = NormalPrior(hyper_prior_mean, torch.ones_like(hyper_prior_mean)) ## no correlation between hypers
        #self.register_prior('prior_log_theta', log_hyper_prior, 'X')

        self.log_theta = LogHyperVariationalDist(hyper_dim, log_hyper_prior, self.n, self.data_dim)
        
    def forward(self, x, log_theta=1.0):
        mean_x = self.mean_module(x)
        theta = torch.nn.functional.softplus(log_theta)
        self.update_covar_module_at_theta(theta)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def update_covar_module_at_theta(self, theta):
        self.covar_module.outputscale = theta[0]
        self.covar_module.base_kernel.lengthscale = theta[1:self.data_dim+1]
        self.likelihood.noise_covar.noise = theta[-1]
        return self.covar_module
    
    def sample_variational_log_hyper(self, num_samples):
        
        return self.log_theta(num_samples)
    
    def get_inducing_prior(self):
        
        Kmm = self.covar_module._inducing_mat
        return torch.distributions.MultivariateNormal(ZeroMean(), Kmm)
            
    def train_model(self, optimizer, train_loader, minibatch_size=100, num_epochs=25, combine_terms=True):
        
        self.train()
        self.likelihood.train()
        elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=len(self.train_y))
        
        losses = []
        #iterator = trange(1000, leave=True)
        for i in range(num_epochs):
            with tqdm(train_loader, unit="batch", leave=True) as minibatch_iter:
                for x_batch, y_batch in minibatch_iter:

                    minibatch_iter.set_description(f"Epoch {i}")

                    optimizer.zero_grad()
                    log_hyper_sample = self.sample_variational_log_hyper(num_samples=1)
                    #print('hyper_sample :' + str(log_hyper_sample))
                    output = self(x_batch, log_theta=log_hyper_sample.flatten())
                    #print(self.covar_module.base_kernel.lengthscale)
                    loss = -elbo(output, y_batch).sum()
                    losses.append(loss.item())
                    # if i%100 == 0:
                    #       print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %.3f   noise: %.3f' % (
                    #       i + 1, 1000, loss.item(),
                    #       self.covar_module.outputscale.item(),
                    #       self.covar_module.base_kernel.lengthscale,
                    #       self.likelihood.noise.item()))
                    loss.backward()
                    optimizer.step()
                    minibatch_iter.set_postfix(loss=loss.item())
        return losses

    def _get_batch_idx(self, batch_size):
           
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
    
    def optimization_trace(self):
        return;
        
    def mixture_posterior_predictive(self, test_x):
        
        ''' Returns the mixture posterior predictive - where samples are from the 
            variational distribution of the hyperparameters.
        '''
        
        self.eval()
        self.likelihood.eval()
        
        log_hyper_samples = self.sample_variational_log_hyper(num_samples=100)

        # Get predictive distributions by feeding model through likelihood
        
        list_of_y_pred_dists = []
        
        for i in range(len(log_hyper_samples)):
            
            theta = torch.nn.functional.softplus(log_hyper_samples[i])
            
            #self.update_covar_module_at_theta(theta)
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                list_of_y_pred_dists.append(self.likelihood(self(test_x, theta)))
        
        return list_of_y_pred_dists
    

if __name__ == '__main__':

    N = 1000  # Number of training observations

    X = torch.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n][:,None]
    train_y = Y[:train_n].contiguous()
    
    test_x = X[train_n:][:,None]
    test_y = Y[train_n:].contiguous()
        
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Initial inducing points
    Z_init = torch.randn(25)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BayesianStochasticVariationalGP(train_x, train_y, likelihood, Z_init)
    
    model_base = StochasticVariationalGP(train_x, train_y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    # Train
    losses = model.train_model(optimizer, train_loader, 
                              minibatch_size=500, num_epochs=200,  combine_terms=True)
        
    # Test 
    test_x = torch.linspace(-8, 8, 1000)
    test_y = func(test_x)
    
    model.eval()
    model.likelihood.eval()
    
    ####
    Y_train_pred = model_base.posterior_predictive(train_x)
    Y_test_pred = model_base.posterior_predictive(test_x)
      
    log_hyper_samples = model.sample_variational_log_hyper(num_samples=100)

      # Get predictive distributions by feeding model through likelihood
      
    list_of_y_pred_dists = []
      
    for i in range(len(log_hyper_samples)):
        
        #theta = torch.nn.functional.softplus(log_hyper_samples[i])
        
        #self.update_covar_module_at_theta(theta)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            list_of_y_pred_dists.append(model(test_x, log_theta=log_hyper_samples[i], prior=False))
    
    
    y_mix_loc = [np.array(dist.loc) for dist in list_of_y_pred_dists]

    plt.plot(test_x, np.mean(y_mix_loc, axis=0))
    plt.plot(train_x, train_y, 'bo')
    plt.plot(test_x, Y_test_pred.loc)
    
    # #y_star = model.mixture_posterior_predictive(test_x)
    
    # Visualise 
    
    #model.visualise_posterior(test_x, y_star)
    
    # Compute metrics
    #rmse = model.rmse(y_star, test_y)
    #nll = model.neg_test_log_likelihood(y_star, test_y)
    


   