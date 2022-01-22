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
import matplotlib.pyplot as plt
from gpytorch.models import ApproximateGP
from torch.distributions import kl_divergence
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.priors import NormalPrior
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls.added_loss_term import AddedLossTerm

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

class VariationalHyperDist(torch.nn.Module):
    
    def __init__(self, hyper_dim, hyper_prior, n, input_dim):
        super().__init__()
        
        self.hyper_dim = hyper_dim
        self.hyper_prior = hyper_prior
        self.n = n
        self.input_dim = input_dim
        
        num_elements_cholesky = hyper_dim*(hyper_dim + 1)/2

        # Global variational params
        self.q_mu = torch.nn.Parameter(torch.randn(hyper_dim))
        self.q_sigma_vec = torch.nn.Parameter(torch.randn(num_elements_cholesky))
        
        self.jitter = torch.eye(hyper_dim).unsqueeze(0)*1e-5

        # This will add the KL divergence KL(q(theta) || p(theta)) to the loss
        self.register_added_loss_term("theta_kl")
        
    def construct_sigma(self):
       
       row_ids, col_ids = torch.tril_indices(self.hyper_dim, self.hyper_dim)
       lower_sigma = torch.eye(self.hyper_dim)
       k = 0
       for i,j in zip(row_ids, col_ids):
               lower_sigma[i,j] = self.q_sigma_vec[k]
               k +=1
       sigma = torch.matmul(lower_sigma, lower_sigma.T)
       sigma += self.jitter
       return sigma  

    def forward(self, num_samples, batch_idx=None):
        
        self.q_sigma = self.construct_sigma()
        
        if batch_idx is None:
            batch_idx = np.arange(self.n) 
        
        q_mu_batch = self.q_mu[batch_idx, ...]
        q_sigma_batch = self.q_sigma[batch_idx, ...]

        q_theta = torch.distributions.MultivariateNormal(q_mu_batch, q_sigma_batch)

        self.hyper_prior.loc = self.hyper_prior.loc[:len(batch_idx), ...]
        self.hyper_prior.scale = self.hyper_prior.covariance_matrix[:len(batch_idx), ...]
        theta_kl = kl_gaussian_loss_term(q_theta, self.hyper_prior, len(batch_idx), 1)        
        self.update_added_loss_term('theta_kl', theta_kl)
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
        self.input_dim = train_x.shape[1]
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(self.num_inducing) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(BayesianStochasticVariationalGP, self).__init__(q_f)
        
        self.likelihood = likelihood
        self.train_x = train_x
        self.train_y = train_y
       
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))
        
        # Hyperparameter variational distribution
        
        hyper_dim = self.data_dim + 2 # lengthscale per dim, sig var and noise var
        hyper_prior_mean = torch.zeros(hyper_dim)
        log_hyper_prior = NormalPrior(hyper_prior_mean, torch.ones_like(hyper_prior_mean)*0.4) ## no correlation between hypers

        self.log_theta = VariationalHyperDist(hyper_dim, log_hyper_prior, self.n, self.input_dim)
        
    def forward(self, x, log_theta=1.0):
        mean_x = self.mean_module(x)
        theta = torch.exp(log_theta)
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

                    optimizer.zero_grad()
                    log_hyper_sample = self.sample_variational_log_hyper(num_samples=1)
                    output = self(x_batch, log_theta=log_hyper_sample.flatten())
                    loss = -elbo(output, y_batch).sum()
                    
                    if i%100 == 0:
                        minibatch_iter.set_description(f"Epoch {i}")
                        minibatch_iter.set_postfix(loss=loss.item())
                        
                        #print(self.log_theta.q_mu.detach())
                        #print('hyper_sample :' + str(log_hyper_sample))
                        #print(self.covar_module.base_kernel.lengthscale)    
                        
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    
        return losses

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
    
    #train_index = np.where((X < -2) | (X > 2))

    #train_x = X[train_index][:,None]
    #train_y = Y[train_index]

    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n][:,None].double()
    train_y = Y[:train_n].contiguous().double()
    
    test_x = X[train_n:][:,None].double()
    test_y = Y[train_n:].contiguous().double()
        
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    
    # Initial inducing points
    index_inducing = np.random.randint(0,len(train_x), 25)
    Z_init = train_x[index_inducing].double()
        
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = BayesianStochasticVariationalGP(train_x, train_y, likelihood, Z_init)
    
    model = model.double()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Train
    losses = model.train_model(optimizer, train_loader, 
                              minibatch_size=100, num_epochs=50,  combine_terms=True)
        
    # # Test 
    # test_x = torch.linspace(-8, 8, 1000).double()
    # test_y = func(test_x).double()
    
    # ####
      
    # log_hyper_samples = model.sample_variational_log_hyper(num_samples=100)
    
    # # Get predictive distributions by feeding model through likelihood
    
    # list_of_y_pred_dists = []
    
    # model.eval()
    # model.likelihood.eval()
    
    # for i in range(len(log_hyper_samples)):
    
    #     #theta = torch.nn.functional.softplus(log_hyper_samples[i])
    
    #     #self.update_covar_module_at_theta(theta)
    
    #     with torch.no_grad():
    #         list_of_y_pred_dists.append(likelihood(model(test_x, log_theta=log_hyper_samples[i], prior=False)))
            
    
    # y_mix_loc = np.array([np.array(dist.loc) for dist in list_of_y_pred_dists])
    # y_mix_std = np.array([np.array(dist.covariance_matrix.diag()) for dist in list_of_y_pred_dists])
    
    # plt.figure()
    # plt.plot(y_mix_std.T)
    
    # plt.figure()
    # plt.plot(test_x, np.mean(y_mix_loc, axis=0),color='r')
    # plt.plot(test_x, test_y)
    # #plt.plot(test_x, np.array(y_mix_loc).T, alpha=0.4, color='b')
    # plt.plot(train_x, train_y, 'bo')
    # plt.scatter(model.variational_strategy.inducing_points.detach(), [-2.0]*model.num_inducing, c='g', marker='x', label='Inducing')

    # ###

    # prior_samples = model.log_theta.hyper_prior.sample_n(1000).numpy()
    # posterior_samples = model.sample_variational_log_hyper(1000).detach().numpy()
    
    # plt.figure()
    # plt.hist(prior_samples[:,0], bins=40, alpha=0.5)
    # plt.hist(posterior_samples[:,0], bins=40, alpha=0.5)

    
    # # plt.figure()
    # # plt.hist(prior_samples[:,0], bins=40)
    # # plt.hist(prior_samples[:,1], bins=40)

    # # #y_star = model.mixture_posterior_predictive(test_x)
    
    # # Visualise 
    
    # #model.visualise_posterior(test_x, y_star)
    # # # Compute metrics
    # from utils.metrics import rmse, nlpd_mixture

    # rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)),test_y, torch.tensor([1.0]))
    # nll_test = nlpd_mixture(test_y, y_mix_loc, y_mix_std)
    
    # print('Test RMSE: ' + str(rmse_test))
    # print('Test NLPD: ' + str(nll_test))
    
