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
from gpytorch.priors import MultivariateNormalPrior
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls.added_loss_term import AddedLossTerm

torch.manual_seed(42)
np.random.seed(37)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

class VariationalHyperDist(gpytorch.Module):
    
    def __init__(self, hyper_dim, hyper_prior, n, input_dim):
        super().__init__()
        
        self.hyper_dim = hyper_dim
        self.hyper_prior = hyper_prior
        self.n = n
        self.input_dim = input_dim
        
        num_elements_cholesky = int(hyper_dim*(hyper_dim + 1)/2)

        # Global variational params
        self.q_mu = torch.nn.Parameter(torch.randn(hyper_dim)*1e-3)
        self.q_sigma_vec = torch.nn.Parameter(torch.randn(num_elements_cholesky)*1e-3)
        
        self.jitter = torch.eye(hyper_dim)*1e-5

        # This will add the KL divergence KL(q(log_theta) || p(log_theta)) to the loss
        self.register_added_loss_term("log_theta_kl")
        
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

    def forward(self, num_samples):
        
        self.q_sigma = self.construct_sigma()
        
        q_log_theta = torch.distributions.MultivariateNormal(self.q_mu, self.q_sigma)
        
        log_theta_kl = kl_gaussian_loss_term(q_log_theta, self.hyper_prior, self.n, 1)        
        self.update_added_loss_term('log_theta_kl', log_theta_kl)
        return q_log_theta.rsample(sample_shape=torch.Size([num_samples]))
    
class kl_gaussian_loss_term(AddedLossTerm):
    
    def __init__(self, q_log_theta, log_hyper_prior, n, input_dim):
        self.q_log_theta = q_log_theta
        self.p_log_theta = log_hyper_prior
        self.n = n
        self.input_dim = input_dim
        
    def loss(self): 
        kl_per_hyper_dim = kl_divergence(self.q_log_theta, self.p_log_theta).sum(axis=0) # vector of size hyper_dim
        kl_per_point = kl_per_hyper_dim.sum()/self.n # scalar
        return kl_per_point


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
        
        hyper_dim = self.input_dim + 2 # lengthscale per dim, sig var and noise var
        hyper_prior_mean = torch.zeros(hyper_dim)
        log_hyper_prior = MultivariateNormalPrior(hyper_prior_mean, torch.eye(hyper_dim)*0.01) ## no correlation between hypers

        self.log_theta = VariationalHyperDist(hyper_dim, log_hyper_prior, self.n, self.input_dim)
        
    def forward(self, x, log_theta=None):
        mean_x = self.mean_module(x)
        if log_theta is not None:
            theta = torch.exp(log_theta)
            self.update_covar_module_at_theta(theta)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def update_covar_module_at_theta(self, theta):
        self.covar_module.outputscale = theta[0]
        self.covar_module.base_kernel.lengthscale = theta[1:-1]
        self.likelihood.noise_covar.noise = theta[-1]**2
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
        
        epoch_losses = []
        #iterator = trange(1000, leave=True)
        for i in range(num_epochs):
            with tqdm(train_loader, unit="batch", leave=True) as minibatch_iter:
                batch_losses = []
                for x_batch, y_batch in minibatch_iter:

                    optimizer.zero_grad()
                    
                    loss = 0.0
                    for _ in range(5):
                        
                        log_hyper_sample = self.sample_variational_log_hyper(num_samples=1)
    
                        #while np.exp(log_hyper_sample[0][0].detach().numpy()) > 2:
    
                        output = self(x_batch, log_theta=log_hyper_sample.flatten())
                        loss += -elbo(output, y_batch).sum()/5
                        
                        #print(self.log_theta.q_mu.detach())
                        #print('hyper_sample :' + str(np.exp(log_hyper_sample.detach().numpy())))
                        #print(self.covar_module.base_kernel.lengthscale)    
                        
                    batch_losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    
                minibatch_iter.set_description(f"Epoch {i} {loss.item()}")
                #minibatch_iter.set_postfix(loss=loss.item())
            epoch_losses.append(np.sum(batch_losses))
                    
        return epoch_losses, batch_losses

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
    train_x = X[:train_n][:,None]
    train_y = Y[:train_n].contiguous()
    
    test_x = X[train_n:][:,None]
    test_y = Y[train_n:].contiguous()
        
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    
    # Initial inducing points
    index_inducing = np.random.randint(0,len(train_x), 25)
    Z_init = train_x[index_inducing]
        
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BayesianStochasticVariationalGP(train_x, train_y, likelihood, Z_init)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Train
    
    losses = model.train_model(optimizer, train_loader, 
                              minibatch_size=100, num_epochs=1000,  combine_terms=True)
        
    # Test 
    test_x = torch.linspace(-8, 8, 1000)
    test_y = func(test_x)
    
    # ####
      
    log_hyper_samples = model.sample_variational_log_hyper(num_samples=200)
    
    # Get predictive distributions by feeding model through likelihood
    
    list_of_y_pred_dists = []
    
    model.eval()
    model.likelihood.eval()
    
    for i in range(len(log_hyper_samples)):
    
        #theta = torch.nn.functional.softplus(log_hyper_samples[i])
    
        #self.update_covar_module_at_theta(theta)
    
        with torch.no_grad():
            list_of_y_pred_dists.append(likelihood(model(test_x, log_theta=log_hyper_samples[i], prior=False)))
            
    
    y_mix_loc = np.array([np.array(dist.loc) for dist in list_of_y_pred_dists])
    y_mix_std = np.array([np.array(dist.covariance_matrix.diag()) for dist in list_of_y_pred_dists])
    
    #plt.figure()
    #plt.plot(y_mix_std.T)
    
    plt.figure(figsize=(8,5))
    
    plt.subplot(1,2,1)
    plt.plot(train_x, train_y, 'bo')
    plt.plot(test_x, np.mean(y_mix_loc, axis=0),color='r')
    plt.plot(test_x, test_y)
    plt.scatter(model.variational_strategy.inducing_points.detach(), [-2.0]*model.num_inducing, c='g', marker='x', label='Inducing')
    plt.title('Doubly Stochastic SVGP', fontsize='small')
    
    plt.subplot(1,2,2)
    plt.plot(test_x, y_mix_loc.T ,color='b', alpha=0.3)
    plt.title('Means of predictive mixture', fontsize='small')

    # ###

    # prior_samples = model.log_theta.hyper_prior.sample_n(200).numpy()
    
    # plt.figure()
    # plt.hist(prior_samples[:,0], bins=40, alpha=0.5)
    # plt.hist(log_hyper_samples[:,0].detach().numpy(), bins=40, alpha=0.5)

    
    # # plt.figure()
    # # plt.hist(prior_samples[:,0], bins=40)
    # # plt.hist(prior_samples[:,1], bins=40)
    
    ### draw samples form the prior
    
    # mean_module = ZeroMean()
    # rbf_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
    
    # rbf_module.outputscale = model.covar_module.outputscale
    # rbf_module.base_kernel.lengthscale = model.covar_module.base_kernel.lengthscale
    
    # kernel_matrix = rbf_module(test_x)
    
    # f = gpytorch.distributions.MultivariateNormal(mean_module(test_x), kernel_matrix).sample(torch.Size([100]))
    
    # plt.figure()
    # plt.plot(test_x, f.T)

    # # #y_star = model.mixture_posterior_predictive(test_x)
    
    # # Visualise 
    
    # #model.visualise_posterior(test_x, y_star)
    # # # Compute metrics
    from utils.metrics import rmse, nlpd_mixture

    rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)),test_y, torch.tensor([1.0]))
    nll_test = nlpd_mixture(list_of_y_pred_dists, test_y,  torch.tensor([1.0]))
    
    print('Test RMSE: ' + str(rmse_test))
    print('Test NLPD: ' + str(nll_test))
    
