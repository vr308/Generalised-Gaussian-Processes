#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian SVGP

"""

import gpytorch as gpytorch
import torch as torch
import numpy as np
import tqdm
from math import floor
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

class HyperVariationalDist(gpytorch.Module):
    
     def __init__(self, hyper_dim, hyper_prior, data_dim):
        super().__init__()
        
        self.hyper_dim = hyper_dim
        self.hyper_prior = hyper_prior
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Global variational params
        self.q_mu = torch.nn.Parameter(torch.randn(hyper_dim))
        self.q_log_sigma = torch.nn.Parameter(torch.randn(hyper_dim))     
        # This will add the KL divergence KL(q(theta) || p(theta)) to the loss
        self.register_added_loss_term("theta_kl")

     def forward(self):
        # Variational distribution over the hyper variable q(x)
        q_theta = torch.distributions.Normal(self.q_mu, torch.nn.functional.softplus(self.q_log_sigma))
        theta_kl = kl_gaussian_loss_term(q_theta, self.hyper_prior, self.hyper_dim)
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
        
         # Register priors
        
        self.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")

        # Hyperparameter Variational distribution
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
        theta = HyperVariationalDist(n, data_dim, latent_dim, X_init, prior_x)
        

    def forward(self, x, theta):
        mean_x = self.mean_module(x)
        covar_x = self.get_covar_at_theta(theta)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_covar_at_theta(self, theta):
        return
        
    def get_inducing_prior(self):
        
        Kmm = self.covar_module._inducing_mat
        return torch.distributions.MultivariateNormal(ZeroMean(), Kmm)
    
    def elbo(self, output, y):
        
        Knn = model.base_covar_module(self.train_x).evaluate()
        Knm = model.base_covar_module(self.train_x, self.Z_init).evaluate()
        lhs = torch.matmul(Knm, self.covar_module._inducing_mat.inverse())
        Qnn = torch.matmul(lhs, Knm.evaluate().T)
        
        shape = Knn.shape[:-1]
        noise_diag = self.likelihood._shaped_noise_covar(shape).diag()
        S = self.q_u.forward().covariance_matrix
        Lambda = torch.matmul(lhs.T, lhs)
        #p_y = model.likelihood(output)
        p_y = gpytorch.torch.MultivariateNormal(lhs,noise_diag)
        expected_log_lik = p_y.log_prob(y)
        shape = Knn.shape[:-1]
        diag_1 = Knn.diag() - Qnn.diag()
        trace_term_1 = 0.5*(diag_1/noise_diag).sum() 
        
        diag_2 = torch.matmul(S, Lambda).diag()
        trace_term_2 = 0.5*(diag_2/noise_diag).sum() 
        kl_term = self.q_f.kl_divergence()
        return expected_log_lik, trace_term_1, trace_term_2, kl_term
            
    def train_model(self, likelihood, optimizer, train_loader, minibatch_size=100, num_epochs=25, combine_terms=True):
        
        self.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(self.train_y))
        
        losses = []
        for i in range(num_epochs):
            minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=True)
            for x_batch, y_batch in minibatch_iter:
                  optimizer.zero_grad()
                  output = self(x_batch)
                  if combine_terms:
                      loss = -mll(output, y_batch)
                  else:
                      loss = -self.elbo(output, y_batch)
                  losses.append(loss)
                  loss.backward()
                  if i%10 == 0:
                            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, 1000, loss.item(),
                            self.covar_module.base_kernel.lengthscale.item(),
                            likelihood.noise.item()))
                  optimizer.step()
        return losses
    
    def optimization_trace(self):
        return;
        
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
        ax.plot(self.inducing_inputs.detach(), [-2.5]*self.num_inducing, 'rx')
        ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
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
    model = StochasticVariationalGP(train_x, train_y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    # Train
    losses = model.train_model(likelihood, optimizer, train_loader, 
                               minibatch_size=100, num_epochs=100,  combine_terms=True)

    # Test 
    test_x = torch.linspace(-8, 8, 1000)
    test_y = func(test_x)
    
    y_star = model.posterior_predictive(test_x)
    
    # Visualise 
    
    model.visualise_posterior(test_x, y_star)
    
    # Compute metrics
    rmse = model.rmse(y_star, test_y)
    nll = model.neg_test_log_likelihood(y_star, test_y)
    

#  num_epochs = 100
 # losses = []
 # epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
 # for i in epochs_iter:
 #     # Within each iteration, we will go over each minibatch of data
 #     print('Finished 1 loop')
 #     minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=True)
 #     for x_batch, y_batch in minibatch_iter:
 #         optimizer.zero_grad()
 #         output = model(x_batch)
 #         loss = -mll(output, y_batch)
 #         losses.append(loss)
 #         minibatch_iter.set_postfix(loss=loss.item())
 #         loss.backward()
 #         optimizer.step()
 
   