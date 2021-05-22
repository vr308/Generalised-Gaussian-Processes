#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doubly collapsed SGPR with HMC

"""
#TODO: Mixture Posterior Predictive
#TODO: Only assigning lengthscale from hmc samples as a MWE (need to write a function to assign)

import gpytorch 
import torch
import numpy as np
from prettytable import PrettyTable
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import UniformPrior
import matplotlib.pyplot as plt
import scipy.stats as st
import pymc3 as pm


def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

class BayesianSparseGPR_HMC(gpytorch.models.ExactGP):
    
   """ The sparse GP class for regression with the doubly 
        collapsed stochastic bound.
        q(u) is implicit 
   """
      
   def __init__(self, train_x, train_y, likelihood, Z_init):
        
        super(BayesianSparseGPR_HMC, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.inducing_points = Z_init
        self.num_inducing = len(Z_init)                                                                                             
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=likelihood)

   def forward(self, x): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
   def freeze_kernel_hyperparameters(self):
       
        for name, parameter in self.named_hyperparameters():
           if (name != 'covar_module.inducing_points'):
               parameter.requires_grad = False
        
   def sample_optimal_variational_hyper_dist(self, n_samples, input_dim, Z_opt):
       
       with pm.Model() as model:
           
            ls = pm.Gamma("ls", alpha=2, beta=1)
            sig_f = pm.HalfCauchy("sig_f", beta=5)
        
            cov = sig_f ** 2 * pm.gp.cov.ExpQuad(input_dim=input_dim, ls=ls)
            gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")
                
            sig_n = pm.HalfCauchy("sig_n", beta=5)
            
            # Z_opt is the intermediate inducing points from the optimisation stage
            y_ = gp.marginal_likelihood("y", X=self.train_x.numpy(), Xu=Z_opt, y=self.train_y.numpy(), noise=sig_n)
        
            trace = pm.sample(n_samples, tune=500, chains=1)
        
       return trace
   
   def update_elbo_with_hyper_samples(self, elbo, trace_hyper):
       
       elbo.likelihood.noise_covar.noise = trace_hyper['sig_n'][-1]**2
       elbo.model.base_covar_module.outputscale = trace_hyper['sig_f'][-1]**2
       elbo.model.base_covar_module.base_kernel.lengthscale = trace_hyper['ls'][-1]
               
   def train_model(self, optimizer):

        self.train()
        self.likelihood.train()
        elbo = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        losses = []
        for i in range(1000):
          optimizer.zero_grad()
          output = self(self.train_x)
          self.freeze_kernel_hyperparameters()
          #print(elbo.model.base_covar_module.base_kernel.lengthscale)
          loss = -elbo(output, self.train_y)
          losses.append(loss)
          loss.backward()
          if i%100 == 0:
                    print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %.3f   noise: %.3f' % (
                    i + 1, 1000, loss.item(),
                    self.base_covar_module.outputscale.item(),
                    self.base_covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item()))
                    Z_opt = self.inducing_points.numpy()[:,None]
                    trace_hyper = self.sample_optimal_variational_hyper_dist(200, 1, Z_opt)  
                    self.update_elbo_with_hyper_samples(elbo, trace_hyper)
          optimizer.step()
          #print(elbo.model.base_covar_module.base_kernel.lengthscale)
        return losses, trace_hyper
               
   def optimal_q_u(self):
       return self(self.covar_module.inducing_points)
    
   def mixture_posterior_predictive(self, test_x, trace_hyper):
        
        ''' Returns the posterior predictive multivariate normal '''
        
        self.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        
        list_of_y_pred_dists = []
        for i in range(len(trace_hyper)):
            self.likelihood.noise_covar.noise = trace_hyper['sig_n'][i]**2
            self.base_covar_module.outputscale = trace_hyper['sig_f'][i]**2
            self.base_covar_module.base_kernel.lengthscale = trace_hyper['ls'][i]
    
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                list_of_y_pred_dists.append(self.likelihood(self(test_x)))
        
        return list_of_y_pred_dists
    
   def get_posterior_predictive_mean(sample_means):
        return np.average(sample_means, axis=0)


   def compute_log_marginal_likelihood(K_noise, y):
        return np.log(st.multivariate_normal.pdf(y, cov=K_noise.eval()))
    
    
   def get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds):
        # Fixed at 95% CI
    
        n_test = sample_means.shape[-1]
        components = sample_means.shape[0]
        lower_ = []
        upper_ = []
        for i in np.arange(n_test):
            print(i)
            mix_idx = np.random.choice(np.arange(components), size=2000, replace=True)
            mixture_draws = np.array(
                [st.norm.rvs(loc=sample_means.iloc[j, i], scale=sample_stds.iloc[j, i]) for j in mix_idx])
            lower, upper = st.scoreatpercentile(mixture_draws, per=[2.5, 97.5])
            lower_.append(lower)
            upper_.append(upper)
        return np.array(lower_), np.array(upper_)
    
# if __name__ == '__main__':

#     N = 1000  # Number of training observations

#     X = torch.randn(N) * 2 - 1  # X values
#     Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

#     # Initial inducing points
#     Z_init = torch.randn(12)
    
#     # Initialise model and likelihood
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     model = BayesianSparseGPR_HMC(X[:,None], Y, likelihood, Z_init)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        
#     # Train
#     #losses = model.train_model(likelihood, optimizer, combine_terms=True)
    
#     model.train()
#     likelihood.train()
#     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
      
#     #model.set_hyper_priors()
            
#     losses = []
#     for i in range(5000):
#       optimizer.zero_grad()
#       output = model(model.train_x)
#       model.freeze_kernel_hyperparameters()
#       loss = -mll(output, model.train_y)
#       losses.append(loss)
#       loss.backward()
#       if i%200 == 0:
#                 print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#                 i + 1, 5000, loss.item(),
#                 model.base_covar_module.base_kernel.lengthscale.item(),
#                 likelihood.noise.item()))
#                 Z_opt = model.inducing_points.numpy()[:,None]
#                 trace_hyper = model.sample_optimal_variational_hyper_dist(200, 1, Z_opt)  
#                 model.update_elbo_with_hyper_samples(mll, trace_hyper)
#       optimizer.step()
#     #return losses


    # Test 
    #test_x = torch.linspace(-8, 8, 1000)
    #test_y = func(test_x)
    
    #y_star = model.posterior_predictive(test_x)
    
    # Visualise 
    
    #model.visualise_posterior(test_x, y_star)
    
    # Compute metrics
    #rmse = model.rmse(y_star, test_y)
    #nll = model.neg_test_log_likelihood(y_star, test_y)
    

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
 
 
 
   