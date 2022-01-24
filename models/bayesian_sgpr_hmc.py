#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doubly collapsed SGPR with HMC

"""
#TODO: Mixture Posterior Predictive
#TODO: Persist sampler hypers and cancel tuning

import gpytorch 
import torch
import numpy as np
import copy
from prettytable import PrettyTable
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import UniformPrior
import matplotlib.pyplot as plt
import scipy.stats as st
import pymc3 as pm

torch.manual_seed(45)
np.random.seed(37)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

class BayesianSparseGPR_HMC(gpytorch.models.ExactGP):
    
   """ The sparse GP class for regression with the doubly 
        collapsed stochastic bound.
        q(u) is implicit 
        theta is sampled using HMC based on pre-specified intervals
   """
      
   def __init__(self, train_x, train_y, likelihood, Z_init):
        
        super(BayesianSparseGPR_HMC, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.inducing_points = Z_init
        self.num_inducing = len(Z_init)  
        self.data_dim = self.train_x.shape[1]                                                                                          
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.data_dim))
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
       
       with pm.Model() as model_pymc3:
           
            ls = pm.Gamma("ls", alpha=2, beta=1, shape=(input_dim,))
            sig_f = pm.HalfCauchy("sig_f", beta=1)
        
            cov = sig_f ** 2 * pm.gp.cov.ExpQuad(input_dim, ls=ls)
            gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")
                
            sig_n = pm.HalfCauchy("sig_n", beta=1)
            
            # Z_opt is the intermediate inducing points from the optimisation stage
            y_ = gp.marginal_likelihood("y", X=self.train_x.numpy(), Xu=Z_opt, y=self.train_y.numpy(), noise=sig_n)
        
            trace = pm.sample(n_samples, tune=100, chains=1, return_inferencedata=False)   
            
       return trace
   
   def update_model_to_hyper(self, elbo, hyper_sample):
                
        elbo.likelihood.noise_covar.noise = hyper_sample['sig_n']**2
        elbo.model.base_covar_module.outputscale = hyper_sample['sig_f']**2
        elbo.model.base_covar_module.base_kernel.lengthscale = hyper_sample['ls']
           
   def train_model(self, optimizer, max_steps=10000, break_for_hmc=[200,500,1000,1500]):

        self.train()
        self.likelihood.train()
        elbo = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        losses = []
        trace_hyper = None
        
        for n_iter in range(max_steps):
            
          optimizer.zero_grad()
          
          ## Make sure to always freeze hypers before optimising for inducing locations
          self.freeze_kernel_hyperparameters()

          if n_iter < break_for_hmc[0]: ## the iterations before sampling (no sampling has occured, just optimise as normal)
                           
              output = self(self.train_x)
              loss = -elbo(output, self.train_y)
              
              losses.append(loss.item())
              loss.backward()
              optimizer.step()
        
          else:
              ### Compute stochastic elbo loss 
              if trace_hyper is not None:
                  loss = 0.0
                  for i in range(10):
                      hyper_sample = trace_hyper[i]
                      self.update_model_to_hyper(elbo, hyper_sample)
                      output = self(self.train_x)
                      loss += -elbo(output, self.train_y).sum()/len(trace_hyper)
                  print('Iter %d/%d - Loss: %.3f ' % (n_iter, max_steps, loss.item())),
                  losses.append(loss.item())
                  loss.backward()
                  optimizer.step()
              
          if n_iter in break_for_hmc: ## alternate to hmc sampling of hypers
                    print('---------------HMC step start------------------------------')
                    print('Iter %d/%d - Loss: %.3f ' % (n_iter, max_steps, loss.item()) + '\n'),
                    Z_opt = self.inducing_points.numpy()#[:,None]
                    trace_hyper = self.sample_optimal_variational_hyper_dist(100, self.data_dim, Z_opt)  
                    print('---------------HMC step finish------------------------------')
                    
        return losses, trace_hyper, Z_opt
               
   def optimal_q_u(self):
       return self(self.covar_module.inducing_points)
   

   def posterior_predictive(self, test_x):

        ''' Returns the posterior predictive multivariate normal '''

        self.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad():
            y_star = self.likelihood(self(test_x))
        return y_star
    
def mixture_posterior_predictive(model, test_x, trace_hyper):
     
      ''' Returns the posterior predictive multivariate normal '''
     
      model.eval()
      model.likelihood.eval()
      
      # Make predictions by feeding model through likelihood
     
      list_of_y_pred_dists = []
      
      for i in range(len(trace_hyper)):
          
         hyper_sample = trace_hyper[i]
         model.update_model_to_hyper(model, hyper_sample)
         with torch.no_grad():
              pred = model.likelihood(model(test_x))
              try:
                    chol = torch.linalg.cholesky(pred.covariance_matrix + torch.eye(len(test_x))*1e-5)
                    list_of_y_pred_dists.append(pred)
              except RuntimeError:
                   print('Not psd for sample ' + str(i))
     
      return list_of_y_pred_dists
       
if __name__ == '__main__':
    
    from utils.experiment_tools import get_dataset_class
    from utils.metrics import rmse, nlpd_mixture, nlpd

    dataset = get_dataset_class('Yacht')(split=0, prop=0.8)
    X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
    
    ###### Initialising model class, likelihood, inducing inputs ##########
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    ## Fixed at X_train[np.random.randint(0,len(X_train), 200)]
    #Z_init = torch.randn(num_inducing, input_dim)
    Z_init = X_train[np.random.randint(0,len(X_train), 100)]

    model = BayesianSparseGPR_HMC(X_train,Y_train, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    ####### Custom training depending on model class #########
    
    break_for_hmc = [200,500,700,1500,2000,2500, 3000,3999]
    #losses, trace_hyper, Z_opt = model.train_model(optimizer, max_steps=4000, break_for_hmc=break_for_hmc)
    
    # loss_greedy_protocol = losses
    
    # plt.figure()
    # plt.plot(loss_greedy_protocol)
    
    model.eval()
    likelihood.eval()
    
    Y_test_pred = model.posterior_predictive(X_test)

    # # ### Compute Metrics ###########
    
    rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
   
    # ### Convert everything back to float for Naval 
    
    nlpd_test = np.round(nlpd(Y_test_pred, Y_test, dataset.Y_std).item(), 4)
    
    print('Test RMSE: ' + str(rmse_test))
    print('Test NLPD: ' + str(nlpd_test))
    
    with pm.Model() as sampling_model:
           
           ls = pm.Gamma("ls", alpha=2, beta=1, shape=(6,))
           sig_f = pm.HalfCauchy("sig_f", beta=1)
       
           cov = sig_f ** 2 * pm.gp.cov.ExpQuad(6, ls=ls)
           gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")
               
           sig_n = pm.HalfCauchy("sig_n", beta=1)
           #Z_opt = pm.Normal("Xu", shape=(num_inducing, input_dim))
   
           # Z_opt is the intermediate inducing points from the optimisation stage
           y_ = gp.marginal_likelihood("y", X=X_train.numpy(), Xu=Z_opt, y=Y_train.numpy(), noise=sig_n)
       
           trace = pm.sample(100, tune=500, chains=1)
    
    ############ Mixture stuff
    
    # Y_test_pred_list = mixture_posterior_predictive(model, X_test, trace_hyper) ### a list of predictive distributions

    # y_mix_loc = np.array([np.array(dist.loc.detach()) for dist in Y_test_pred_list])
    # y_mix_std = np.array([np.array(dist.covariance_matrix.diag().detach()) for dist in Y_test_pred_list])
    # y_mix_std = np.nan_to_num(y_mix_std, nan=1e-5)
    
    # # # # ### Compute Metrics  ###########
    
    # rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), Y_test, dataset.Y_std)
    
    # ####
    
    # # # # ### Convert everything back to float for Naval 
    
    # # # # nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
    # nlpd_test = np.round(nlpd_mixture(Y_test_pred_list, Y_test, dataset.Y_std).item(), 4)
    
    # print('Test RMSE: ' + str(rmse_test))
    # print('Test NLPD: ' + str(nlpd_test))
    
    ################################################

    # N = 1000  # Number of training observations

    # X = torch.randn(N) * 2 - 1  # X values
    # Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

    # # Initial inducing points
    # Z_init = torch.randn(12)
    
    # # Initialise model and likelihood
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # model = BayesianSparseGPR_HMC(X[:,None], Y, likelihood, Z_init)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        
    # # Train
    # #losses = model.train_model(likelihood, optimizer, combine_terms=True)
    
    # model.train()
    # likelihood.train()
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
      
    # # # Test 
    # test_x = torch.linspace(-8, 8, 1000)
    # test_y = func(test_x)
    
    # list_of_y_pred_dists = model.mixture_posterior_predictive(test_x, trace_hyper) ## a list of predictive dists.
    
    # # ## Extracting list of means 
    
    # y_mix_loc = np.array([np.array(dist.loc) for dist in list_of_y_pred_dists])
    # y_mix_std = np.array([np.array(dist.covariance_matrix.diag().sqrt()) for dist in list_of_y_pred_dists])
    # y_mix_covar = [dist.covariance_matrix for dist in list_of_y_pred_dists]
    
    # for i in np.arange(len(y_mix_covar)):
    #     torch.linalg.cholesky(y_mix_covar[i] + torch.eye(1000)*1e-4)
        
    
    # y_mix_std = np.nan_to_num(y_mix_std, nan=1e-5)
    
    # skip_rows = np.unique(np.where(y_mix_std == 1e-5)[0])
    # y_mix_loc = np.delete(y_mix_loc, skip_rows, axis=0)
    # y_mix_std = np.delete(y_mix_std, skip_rows, axis=0)
    # y_mix_covar = np.delete(np.array(y_mix_covar), skip_rows, axis=0)
    
    # # # Visualise 
    # from utils.visualisation import visualise_posterior
    
    # visualise_posterior(model, test_x, test_y, list_of_y_pred_dists, mixture=True, title=None, new_fig=True)
    
    # #Compute metrics
    
    # from utils.metrics import rmse, nlpd_mixture
    
    # y_std = torch.tensor([1.0]) ## did not scale y-values
    
    # rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), test_y, y_std)
    # nll_test = nlpd_mixture(test_y, list_of_y_pred_dists, y_std)
    
    # print('Test RMSE: ' + str(rmse_test))
    # print('Test NLPD: ' + str(nll_test))
    
 
 
   