#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
allinHMC sampling both Z and theta (Rossi et al, 2021)

@author: vr308
"""

import gpytorch 
import torch
import numpy as np
import copy
from prettytable import PrettyTable
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import scipy.stats as st
import pymc3 as pm

torch.manual_seed(45)
np.random.seed(45)

class all_in_HMC(gpytorch.models.ExactGP):
    
   """ The GP class for regression with 
        theta sampled using NUTS
   """
      
   def __init__(self, train_x, train_y, likelihood, Z_init):
        
        super(all_in_HMC, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.data_dim = self.train_x.shape[1]                                                                                          
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.data_dim))
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=likelihood)

   def forward(self, x): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
           
   def sample(self, n_samples, input_dim, tune):
       
       with pm.Model() as model_pymc3:
           
            ls = pm.Gamma("ls", alpha=2, beta=1, shape=(input_dim,))
            sig_f = pm.HalfCauchy("sig_f", beta=1)
        
            cov = sig_f ** 2 * pm.gp.cov.ExpQuad(input_dim, ls=ls)
            gp = pm.gp.MarginalSparse(cov_func=cov, approx='VFE')
                
            sig_n = pm.HalfCauchy("sig_n", beta=1)
            
            Z = pm.Normal("Z", mu=0, sigma=1, shape=(100,input_dim))
            
            y_ = gp.marginal_likelihood("y", X=self.train_x.numpy(), Xu=Z, y=self.train_y.numpy(), noise=sig_n)
            trace = pm.sample(n_samples, tune=tune, chains=1, return_inferencedata=False)   
            
       return trace
       
   def train_model(self):
       
        self.train()
        self.likelihood.train()
        
        trace_step_size = []
        trace_perf_time = []
        
        print('---------------HMC step start------------------------------')                    
        num_tune = 500
        num_samples = 100
                   
        trace_hyper = self.sample(num_samples, self.data_dim, num_tune)  
        
        print('---------------HMC step finish------------------------------')
        trace_step_size.append(trace_hyper.get_sampler_stats('step_size')[0])
        trace_perf_time.append(trace_hyper.get_sampler_stats('perf_counter_diff').sum())  
        
        return trace_hyper, trace_step_size, trace_perf_time
    
               
   def optimal_q_u(self):
       return self(self.covar_module.inducing_points)
       
def full_mixture_posterior_predictive(model, test_x, trace_hyper):
     
      ''' Returns the mixture posterior predictive multivariate normal '''
     
      # Make predictions by feeding model through likelihood
     
      list_of_y_pred_dists = []
      
      for i in range(len(trace_hyper)):
          
         hyper_sample = trace_hyper[i]
         
         ## Training mode for overwriting hypers 
         model.train()
         model.likelihood.train()
         
         if hyper_sample['sig_n']**2 < 1e-4:
             hyper_sample['sig_n'] = 0.01
             
         model.likelihood.noise_covar.noise = hyper_sample['sig_n']**2
         model.base_covar_module.outputscale = hyper_sample['sig_f']**2
         model.base_covar_module.base_kernel.lengthscale = hyper_sample['ls']
         model.covar_module.inducing_points = torch.nn.Parameter(torch.Tensor(hyper_sample['Z']).double())
         
         with torch.no_grad():
             
              ## Testing mode for computing the posterior predictive 
              model.eval()
              model.likelihood.eval()
              pred = model.likelihood(model(test_x))
              
              try:
                    chol = torch.linalg.cholesky(pred.covariance_matrix + torch.eye(len(test_x))*1e-5)
                    list_of_y_pred_dists.append(pred)
              except RuntimeError:
                   print('Not psd for sample ' + str(i))
     
      return list_of_y_pred_dists
  
if __name__ == '__main__':
    
    ## Test
    
    from utils.experiment_tools import get_dataset_class
    from utils.metrics import rmse, nlpd_mixture, nlpd

    dataset = get_dataset_class('Boston')(split=0, prop=0.8)
    X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
    
    ###### Initialising model class, likelihood, inducing inputs ##########
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    Z_init = X_train[np.random.randint(0, len(X_train), 100)]

    model = all_in_HMC(X_train,Y_train, likelihood, Z_init)
    
    ####### Custom training depending on model class #########
    
    trace_hyper, step_sizes, perf_time = model.train_model()
      
    ##### Predictions ###########
    
    Y_test_pred_list = full_mixture_posterior_predictive(model, X_test, trace_hyper) ### a list of predictive distributions
    y_mix_loc = np.array([np.array(dist.loc.detach()) for dist in Y_test_pred_list])    
    
    #### Compute Metrics  ###########
    
    rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), Y_test, dataset.Y_std)
    nlpd_test = np.round(nlpd_mixture(Y_test_pred_list, Y_test, dataset.Y_std).item(), 4)

    print('Test RMSE: ' + str(rmse_test))
    print('Test NLPD: ' + str(nlpd_test))