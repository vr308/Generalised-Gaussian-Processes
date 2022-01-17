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
        theta is sampled using HMC every 200 iterations
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
       
       with pm.Model() as model:
           
            ls = pm.Gamma("ls", alpha=2, beta=1)
            sig_f = pm.HalfCauchy("sig_f", beta=1)
        
            cov = sig_f ** 2 * pm.gp.cov.ExpQuad(input_dim, ls=ls)
            gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")
                
            sig_n = pm.HalfCauchy("sig_n", beta=1)
            
            # Z_opt is the intermediate inducing points from the optimisation stage
            y_ = gp.marginal_likelihood("y", X=self.train_x.numpy(), Xu=Z_opt, y=self.train_y.numpy(), noise=sig_n)
        
            trace = pm.sample(n_samples, tune=50, chains=1)
        
       return trace
   
   def update_elbo_with_hyper_samples(self, elbo, trace_hyper):
       
       elbo.likelihood.noise_covar.noise = trace_hyper['sig_n']**2
       elbo.model.base_covar_module.outputscale = trace_hyper['sig_f']**2
       elbo.model.base_covar_module.base_kernel.lengthscale = trace_hyper['ls']
       
   def find_optimal_hyper_from_trace(self, loss, elbo, output, trace_hyper):
       
       print('----------Finding optimal hypers from trace-------')
       with torch.no_grad():
           elbos_from_trace = []
           for i in np.arange(len(trace_hyper)):
               self.update_elbo_with_hyper_samples(elbo, trace_hyper[i])
               elbos_from_trace.append(-elbo(output, self.train_y))
            
           ## Check if the best elbo is better than the loss at this juncture
           min_elbo = np.argmin(elbos_from_trace)
           if min_elbo < loss:
                return min_elbo
           else: 
                return None
               
               
   def train_model(self, optimizer, max_steps=10000, break_for_hmc=[500,1000]):

        self.train()
        self.likelihood.train()
        elbo = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        losses = []
        for i in range(max_steps):
          optimizer.zero_grad()
          output = self(self.train_x)
          self.freeze_kernel_hyperparameters()
          #print(elbo.model.base_covar_module.base_kernel.lengthscale)
          loss = -elbo(output, self.train_y)
          losses.append(loss.item())
          loss.backward()
          if i in break_for_hmc: ##== 0: ## alternate to hmc sampling of hypers
                    print('Iter %d/%d - Loss: %.3f   outputscale: %.3f lengthscale: %s noise: %.3f' % (
                    i + 1, max_steps, loss.item(),
                    self.base_covar_module.outputscale.item(),
                    self.base_covar_module.base_kernel.lengthscale,
                    self.likelihood.noise.item()) + '\n')
                    Z_opt = self.inducing_points.numpy()[:,None]
                    trace_hyper = self.sample_optimal_variational_hyper_dist(100, self.data_dim, Z_opt)  
                    optimal_hyper_index = self.find_optimal_hyper_from_trace(loss, elbo, output, trace_hyper)
                    if optimal_hyper_index is not None:
                        self.update_elbo_with_hyper_samples(elbo, trace_hyper[optimal_hyper_index])
                        print('Optimal hypers found in this round')
                    else:
                        print('No optimal hypers found - continue with optimisation')
          optimizer.step()
        return losses, trace_hyper
               
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
         model.likelihood.noise_covar.noise = trace_hyper['sig_n'][i]**2
         model.base_covar_module.outputscale = trace_hyper['sig_f'][i]**2
         model.base_covar_module.base_kernel.lengthscale = trace_hyper['ls'][i]
 
      #with torch.no_grad():
         pred = model.likelihood(model(test_x))
       #       try:
                  #chol = torch.linalg.cholesky(pred.covariance_matrix)
         list_of_y_pred_dists.append(pred)
      #        except RuntimeError:
      #            print('Not psd for ' + str(trace_hyper[i]) + ' ' + str(i))
     
      return list_of_y_pred_dists
       
if __name__ == '__main__':
    
    torch.manual_seed(13037762642999675754)
    
    from utils.experiment_tools import get_dataset_class
    from utils.metrics import rmse, nlpd_mixture, nlpd

    dataset = get_dataset_class('Power')(split=0, prop=0.8)
    X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
    
    ###### Initialising model class, likelihood, inducing inputs ##########
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    ## Fixed at X_train[np.random.randint(0,len(X_train), 200)]
    #Z_init = torch.randn(num_inducing, input_dim)
    Z_init = X_train[np.random.randint(0,len(X_train), 50)]

    model = BayesianSparseGPR_HMC(X_train,Y_train, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    ####### Custom training depending on model class #########
    
    break_for_hmc = [200,500,1500,2000,2999]
    losses, trace_hyper = model.train_model(optimizer, max_steps=3000, break_for_hmc=break_for_hmc)
    
    loss_list = [x.detach().item() for x in losses]
    
    plt.figure()
    plt.plot(loss_list)
    
    Y_test_pred = model.posterior_predictive(X_test)

    # ### Compute Metrics ###########
    
    #rmse_train = np.round(rmse(Y_train_pred.loc, Y_train, dataset.Y_std).item(), 4)
    rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
   
    # ### Convert everything back to float for Naval 
    
    # nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
    nlpd_test = np.round(nlpd(Y_test_pred, Y_test, dataset.Y_std).item(), 4)
    
    print('Test RMSE: ' + str(rmse_test))
    print('Test NLPD: ' + str(nlpd_test))
    
    ############ Mixture stuff
    
    # Y_test_pred_list = model.mixture_posterior_predictive(X_test, trace_hyper) ### a list of predictive distributions

    # y_mix_loc = np.array([np.array(dist.loc) for dist in Y_test_pred_list])
    # y_mix_std = np.array([np.array(dist.covariance_matrix.diag()) for dist in Y_test_pred_list])
    # y_mix_std = np.nan_to_num(y_mix_std, nan=1e-5)
    
    # # # ### Compute Metrics  ###########
    
    # rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), Y_test, dataset.Y_std)
    
    # # # ### Convert everything back to float for Naval 
    
    # # # nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
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
      
    # #model.set_hyper_priors()
            
    # losses = []
    # for i in range(5000):
    #   optimizer.zero_grad()
    #   output = model(model.train_x)
    #   model.freeze_kernel_hyperparameters()
    #   loss = -mll(output, model.train_y)
    #   losses.append(loss)
    #   loss.backward()
    #   if i%200 == 0:
    #             print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #             i + 1, 5000, loss.item(),
    #             model.base_covar_module.base_kernel.lengthscale.item(),
    #             likelihood.noise.item()))
    #             Z_opt = model.inducing_points.numpy()[:,None]
    #             trace_hyper = model.sample_optimal_variational_hyper_dist(200, 1, Z_opt)  
    #             model.update_elbo_with_hyper_samples(mll, trace_hyper)
    #   optimizer.step()
    # #return losses

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
 
 
 
   