#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doubly collapsed SGPR with HMC

"""


import gpytorch 
import torch
import numpy as np
from prettytable import PrettyTable
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import UniformPrior
import matplotlib.pyplot as plt
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
            y_ = gp.marginal_likelihood("y", X=self.train_x.numpy()[:,None], Xu=Z_opt, y=self.train_y.numpy(), noise=sig_n)
        
            trace = pm.sample(n_samples, tune=500, chains=1)
        
       return trace
   
   def update_elbo_with_hyper_samples(self, mll, trace_hyper):
       
       mll.likelihood.noise_covar.noise = trace_hyper['sig_n'][-1]**2
       mll.model.base_covar_module.outputscale = trace_hyper['sig_f'][-1]**2
       mll.model.base_covar_module.base_kernel.lengthscale = trace_hyper['ls'][-1]
               
   def train_model(self, likelihood, optimizer):

        self.train()
        likelihood.train()
        elbo = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
             
        losses = []
        for i in range(1000):
          optimizer.zero_grad()
          output = self(self.train_x)
          self.freeze_kernel_hyperparameters()
          loss = -elbo(output, self.train_y)
          losses.append(loss)
          loss.backward()
          if i%100 == 0:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, 1000, loss.item(),
                    self.base_covar_module.base_kernel.lengthscale.item(),
                    likelihood.noise.item()))
                    Z_opt = model.inducing_points.numpy()
                    trace_hyper = self.sample_optimal_variational_hyper_dist(Z_opt)
          optimizer.step()
        return losses
               
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

    # Initial inducing points
    Z_init = torch.randn(12)
    
    # Initialise model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DoublyCollapsedSparseGPR(X, Y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        
    # Train
    #losses = model.train_model(likelihood, optimizer, combine_terms=True)
    
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
      
    #model.set_hyper_priors()
            
    losses = []
    for i in range(5000):
      optimizer.zero_grad()
      output = model(model.train_x)
      model.freeze_kernel_hyperparameters()
      loss = -mll(output, model.train_y)
      losses.append(loss)
      loss.backward()
      if i%200 == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, 5000, loss.item(),
                model.base_covar_module.base_kernel.lengthscale.item(),
                likelihood.noise.item()))
                Z_opt = model.inducing_points.numpy()[:,None]
                trace_hyper = model.sample_optimal_variational_hyper_dist(200, 1, Z_opt)  
                model.update_elbo_with_hyper_samples(mll, trace_hyper)
      optimizer.step()
    #return losses


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
 
   