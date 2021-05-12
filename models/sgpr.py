#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGPR (Titsias, 2009)

"""

#TODO: Collect optimisation trace

import gpytorch 
import torch
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

class SparseGPR(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, Z_init):
        
        """The sparse GP class for regression with the collapsed bound.
           q*(u) is implicit. 
        """
        super(SparseGPR, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.inducing_points = Z_init
        self.num_inducing = len(Z_init)   
        self.likelihood = likelihood                                                                                          
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=self.likelihood)

    def forward(self, x): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def elbo(self, output, y):
        
        Knn = self.base_covar_module(self.train_x).evaluate()
        Knm = self.base_covar_module(self.train_x, self.inducing_points).evaluate()
        lhs = self.matmul(Knm, self.covar_module._inducing_mat.inverse())
        Qnn = torch.matmul(lhs, Knm.T)

        shape = Knn.shape[:-1]
        noise_covar = self.likelihood._shaped_noise_covar(shape).evaluate()
        noise_diag = noise_covar.diag()

        #p_y = gpytorch.distributions.MultivariateNormal(torch.Tensor([0]*len(self.train_x)), Qnn + noise_covar)
        p_y = self.likelihood(output).log_prob(y)
        expected_log_lik = p_y.log_prob(y)
       
        diag = Knn.diag() - Qnn.diag()
        trace_term = 0.5*(diag/noise_diag).sum() 
    
        return expected_log_lik, trace_term
            
    def train_model(self, optimizer, combine_terms=True):

        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        losses = []
        for i in range(1000):
          optimizer.zero_grad()
          output = self(self.train_x)
          if combine_terms:
              loss = -mll(output, self.train_y)
          else:
              loss = -self.elbo(output, self.train_y)
          losses.append(loss)
          loss.backward()
          if i%100 == 0:
                    print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %.3f   noise: %.3f' % (
                    i + 1, 1000, loss.item(),
                    self.base_covar_module.outputscale.item(),
                    self.base_covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item()))
          optimizer.step()
        return losses
    
    def optimization_trace(self):
        return;
        
    def optimal_q_u(self):
       return self(self.covar_module.inducing_points)
    
    def posterior_predictive(self, test_x):
        
        ''' Returns the posterior predictive multivariate normal '''
        
        self.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad():
            y_star = self.likelihood(self(test_x))
        return y_star
     
# Verify: elbo, q*(u), p(f*|y)

# # q*(u) - mean
# Z = model.covar_module.inducing_points
# K_nm = model.base_covar_module(X,Z).evaluate()
# K_mm = model.covar_module._inducing_mat
# noise = likelihood.noise
# K_mn_K_nm = torch.matmul(K_nm.T,K_nm)
# W = (K_mm + K_mn_K_nm/noise).inverse()
# lhs = K_mm/noise
# rhs = torch.matmul(K_nm.T, Y)
# l1 = torch.matmul(lhs, W)
# final = torch.matmul(l1, rhs)

# K_star_m = model.base_covar_module(test_x, Z).evaluate()
# K_mm_inv = model.covar_module._inducing_mat.inverse()
# H = torch.matmul(K_star_m, K_mm_inv)
# pred_mean = torch.matmul(H, final)

# l1 = torch.matmul(K_mm, W)
# sigma = torch.matmul(l1, K_mm)

#sigma = model.optimal_q_u().covariance_matrix
# Knn = model.base_covar_module(model.train_x).evaluate()
# lhs = torch.matmul(K_nm, model.covar_module._inducing_mat.inverse())
# Qnn = torch.matmul(lhs, K_nm.evaluate().T)
#K_ss = model.base_covar_module(test_x)
#lh = torch.matmul(K_nm, K_mm_inv)
#third_term = torch.matmul(torch.matmul(lh, sigma), lh.T)

#pred_covar = K_ss.evaluate() - torch.matmul(H, K_star_m.T) + third_term

