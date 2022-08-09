#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGPR
Reference: Michalis Titsias 2009, Sparse Gaussian processes using inducing points.

"""

import gpytorch
import torch
import numpy as np
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
#from utils.metrics import get_trainable_param_names
torch.manual_seed(45)
np.random.seed(37)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

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
        self.base_covar_module = ScaleKernel(RBFKernel(ard_num_dims = self.train_x.shape[-1]))
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
    
    @staticmethod
    def optimization_trace(trace_states, states, grad_params):
        trace_states.append({param_name: param.numpy() for param_name, param in states.items() if param_name in grad_params})
        return trace_states


    # def train_model(self, optimizer, combine_terms=True, n_restarts=10, num_steps=1000):

    #     self.train()
    #     self.likelihood.train()
    #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

    #     grad_params = get_trainable_param_names(self)

    #     trace_states = []
    #     losses = torch.zeros(n_restarts, num_steps)
    #     for i in range(n_restarts):

    #         ## re-initialise model
    #         self.initialise_parameters()

    #         for j in range(num_steps):
    #           optimizer.zero_grad()
    #           output = self(self.train_x)
    #           if combine_terms:
    #               loss = -mll(output, self.train_y)
    #           else:
    #               loss = -self.elbo(output, self.train_y)
    #           losses[i, j] = loss.item()
    #           loss.backward()
    #           if j%100 == 0:
    #                     print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %.3f   noise: %.3f' % (
    #                     j + 1, 1000, loss.item(),
    #                     self.base_covar_module.outputscale.item(),
    #                     self.base_covar_module.base_kernel.lengthscale.item(),
    #                     self.likelihood.noise.item()))
    #           optimizer.step()
    #           states = self.state_dict().copy()
    #           trace_states = SparseGPR.optimization_trace(trace_states, states, grad_params)
    #         #hyper_trace_dict_i = {param_name: param for param_name, param in model_states[i] if 'covar_module' in param_name}
       
    #     print('Preserving final model state for restart iteration ' + str(i))
    #     final_states.append({param_name: param.detach() for param_name, param in model_ml.state_dict().items()})
    
    #     return losses, trace_states
    
    def train_model(self, optimizer, combine_terms=True, n_restarts=10, max_steps=10000):

        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        #grad_params = get_trainable_param_names(self)

        #trace_states = []
        losses = []

        for j in range(max_steps):
              optimizer.zero_grad()
              output = self.forward(self.train_x)
              if combine_terms:
                  loss = -mll(output, self.train_y).sum()
              else:
                  loss = -self.elbo(output, self.train_y)
              losses.append(loss.item())
              loss.backward()
              if j%1000 == 0:
                        print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %s   noise: %.3f ' % (
                        j + 1, max_steps, loss.item(),
                        self.base_covar_module.outputscale.item(),
                        self.base_covar_module.base_kernel.lengthscale,
                        self.likelihood.noise.item()))
                        #self.covar_module.inducing_points[0:5]))
              optimizer.step()
              #if len(losses) > 2:
              #    if (losses[-2] - losses[-1] < 1e-5):
              #        break;
              #states = self.state_dict().copy()
              #trace_states = SparseGPR.optimization_trace(trace_states, states, grad_params)
              #hyper_trace_dict_i = {param_name: param for param_name, param in model_states[i] if 'covar_module' in param_name}
        return losses


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
    

if __name__ == "__main__":
    
    
    
    
    N = 1000  # Number of training observations

    X = torch.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

    # Initial inducing points
    Z_init = torch.randn(12)
    
    # Initialise model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SparseGPR(X[:,None], Y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        
    # Train
    losses = model.train_model(optimizer)
    
    # # # Test 
    # test_x = torch.linspace(-8, 8, 1000)
    # test_y = func(test_x)
    
    # ## predictions
    # test_pred = model.posterior_predictive(test_x)
    
    # from utils.metrics import rmse, nlpd
    # from utils.visualisation import visualise_posterior
    
    # y_std = torch.tensor([1.0]) ## did not scale y-values

    # rmse_test = np.round(rmse(test_pred.loc, test_y,y_std).item(), 4)
    # nlpd_test = np.round(nlpd(test_pred, test_y, y_std).item(), 4)
    
    # visualise_posterior(model, test_x, test_y, test_pred, mixture=False, title=None, new_fig=True)


    ########### Elevator example
    
    from utils.experiment_tools import get_dataset_class
    import numpy as np
    from utils.metrics import rmse, nlpd, nlpd_marginal

    dataset = get_dataset_class('Concrete')(split=0, prop=0.8)
    X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
    
    ###### Initialising model class, likelihood, inducing inputs ##########
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #likelihood = gpytorch.likelihoods.PoissonLikelihood()
    
    ## Fixed at X_train[np.random.randint(0,len(X_train), 200)]
    #Z_init = torch.randn(num_inducing, input_dim)
    Z_init = X_train[np.random.randint(0,len(X_train), 500)]
    
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        X_test = X_test.cuda()
        Y_train = Y_train.cuda()
        Y_test = Y_test.cuda()
        Z_init = Z_init.cuda()

    model = SparseGPR(X_train, Y_train.flatten(), likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
       
    ####### Custom training depending on model class #########
    
    losses = model.train_model(optimizer, max_steps=4000)
    
    #Y_train_pred = model.posterior_predictive(X_train)
    Y_test_pred = model.posterior_predictive(X_test)

    # ### Compute Metrics  ###########
    # import scipy.stats as st
    
    # #rmse_train = np.round(rmse(Y_train_pred.loc, Y_train, dataset.Y_std).item(), 4)
    # rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
   
    # # ### Convert everything back to float for Naval 
    
    # # nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
    # nlpd_test = np.round(nlpd_marginal(Y_test_pred, Y_test, dataset.Y_std).item(), 4)

    print('Test RMSE: ' + str(rmse_test))
    print('Test NLPD: ' + str(nlpd_test))


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

#double_dist = gpytorch.distributions.MultivariateNormal(mean=torch.zeros(9700).double(), covariance_matrix=torch.eye(9700).double())
#double_dist.log_prob(torch.ones(9700).double())

