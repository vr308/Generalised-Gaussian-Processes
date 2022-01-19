#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVGP Model gpytorch
Reference: Hensman et al. 2015 Scalable Variational Gaussian process classification.

"""

import gpytorch as gpytorch
import torch as torch
import numpy as np
from math import floor
from tqdm import tqdm
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, MultitaskVariationalStrategy
from gpytorch.means import ZeroMean
from torch.utils.data import TensorDataset, DataLoader
from utils.metrics import rmse, nlpd

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14)

class StochasticVariationalGP(ApproximateGP):

    """ The sparse GP class for regression with the uncollapsed stochastic bound.
         The parameters of q(u) \sim N(m, S) are learnt numerically.
    """

    def __init__(self, train_x, train_y, likelihood, Z_init):

        # Locations Z corresponding to u, they can be randomly initialized or
        # regularly placed.
        self.inducing_inputs = Z_init
        self.num_inducing = len(Z_init)
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(self.num_inducing)
        
        if Z_init.shape[1] > 1: ## we are probably doing multi-class classification
            base_variational_strategy = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
            q_f = MultitaskVariationalStrategy(base_variational_strategy, num_tasks=3)
        else:
            q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(StochasticVariationalGP, self).__init__(q_f)
        self.likelihood = likelihood
        self.train_x = train_x
        self.train_y = train_y

        self.mean_module = ZeroMean()
        #self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_inducing_prior(self):

        Kmm = self.covar_module._inducing_mat
        return torch.distributions.MultivariateNormal(ZeroMean(), Kmm)

    def elbo(self, output, y):

        Knn = self.base_covar_module(self.train_x).evaluate()
        Knm = self.base_covar_module(self.train_x, self.Z_init).evaluate()
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

    def train_model(self, optimizer, train_loader, minibatch_size=100, num_epochs=25, combine_terms=True):

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=len(self.train_y))

        losses = []
        for i in range(num_epochs):
            with tqdm(train_loader, unit="batch", leave=True) as minibatch_iter:
                for x_batch, y_batch in minibatch_iter:
                      
                      self.train()
                      self.likelihood.train()
                      
                      minibatch_iter.set_description(f"Epoch {i}")
                      
                      optimizer.zero_grad()
                      
                      output = self(x_batch)
                      if combine_terms:
                          loss = -mll(output, y_batch)
                      else:
                          loss = -self.elbo(output, y_batch)
                      losses.append(loss.item())
                      loss.backward()
                      
                      #if i%100 == 0:
                      #          print('Iter %d/%d - Loss: %.3f  outputscale: %.3f  lengthscale: %s   noise: %.3f' % (
                      #          i + 1, 1000, loss.item(),
                      #          self.covar_module.outputscale.item(),
                      #          self.covar_module.base_kernel.lengthscale,
                      #          self.likelihood.noise.item()))
                      
                      optimizer.step()
                      
                      ## Training metrics at end of each epoch
                      #Y_pred = self.posterior_predictive(x_batch)
                      #train_rmse = rmse(Y_pred.loc, y_batch, y_batch.std())
                      #train_nlpd = nlpd(Y_test_pred, Y_test, Y_std)
                      #minibatch_iter.set_postfix(loss=loss.item(), rmse=train_rmse.item())
                      minibatch_iter.set_postfix(loss=loss.item())
        return losses

    def optimization_trace(self):
        return;

    def posterior_predictive(self, test_x):

        ''' Returns the posterior predictive multivariate normal '''

        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_star = self.likelihood(self(test_x))
        return y_star


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
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Initial inducing points
    index_inducing = np.random.randint(0,len(train_x), 12)
    Z_init = train_x[index_inducing]
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = StochasticVariationalGP(train_x, train_y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train
    losses = model.train_model(optimizer, train_loader, 
                              minibatch_size=100, num_epochs=200,  combine_terms=True)
    
    ####
    # Test 
    test_x = torch.linspace(-8, 8, 1000)
    test_y = func(test_x)
    
    #Y_train_pred = model.posterior_predictive(train_x)
    test_pred = model.posterior_predictive(test_x)
    
    import matplotlib.pylab as plt
    
    plt.figure()
    plt.plot(test_x, test_y)
    plt.plot(train_x, train_y, 'bo')
    plt.plot(test_x, test_pred.loc)
    #plt.scatter(model.inducing_inputs.detach(), [-2.0]*model.num_inducing, c='r', marker='x', label='Inducing')
    plt.scatter(model.variational_strategy.inducing_points.detach(), [-2.0]*model.num_inducing, c='g', marker='x', label='Inducing')

    # # Compute metrics
    
    rmse_test = rmse(test_pred.loc, test_y, torch.tensor([1.0]))
    nll_test = nlpd(test_pred, test_y, torch.tensor([1.0]))
    
    print('Test RMSE: ' + str(rmse_test))
    print('Test NLPD: ' + str(nll_test))
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #from utils.experiment_tools import get_dataset_class
    
    #dataset = get_dataset_class('Boston')(split=0, prop=0.8)
    #X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
    
    #num_inducing = 100
    # N = 1000  # Number of training observations

    # X = torch.randn(N) * 2 - 1  # X values
    # Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

    # train_n = int(np.floor(0.8 * len(X)))
    # train_x = X[:train_n][:,None]
    # train_y = Y[:train_n].contiguous()

    # test_x = X[train_n:][:,None]
    # test_y = Y[train_n:].contiguous()

    # train_dataset = TensorDataset(X_train, Y_train)
    # train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

    # #test_dataset = TensorDataset(X_test, Y_test)
    # #test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # # Initial inducing points
    # Z_init = X_train[np.random.randint(0,len(X_train), num_inducing)]

    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # model = StochasticVariationalGP(X_train, Y_train, likelihood, Z_init).double()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # # Train
    # losses = model.train_model(optimizer, train_loader,
    #                             minibatch_size=100, num_epochs=200,  combine_terms=True)
    
    
    # Y_train_pred = model.posterior_predictive(X_train)
    # Y_test_pred = model.posterior_predictive(X_test)

    # ### Compute Metrics  ###########
    
    # rmse_train = np.round(rmse(Y_train_pred.loc, Y_train, dataset.Y_std).item(), 4)
    # rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
   
    # ### Convert everything back to float for Naval 
    
    # nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
    # nlpd_test = np.round(nlpd(Y_test_pred, Y_test, dataset.Y_std).item(), 4)


    # Test
    # test_x = torch.linspace(-8, 8, 1000)
    # test_y = func(test_x)

    # y_star = model.posterior_predictive(test_x)

    # # Visualise
    # # Compute metrics
    # rmse = model.rmse(y_star, test_y)
    # nll = model.neg_test_log_likelihood(y_star, test_y)


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

