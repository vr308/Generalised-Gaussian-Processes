#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Demo 

Baseline models: Canonical SGPR, SVGP
New: BayesianSVGP and BayesianSGPR_with_HMC

"""
import numpy as np
import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader

# Models
from models.sgpr import SparseGPR
from models.svgp import StochasticVariationalGP
from models.bayesian_svgp import BayesianStochasticVariationalGP
from models.bayesian_sgpr_hmc import BayesianSparseGPR_HMC

# Metrics and Viz
from utils.metrics import *
from utils.visualisation import *

gpytorch.settings.cholesky_jitter(float=1e-5)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 3.14) 

titles = ['SparseGPR', 'SVGP', 'BayesianSVGP', 'BayesianSGPR_HMC']
model_list = [
            SparseGPR,
            StochasticVariationalGP,
            BayesianStochasticVariationalGP,
            BayesianSparseGPR_HMC
        ]

if __name__ == '__main__':
    
    torch.manual_seed(57)
    
    N = 1000  # Number of training observations

    X = torch.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values
    
    train_dataset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    
    test_dataset = TensorDataset(X, Y)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
            
    # Initial inducing points
    Z_init = torch.randn(25)
    
    #for m in range(len(model_list)): 
    m = 3    
    print('Training with model ' + f'{model_list[m]}')
    
    # Initialise model and likelihood
    model_class = model_list[m]
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_class(X[:,None], Y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if titles[m][-4:] == 'SVGP':
        losses = model.train_model(optimizer, train_loader, minibatch_size=100, num_epochs=25, combine_terms=True)     
    else:
        losses = model.train_model(optimizer)

        # Test 
        #test_x = torch.linspace(-8, 8, 1000)
        #test_y = func(test_x)
        
        #y_star = model.posterior_predictive(test_x)
        
        # Visualise 
        
        #titles[m]
        #visualise_posterior(model, test_x, y_star, title)
        
        # Compute metrics
        #rmse = rmse(model, y_star, test_y)
        #nll = neg_test_log_likelihood(model, y_star, test_y)
