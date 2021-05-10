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

# Models
from models.sgpr import SparseGPR
from models.svgp import StochasticVariationalGP
from models.bayesian_svgp import BayesianStochasticVariationalGP
from models.bayesian_sgpr_hmc import DoublyCollapsedSparseGPR

# Metrics and Viz
from utils.metrics import *
from utils.visualisation import *

gpytorch.settings.cholesky_jitter(float=1e-5)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 3.14) 

titles = ['SparseGPR', 'SVGP', 'BayesianSVGP', 'BayesianSGPR_HMC']
models = [
            SparseGPR,
            StochasticVariationalGP,
            BayesianStochasticVariationalGP,
            DoublyCollapsedSparseGPR
        ]
model_map = map(titles, numbers)

if __name__ == '__main__':
    
    torch.manual_seed(57)
    
    N = 1000  # Number of training observations

    X = torch.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

    # Initial inducing points
    Z_init = torch.randn(25)
    
    # Initialise model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in model_list: 
        model = SparseGPR(X, Y, likelihood, Z_init)
    
        # Train
        losses = model.train_model(likelihood, optimizer, combine_terms=True)
    
        # Test 
        test_x = torch.linspace(-8, 8, 1000)
        test_y = func(test_x)
        
        y_star = model.posterior_predictive(test_x)
        
        # Visualise 
        
        title 
        visualise_posterior(model, test_x, y_star, title)
        
        # Compute metrics
        rmse = rmse(model, y_star, test_y)
        nll = neg_test_log_likelihood(model, y_star, test_y)
