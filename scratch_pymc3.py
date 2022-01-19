#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy classification 

SVGP and BayesianSVGP -> gpytorch
Full HMC model -> pymc3 GP classification (generalised likelihood)
SGPMC -> gpflow? 

"""

import pymc3 as pm
import gpytorch 
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models.svgp import StochasticVariationalGP
from models.bayesian_svgp import BayesianStochasticVariationalGP
import matplotlib.pyplot as plt
import scipy.stats as st
import pymc3 as pm

if __name__ == '__main__':

  ## Load Banana dataset
  
  from utils.dataset import get_classification_data
  from utils.metrics import rmse, nlpd
  
  dataset = get_classification_data('banana')
  X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()

  #### SVGP model 
  
  num_inducing = 15
    
  train_dataset = TensorDataset(X_train, Y_train)
  train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    
  test_dataset = TensorDataset(X_test, Y_test)
  test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

  # Initial inducing points
  Z_init = X_train[np.random.randint(0,len(X_train), num_inducing)]

  likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=dataset.K, mixing_weights=False)
  model = StochasticVariationalGP(X_train, Y_train, likelihood, Z_init).double()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
  # Train
  losses = model.train_model(optimizer, train_loader,
                                minibatch_size=100, num_epochs=50,  combine_terms=True)
     
   #Y_train_pred = model.posterior_predictive(X_train)
   f_test_pred = model.posterior_predictive(X_test)
   

  # ### Compute Metrics  ###########
  
  # rmse_train = np.round(rmse(Y_train_pred.loc, Y_train, dataset.Y_std).item(), 4)
  # rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
 
  # ### Convert everything back to float for Naval 
  
  # nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
  # nlpd_test = np.round(nlpd(Y_test_pred, Y_test, dataset.Y_std).item(), 4)
  
  #################
  
  torch.manual_seed(45)
  
  from utils.experiment_tools import get_dataset_class
  from utils.metrics import rmse, nlpd_mixture, nlpd

  dataset = get_dataset_class('Yacht')(split=0, prop=0.8)
  X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
  
  Z_opt = X_train[np.random.randint(0,len(X_train), 50)]
  
  with pm.Model() as model:
        
        ls = pm.Gamma("ls", alpha=2, beta=1)
        sig_f = pm.HalfCauchy("sig_f", beta=1)
    
        cov = sig_f ** 2 * pm.gp.cov.ExpQuad(6, ls=ls)
        gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")
            
        sig_n = pm.HalfCauchy("sig_n", beta=1)
        Z_opt = pm.Flat("Xu", shape=(50,6))

        # Z_opt is the intermediate inducing points from the optimisation stage
        y_ = gp.marginal_likelihood("y", X=X_train.numpy(), Xu=Z_opt, y=Y_train.numpy(), noise=sig_n)
    
        trace = pm.sample(100, tune=100, chains=1)
        mp = pm.find_MAP()
        
   with model:
       
    f_pred = gp.conditional("f_pred", X_test)
    
  with model:
      
    pred_samples = pm.sample_posterior_predictive(trace, var_names=['f_pred'], samples=1000)
