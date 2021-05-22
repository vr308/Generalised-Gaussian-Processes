#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computation of metrics 

"""

import torch 
from prettytable import PrettyTable
import numpy as np
import scipy.stats as st

def get_trainable_param_names(model):
    
    ''' Prints a list of parameters (model + variational) which will be 
    learnt in the process of optimising the objective '''
    
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
        

def neg_test_log_likelihood(model, y_star, test_y):
     
      lpd = y_star.log_prob(test_y)
      # return the average
      return -torch.mean(lpd).detach()
 
def rmse(y_star, test_y):
     
      return torch.sqrt(torch.mean((y_star.loc - test_y)**2)).detach()
  
def posterior_predictive_samples(post_mean, post_cov):
    
    return np.random.multivariate_normal(post_mean, post_cov, 20)


def log_predictive_density(predictive_density):

      return np.round(np.sum(np.log(predictive_density)), 3)

def log_predictive_mixture_density(f_star, list_means, list_cov):
      
      components = []
      for i in np.arange(len(list_means)):
            components.append(st.multivariate_normal.pdf(f_star, list_means[i].eval(), list_cov[i].eval(), allow_singular=True))
      return np.round(np.sum(np.log(np.mean(components))),3)