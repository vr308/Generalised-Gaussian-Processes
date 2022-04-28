#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computation of metrics

"""

import torch
import scipy.stats as st
import numpy as np
from prettytable import PrettyTable
import scipy.stats as stats

def print_trainable_param_names(model):

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


def get_trainable_param_names(model):

    ''' Returns a list of names for the model which are learnable '''
    grad_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        grad_params.append(name)
    return grad_params

def rmse(Y_pred_mean, Y_test, Y_std):

      return Y_std.item()*torch.sqrt(torch.mean((Y_pred_mean - Y_test)**2)).detach()
  
def nlpd(Y_test_pred, Y_test, Y_std):
    
      lpd = Y_test_pred.log_prob(Y_test)
      # return the average
      avg_lpd_rescaled = lpd.detach()/len(Y_test) - torch.log(torch.Tensor(Y_std))
      return -avg_lpd_rescaled
  
def nlpd_marginal(Y_test_pred, Y_test, Y_std):
    
    test_nlpds_per_point = []
    Y_test = Y_test.cpu().numpy()
    means = Y_test_pred.loc.detach().cpu().numpy()
    var = Y_test_pred.covariance_matrix.diag().detach().cpu().numpy()
    for i in np.arange(len(Y_test)):
        pp_lpd = stats.norm.logpdf(Y_test[i], loc = means[i], scale = np.sqrt(var[i])) - np.log(Y_std)
        test_nlpds_per_point.append(pp_lpd)
    return -np.mean(test_nlpds_per_point)
    
  
def nlpd_mixture(Y_test_pred_list, Y_test, Y_std):
      
      components = []
      for i in np.arange(len(Y_test_pred_list)):
            test_pred = Y_test_pred_list[i]
            components.append(nlpd(test_pred, Y_test, Y_std).detach().item())
      return np.mean(components)

def negative_log_predictive_mixture_density(Y_test, y_mix_loc, y_mix_std, Y_std):
      
      lppd_per_point = []
      for i in np.arange(len(Y_test)):
            components = []
            for j in np.arange(len(y_mix_loc)):
                  components.append(st.norm.logpdf(Y_test[i], y_mix_loc[:,i][j], y_mix_std[:,i][j]) - np.log(Y_std))
            lppd_per_point.append(np.mean(components))
      return -np.round(np.mean(lppd_per_point),3)

# def nlpd_mixture(Y_test, mix_means, mix_std, num_mix):
      
#       mix = torch.distributions.Categorical(torch.ones(num_mix,))
    
#       lppd_per_point = []
#       for i in np.arange(len(Y_test)):
#             comp = torch.distributions.Normal(torch.tensor([mix_means[:,i]]), torch.tensor([mix_std[:,i]]))
#             gmm = torch.distributions.MixtureSameFamily(mix, comp)
#             lppd_per_point.append(gmm.log_prob(Y_test[i]).item())
#       return -np.round(np.mean(lppd_per_point),3)
    