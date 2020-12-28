#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

Data loading utilities - synthetic and benchmark datasets

"""
import numpy as np
import pandas as pd
import torch
import sys
import gpytorch
import scipy.stats as st
import matplotlib.pyplot as plt

#reproducibility
seed = 4321
np.random.seed(seed)
torch.manual_seed(seed)

class KernelConfig(object):
    
    ''' Config class for declaring kernels for drawing ground truth functions from'''
    
    def __init__(self,
               input_dim: int = 1,
               inputs: torch.Tensor = torch.linspace(0,10,200),
               kernel_func: gpytorch.kernels = gpytorch.kernels.RBFKernel(input_dim=1),
               **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.inputs = inputs
        self.kernel_func = kernel_func
        for item in kwargs:
            setattr(self.kernel_func, item, kwargs[item])
        self.K = kernel_func(inputs)

class DataConfig(object):
    
    ''' Config class for generating 1d or 2d synthetic data'''
    
    def __init__(self,
               n_train: int = 40,
               input_distribution: str = 'uniform',
               cluster_centers: tuple = (2,7),
               likelihood = 'gaussian',
               train_sigma: bool = False,
               noise_var: float = 1,
               classes = 1):
        super().__init__()
        self.n_train = n_train
        self.input_distribution = input_distribution   
        self.cluster_centers = cluster_centers
        self.likelihood = likelihood
        self.train_sigma = train_sigma
        self.noise_var = noise_var
        self.classes = classes
    
# link function
def invlogit(x: float, eps: float = sys.float_info.epsilon):
      return (1.0 + 2.0 * eps) / (1.0 + torch.exp(-x)) + eps
  
def probit(x: float):
    return gpytorch.distributions.base_distributions.Normal(0,1).cdf(torch.Tensor([x]))
  
def load_1d_synthetic(data_config, kernel_config):
    
    #TODO: can generate multiple datasets by drawing multiple latent processes from same prior
     
    ''' Extracting data and kernel details from the data / kernel config to 
    generate outputs y from a synthetic 1d function drawn from a GP prior''' 
    
    X = kernel_config.inputs
    n = len(X)
    K = kernel_config.K
    mean = torch.zeros(X.shape[0])
    mvn = gpytorch.distributions.MultivariateNormal(mean, K + 1e-6 * torch.eye(n))
    latent_f = mvn.sample(sample_shape=torch.Size([1])).flatten()
    
    if data_config.input_distribution == 'uniform':
            X_train = np.random.choice(X, data_config.n_train, replace=False)
    else:
            pdf = 0.5*st.norm.pdf(X, data_config.cluster_centers[0], 0.5) 
            + 0.3*st.norm.pdf(X, data_config.cluster_centers[1], 1)
            prob = pdf/np.sum(pdf)
            X_train = np.random.choice(X, data_config.n_train, replace=False,  p=prob.ravel())
    
    if data_config.likelihood == 'gaussian':
        if data_config.train_sigma:
            noise = torch.randn(X.shape[0])*data_config.noise_var
            y = latent_f + noise
        else:
            y = latent_f
    elif data_config.likelihood == 'binary':
        y = torch.bernoulli(input=torch.Tensor(invlogit(latent_f)))
    elif data_config.likelihood == 'poisson':
        y = torch.poisson(input=torch.Tensor(torch.exp(latent_f)))
    elif data_config.likelihood == 'multi-class':
         # Need to draw latent function per class
          latent_f = mvn.sample(sample_shape=torch.Size([data_config.classes])).T
          y = torch.argmax(latent_f, 1)
    else: 
        print('Please specify one out of binary / multi-class / poisson')
        return None

    train_index = []
    for i in X_train:
          #print(X.tolist().index(i))
          train_index.append(X.tolist().index(i))
         
    f_train = latent_f[train_index]
    y_train = y.flatten()[train_index]
    
    test_index = list(set(np.arange(len(X))) - set(train_index))
    X_test = X[test_index]
    f_test = latent_f[test_index]
    y_test = y.flatten()[test_index]
    
    return X_train, y_train, X_test, y_test, f_train, f_test, train_index, test_index

# def load_2d_synthetic(X_all, f_all, n_train, uniform, seed):
    
#        ''' Extracting data and kernel details from the data config to 
#        generate outputs y from a synthetic 2d function drawn from a GP prior''' 
    
#       h_x = 0.2
#       h_y = 0.2
#       x_min, x_max = 0.5, 15 
#       y_min, y_max = 0.5,  15
#       xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x),
#                          np.arange(y_min, y_max, h_y))
#       X_eval = np.vstack((xx.reshape(-1), 
#                         yy.reshape(-1))).T
    
#     if uniform == True:
#              train_index = np.random.choice()
#              X = np.random.choice(X_all.ravel(), n_train, replace=False)
#     else:
#              pdf = 0.5*st.norm.pdf(X_all, 1, 0.5) + 0.3*st.norm.pdf(X_all, 4.5, 1)
#              prob = pdf/np.sum(pdf)
#              X = np.random.choice(X_all.ravel(), n_train, replace=False,  p=prob.ravel())
#     train_index = []
#     for i in X:
#           train_index.append(X_all.ravel().tolist().index(i))
#     X = X_all[train_index]
#     f = f_all[train_index]
#     y = pm.Bernoulli.dist(p=invlogit(f)).random()
#     return X, y, f, train_index

def load_coal():
    
    return

def load_breast_cancer():
    
    return

def load_pine_saplings():
    
    return
  
if __name__== "__main__":

    dconfig_1d = DataConfig(input_distribution = 'uniform', n_train = 40, likelihood='multi-class', train_sigma=True, classes=3, noise_var=0.1)
    kconfig_1d = KernelConfig(lengthscale=0.5)
    X_train, y_train, X_test, y_test, f_train, f_test, train_index, test_index = load_1d_synthetic(dconfig_1d, kconfig_1d)

    plt.plot(X_test, y_test, 'o')
    plt.plot(X_train, y_train, 'o', color='r')