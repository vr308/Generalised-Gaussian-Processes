#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2d syntehtic experiment to demonstrate SGPR vs Bayesian SGPR

"""

import torch 
import gpytorch 
import numpy as np
import matplotlib.pylab as plt
from models.sgpr import SparseGPR
from models.bayesian_sgpr_hmc import BayesianSparseGPR_HMC
from utils.metrics import rmse, nlpd, nlpd_mixture
#from gpytorch.priors import MultivariateNormalPrior, NormalPrior

def camel_back_function(x: np.array):
    
    """ Returns the function as defined:

    :param x: Nx2 set of x coordinates
    :return: Nx1 array of y values
    
    """
    x1 = x[:, 0]
    x2 = x[:, 1]    
    y = (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
    return y.reshape(len(x),)/torch.std(y)

def get_training_dataset(X, Y, train_index, noise_scale):
    
    train_sizes = [30,100,200,400]   
    X1, X2, X3, X4 = [X[train_index[0:train_sizes[i]]] for i in np.arange(4)]
    Y1, Y2, Y3, Y4 = [Y[train_index[0:train_sizes[i]]] for i in np.arange(4)]
    
    Y1 = Y1 + torch.randn(len(Y1))*noise_scale
    return (X1, X2, X3, X4), (Y1, Y2, Y3, Y4)

def get_grid_mesh():
    
    h_x = 0.05
    h_y = 0.05
    x_min, x_max = -2,2.05  
    y_min, y_max = -1,1.05
        
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h_x),
                                torch.arange(y_min, y_max, h_y))
    return xx, yy

def get_grid_2d_points(xx, yy):
    
    X = torch.vstack((xx.reshape(-1), yy.reshape(-1))).T    
    Y = camel_back_function(X)
    return X, Y

def plot_contour_camel_back(train_index, train_sizes):
    
    h_x = 0.01
    h_y = 0.01
    x_min, x_max = -2,2  
    y_min, y_max = -1,1
        
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x),
                                np.arange(y_min, y_max, h_y))
    X = np.vstack((xx.reshape(-1), 
                               yy.reshape(-1))).T    
    Y = camel_back_function(X)
      
    plt.figure(figsize=(12,5))
    
    for i in np.arange(4):
        
        plt.subplot(1,4,i+1)
        plt.contourf(xx, yy, Y.reshape(xx.shape[0], xx.shape[1]), levels=50, cmap=plt.get_cmap('jet'))
        indices = train_index[0:train_sizes[i]]
        plt.scatter(X[indices][:,0], X[indices][:,1], marker='+', color='k')
        plt.axis('off')
            
if __name__ == "__main__":
    
    xx, yy = get_grid_mesh()
    train_index = np.random.randint(0,xx.shape[0]*xx.shape[1],3000)

    X, Y = get_grid_2d_points(xx, yy)
  
    # Original function
    
    plt.figure()
    plt.contourf(xx, yy, Y.reshape(xx.shape[0], xx.shape[1]), levels=50, cmap=plt.get_cmap('jet'))

    ##### Training data and initial inducing points
    
    (X1,X2,X3,X4), (Y1,Y2,Y3,Y4) = get_training_dataset(X,Y,train_index,1)
    
    X_train = X3
    Y_train = Y3
    
    num_inducing = [20]
    
    model_name = 'SGPR'
    
    if model_name == 'SGPR':
    
        plt.figure(figsize=(12,4))
        
        rmse_ = []
        nlpd_ = []
    
        for i in range(len(num_inducing)):
            
            #Z_init = X_train[np.random.randint(0, len(X_train), num_inducing[i])]
            Z_init = torch.randn(num_inducing[i], 2)
                
            ### Initialize likelihood and model
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = 1e-4  # Some small value, but don't make it too small or numerical performance will suffer. I recommend 1e-4.
            #likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.
            
            model_sgpr = SparseGPR(X_train, Y_train, likelihood, Z_init)
            optimizer = torch.optim.Adam(model_sgpr.parameters(), lr=0.01)
            losses = model_sgpr.train_model(optimizer, max_steps=200)
            
            ## Extract inducing locations and posterior mean
            f_preds = model_sgpr.posterior_predictive(X)
            Z_opt = model_sgpr.inducing_points
            
            ##### Visualisation 
            
            plt.subplot(1,3,i+1)
            plt.contourf(xx, yy, f_preds.loc.reshape(81,41).detach().cpu(), levels=50, cmap=plt.get_cmap('jet'))
            plt.scatter(X_train[:,0].numpy(), X_train[:,1].numpy(), c='k', marker='x')
            plt.scatter(Z_opt[:,0].numpy(), Z_opt[:,1].numpy(), c='cyan', marker='o')
            plt.xlim(-2,2)
            plt.ylim(-1,1)
            plt.title('M='+str(num_inducing[i]), fontsize='small')
    
            # ##### Metrics: RMSE and NLPD
            
            test_pred_mean = f_preds.loc.detach().cpu()
            rmse_.append(rmse(test_pred_mean, Y, torch.tensor([1.0])))
            nlpd_.append(nlpd(f_preds, Y, torch.tensor([1.0])))
        
        plt.suptitle('SGPR with random initialisation', fontsize='small')
        
    else:
    
        ###### SGPR w. HMC
                 
        plt.figure(figsize=(12,4))
        
        rmse_hmc = []
        nlpd_hmc = []
    
        for i in range(len(num_inducing)):
            
            #Z_init = X_train[np.random.randint(0, len(X_train), num_inducing[i])]
            Z_init = torch.randn(num_inducing[i], 2)
            #Z_init = torch.randn(33, 2)

                
            ### Initialize likelihood and model
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = 1e-4  # Some small value, but don't make it too small or numerical performance will suffer. I recommend 1e-4.
            #likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.
            
            model_hmc = BayesianSparseGPR_HMC(X_train, Y_train, likelihood, Z_init)
            optimizer = torch.optim.Adam(model_hmc.parameters(), lr=0.01)
            
            hmc_scheduler = np.arange(10,1500,100)
            losses = model_hmc.train_model(optimizer, max_steps=150, hmc_scheduler=hmc_scheduler)
            
            ## Extract inducing locations and posterior mean
            f_preds = model_hmc.posterior_predictive(X)
            Z_opt = model_hmc.inducing_points
            
            ##### Visualisation 
            
            plt.subplot(1,3,i+1)
            plt.contourf(xx, yy, f_preds.loc.reshape(81,41).detach().cpu(), levels=50, cmap=plt.get_cmap('jet'))
            plt.scatter(X_train[:,0].numpy(), X_train[:,1].numpy(), c='k', marker='x')
            plt.scatter(Z_opt[:,0].numpy(), Z_opt[:,1].numpy(), c='r', marker='o')
            plt.xlim(-2,2)
            plt.ylim(-1,1)
            plt.title('M='+str(num_inducing[i]), fontsize='small')
    
            # ##### Metrics: RMSE and NLPD
            
            test_pred_mean = f_preds.loc.detach().cpu()
            rmse_hmc.append(rmse(test_pred_mean, Y, torch.tensor([1.0])))
            nlpd_hmc.append(nlpd(f_preds, Y, torch.tensor([1.0])))
        
        plt.suptitle('SGPR with HMC', fontsize='small')
    
        
       
