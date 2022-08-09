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
from utils.posterior_predictive import get_posterior_predictive_means_stds, get_posterior_predictive_mean
import gpflow
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import tensorflow_probability as tfp
# Models
from models.sgpr import SparseGPR
from models.bayesian_sgpr_hmc import BayesianSparseGPR_HMC, mixture_posterior_predictive
from models.sgp_hmc import train_sgp_hmc, predict_sgpmc

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

# Metrics and Viz
#from utils.metrics import rmse, nlpd, get_trainable_param_names
from utils.visualisation import visualise_posterior, visualise_mixture_posterior_samples
import matplotlib.pyplot as plt

gpytorch.settings.cholesky_jitter(float=1e-4)
plt.style.use('seaborn-muted')

### Set seed for reproducibility
torch.manual_seed(37)
np.random.seed(45)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 3.14)

titles = ['SparseGPR','BayesianSGPR_HMC'] #'StochasticVariationalGP', 'BayesianStochasticVariationalGP']
model_list = [SparseGPR,
            BayesianSparseGPR_HMC]
            #StochasticVariationalGP,
            #BayesianStochasticVariationalGP]


if __name__ == '__main__':

    torch.manual_seed(45)

    N = 1000  # Number of training observations

    X = torch.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.4 * torch.randn(N)  # Noisy Y values

    train_index = np.where((X < -2) | (X > 2))

    X_train = X[train_index][:,None]
    Y_train = Y[train_index]

    ## Test
    X_test = torch.linspace(-8, 8, 1000)
    Y_test = func(X_test)

    # Initial inducing points
    Z_init = torch.randn(25)

    losses_dict = {}
    model_dict =  {}

    for m in range(len(model_list)):
        
        print('Training with model ' + f'{model_list[m]}')

        # Initialise model and likelihood
        model_class = model_list[m]

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = model_class(X_train, Y_train, likelihood, Z_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


        #if titles[m][-4:] == 'SVGP':
        #    losses = model.train_model(optimizer, train_loader, minibatch_size=100, num_epochs=50, combine_terms=True)
        #else:
        if titles[m][-3:] == 'HMC':
            losses, trace_hyper, step_sizes, perf_times = model.train_model(optimizer, max_steps=2000, hmc_scheduler=[100,200,500,1000,1500,1999])
            Y_test_pred_list = mixture_posterior_predictive(model, X_test, trace_hyper)
        else:
            losses = model.train_model(optimizer, max_steps=2000)
            Y_test_pred = model.posterior_predictive(X_test)

        # save losses
        losses_dict[titles[m]] = losses
        model_dict[titles[m]] = model


    # Visualisation of synthetic example

    plt.style.use('seaborn')

    plt.figure(figsize=(14,4))

    # SGPR
    plt.subplot(131)
    visualise_posterior(model_dict['SparseGPR'], X_test, Y_test, Y_test_pred, mixture=False, title='SGPR', new_fig=False)
    plt.legend(fontsize='small')
    # SGPR w HMC

    plt.subplot(132)
    visualise_posterior(model_dict['BayesianSGPR_HMC'], X_test, Y_test, Y_test_pred_list, mixture=True, title='SGPR + HMC', new_fig=False)

    
    from utils.metrics import rmse, nlpd, nlpd_mixture
    
    sample_means, sample_stds = get_posterior_predictive_means_stds(Y_test_pred_list)
    rmse_sgpr = rmse(Y_test_pred.loc, Y_test, torch.tensor([1.0]))
    rmse_sgpr_hmc = rmse(get_posterior_predictive_mean(sample_means), Y_test,torch.tensor([1.0]))

    print('RMSE SGPR ' + str(rmse_sgpr))
    print('RMSE SGPR HMC ' + str(rmse_sgpr_hmc))
    
    nll_sgpr = nlpd(Y_test_pred, Y_test, torch.tensor([1.0]))
    nll_hmc = nlpd_mixture(Y_test_pred_list, Y_test, torch.tensor([1.0]))
    
    print('NLPD SGPR ' + str(nll_sgpr))
    print('NLPD SGPR HMC ' + str(nll_hmc))
    
    ##### TF Model
    
    plt.subplot(133)
    
    ## train and plot joint hmc
    # Train
    X_train = X[train_index][:,None]
    Y_train = Y[train_index][:,None]
    
    ## Test
    X_test = f64(tf.linspace(-8.0, 8.0, 1000)[:,None])
    Y_test = f64(func(X_test))
    
    data = (X_train, Y_train)
    data_test = (X_test, Y_test)
    
    ## Train model
    hmc = model_dict['BayesianSGPR_HMC']
    Z_init = np.array(hmc.covar_module.inducing_points.detach())
    model, hmc_helper, samples, wall_clock_secs = train_sgp_hmc(data, Z_init, input_dims=X_train.shape[-1], tune=500, num_samples=500)
    
    ## Predictions
    
    pred_mean, y_pred_dists, lower, upper = predict_sgpmc(model, hmc_helper, samples, X_test)
    
    # ## Visualisation
    
    # # lower=lower.detach().numpy()
    # # upper=upper.detach().numpy()
    # # pred_mean = Y_test_pred.mean.numpy()
    plt.subplot(133)
    plt.plot(X_test.numpy(), pred_mean, 'b-', label='Mean')
    plt.scatter(Z_init, [-2.5]*25, c='r', marker='x', label='Inducing')
    plt.fill_between(X_test[:,0], lower, upper, alpha=0.5, label=r'$\pm$2\sigma', color='g')
    plt.scatter(X_train[:,0], Y_train[:,0], c='k', marker='x', alpha=0.7, label='Train')
    plt.plot(X_test[:,0], Y_test[:,0], color='b', linestyle='dashed', alpha=0.7, label='True')
    plt.title('Joint HMC')
    
    # plt.plot(losses_dict['SparseGPR'], label='SGPR')
    # plt.plot(losses_dict['BayesianSGPR_HMC'], label='SGPR + HMC')
    # plt.title('Neg. ELBO Loss')
    # plt.legend(fontsize='small')

    plt.figure(figsize=(14,4))

    # Samples
    sgpr = model_dict['SparseGPR']
    hmc = model_dict['BayesianSGPR_HMC']

    from models.bayesian_sgpr_hmc import mixture_posterior_predictive
    
    Y_test_pred_list = mixture_posterior_predictive(hmc, X_test, trace_hyper)
    
    plt.subplot(131)
    visualise_mixture_posterior_samples(hmc, X_test, Y_test_pred_list, title='GP Mixture Samples', new_fig=False)
    plt.title('SGPR + HMC')

    plt.subplot(132)
    plt.hist(trace_hyper['ls'], bins=25)
    plt.axvline(sgpr.base_covar_module.base_kernel.lengthscale.detach(), color='r', label='ML-II')
    plt.title('Lengthscale identification', fontsize='medium')
    plt.legend()

    plt.subplot(133)
    plt.hist(trace_hyper['sig_n'], bins=25)
    plt.axvline(sgpr.likelihood.noise.detach(), color='r', label='ML-II')
    plt.axvline(x=0.4, label='Truth', c='k', linestyle='--')
    plt.title('Noise sd identification', fontsize='medium')
    plt.legend()

    #Compute metrics
    
    joint_hmc = rmse(torch.tensor(pred_mean), np.array(Y_test), torch.tensor([1.0]))
    print('RMSE Joint_HMC ' + str(joint_hmc))

    nll_joint_hmc = nlpd_mixture(y_pred_dists, torch.tensor(np.array(Y_test).squeeze()), torch.tensor([1.0]))
    print('NLPD Joint HMC ' + str(nll_joint_hmc))

