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

# Models
from models.sgpr import SparseGPR
#from models.svgp import StochasticVariationalGP
#from models.bayesian_svgp import BayesianStochasticVariationalGP
from models.bayesian_sgpr_hmc import BayesianSparseGPR_HMC

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

    torch.manual_seed(57)

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
    Z_init = torch.randn(10)

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
            losses, trace_hyper = model.train_model(optimizer, max_steps=5000, break_for_hmc=[100,200,500,1000,1900,2000,3000,4000,4900])
            Y_test_pred_list = model.posterior_predictive(X_test)
        else:
            losses = model.train_model(optimizer, max_steps=5000)
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

    # SGPR w HMC

    plt.subplot(132)
    visualise_posterior(model_dict['BayesianSGPR_HMC'], X_test, Y_test, Y_test_pred_list, mixture=False, title='SGPR + HMC', new_fig=False)

    plt.subplot(133)
    plt.plot(losses_dict['SparseGPR'], label='SGPR')
    plt.plot(losses_dict['BayesianSGPR_HMC'], label='SGPR + HMC')
    plt.title('Neg. ELBO Loss')
    plt.legend(fontsize='small')

    plt.figure(figsize=(14,4))

    # Samples

    sgpr = model_dict['SparseGPR']
    hmc = model_dict['BayesianSGPR_HMC']

    from models.bayesian_sgpr_hmc import mixture_posterior_predictive
    
    Y_test_pred_list = mixture_posterior_predictive(model, X_test, trace_hyper)
    
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

    # import pickle as pkl

    # with open('pre_trained_models/hmc.pkl', 'wb') as file:
    #     pkl.dump((hmc.state_dict(), likelihood.state_dict()), file)

    #Compute metrics
    
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
