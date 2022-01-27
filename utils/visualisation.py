#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualisation utilities for data and experiments

"""
import matplotlib.pylab as plt
import torch
from utils.posterior_predictive import get_posterior_predictive_means_stds, get_posterior_predictive_uncertainty_intervals

def visualise_mixture_posterior_samples(model, X_test, Y_test_pred_list, title=None, new_fig=False):

    ''' Visualising posterior predictive samples'''
    
    if new_fig:
        plt.figure(figsize=(5,5))
    #f, ax = plt.subplots(1, 1, figsize=(8, 8))
    for y_star in Y_test_pred_list[::2]:
        plt.plot(X_test.numpy(), y_star.mean.detach().numpy(), c='magenta', alpha=0.2)
        #plt.plot(model.covar_module.inducing_points.detach(), [-2.5]*model.num_inducing, 'rx')
        lower, upper = y_star.confidence_region()
        plt.fill_between(X_test.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.1, color='blue')
    plt.legend(['Mean', r'$\pm$2\sigma'])


def visualise_posterior(model, X_test, Y_test, Y_test_pred, mixture=True, title=None, new_fig=False):

    ''' Visualising posterior predictive (a single distribution) '''

    if new_fig:
        plt.figure(figsize=(5,5))
    #f, ax = plt.subplots(1, 1, figsize=(8, 8))
    if mixture:
        sample_means, sample_stds = get_posterior_predictive_means_stds(Y_test_pred)

        sample_stds = torch.nan_to_num(sample_stds, nan=1e-5)
        lower, upper = get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds)
        pred_mean = sample_means.mean(dim=0)
    else:
        lower, upper = Y_test_pred.confidence_region()
        lower=lower.detach().numpy()
        upper=upper.detach().numpy()
        pred_mean = Y_test_pred.mean.numpy()

    plt.plot(X_test.numpy(), pred_mean, 'b-', label='Mean')
    plt.scatter(model.covar_module.inducing_points.detach(), [-2.5]*model.num_inducing, c='r', marker='x', label='Inducing')
    plt.fill_between(X_test.detach().numpy(), lower, upper, alpha=0.5, label=r'$\pm$2\sigma', color='g')
    plt.scatter(model.train_x.numpy(), model.train_y.numpy(), c='k', marker='x', alpha=0.7, label='Train')
    plt.plot(X_test, Y_test, color='b', linestyle='dashed', alpha=0.7, label='True')
    #ax.set_ylim([-3, 3])
    plt.title(title)

def visualise_train(model, new_fig=True):

    ''' Visualise training points '''

    if new_fig is True:
        plt.figure()
    plt.plot(model.train_x, model.train_y, 'bx', label='Train')
    plt.legend()
