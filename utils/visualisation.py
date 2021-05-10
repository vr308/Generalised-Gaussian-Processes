#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualisation utilities for data and experiments

"""

import matplotlib.pylab as plt

def visualise_posterior(model, test_x, y_star):

    ''' Visualising posterior predictive '''
    
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    lower, upper = y_star.confidence_region()
    ax.plot(model.train_x.numpy(), model.train_y.numpy(), 'kx')
    ax.plot(test_x.numpy(), y_star.mean.numpy(), 'b-')
    ax.plot(model.covar_module.inducing_points.detach(), [-2.5]*model.num_inducing, 'rx')
    ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    #ax.set_ylim([-3, 3])
    ax.legend(['Train', 'Mean', 'Inducing inputs', r'$\pm$2\sigma'])

def visualise_train(model):
    
    ''' Visualise training points '''
    
    plt.figure()
    plt.plot(model.train_x, model.train_y, 'bx', label='Train')
    plt.legend()
