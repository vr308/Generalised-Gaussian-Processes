#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Posterior predictive utils

"""

import numpy as np
import scipy.stats as st
import torch

def get_posterior_predictive_means_stds(Y_test_pred_list):

     sample_means = []
     sample_stds = []
     for dist in Y_test_pred_list:
         sample_means.append(dist.loc.detach())
         sample_stds.append(dist.covariance_matrix.diag().sqrt())

     sample_means = torch.stack(sample_means)
     sample_stds = torch.stack(sample_stds)
     return sample_means,sample_stds

def get_posterior_predictive_mean(sample_means):
      return torch.mean(sample_means, axis=0)

def compute_log_marginal_likelihood(K_noise, y):
      return np.log(st.multivariate_normal.pdf(y, cov=K_noise.eval()))

def get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds):

    # Fixed at 95% CI
      n_test = sample_means.shape[-1]
      components = sample_means.shape[0]
      lower_ = []
      upper_ = []
      for i in np.arange(n_test):
         print(i)
         mix = torch.distributions.Categorical(torch.ones(components,))
         comp = torch.distributions.Normal(torch.tensor(sample_means[:,i]), torch.tensor(sample_stds[:,i]))
         gmm = torch.distributions.MixtureSameFamily(mix, comp)
         mixture_draws = gmm.sample((1000,))
         lower, upper = st.scoreatpercentile(mixture_draws, per=[2.5, 97.5])
         lower_.append(lower)
         upper_.append(upper)
      return np.array(lower_), np.array(upper_)

def posterior_predictive_samples(post_mean, post_cov):

    return np.random.multivariate_normal(post_mean, post_cov, 20)

def log_predictive_density(predictive_density):

      return np.round(np.sum(np.log(predictive_density)), 3)

def log_predictive_mixture_density(f_star, list_means, list_cov):

      components = []
      for i in np.arange(len(list_means)):
            components.append(st.multivariate_normal.pdf(f_star, list_means[i].eval(), list_cov[i].eval(), allow_singular=True))
      return -np.round(np.sum(np.log(np.mean(components))),3)


