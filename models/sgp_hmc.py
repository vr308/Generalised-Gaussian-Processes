#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint HMC Model -> Hensman et al, 2015

"""
import gpflow
import numpy as np
import matplotlib.pylab as plt
import numpy as np
from tensorflow_probability import distributions as tfd
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
from gpflow.utilities import print_summary
from utils.posterior_predictive import get_posterior_predictive_uncertainty_intervals

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

# Generate data by sampling from SquaredExponential kernel, and classifying with the argmax
rng = np.random.RandomState(41)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

if __name__ == '__main__':

    ## Creating 1d synthetic data

    N = 1000
    
    X = rng.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.4 * rng.randn(N)  # Noisy Y values
        
    train_index = np.where((X < -2) | (X > 2))
    
    # Train
    X_train = X[train_index][:,None]
    Y_train = Y[train_index][:,None]
    
    ## Test
    X_test = tf.linspace(-8, 8, 1000)[:,None]
    Y_test = func(X_test)
    
    data = (X_train, Y_train)
    
    # Instantiating kernel and model 
    
    kernel = gpflow.kernels.SquaredExponential(lengthscales=0.1)
    
    model = gpflow.models.SGPMC(
        data,
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=X_train[::7].copy()
    )
    
    # Setting priors for params 
    
    model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
    model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
    model.kernel.lengthscales.prior = tfd.Gamma(f64(2.0), f64(2.0))
    
    gpflow.utilities.print_summary(model, fmt="notebook")
    set_trainable(model.inducing_variable, False) ## why (Hensman trains this somehow)
    
    # Initial warm-up (as per gpflow tutorial)
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables, options={"maxiter": 100})
    
    ## HMC step
    num_burnin_steps = ci_niter(100)
    num_samples = ci_niter(500)
    
    # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )
    
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
    )
    
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=10, target_accept_prob=f64(0.80), adaptation_rate=0.1
    )
    
    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
       
    samples, _ = run_chain_fn()
    constrained_samples = hmc_helper.convert_to_constrained_values(samples)
    
    ## Predictions 
    f_means = [] ## we want to extract the mean of the latent function values and
    y_stds = []   ## prediction intervals around the data. 
    f_samples = []
    
    for i in range(10):
    # Note that hmc_helper.current_state contains the unconstrained variables
        for var, var_samples in zip(hmc_helper.current_state, samples):
            var.assign(var_samples[i])
            
        f_mean, f_var = model.predict_f(X_test)
        y_mean, y_var = model.predict_y(X_test)
        
        f_means.append(np.array(f_mean).T)
        f_samples.append(np.array(model.predict_f_samples(X_test,3)))
        y_stds.append(np.array(np.sqrt(y_var)).T)
        
    f_means = np.vstack(f_means)
    f_samples = np.vstack(f_samples)
    y_stds = np.vstack(y_stds)
    
    
    ## prediction intervals
    lower , upper = get_posterior_predictive_uncertainty_intervals(f_means, y_stds)
    pred_mean = np.mean(f_means, axis=0)
    
    ## Visualisation
    
    # lower=lower.detach().numpy()
    # upper=upper.detach().numpy()
    # pred_mean = Y_test_pred.mean.numpy()

    plt.plot(X_test.numpy(), pred_mean, 'b-', label='Mean')
    # plt.scatter(model.covar_module.inducing_points.detach(), [-2.5]*model.num_inducing, c='r', marker='x', label='Inducing')
    plt.fill_between(X_test[:,0], lower[:,0], upper[:,0], alpha=0.5, label=r'$\pm$2\sigma', color='g')
    plt.fill_between(X_test[:,0], lower2, upper2, alpha=0.5, label=r'$\pm$2\sigma', color='r')
    plt.scatter(X_train[:,0], Y_train[:,0], c='k', marker='x', alpha=0.7, label='Train')
    plt.plot(X_test[:,0], Y_test[:,0], color='b', linestyle='dashed', alpha=0.7, label='True')
    # #ax.set_ylim([-3, 3])
    # plt.legend(fontsize='small')
    # plt.title(title)
    
    
    
    
    