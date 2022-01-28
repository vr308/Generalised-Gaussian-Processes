#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint HMC Model -> Hensman et al, 2015

"""
import gpflow
import numpy as np
import matplotlib.pylab as plt
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
import time
import torch
from utils.posterior_predictive import get_posterior_predictive_uncertainty_intervals

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-5)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

### Set seed for reproducibility
tf.random.set_seed(37)
np.random.seed(45)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

def train_sgp_hmc(data, Z_init, input_dims, tune, num_samples):
    
    # Instantiating kernel and model 
    
    kernel = gpflow.kernels.SquaredExponential(variance=np.log(2)**2, lengthscales=np.array([np.log(2)]*input_dims))
    
    model = gpflow.models.SGPMC(
        data,
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=Z_init
    )
    
    # Setting priors for params 
    
    model.likelihood.variance.prior = tfd.Gamma(f64(2.0), f64(1.0))
    model.kernel.variance.prior = tfd.Gamma(f64(2.0), f64(1.0))
    model.kernel.lengthscales.prior = tfd.Gamma(f64([2.0]*input_dims), rate=f64([1.0]*input_dims))
    
    gpflow.utilities.print_summary(model, fmt="notebook")
    
    # Initial warm-up (as per gpflow tutorial)
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(model.training_loss, model.trainable_variables, options={"maxiter": 100})
    set_trainable(model.inducing_variable, False)

    ## HMC step
    num_burnin_steps = ci_niter(tune)
    num_samples = ci_niter(num_samples)
    
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
      
    print('Sampling step')
    start = time.time()
    samples, _ = run_chain_fn()
    wall_clock_secs = time.time() - start
    constrained_samples = hmc_helper.convert_to_constrained_values(samples)
    
    return model, hmc_helper, samples, wall_clock_secs

def predict_sgpmc(model, hmc_helper, samples, X_test):
    
    ## Predictions 
    f_means = [] ## we want to extract the mean of the latent function values and
    y_pred_dists = []   ## prediction intervals around the data. 
    y_stds = []
    #f_samples = []
    
    for i in range(50):
    # Note that hmc_helper.current_state contains the unconstrained variables
        for var, var_samples in zip(hmc_helper.current_state, samples):
            var.assign(var_samples[i])
            
        f_mean, f_cov = model.predict_f(X_test, full_cov=True)
        y_cov = f_cov + np.eye(len(X_test))*model.likelihood.variance.numpy()
        
        f_means.append(np.array(f_mean).T)
        #f_samples.append(np.array(model.predict_f_samples(X_test,3)))
        y_stds.append(np.array(np.sqrt(np.diag(y_cov[0]))).T)
        
        ## torch multivariate normal to compute metrics
        torch_mean = torch.tensor(np.array(f_mean)).squeeze()
        torch_cov = torch.tensor(np.array(y_cov[0]))
        y_pred = torch.distributions.MultivariateNormal(torch_mean, torch_cov)
        y_pred_dists.append(y_pred)

    f_means = np.vstack(f_means)
    #f_samples = np.vstack(f_samples)
    y_stds = np.vstack(y_stds)
    
    ## prediction intervals
    lower, upper = get_posterior_predictive_uncertainty_intervals(f_means, y_stds)
    pred_mean = np.mean(f_means, axis=0)
    return pred_mean, y_pred_dists, lower, upper

if __name__ == '__main__':
    
    from utils.experiment_tools import get_dataset_class
    from utils.metrics import rmse, nlpd_mixture, nlpd

    dataset = get_dataset_class('Yacht')(split=7, prop=0.8)
    X_train, Y_train, X_test, Y_test = f64(dataset.X_train), f64(dataset.Y_train)[:,None], f64(dataset.X_test), f64(dataset.Y_test)[:,None]


    ## Creating 1d synthetic data

    # N = 1000
    
    # rng = np.random.RandomState(45)
    
    # X = rng.randn(N) * 2 - 1  # X values
    # Y = func(X) + 0.4 * rng.randn(N)  # Noisy Y values
        
    # train_index = np.where((X < -2) | (X > 2))
    
    # # Train
    # X_train = X[train_index][:,None]
    # Y_train = Y[train_index][:,None]
    
    # ## Test
    # X_test = f64(tf.linspace(-8.0, 8.0, 1000)[:,None])
    # Y_test = f64(func(X_test))
    
    data = (X_train, Y_train)
    data_test = (X_test, Y_test)
    
    ## Train model
    Z_init = np.array(X_train)[np.random.randint(0, len(X_train), 100)]
    model, hmc_helper, samples, wall_clock_secs = train_sgp_hmc(data, Z_init, input_dims=X_train.shape[-1], tune=500, num_samples=1000)
    
    # ## Predictions
    
    pred_mean, y_pred_dists, lower, upper = predict_sgpmc(model, hmc_helper, samples, X_test)
    
    # # ## Visualisation
    
    # # # lower=lower.detach().numpy()
    # # # upper=upper.detach().numpy()
    # # # pred_mean = Y_test_pred.mean.numpy()
    # plt.subplot(133)
    # plt.plot(X_test.numpy(), pred_mean, 'b-', label='Mean')
    # plt.scatter(Z_init, [-2.5]*25, c='r', marker='x', label='Inducing')
    # plt.fill_between(X_test[:,0], lower, upper, alpha=0.5, label=r'$\pm$2\sigma', color='g')
    # plt.scatter(X_train[:,0], Y_train[:,0], c='k', marker='x', alpha=0.7, label='Train')
    # plt.plot(X_test[:,0], Y_test[:,0], color='b', linestyle='dashed', alpha=0.7, label='True')
    # # #ax.set_ylim([-3, 3])
    # plt.legend(fontsize='small')
    #plt.title(title)
    
    ## Metrics
    from utils.metrics import rmse, nlpd_mixture
    import torch
        
    joint_hmc = rmse(torch.tensor(pred_mean), np.array(Y_test).flatten(), dataset.Y_std)

    print('RMSE Joint_HMC ' + str(joint_hmc))

    nll_hmc = nlpd_mixture(y_pred_dists, torch.tensor(np.array(Y_test).squeeze()), dataset.Y_std)
    
    print('NLPD Joint HMC ' + str(nll_hmc))
        
    
    
    
    
    