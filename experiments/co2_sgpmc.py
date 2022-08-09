#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint HMC Model for co2

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
from utils.config import DATASET_DIR
import pandas as pd
from utils.posterior_predictive import get_posterior_predictive_uncertainty_intervals

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

### Set seed for reproducibility
tf.random.set_seed(40)
np.random.seed(42)

def load_co2_dataset():
    
    df = pd.read_table(str(DATASET_DIR) + '/mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)
    
    # creat a date index for the data - convert properly from the decimal year 
    
    #df.index = pd.date_range(start='1958-01-15', periods=len(df), freq='M')
    
    df.dropna(inplace=True)
    
    first_co2 = df['co2'][0]
    std_co2 = np.std(df['co2'])   
    
    # normalize co2 levels
       
    y = (df['co2'] - first_co2)/std_co2
    t = df['year'] - df['year'][0]
    
    sep_idx = 600
    
    y_train = y[0:sep_idx].values
    y_test = y[sep_idx:].values
    t_train = t[0:sep_idx].values[:,None]
    t_test = t[sep_idx:].values[:,None]
    return f64(y_train), f64(t_train), f64(y_test), f64(t_test), f64(std_co2)

def train_sgp_hmc(data, Z_init, input_dims, tune, num_samples):
    
    # Instantiating kernel and model 
 
    mean = gpflow.mean_functions.Linear()
    #mean = gpflow.mean_functions.Zero()
    mean.A.prior = tfd.Normal(f64(0),f64(3))
    mean.b.prior = tfd.Normal(f64(0),f64(3))
    
    cov_seasonal = gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential(), period=f64(1.0))*gpflow.kernels.Matern52()
    cov_medium = gpflow.kernels.RationalQuadratic()
    cov_trend = gpflow.kernels.SquaredExponential(variance=np.log(2)**2)
    cov_noise = gpflow.kernels.Matern52() + gpflow.kernels.White()
    
    ## don't train for period
    set_trainable(cov_seasonal.kernels[0].period, False)
    set_trainable(cov_seasonal.kernels[1].variance, False)

    # Setting priors for params 

    cov_seasonal.kernels[0].base_kernel.variance.prior = tfd.HalfNormal(scale=f64(2.0))
    cov_seasonal.kernels[0].base_kernel.lengthscales.prior = tfd.Gamma(f64(4.0), f64(3.0)) 
    cov_seasonal.kernels[1].lengthscales.prior = tfd.Gamma(f64(10), f64(0.075))
    
    cov_medium.variance.prior = tfd.HalfNormal(scale=f64(0.5))
    cov_medium.lengthscales.prior = tfd.Gamma(f64(2), f64(0.75))
    cov_medium.alpha.prior = tfd.Gamma(f64(5), f64(2))
    
    cov_trend.variance.prior = tfd.HalfNormal(scale=f64(2))
    cov_trend.lengthscales.prior = tfd.Gamma(concentration=f64(4), rate=f64(0.1)) 
    
    cov_noise.kernels[0].variance.prior = tfd.HalfNormal(scale=f64(0.5))
    cov_noise.kernels[0].lengthscales.prior = tfd.Gamma(concentration=f64(2), rate=f64(4))
    cov_noise.kernels[1].variance.prior = tfd.HalfNormal(scale=f64(0.25))
    
    # The Gaussian process is a sum of these three components
    kernel = cov_seasonal + cov_medium + cov_trend + cov_noise
         
    model = gpflow.models.SGPMC(
        data,
        kernel=kernel,
        mean_function = mean,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=Z_init
    )
    gpflow.utilities.print_summary(model, fmt="notebook")
    
    # Initial warm-up (as per gpflow tutorial)
    #optimizer = gpflow.optimizers.Scipy()
    #set_trainable(model.inducing_variable, False)
    #optimizer.minimize(model.training_loss, model.trainable_variables, options={"maxiter": 100})
    set_trainable(model.inducing_variable, False)
    model.likelihood.variance.prior = tfd.Gamma(f64(2.0), f64(1.0))

    ## HMC step
    num_burnin_steps = ci_niter(tune)
    num_samples = ci_niter(num_samples)
    
    # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )
    
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=20, step_size=0.005
    )
    
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=20, target_accept_prob=f64(0.80), adaptation_rate=0.05
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
    
    from utils.metrics import rmse, nlpd_mixture, nlpd

    y_train, t_train, y_test, t_test, std_co2 = load_co2_dataset()
    
    data = (t_train, y_train)
    data_test = (t_test, y_test)
    
    Z_init = np.array(t_train)[np.random.randint(0, len(t_train), 200)]
     
    ## Train model
    model, hmc_helper, samples, wall_clock_secs = train_sgp_hmc(data, Z_init, input_dims=t_train.shape[-1], tune=100, num_samples=100)
    
#     # ## Predictions
    
    pred_mean, y_pred_dists, lower, upper = predict_sgpmc(model, hmc_helper, samples, t_test)
    