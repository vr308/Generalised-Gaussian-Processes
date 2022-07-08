#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co2 experiment - gpr + hmc, sgpr + hmc (Fixed Z)

"""

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import torch
import scipy.stats as st
from utils.metrics import rmse, nlpd_mixture, nlpd, negative_log_predictive_mixture_density
from utils.config import DATASET_DIR, LOG_DIR

def mixture_posterior_predictive(model: pm.model.Model, test_x: np.ndarray, trace_hyper:pm.backends.base.MultiTrace) -> list:
     
      ''' Returns the mixture posterior predictive multivariate normal '''
     
      # Make predictions by feeding model through likelihood
     
      list_of_y_pred_dists = []
      for i in range(len(trace_hyper)):
         hyper_sample = trace_hyper[i]
         post_mean, post_covar = gp.predict(test_x, point=hyper_sample, pred_noise=True) 
         # cast as a torch Multivariate normal (as thats what the metrics calculation expect)
         pred = torch.distributions.MultivariateNormal(torch.Tensor(post_mean), torch.Tensor(post_covar))
         try:
            chol = torch.linalg.cholesky(pred.covariance_matrix + torch.eye(len(test_x))*1e-6)
            list_of_y_pred_dists.append(pred)
         except RuntimeError:
            print('Not psd for sample ' + str(i))
      return list_of_y_pred_dists
  

def get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds):
      
      # Fixed at 95% CI
      
      n_test = sample_means.shape[-1]
      components = sample_means.shape[0]
      lower_ = []
      upper_ = []
      for i in np.arange(n_test):
            print(i)
            mix_idx = np.random.choice(np.arange(components), size=2000, replace=True)
            mixture_draws = np.array([st.norm.rvs(loc=sample_means[j,i], scale=sample_stds[j,i]) for j in mix_idx])
            lower, upper = st.scoreatpercentile(mixture_draws, per=[2.5,97.5])
            lower_.append(lower)
            upper_.append(upper)
      return np.array(lower_), np.array(upper_)
  
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
    
    sep_idx = 545
    
    y_train = y[0:sep_idx].values
    y_test = y[sep_idx:].values
    t_train = t[0:sep_idx].values[:,None]
    t_test = t[sep_idx:].values[:,None]
    return y_train, t_train, y_test, t_test, std_co2

if __name__ == "__main__":
    
    y_train, t_train, y_test, t_test, std_co2 = load_co2_dataset()

    ## SGPR + HMC / GPR + HMC (fixed Z)
    
    with pm.Model() as model:
        
        
        coef = pm.Normal('coef', 0, 3)
        c = pm.Normal('c', 0, 3)
        mean = pm.gp.mean.Linear(coeffs=coef, intercept=c)
        # yearly periodic component x long term trend
        n_per = pm.HalfCauchy("n_per", beta=2, testval=1.0)
        l_pdecay = pm.Gamma("l_pdecay", alpha=10, beta=0.075)
        period = pm.Normal("period", mu=1, sigma=0.05)
        l_psmooth = pm.Gamma("l_psmooth ", alpha=4, beta=3)
        cov_seasonal = (
            n_per ** 2 * pm.gp.cov.Periodic(1, period, l_psmooth) * pm.gp.cov.Matern52(1, l_pdecay)
        )
        # small/medium term irregularities
        n_med = pm.HalfCauchy("n_med", beta=0.5, testval=0.1)
        l_med = pm.Gamma("l_med", alpha=2, beta=0.75)
        alpha = pm.Gamma("αlpha", alpha=5, beta=2)
        cov_medium = n_med ** 2 * pm.gp.cov.RatQuad(1, l_med, alpha)
    
        # long term trend
        η_trend = pm.HalfCauchy("η_trend", beta=2, testval=2.0)
        ℓ_trend = pm.Gamma("ℓ_trend", alpha=4, beta=0.1)
        cov_trend = η_trend ** 2 * pm.gp.cov.ExpQuad(1, ℓ_trend)
    
        # noise model
        η_noise = pm.HalfNormal("η_noise", sigma=0.5, testval=0.05)
        ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
        sigma = pm.HalfNormal("sigma", sigma=0.25, testval=0.05)
        cov_noise = η_noise ** 2 * pm.gp.cov.Matern32(1, ℓ_noise) + pm.gp.cov.WhiteNoise(sigma)
        cov_noise = pm.gp.cov.WhiteNoise(sigma)
        
        # The Gaussian process is a sum of these three components
        cov_final = cov_seasonal + cov_medium + cov_trend + cov_noise
        gp = pm.gp.MarginalSparse(cov_func=cov_final, mean_func=mean, approx='VFE')
        Xu = t_train[::4].copy()
        
        # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
        y_ = gp.marginal_likelihood("y", X=t_train, Xu=Xu, y=y_train, noise=sigma)
        trace = pm.sample(100, tune=100, chains=1, return_inferencedata=False)
        
    # with model:
        
    #     fnew = gp.conditional("fnew", Xnew=t_test)
    #     ppc = pm.sample_posterior_predictive(trace, samples=100, var_names=["fnew"])
        
    
    ##### Predictions ###########
    
    Y_test_pred_list = mixture_posterior_predictive(model, t_test, trace[::5]) ### a list of predictive distributions
    y_mix_loc = np.array([np.array(dist.loc.detach()) for dist in Y_test_pred_list])    
    y_mix_std = np.array([np.array(np.sqrt(x.covariance_matrix.diag())) for x in Y_test_pred_list])
    
    #### Compute Metrics  ###########
    
    rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), torch.tensor(y_test), std_co2)
    nlpd_test = np.round(nlpd_mixture(Y_test_pred_list, torch.tensor(y_test, dtype=torch.float32), np.array(std_co2)).item(), 4)

    #nlpd_marginal = negative_log_predictive_mixture_density(y_test, y_mix_loc, y_mix_std, np.array(std_co2))
    
    print('Test RMSE: ' + str(rmse_test))
    print('Test NLPD: ' + str(nlpd_test))
    
    lower, upper = get_posterior_predictive_uncertainty_intervals(y_mix_loc, y_mix_std)
    
    ##### Save the means and stds ############# 

    #np.savetxt(fname=str(LOG_DIR)+'/means_full_gpr.txt', X=y_mix_loc, delimiter=',')
    #np.savetxt(fname=str(LOG_DIR)+'/stds_full_gpr.txt', X=y_mix_std, delimiter=',')
    #np.savetxt(fname=str(LOG_DIR)+'/ci_full_gpr.txt', X=np.vstack((lower,upper)).T, delimiter=',')

    np.savetxt(fname=str(LOG_DIR)+'/means_sparse_gpr.txt', X=y_mix_loc, delimiter=',')
    np.savetxt(fname=str(LOG_DIR)+'/stds_sparse_gpr.txt', X=y_mix_std, delimiter=',')
    np.savetxt(fname=str(LOG_DIR)+'/ci_sparse_gpr.txt', X=np.vstack((lower,upper)).T, delimiter=',')
    
    with pm.Model() as co2_model:
      
       log_l2 = pm.Normal('log_l2', mu=0, sd=3)
       log_l4 = pm.Normal('log_l4', mu=0, sd=3)
       log_l5 = pm.Normal('log_l5', mu=0, sd=3)
       log_l7 = pm.Normal('log_l7', mu=0, sd=3)
       log_l10 = pm.Normal('log_l10', mu=0, sd=3)

       ls_2 = pm.Deterministic('ls_2', tt.exp(log_l2))
       ls_4 = pm.Deterministic('ls_4', tt.exp(log_l4))
       ls_5 = pm.Deterministic('ls_5', tt.exp(log_l5))
       ls_7 = pm.Deterministic('ls_7', tt.exp(log_l7))
       ls_10 = pm.Deterministic('ls_10', tt.exp(log_l10))
       

       # prior on amplitudes

       log_s1 = pm.Normal('log_s1', mu=0, sd=3)
       log_s3 = pm.Normal('log_s3', mu=0, sd=3)
       log_s6 = pm.Normal('log_s6', mu=0, sd=3)
       log_s9 = pm.Normal('log_s9', mu=0, sd=3)

       s_1 = pm.Deterministic('s_1', tt.exp(log_s1))
       s_3 = pm.Deterministic('s_3', tt.exp(log_s3))
       s_6 = pm.Deterministic('s_6', tt.exp(log_s6))
       s_9 = pm.Deterministic('s_9', tt.exp(log_s9))
       
       #s_3 = 2.59
       #s_9 = 0.169
      
       # prior on alpha
      
       log_alpha8 = pm.Normal('log_alpha8', mu=0, sd=0.1)
       alpha_8 = pm.Deterministic('alpha_8', tt.exp(log_alpha8))
       #alpha_8 = 0.121
       
       # prior on noise variance term
      
       log_n11 = pm.Normal('log_n11', mu=0, sd=3)
       n_11 = pm.Deterministic('n_11', tt.exp(log_n11))
       
       #n_11 = 0.195
       
       # Specify the covariance function
       
       k1 = pm.gp.cov.Constant(s_1**2)*pm.gp.cov.ExpQuad(1, ls_2) 
       k2 = pm.gp.cov.Constant(s_3**2)*pm.gp.cov.ExpQuad(1, ls_4)*pm.gp.cov.Periodic(1, period=1, ls=ls_5)
       k3 = pm.gp.cov.Constant(s_6**2)*pm.gp.cov.RatQuad(1, alpha=alpha_8, ls=ls_7)
       k4 = pm.gp.cov.Constant(s_9**2)*pm.gp.cov.ExpQuad(1, ls_10) +  pm.gp.cov.WhiteNoise(n_11)

       k =  k1 + k2 + k3 +k4
          
       gp = pm.gp.MarginalSparse(cov_func=k, approx='VFE')
       
       trace_prior = pm.sample(draws=500)
            
       # Marginal Likelihood
       y_ = gp.marginal_likelihood("y", X=t_train, Xu = t_train[::4], y=y_train, noise=n_11) 
              
with co2_model:
      
      # HMC Nuts auto-tuning implementation

      trace_hmc = pm.sample(draws=100, tune=100, chains=1)
