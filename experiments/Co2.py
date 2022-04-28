#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co2 experiment - sgpr, gpr + hmc, sgpr + hmc, joint hmc

"""

import pandas as pd
import numpy as np
import pymc3 as pm
from utils.config import DATASET_DIR

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
    
    return y_train, t_train, y_test, t_test 


if __name__ == "__main__":
    
    y_train, t_train, y_test, t_test = load_co2_dataset()

    ## SGPR + HMC (fixed Z)
    
    with pm.Model() as model:
        
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
        #η_noise = pm.HalfNormal("η_noise", sigma=0.5, testval=0.05)
        #ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
        sigma = pm.HalfNormal("sigma", sigma=0.25, testval=0.05)
        #cov_noise = η_noise ** 2 * pm.gp.cov.Matern32(1, ℓ_noise) + pm.gp.cov.WhiteNoise(σ)
        cov_noise = pm.gp.cov.WhiteNoise(sigma)
        
        # The Gaussian process is a sum of these three components
        cov_final = cov_seasonal + cov_medium + cov_trend + cov_noise
        gp = pm.gp.MarginalSparse(cov_func=cov_final, approx="VFE")
        #Xu = t_train[::5].copy()
        Xu = pm.HalfNormal('Z',sigma=3, shape=100)
        # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
        y_ = gp.marginal_likelihood("y", X=t_train, Xu=Xu[:,None], y=y_train, noise=sigma)
    
        trace = pm.sample(100, tune=100, chains=1, return_inferencedata=False)
        
    
    ## SGPR + HMC model 
    
    


