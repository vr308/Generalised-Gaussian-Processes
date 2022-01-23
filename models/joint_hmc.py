#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint HMC model -> sample inducing locations Z and hyperparameters

"""
import pymc3 as pm
import numpy as np

def get_trace_and_map_estimate(input_dim, num_inducing):
    
     with pm.Model() as model:
            
            ls = pm.Gamma("ls", alpha=2, beta=1)
            sig_f = pm.HalfCauchy("sig_f", beta=1)
        
            cov = sig_f ** 2 * pm.gp.cov.ExpQuad(6, ls=ls)
            gp = pm.gp.MarginalSparse(cov_func=cov, approx="VFE")
                
            sig_n = pm.HalfCauchy("sig_n", beta=1)
            #Z_opt = pm.Normal("Xu", shape=(num_inducing, input_dim))
    
            # Z_opt is the intermediate inducing points from the optimisation stage
            y_ = gp.marginal_likelihood("y", X=X_train.numpy(), Xu=Z_opt, y=Y_train.numpy(), noise=sig_n)
        
            trace = pm.sample(100, tune=500, chains=1)
            #mp = pm.find_MAP()
        
     return model, trace
 
