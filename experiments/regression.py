#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression training over 10 splits different models.
Full-GP, SGPR, SVGP, bSGPR-HMC, bSVGP-DS

"""

import numpy as np
from joblib import delayed, Parallel
import argparse
import json
from itertools import product
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

import torch 
import gpytorch 

from utils.config import LOG_DIR

# Models
from models.sgpr import SparseGPR
from models.svgp import StochasticVariationalGP
from models.bayesian_svgp import BayesianStochasticVariationalGP
from models.bayesian_sgpr_hmc import BayesianSparseGPR_HMC, mixture_posterior_predictive

# Metrics and Viz
from utils.metrics import rmse, nlpd, nlpd_mixture
from utils.experiment_tools import get_dataset_class, experiment_name
import matplotlib.pyplot as plt

gpytorch.settings.cholesky_jitter(float=1e-4)
plt.style.use('seaborn-muted')

#### Global variables 

DATASETS = ["Boston", "Concrete", "Energy", "Yacht", "WineRed"]

MODEL_NAMES = ['SGPR', 'Bayesian_SGPR_HMC', 'SVGP', 'Bayesian_SVGP']
MODEL_CLASS = [
            SparseGPR,
            BayesianSparseGPR_HMC,
            StochasticVariationalGP,
            BayesianStochasticVariationalGP
            ]
SPLIT_INDICES = [*range(10)]

model_dictionary = dict(zip(MODEL_NAMES, MODEL_CLASS))

##### Methods

def single_run(
        
    dataset_name: str,
    split_index: int,
    model_name: str,
    num_inducing: int,
    train_test_split: float,
    max_iter: int,
    num_epochs : int,
    batch_size: int,
    date_str: str,
   
):
        print(f"Running on {dataset_name} (Split {split_index})" )
    
        ###### Loading a data split ########
        
        dataset = get_dataset_class(dataset_name)(split=split_index, prop=train_test_split)
        X_train, Y_train, X_test, Y_test = dataset.X_train, dataset.Y_train, dataset.X_test, dataset.Y_test
        
        ###### Initialising model class, likelihood, inducing inputs ##########
        
        model_class = model_dictionary[model_name]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        ## Init at X_train[np.random.randint(0,len(X_train), 200)]
        #Z_init = torch.randn(num_inducing, input_dim)
        Z_init = X_train[np.random.randint(0,len(X_train), num_inducing)]

        model = model_class(X_train, Y_train.flatten(), likelihood, Z_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        ####### Custom training call depending on model class #########
        
        if model_name != 'Bayesian_SGPR_HMC':

            if model_name == 'SGPR':
            
                losses = model.train_model(optimizer, num_steps=max_iter)
                
            elif model_name in ('SVGP', 'Bayesian_SVGP'):
                
                train_dataset = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                losses = model.train_model(optimizer, train_loader, minibatch_size=batch_size, num_epochs=num_epochs, combine_terms=True)
            
            ### Predictions
            
            Y_train_pred = model.posterior_predictive(X_train)
            Y_test_pred = model.posterior_predictive(X_test)
    
            ### Compute Metrics  
            
            rmse_train = np.round(rmse(Y_train_pred.loc, Y_train, dataset.Y_std).item(), 4)
            rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
           
            nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
            nlpd_test = np.round(nlpd(Y_test_pred, Y_test, dataset.Y_std).item(), 4)
        
        else:
            
            break_for_hmc = np.concatenate((np.arange(100,1500,50), np.array([max_iter-1])))
            losses, trace_hyper, step_sizes = model.train_model(optimizer, max_steps=max_iter, hmc_scheduler=break_for_hmc)
       
            ### Predictions
            
            Y_test_pred_list = mixture_posterior_predictive(model, X_test, trace_hyper)
    
            ### Compute Metrics  
            
            rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
            nlpd_test = np.round(nlpd_mixture(Y_test_pred_list, Y_test, dataset.Y_std).item(), 4)
            
            # Saving trace summary
            #loss_path = LOG_DIR / "loss" / experiment_name(**exp_info)
            #np.savetxt(fname=f"{loss_path}.csv", X=losses)
      
        
        metrics = {
                'test_rmse': rmse_test,
                  'test_nlpd': nlpd_test,
                  'train_rmse': rmse_train,
                  'train_nlpd': nlpd_train
                  }
     
        exp_info = {
             "date_str": date_str,
             "split_index": split_index,
             "dataset_name": dataset_name,
             "model_name":model_name,
             "num_inducing": num_inducing,
             "max_iter": max_iter,
             "num_epochs": num_epochs,
             "batch_size": batch_size,
             "train_test_split": train_test_split ,
             "step_sizes": np.array(step_sizes)
             }
        
        # One dict with all info + metrics
        experiment_dict = {**exp_info,
                            **metrics}
         
        print(dataset_name, ': Test RMSE:', rmse_test)
        print(dataset_name, ': Test NLPD:', nlpd_test)
        
        ###### Saving losses / performance metrics for analysis and reporting #########
        
        date_str = datetime.now().strftime('%b_%d')
        
        log_path = LOG_DIR / date_str / experiment_name(**exp_info)
        #loss_path = LOG_DIR / "loss" / experiment_name(**exp_info)
        
        #np.savetxt(fname=f"{loss_path}.csv", X=losses)
        
        results_filename = f"{log_path}__.json"
        with open(results_filename, "w") as fp:
             json.dump(experiment_dict, fp, indent=4)
       
def main(args: argparse.Namespace):
        
        np.random.seed(args.seed)
    
        date_str = datetime.now().strftime('%b_%d')
        save_dir = LOG_DIR / date_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        #loss_dir = LOG_DIR / "loss" 
        #loss_dir.mkdir(parents=True, exist_ok=True)
    
        print("Training GPR in parallel. GPUs may run out of memory, use CPU if this happens. "
            "To use CPU only, set environment variable CUDA_VISIBLE_DEVICES=-1")
    
        Parallel(args.n_jobs)(
            delayed(single_run)(
                dataset_name,
                split_index,
                args.model_name,
                args.num_inducing,
                args.test_train_split,
                args.max_iters,
                args.num_epochs,
                args.batch_size,
                date_str
            ) for dataset_name, split_index in product(DATASETS, SPLIT_INDICES)
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='SVGP')
    parser.add_argument("--num_inducing", type=int, default=100)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--test_train_split", type=float, default=0.8)
    parser.add_argument("--n_jobs", type=int, default=1)
    
    arguments = parser.parse_args()
    
    main(arguments)
    
    #single_run('Boston', 4, 'SparseGPR', 50, 0.9, 2000, '30_12')
    #single_run('Boston',4, 'StochasticVariationalGP',506, 0.9, 20000, '06_10')
    #single_run('Boston',4, 'SparseGPR',500, 0.9, 20000, '06_10')
    
 