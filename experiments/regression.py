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
import time
import torch 
import gpytorch 
import pymc3 as pm
from utils.config import LOG_DIR

# Models
from models.sgpr import SparseGPR
from models.svgp import StochasticVariationalGP
from models.bayesian_svgp import BayesianStochasticVariationalGP
from models.bayesian_sgpr_hmc import BayesianSparseGPR_HMC, mixture_posterior_predictive
from models.gpr_hmc import GPR_HMC, full_mixture_posterior_predictive

# Metrics and Viz
from utils.metrics import rmse, nlpd, nlpd_mixture
from utils.experiment_tools import get_dataset_class, experiment_name
import matplotlib.pyplot as plt

gpytorch.settings.cholesky_jitter(float=1e-4)
plt.style.use('seaborn-muted')

#### Global variables 

DATASETS = ["Boston", "Concrete", "Energy", "WineRed", "Yacht"]
MODEL_NAMES = ['SGPR', 'Bayesian_SGPR_HMC', 'SVGP', 'Bayesian_SVGP', 'GPR_HMC']
MODEL_CLASS = [
            SparseGPR,
            BayesianSparseGPR_HMC,
            StochasticVariationalGP,
            BayesianStochasticVariationalGP,
            GPR_HMC
            ]
SPLIT_INDICES = [*range(10)]
model_dictionary = dict(zip(MODEL_NAMES, MODEL_CLASS))

## Dry-run
#DATASETS = ['Yacht']
#SPLIT_INDICES = [2,3,4,5,6,7,8,9]

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
        X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
        
        ###### Initialising model class, likelihood, inducing inputs ##########
        
        model_class = model_dictionary[model_name]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        ## Init at X_train[np.random.randint(0,len(X_train), 200)]
        Z_init = X_train[np.random.randint(0,len(X_train), num_inducing)]

        if model_name != 'GPR_HMC':
            model = model_class(X_train, Y_train.flatten(), likelihood, Z_init)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        else:
            model = model_class(X_train, Y_train.flatten(), likelihood)
        
        ####### Custom training call depending on model class #########
        
        if model_name not in ('Bayesian_SGPR_HMC', 'GPR_HMC'):

            step_sizes = None
            trace_hyper = None  ## These variables are unused for the methods in this block.
            perf_times = None
            
            if model_name == 'SGPR':
                
                start_time = time.time()
                losses = model.train_model(optimizer, num_steps=max_iter)
                wall_clock_secs = time.time() - start_time

            elif model_name in ('SVGP', 'Bayesian_SVGP'):
                
                train_dataset = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                start_time = time.time()
                losses = model.train_model(optimizer, train_loader, minibatch_size=batch_size, num_epochs=num_epochs, combine_terms=True)
                wall_clock_secs = time.time() - start_time
                
            ### Predictions ###
            
            #Y_train_pred = model.posterior_predictive(X_train)
            Y_test_pred = model.posterior_predictive(X_test)
    
            ### Compute Metrics ###
            
            #rmse_train = np.round(rmse(Y_train_pred.loc, Y_train, dataset.Y_std).item(), 4)
            rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
           
            #nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
            nlpd_test = np.round(nlpd(Y_test_pred, Y_test, dataset.Y_std).item(), 4)
        
        else:
            
            if model_name == 'Bayesian_SGPR_HMC':
            
                break_for_hmc = np.concatenate((np.arange(200,max_iter-500,100), np.array([max_iter-1])))
                
                start_time = time.time()
                #losses, trace_hyper, step_sizes, perf_times = model.train_model(optimizer, max_steps=max_iter, hmc_scheduler=break_for_hmc)
                trace_hyper, step_sizes, perf_times = model.train_fixed_model()
                wall_clock_secs = time.time() - start_time
                
                ### Predictions
                Y_test_pred_list = mixture_posterior_predictive(model, X_test, trace_hyper)
                
            else:
                
                print('Full GPR with HMC')
                start_time = time.time()
                trace_hyper, step_sizes, perf_times = model.train_model()
                wall_clock_secs = time.time() - start_time
                
                ### Predictions
                Y_test_pred_list = full_mixture_posterior_predictive(model, X_test, trace_hyper)
    
            ### Compute Metrics  
            y_mix_loc = np.array([np.array(dist.loc.detach()) for dist in Y_test_pred_list])    
            rmse_test = np.round(rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), Y_test, dataset.Y_std).item(), 4)
            nlpd_test = np.round(nlpd_mixture(Y_test_pred_list, Y_test, dataset.Y_std).item(), 4)
            
        
        metrics = {
                'test_rmse': rmse_test,
                  'test_nlpd': nlpd_test,
                  'wall_clock_secs': wall_clock_secs,
                  'perf_times': perf_times
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
             "step_sizes": step_sizes
             }
        
        # One dict with all info + metrics
        experiment_dict = {**exp_info,
                            **metrics}
         
        print(dataset_name, ': Test RMSE:', rmse_test)
        print(dataset_name, ': Test NLPD:', nlpd_test)
        
        ###### Saving losses / performance metrics / traces for analysis and reporting #########
        
        date_str = datetime.now().strftime('%b_%d')
        
        log_path = LOG_DIR / date_str / experiment_name(**exp_info)
        #loss_path = LOG_DIR / "loss" / experiment_name(**exp_info)
        #np.savetxt(fname=f"{loss_path}.csv", X=losses)
        
        if trace_hyper is not None:
            # Saving trace summary
            trace_path = LOG_DIR / "trace" / experiment_name(**exp_info)
            pm.summary(trace_hyper).to_csv(f"{trace_path}.csv")
        
        results_filename = f"{log_path}__.json"
        with open(results_filename, "w") as fp:
             json.dump(experiment_dict, fp, indent=4)
       
def main(args: argparse.Namespace):
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
        date_str = datetime.now().strftime('%b_%d')
        save_dir = LOG_DIR / date_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        #loss_dir = LOG_DIR / "loss" 
        #loss_dir.mkdir(parents=True, exist_ok=True)
        
        trace_dir = LOG_DIR / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
    
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
    parser.add_argument("--model_name", type=str, default='Bayesian_SGPR_HMC')
    parser.add_argument("--num_inducing", type=int, default=100)
    parser.add_argument("--seed", type=int, default=45)
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
    
 