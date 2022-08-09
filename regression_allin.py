#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression training over 10 splits different models.
Sampling both Z and theta - based on Rossi et al, 2021.

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
from models.all_in_HMC import all_in_HMC, full_mixture_posterior_predictive

# Metrics and Viz
from utils.metrics import rmse, nlpd, nlpd_mixture
from utils.experiment_tools import get_dataset_class, experiment_name_allin
import matplotlib.pyplot as plt

gpytorch.settings.cholesky_jitter(float=1e-5)
plt.style.use('seaborn-muted')

#### Global variables 

DATASETS = ["Boston", "Concrete", "Energy", "WineRed", "Yacht"]
SPLIT_INDICES = [*range(10)]

## Dry-run
#DATASETS = ['WineRed','Yacht']
#SPLIT_INDICES = [0,1,2,3,4,5,6,7,8,9]

##### Methods

def single_run(
        
    dataset_name: str,
    split_index: int,
    model_name: str,
    num_inducing: int,
    train_test_split: float,
    num_tune: int,
    num_samples : int,
    date_str: str,
   
):
        print(f"Running on {dataset_name} (Split {split_index})" )
    
        ###### Loading a data split ########
        
        dataset = get_dataset_class(dataset_name)(split=split_index, prop=train_test_split)
        X_train, Y_train, X_test, Y_test = dataset.X_train.double(), dataset.Y_train.double(), dataset.X_test.double(), dataset.Y_test.double()
        
        ###### Initialising model class, likelihood, inducing inputs ##########
    
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        Z_init = X_train[np.random.randint(0, len(X_train), 100)]
    
        model = all_in_HMC(X_train,Y_train, likelihood, Z_init)
        
        ####### Custom training depending on model class #########
        
        start_time = time.time()
        trace_hyper, step_sizes, perf_times = model.train_model()
        wall_clock_secs = time.time() - start_time

        ##### Predictions ###########
        
        Y_test_pred_list = full_mixture_posterior_predictive(model, X_test, trace_hyper) ### a list of predictive distributions
        y_mix_loc = np.array([np.array(dist.loc.detach()) for dist in Y_test_pred_list])    
        
        #### Compute Metrics  ###########
        
        rmse_test = np.round(rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), Y_test, dataset.Y_std).item(),4)
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
             "num_tune": num_tune,
             "num_samples": num_samples,
             "train_test_split": train_test_split ,
             }
        
        # One dict with all info + metrics
        experiment_dict = {**exp_info,
                            **metrics}
         
        print(dataset_name, ': Test RMSE:', rmse_test)
        print(dataset_name, ': Test NLPD:', nlpd_test)
        
        ###### Saving losses / performance metrics / traces for analysis and reporting #########
        
        date_str = datetime.now().strftime('%b_%d')
        
        log_path = LOG_DIR / date_str / experiment_name_allin(**exp_info)
        
        if trace_hyper is not None:
            # Saving trace summary
            trace_path = LOG_DIR / "trace" / experiment_name_allin(**exp_info)
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
        
        Parallel(args.n_jobs)(
            delayed(single_run)(
                dataset_name,
                split_index,
                args.model_name,
                args.num_inducing,
                args.test_train_split,
                args.num_tune,
                args.num_samples,
                date_str
            ) for dataset_name, split_index in product(DATASETS, SPLIT_INDICES)
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='all_in_HMC')
    parser.add_argument("--num_inducing", type=int, default=100)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num_tune", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--test_train_split", type=float, default=0.8)
    parser.add_argument("--n_jobs", type=int, default=1)
    
    arguments = parser.parse_args()
    
    main(arguments)
        
 