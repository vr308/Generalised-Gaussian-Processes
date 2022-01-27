
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGPMC Regression experiment

"""

import numpy as np
from joblib import delayed, Parallel
import argparse
import json
from itertools import product
from datetime import datetime
import time
import torch 
import gpytorch 
import pymc3 as pm
from utils.config import LOG_DIR

import gpflow
import numpy as np
import matplotlib.pylab as plt

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

# Models
from models.sgp_hmc import train_sgp_hmc, predict_sgpmc

# Metrics and Viz
from utils.metrics import rmse, nlpd_mixture
from utils.experiment_tools import get_dataset_class, experiment_name_tf

gpytorch.settings.cholesky_jitter(float=1e-4)

plt.style.use('seaborn-muted')

#### Global variables 

#DATASETS = ["Boston", "Concrete", "Energy", "WineRed", "Yacht"]
#SPLIT_INDICES = [*range(10)]

## Dry-run
DATASETS = ['Yacht']
SPLIT_INDICES = [3,4,5,6,7,8,9]

##### Methods

def single_run(
        
    dataset_name: str,
    split_index: int,
    model_name: str,
    num_inducing: int,
    train_test_split: float,
    date_str: str,
   
):
        print(f"Running on {dataset_name} (Split {split_index})" )
    
        ###### Loading a data split ########
        
        dataset = get_dataset_class(dataset_name)(split=split_index, prop=train_test_split)
        X_train, Y_train, X_test, Y_test = f64(dataset.X_train), f64(dataset.Y_train)[:,None], f64(dataset.X_test), f64(dataset.Y_test)[:,None]
        
        ###### Initialising model class, likelihood, inducing inputs ##########
        
        data = (X_train, Y_train)
        #data_test = (X_test, Y_test)
        
        ##  Train model ###
        Z_init = np.array(X_train)[np.random.randint(0, len(X_train), num_inducing)]
        
        start = time.time()
        model, hmc_helper, samples, sampling_secs = train_sgp_hmc(data, Z_init, input_dims=X_train.shape[-1], tune=500, num_samples=500)
        wall_clock_secs = time.time() - start
        
        ### Predictions ###
        pred_mean, y_pred_dists, lower, upper = predict_sgpmc(model, hmc_helper, samples, X_test)

        ### Metrics ###
        
        rmse_test = np.round(rmse(torch.tensor(pred_mean), np.array(Y_test).flatten(), dataset.Y_std), 4).item()
        nlpd_test = np.round(nlpd_mixture(y_pred_dists, torch.tensor(np.array(Y_test).squeeze()), dataset.Y_std), 4).item()
        
        metrics = {
                'test_rmse': rmse_test,
                'test_nlpd': nlpd_test,
                'wall_clock_secs': wall_clock_secs,
                'perf_times': sampling_secs}
     
        exp_info = {
             "date_str": date_str,
             "dataset_name": dataset_name,
             "model_name":model_name,
             "split_index": split_index,
             "train_test_split": train_test_split,
             "num_inducing": num_inducing
             }
        
        # One dict with all info + metrics
        experiment_dict = {**exp_info,
                            **metrics}
         
        print(dataset_name, ': Test RMSE:', rmse_test)
        print(dataset_name, ': Test NLPD:', nlpd_test)
        
        ###### Saving losses / performance metrics / traces for analysis and reporting #########
        
        date_str = datetime.now().strftime('%b_%d')
        
        log_path = LOG_DIR / date_str / experiment_name_tf(**exp_info)
      
        results_filename = f"{log_path}__.json"
        with open(results_filename, "w") as fp:
             json.dump(experiment_dict, fp, indent=4)
       
def main(args: argparse.Namespace):
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
        date_str = datetime.now().strftime('%b_%d')
        save_dir = LOG_DIR / date_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("Training GPR in parallel. GPUs may run out of memory, use CPU if this happens. "
            "To use CPU only, set environment variable CUDA_VISIBLE_DEVICES=-1")
    
        Parallel(args.n_jobs)(
            delayed(single_run)(
                dataset_name,
                split_index,
                args.model_name,
                args.num_inducing,
                args.test_train_split,
                date_str
            ) for dataset_name, split_index in product(DATASETS, SPLIT_INDICES)
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='SGPMC')
    parser.add_argument("--num_inducing", type=int, default=100)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--test_train_split", type=float, default=0.8)
    parser.add_argument("--n_jobs", type=int, default=1)
    
    arguments = parser.parse_args()
    
    main(arguments)
    
    #single_run('Boston', 4, 'SparseGPR', 50, 0.9, 2000, '30_12')
    #single_run('Boston',4, 'StochasticVariationalGP',506, 0.9, 20000, '06_10')
    #single_run('Boston',4, 'SparseGPR',500, 0.9, 20000, '06_10')
    
 