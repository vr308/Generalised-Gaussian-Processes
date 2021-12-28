#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression training over 10 splits different models.
Baseline, SGPR, SGHMC

"""

import numpy as np
from typing import Dict, List, Type
from joblib import delayed, Parallel
import argparse
import json
from collections import defaultdict
from itertools import product
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

import torch 
import gpytorch 
import utils.dataset as uci_datasets
from utils.dataset import Dataset
from utils.config import LOG_DIR

# Models
from models.sgpr import SparseGPR
from models.svgp import StochasticVariationalGP
from models.bayesian_svgp import BayesianStochasticVariationalGP
from models.bayesian_sgpr_hmc import BayesianSparseGPR_HMC

# Metrics and Viz
from utils.metrics import rmse, nlpd, get_trainable_param_names
from utils.visualisation import visualise_posterior, visualise_mixture_posterior_samples
import matplotlib.pyplot as plt

gpytorch.settings.cholesky_jitter(float=1e-4)
plt.style.use('seaborn-muted')

#### Global variables 

DATASETS = ["Boston", "Concrete", "Energy", "Power", "Kin8mn" ,"Naval", "Yacht", "WineRed"]
MODEL_NAMES = ['SparseGPR', 'BayesianSGPR_HMC', 'StochasticVariationalGP', 'BayesianStochasticVariationalGP']
MODEL_CLASS = [
            SparseGPR,
            BayesianSparseGPR_HMC,
            StochasticVariationalGP,
            BayesianStochasticVariationalGP
            ]
SPLIT_INDICES = [*range(10)]

model_dictionary = dict(zip(MODEL_NAMES, MODEL_CLASS))

##### Methods

class ExperimentName:
    def __init__(self, base):
        self.s = base

    def add(self, name, value):
        self.s += f"_{name}-{value}"
        return self

    def get(self):
        return self.s

def experiment_name(
    date_str,
    dataset_name,
    model_name,
    split_index,
    train_test_split,
    num_inducing,
    max_iter
):
    return (
        ExperimentName(date_str)
        .add("dataset", dataset_name)
        .add("model_name", model_name)
        .add("split", split_index)
        .add("frac", train_test_split)
        .add("num_inducing", num_inducing)
        .add("max_iter", max_iter)
        .get()
    )

def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)

def single_run(
    dataset_name: str,
    split_index: int,
    model_name: str,
    num_inducing: int,
    train_test_split: float,
    max_iter: int,
    date_str: str,
   
):
        print(f"Running on {dataset_name} (Split {split_index})" )
    
        ###### Loading a data split #################
        
        dataset = get_dataset_class(dataset_name)(split=split_index, prop=train_test_split)
        X_train, Y_train, X_test, Y_test = dataset.X_train, dataset.Y_train, dataset.X_test, dataset.Y_test
        
        model_class = model_dictionary[model_name]
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        ## Fixed at X_train[np.random.randint(0,len(X_train), 200)]
        input_dim = X_train.shape[-1]
        #Z_init = torch.randn(num_inducing, input_dim)
        Z_init = X_train[np.random.randint(0,len(X_train), 200)]

        model = model_class(X_train, Y_train.flatten(), likelihood, Z_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        if model_name == 'SparseGPR':
        
            losses = model.train_model(optimizer, num_steps=max_iter)
            Y_test_pred = model.posterior_predictive(X_test)
            
        elif model_name in ('StochasticVariationalGP', 'BayesianStochasticVariationalGP'):
            
            train_dataset = TensorDataset(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

            test_dataset = TensorDataset(X_test, Y_test)
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
            
            losses = model.train_model(optimizer, train_loader, minibatch_size=100, num_epochs=50, combine_terms=True)
            
        ### Predictions
        
        Y_train_pred = model.posterior_predictive(X_train)
        Y_test_pred = model.posterior_predictive(X_test)

        ### Compute Metrics  
        
        rmse_train = np.round(rmse(Y_train_pred.loc, Y_train, dataset.Y_std).item(), 4)
        rmse_test = np.round(rmse(Y_test_pred.loc, Y_test, dataset.Y_std).item(), 4)
       
        nlpd_train = np.round(nlpd(Y_train_pred, Y_train, dataset.Y_std).item(), 4)
        nlpd_test = np.round(nlpd(Y_test_pred, Y_test, dataset.Y_std).item(), 4)
        
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
             "train_test_split": train_test_split 
             }
        
        # One dict with all info + metrics
        experiment_dict = {**exp_info,
                            **metrics}
         
        print(dataset_name, ': Test RMSE:', rmse_test)
        print(dataset_name, ': Test NLPD:', nlpd_test)
        
        date_str = datetime.now().strftime('%b_%d')
        
        log_path = LOG_DIR / date_str / experiment_name(**exp_info)
        filename = f"{log_path}__.json"
        
        save_dir = LOG_DIR / date_str
        save_dir.mkdir(parents=True, exist_ok=True)
         
        with open(filename, "w") as fp:
             json.dump(experiment_dict, fp, indent=4)
       
# def main(args: argparse.Namespace):
#         np.random.seed(args.seed)
#         tf.random.set_seed(args.seed)
    
#         date_str = args.date_str or datetime.now().strftime('%b%d')
#         save_dir = LOG_DIR / date_str / "results"
#         save_dir.mkdir(parents=True, exist_ok=True)
    
#         if args.rbf_baseline_experiment:
#             expressions = defaultdict(lambda: ([["RBF"]], [1.0], 0.0))
#             expressions_save_path = None
#         else:
#             if args.decoder_name is None:
#                 # Use classifier-transformer only
#                 suffix = "expressions" + "_max_prod_" + str(args.max_terms) + ".json"
#                 expressions_save_path = save_dir / suffix
#             else:
#                 # Use full-KITT
#                 suffix = "captions" + "_max_prod_" + str(args.max_terms) + ".json"
#                 expressions_save_path = save_dir / suffix
#             with open(expressions_save_path, "r") as f:
#                 expressions = json.load(f)
    
#         args_save_path = save_dir / "run_multi_train_parallel_arguments.txt"
#         with open(str(args_save_path), "w") as file:
#             file.write(get_args_string(args))
    
#         print(
#             "Training GPR in parallel. GPUs may run out of memory, use CPU if this happens. "
#             "To use CPU only, set environment variable CUDA_VISIBLE_DEVICES=-1"
#         )
    
#         Parallel(args.n_jobs)(
#             delayed(single_run)(
#                 dsn,
#                 i,
#                 args.test_train_split,
#                 *expressions[f"{dsn.lower()}-{i}"],
#                 args.max_iters,
#                 date_str,
#                 args.encoder_name,
#                 args.decoder_name,
#                 str(expressions_save_path),
#                 args.subset,
#                 args.rbf_baseline_experiment
#             ) for dsn, i in product(DATASETS, SPLIT_INDICES)
#         )

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=5)
    # parser.add_argument("--max_iters", type=int, default=1000)
    # parser.add_argument("--test_train_split", type=float, default=0.9)
    # parser.add_argument("--date_str", type=str, default=None)
    # parser.add_argument("--n_jobs", type=int, default=-1)
    # arguments = parser.parse_args()
    # main(arguments)
    single_run('Boston',4, 'BayesianStochasticVariationalGP',506, 0.9, 20000, '06_10')
    #single_run('Boston',4, 'StochasticVariationalGP',506, 0.9, 20000, '06_10')
    #single_run('Boston',4, 'SparseGPR',500, 0.9, 20000, '06_10')