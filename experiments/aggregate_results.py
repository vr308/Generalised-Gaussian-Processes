#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script with helper functions to aggregate metrics stored in log / *date_str* / 

in a tabular format - mainly aggregating over splits (there is a .json for each split)

"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import json
import numpy as np
import pandas as pd

from utils import dataset as uci_datasets
from utils.config import LOG_DIR

def parse_command_line_args() -> Namespace:
    """ As it says on the tin """
    parser = ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        help="Identify which model jsons to read-up e.g. 'StochasticVariationalGP",
        default='SparseGPR'
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="*",
        help="Name of the log directories containing experimental results, format is e.g. Feb_12."
             "Specify more than one in case the experiment run overnight.",
        default=["Dec_30", "Dec_31"]
    )

    return parser.parse_args()


def get_dataset_class(dataset):
    return getattr(uci_datasets, dataset)


def standard_error(x):
    return np.std(x) / np.sqrt(len(x))


if __name__ == "__main__":
    
    args = parse_command_line_args()

    # Read the results files
    data = []
    paths = []
    for date in args.dates:
        results_dir = LOG_DIR / date
        results_files = list(results_dir.glob(f'*{args.model_id}*.json'))
        paths.extend(results_files)
        for result_file in results_files:
            with open(result_file) as json_file:
                data.append(json.load(json_file))

    if len(paths) == 0:
        print("No results found - try a different results suffix")
        sys.exit(2)

    # Process the data
    df = pd.DataFrame.from_records(data)
    df["paths"] = paths
    df["N"] = [get_dataset_class(d).N for d in df.dataset_name]
    df["D"] = [get_dataset_class(d).D for d in df.dataset_name]
    
    # # Aggregate the tables
    metrics = [col for col in list(df.columns) if (col.endswith('_rmse') or col.endswith('_nlpd'))]  
    summary = df.groupby(['dataset_name'])[metrics].agg(['mean', standard_error])

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(summary)
